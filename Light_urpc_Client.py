import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.dataset import TwoStreamBatchSampler
from val_2D import test_single_volume_ds
import torch.nn.functional as F
from networks.net_factory import Light_LOSS,New_Loss
from torch.optim.swa_utils import AveragedModel, SWALR
import copy
from networks.net_factory import model_norm

def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)

class Client_train(object):
    # 构造函数
    def __init__(self, conf, model, train_dataset, val_dataset, id=-1):
        # 读取配置文件
        self.conf = conf
        self.num_classes = conf["num_classes"]
        # 根据配置文件获取客户端本地模型（一般由服务器传输）
        self.local_model = copy.deepcopy(model)
        # self.prev_model = copy.deepcopy(model)
        # 客户端ID
        self.client_id = id
        self.batch_size = conf["batch_size"]
        self.labeled_bs = conf["labeled_bs"]
        self.lambid = conf["lambid"]
        # 客户端本地数据集
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.base_lr = conf['lr']
        total_num = len(self.train_dataset)
        self.labeled_num = int(total_num*conf["labeled_num"])
        self.optimizer = optim.AdamW(self.local_model.parameters(), lr=self.conf['lr'],weight_decay=1e-3)
        self.scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.conf["local_epochs"][0])
        self.scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.conf["local_epochs"][1])

        labeled_idxs = list(range(0, self.labeled_num))
        unlabeled_idxs = list(range(self.labeled_num, total_num))
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, self.batch_size, self.batch_size - self.labeled_bs)
        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler,
                                       num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                                     num_workers=1)
        self.total_epoch_num = 0

    def light_local_train(self, signal, model):
        best_performance = 0
        snapshot_path = r'../model/FL_model/Light_URPC/safety-21'
        # r'../model/FL_model/URPC/ALL' 3_client
        if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
        # if self.conf['old_nets_pool'] and self.total_epoch_num > 0:
        #     for name, param in self.local_model.state_dict().items():
        #         # 用本地模型覆盖上一轮次的本地模型
        #         self.prev_model.state_dict()[name].copy_(param.clone())
        for name, param in model.state_dict().items():
            # 用服务器下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        iterator = tqdm(range(self.conf["local_epochs"][signal]), ncols=70)
        self.local_model.train()

        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.train_loader):
                volume_batch, label1_batch, label2_batch = \
                    sampled_batch['image'], sampled_batch['label1'], sampled_batch['label2']
                volume_batch, label1_batch, label2_batch = \
                    volume_batch.cuda(), label1_batch.cuda(), label2_batch.cuda()
                # loss开始
                if self.conf['old_nets_pool']:
                    outputs_d1, outputs_d2 = self.local_model(volume_batch)
                    m1, m2 = model(volume_batch[self.labeled_bs:])
                    loss_d1 = New_Loss(self.conf, outputs_d1, m1, self.num_classes[0], label1_batch, epoch_num)
                    loss_d2 = New_Loss(self.conf, outputs_d2, m2, self.num_classes[1], label2_batch, epoch_num)
                    loss = self.conf['lambid'] * loss_d1 + (1 - self.conf['lambid']) * loss_d2
                else:
                    outputs_d1, outputs_d2 = self.local_model(volume_batch)
                    loss_d2 = Light_LOSS(self.conf, outputs_d2, self.num_classes[1], label2_batch, epoch_num)
                    loss_d1 = Light_LOSS(self.conf, outputs_d1, self.num_classes[0], label1_batch, epoch_num)

                    loss = self.conf['lambid'] * loss_d1 + (1 - self.conf['lambid']) * loss_d2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.total_epoch_num+=1
            if signal:
                self.scheduler_1.step()
            else:
                self.scheduler_2.step()
            if self.conf["dp"]:
                my_model_norm = model_norm(model, self.local_model)

                # norm_scale = min(1, self.conf['C'] / (my_model_norm))
                if my_model_norm != 0:
                    norm_scale = min(1, self.conf['C'] / my_model_norm)
                else:
                    print(f"my_model_norm=0\n")
                    norm_scale = 1  # 设置默认值或采取其他操作
                # print(model_norm, norm_scale)
                for name, layer in self.local_model.named_parameters():
                    clipped_difference = norm_scale * (layer.data - model.state_dict()[name])
                    layer.data.copy_(model.state_dict()[name] + clipped_difference)
            if self.total_epoch_num > 0 and self.total_epoch_num % 30 == 0:
                self.local_model.eval()
                metric_list_d1, metric_list_d2 = 0.0, 0.0
                for i_batch, sampled_batch in enumerate(self.val_loader):
                    metric_i_d1, metric_i_d2 = test_single_volume_ds(
                        sampled_batch["image"], sampled_batch["label1"],
                        sampled_batch["label2"], self.local_model, classes=self.num_classes, sign=0)
                    metric_list_d1 += np.array(metric_i_d1)
                    metric_list_d2 += np.array(metric_i_d2)
                metric_list_d1, metric_list_d2 = metric_list_d1 / len(self.val_loader), metric_list_d2 / len(
                    self.val_loader)

                performance_d1, performance_d2 = np.mean(metric_list_d1, axis=0)[0], \
                                                 np.mean(metric_list_d2, axis=0)[0]
                print(f'loss_d1={loss_d1},loss_d2={loss_d2}')
                # print(f'ce_loss(logits, labels)={ce_loss(logits, labels)}')
                print(f'd1={performance_d1},d2={performance_d2}')

                performance = (performance_d1 + performance_d2) / 2

                mean_hd95_d1, mean_hd95_d2 = np.mean(metric_list_d1, axis=0)[1], \
                                             np.mean(metric_list_d2, axis=0)[1]
                if performance > best_performance:
                    best_performance = performance
                    save_best = os.path.join(snapshot_path,
                                             f'{self.client_id}_local_model.pth')
                    torch.save(self.local_model.state_dict(), save_best)
                self.local_model.train()

        diff = dict()
        for name, data in self.local_model.state_dict().items():
                # 计算训练后与训练前的差值
                diff[name] = (data - model.state_dict()[name])
        diff = sorted(diff.items(), key=lambda item: abs(torch.mean(item[1].float())), reverse=True)


        ret_size = int(self.conf["rate"] * len(diff))
        print("Client %d local train done" % self.client_id)

        return dict(diff[:ret_size])

    def light_local_train_only(self, model):
        best_performance = 0
        snapshot_path = r'../model/FL_model/Light_URPC/ALL111'
        if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)

        for name, param in model.state_dict().items():
            # 用服务器下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        iterator = tqdm(range(self.conf["local_epochs"]), ncols=70)
        swa_model = AveragedModel(model)
        swa_start = 100  # 模型优化50轮之后开始使用swa
        swa_scheduler = SWALR(self.optimizer, swa_lr=1e-6)
        self.local_model.train()

        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.train_loader):
                volume_batch, label1_batch, label2_batch = \
                    sampled_batch['image'], sampled_batch['label1'], sampled_batch['label2']
                volume_batch, label1_batch, label2_batch = \
                    volume_batch.cuda(), label1_batch.cuda(), label2_batch.cuda()
                # loss开始

                outputs_d1, outputs_d2 = self.local_model(volume_batch)
                loss_d2 = Light_LOSS(self.conf, outputs_d2, self.num_classes[1], label2_batch, epoch_num)
                loss_d1 = Light_LOSS(self.conf, outputs_d1, self.num_classes[0], label1_batch, epoch_num)
                loss = self.conf['lambid'] * loss_d1 + (1 - self.conf['lambid']) * loss_d2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.total_epoch_num+=1
            if epoch_num > swa_start:
                swa_model.update_parameters(model)  # 更新SWA变量
                swa_scheduler.step()
            else:
                self.scheduler.step()
            if self.total_epoch_num > 0 and self.total_epoch_num % 30 == 0:
                self.local_model.eval()
                metric_list_d1, metric_list_d2 = 0.0, 0.0
                for i_batch, sampled_batch in enumerate(self.val_loader):
                    metric_i_d1, metric_i_d2 = test_single_volume_ds(
                        sampled_batch["image"], sampled_batch["label1"],
                        sampled_batch["label2"], self.local_model, classes=self.num_classes, sign=0)
                    metric_list_d1 += np.array(metric_i_d1)
                    metric_list_d2 += np.array(metric_i_d2)
                metric_list_d1, metric_list_d2 = metric_list_d1 / len(self.val_loader), metric_list_d2 / len(
                    self.val_loader)

                performance_d1, performance_d2 = np.mean(metric_list_d1, axis=0)[0], \
                                                 np.mean(metric_list_d2, axis=0)[0]
                print(f'loss_d1={loss_d1},loss_d2={loss_d2}')

                print(f'd1={performance_d1},d2={performance_d2}')

                performance = (performance_d1 + performance_d2) / 2

                mean_hd95_d1, mean_hd95_d2 = np.mean(metric_list_d1, axis=0)[1], \
                                             np.mean(metric_list_d2, axis=0)[1]
                if performance > best_performance:
                    best_performance = performance
                    save_best = os.path.join(snapshot_path,
                                             f'{self.client_id}_local_model.pth')
                    torch.save(swa_model.state_dict(), save_best)
                self.local_model.train()

        diff = dict()
        for name, data in self.local_model.state_dict().items():
                # 计算训练后与训练前的差值
                diff[name] = (data - model.state_dict()[name])
        print("Client %d local train done" % self.client_id)
        # 客户端返回差值
        return diff