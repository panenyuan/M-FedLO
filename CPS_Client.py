import random
import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import copy
from dataloaders.dataset import TwoStreamBatchSampler

from utils import losses,  ramps
from val_2D import test_single_volume
import numpy as np
import os
from networks.net_factory import model_norm

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * ramps.sigmoid_rampup(epoch, 240.0)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)

class Client_train(object):
    # 构造函数
    def __init__(self, conf, model1, model2, train_dataset,val_dataset, id=-1):
        # 读取配置文件
        self.conf = conf
        self.num_classes = conf["num_classes"]
        # 根据配置文件获取客户端本地模型（一般由服务器传输）
        self.local_model1 = copy.deepcopy(model1)
        self.local_model2 = copy.deepcopy(model2)
        # 客户端ID
        self.client_id = id
        self.batch_size = conf["batch_size"]
        self.labeled_bs = conf["labeled_bs"]
        # 客户端本地数据集
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.base_lr = conf['lr']
        self.iter_num = 0
        self.optimizer1 = optim.Adam(self.local_model1.parameters(), lr=self.conf['lr'])
        self.scheduler1_1 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, T_max=self.conf["local_epochs"][0])

        self.scheduler1_2 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, T_max=self.conf["local_epochs"][1])

        self.optimizer2 = optim.Adam(self.local_model2.parameters(), lr=self.conf['lr'])
        self.scheduler2_1 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, T_max=self.conf["local_epochs"][0])

        self.scheduler2_2 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, T_max=self.conf["local_epochs"][1])
        total_num = len(self.train_dataset)
        self.labeled_num = int(total_num*conf["labeled_num"])
        labeled_idxs = list(range(0, self.labeled_num))
        unlabeled_idxs = list(range(self.labeled_num, total_num))
        # print(self.batch_size - self.labeled_bs)
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, self.batch_size, self.batch_size - self.labeled_bs)
        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler,
                                       num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                                     num_workers=1)

    def local_train(self, signal, model1, model2):
        best_performance1, best_performance2 = 0, 0
        ce_loss = CrossEntropyLoss()
        dice_loss = losses.DiceLoss(self.num_classes)
        snapshot_path = r'../model/FL_model/CPS/safety-2'
        # r'../model/FL_model/MT/3_client' ALL
        if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
        for name, param in model1.state_dict().items():
            # 用服务器下发的全局模型覆盖本地模型
            self.local_model1.state_dict()[name].copy_(param.clone())
        for name, param in model2.state_dict().items():
            # 用服务器下发的全局模型覆盖本地模型
            self.local_model2.state_dict()[name].copy_(param.clone())

        iterator = tqdm(range(self.conf["local_epochs"][signal]), ncols=70)
        self.local_model1.train()
        self.local_model2.train()
        for e in iterator:
            for batch_id, sampled_batch in enumerate(self.train_loader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                outputs1 = self.local_model1(volume_batch)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                outputs2 = self.local_model2(volume_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)
                consistency_weight = get_current_consistency_weight(e)

                loss1 = 0.5 * (ce_loss(outputs1[:self.labeled_bs], label_batch[:][:self.labeled_bs].long()) + dice_loss(
                    outputs_soft1[:self.labeled_bs], label_batch[:self.labeled_bs].unsqueeze(1)))
                loss2 = 0.5 * (ce_loss(outputs2[:self.labeled_bs], label_batch[:][:self.labeled_bs].long()) + dice_loss(
                    outputs_soft2[:self.labeled_bs], label_batch[:self.labeled_bs].unsqueeze(1)))

                pseudo_outputs1 = torch.argmax(outputs_soft1[self.labeled_bs:].detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(outputs_soft2[self.labeled_bs:].detach(), dim=1, keepdim=False)

                pseudo_supervision1 = ce_loss(outputs1[self.labeled_bs:], pseudo_outputs2)
                pseudo_supervision2 = ce_loss(outputs2[self.labeled_bs:], pseudo_outputs1)

                model1_loss = loss1 + consistency_weight * pseudo_supervision1
                model2_loss = loss2 + consistency_weight * pseudo_supervision2

                loss = model1_loss + model2_loss
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()

                self.iter_num+=1

            if signal:
                self.scheduler1_1.step()
                self.scheduler2_1.step()
            else:
                self.scheduler1_2.step()
                self.scheduler2_2.step()
            if self.conf["dp"]:
                my_model_norm1 = model_norm(model1, self.local_model1)

                my_model_norm2 = model_norm(model2, self.local_model2)
                # norm_scale = min(1, self.conf['C'] / (my_model_norm))
                if my_model_norm1 != 0:
                    norm_scale1 = min(1, self.conf['C'] / my_model_norm1)
                else:
                    print(f"my_model_norm1=0\n")
                    norm_scale1 = 1  # 设置默认值或采取其他操作

                if my_model_norm2 != 0:
                    norm_scale2 = min(1, self.conf['C'] / my_model_norm2)
                else:
                    print(f"my_model_norm2=0\n")
                    norm_scale2 = 1  # 设置默认值或采取其他操作

                # print(model_norm, norm_scale)
                for name, layer in self.local_model1.named_parameters():
                    clipped_difference = norm_scale1 * (layer.data - model1.state_dict()[name])
                    layer.data.copy_(model1.state_dict()[name] + clipped_difference)

                for name, layer in self.local_model2.named_parameters():
                    clipped_difference = norm_scale2 * (layer.data - model2.state_dict()[name])
                    layer.data.copy_(model2.state_dict()[name] + clipped_difference)

            if e > 0 and e % 30 == 0:
                self.local_model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(self.val_loader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"],
                        self.local_model1, classes=self.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(self.val_loader)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                print(f'loss={loss},dice={performance1},hd95={mean_hd951}')

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
                    save_best = os.path.join(snapshot_path,
                                             f'{self.client_id}_local_model1.pth')
                    torch.save(self.local_model1.state_dict(), save_best)
                self.local_model1.train()
            if e > 0 and e % 30 == 0:
                self.local_model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(self.val_loader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"],
                        self.local_model2, classes=self.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(self.val_loader)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                print(f'loss={loss},dice={performance2},hd95={mean_hd951}')

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
                    save_best = os.path.join(snapshot_path,
                                             f'{self.client_id}_local_model2.pth')
                    torch.save(self.local_model2.state_dict(), save_best)
                self.local_model2.train()
        diff1, diff2 = dict(), dict()
        for name, data in self.local_model1.state_dict().items():
            # 计算训练后与训练前的差值
            diff1[name] = (data - model1.state_dict()[name])
        for name, data in self.local_model2.state_dict().items():
            # 计算训练后与训练前的差值
            diff2[name] = (data - model2.state_dict()[name])
        print("Client %d local train done" % self.client_id)
        diff1 = sorted(diff1.items(), key=lambda item: abs(torch.mean(item[1].float())), reverse=True)
        # sum1, sum2 = 0, 0
        # for id, (name, data) in enumerate(diff):
        #     if id < 304:
        #         sum1 += torch.prod(torch.tensor(data.size()))
        #     else:
        #         sum2 += torch.prod(torch.tensor(data.size()))

        ret_size1 = int(self.conf["rate"] * len(diff1))

        diff2 = sorted(diff2.items(), key=lambda item: abs(torch.mean(item[1].float())), reverse=True)
        # sum1, sum2 = 0, 0
        # for id, (name, data) in enumerate(diff):
        #     if id < 304:
        #         sum1 += torch.prod(torch.tensor(data.size()))
        #     else:
        #         sum2 += torch.prod(torch.tensor(data.size()))

        ret_size2 = int(self.conf["rate"] * len(diff2))
        # 客户端返回差值
        # return diff1,diff2
        return dict(diff1[:ret_size1]), dict(diff2[:ret_size2])