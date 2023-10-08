import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from dataloaders.dataset import TwoStreamBatchSampler
from utils import losses, ramps
from val_2D import test_single_volume
from networks.net_factory import model_norm

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * ramps.sigmoid_rampup(epoch, 240.0)

def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)

class Client_train(object):
    # 构造函数
    def __init__(self, conf, model, train_dataset,val_dataset, id=-1):
        self.conf = conf
        self.num_classes = conf["num_classes"]
        # 根据配置文件获取客户端本地模型（一般由服务器传输）
        self.local_model = copy.deepcopy(model)
        # 客户端ID
        self.client_id = id
        self.batch_size = conf["batch_size"]
        self.labeled_bs = conf["labeled_bs"]
        self.lambid = conf["lambda"]
        # 客户端本地数据集
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.base_lr = conf['lr']
        self.total_epoch_num = 0
        total_num = len(self.train_dataset)
        self.labeled_num = int(total_num*conf["labeled_num"])
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])

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

    def local_train(self, signal, model):
        snapshot_path = r'../model/FL_model/EM/safety-2'
        ce_loss = CrossEntropyLoss()
        dice_loss = losses.DiceLoss(self.conf['num_classes'])
        if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
        for name, param in model.state_dict().items():
            # 用服务器下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        iterator = tqdm(range(self.conf["local_epochs"][signal]), ncols=70)
        self.local_model.train()

        for e in iterator:
            for batch_id, sampled_batch in enumerate(self.train_loader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                outputs = self.local_model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = ce_loss(outputs[:self.labeled_bs],
                                  label_batch[:][:self.labeled_bs].long())
                loss_dice = dice_loss(
                    outputs_soft[:self.labeled_bs], label_batch[:self.labeled_bs].unsqueeze(1))
                supervised_loss = 0.5 * (loss_dice + loss_ce)

                consistency_weight = get_current_consistency_weight(self.total_epoch_num)
                consistency_loss = losses.entropy_loss(outputs_soft, C=self.conf['num_classes'])
                loss = supervised_loss + consistency_weight * consistency_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if signal:
                self.scheduler_1.step()
            else:
                self.scheduler_2.step()
            self.total_epoch_num+=1

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
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(self.val_loader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"],
                        self.local_model, classes=self.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(self.val_loader)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                print(f'loss={loss},dice={performance},hd95={mean_hd95}')
                self.local_model.train()

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            # 计算训练后与训练前的差值
            diff[name] = (data - model.state_dict()[name])

        diff = sorted(diff.items(), key=lambda item: abs(torch.mean(item[1].float())), reverse=True)
        # sum1, sum2 = 0, 0
        # for id, (name, data) in enumerate(diff):
        #     if id < 304:
        #         sum1 += torch.prod(torch.tensor(data.size()))
        #     else:
        #         sum2 += torch.prod(torch.tensor(data.size()))

        ret_size = int(self.conf["rate"] * len(diff))
        print("Client %d local train done" % self.client_id)
        # 客户端返回差值
        # return diff

        return dict(diff[:ret_size])