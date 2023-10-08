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
    def __init__(self, conf, model, train_dataset,val_dataset, id=-1):
        # 读取配置文件
        self.conf = conf
        self.ict_alpha = 0.2
        self.num_classes = conf["num_classes"]
        # 根据配置文件获取客户端本地模型（一般由服务器传输）
        self.local_model = copy.deepcopy(model)
        # 客户端ID
        self.client_id = id
        self.batch_size = conf["batch_size"]
        self.labeled_bs = conf["labeled_bs"]
        # 客户端本地数据集
        self.train_dataset = train_dataset
        self.total_epoch_num = 0
        self.val_dataset = val_dataset
        self.base_lr = conf['lr']
        self.iter_num = 0
        self.optimizer = optim.AdamW(self.local_model.parameters(), lr=self.conf['lr'], weight_decay=1e-3)
        self.scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.conf["local_epochs"][0])

        self.scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.conf["local_epochs"][1])
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

    def local_train(self, signal, model):
        best_performance = 0
        ce_loss = CrossEntropyLoss()
        dice_loss = losses.DiceLoss(self.num_classes)
        snapshot_path = r'../model/FL_model/ICT/safety-2'
        # r'../model/FL_model/ICT/3_client' ALL
        if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
        for name, param in model.state_dict().items():
            # 用服务器下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())

        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        iterator = tqdm(range(self.conf["local_epochs"][signal]), ncols=70)
        self.local_model.train()
        for e in iterator:
            for batch_id, sampled_batch in enumerate(self.train_loader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                unlabeled_volume_batch = volume_batch[self.labeled_bs:]
                labeled_volume_batch = volume_batch[:self.labeled_bs]

                # ICT mix factors
                ict_mix_factors = np.random.beta(
                    self.ict_alpha, self.ict_alpha, size=(9, 1, 1, 1))
                ict_mix_factors = torch.tensor(
                    ict_mix_factors, dtype=torch.float).cuda()
                unlabeled_volume_batch_0 = unlabeled_volume_batch[0:(self.batch_size - self.labeled_bs) // 2, ...]
                unlabeled_volume_batch_1 = unlabeled_volume_batch[(self.batch_size - self.labeled_bs) // 2:, ...]
                # print(unlabeled_volume_batch_1.shape,ict_mix_factors.shape)
                # Mix images
                batch_ux_mixed = unlabeled_volume_batch_0 * \
                                 (1.0 - ict_mix_factors) + \
                                 unlabeled_volume_batch_1 * ict_mix_factors
                input_volume_batch = torch.cat(
                    [labeled_volume_batch, batch_ux_mixed], dim=0)
                outputs = self.local_model(input_volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_output_ux0 = torch.softmax(
                        ema_model(unlabeled_volume_batch_0), dim=1)
                    ema_output_ux1 = torch.softmax(
                        ema_model(unlabeled_volume_batch_1), dim=1)
                    batch_pred_mixed = ema_output_ux0 * \
                                       (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors

                loss_ce = ce_loss(outputs[:self.labeled_bs],
                                  label_batch[:self.labeled_bs][:].long())
                loss_dice = dice_loss(
                    outputs_soft[:self.labeled_bs], label_batch[:self.labeled_bs].unsqueeze(1))
                supervised_loss = 0.5 * (loss_dice + loss_ce)
                consistency_weight = get_current_consistency_weight(self.total_epoch_num)
                consistency_loss = torch.mean(
                    (outputs_soft[self.labeled_bs:] - batch_pred_mixed) ** 2)
                loss = supervised_loss + consistency_weight * consistency_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                update_ema_variables(self.local_model, ema_model, 0.99, self.iter_num)
                self.iter_num+=1
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

                # if performance > best_performance:
                #     best_performance = performance
                #     if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
                #     save_best = os.path.join(snapshot_path,
                #                              f'{self.client_id}_local_model.pth')
                #     torch.save(self.local_model.state_dict(), save_best)
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
