import numpy as np
from networks.net_factory import net_factory
import torch
from torch.optim.swa_utils import AveragedModel
from val_2D import test_single_volume, test_single_volume_ds

# 服务器类
class Server(object):
    # 定义构造函数
    def __init__(self, conf, eval_dataset):
        # 导入配置文件
        self.conf = conf
        # 根据配置文件获取模型
        self.global_model = net_factory(self.conf["model_name"], in_chns=1, class_num=self.conf['num_classes'])
        if self.conf['use_swa']:
            self.swa_model = AveragedModel(self.global_model)
        else:
            self.swa_model = None

        # 生成测试集合加载器
        self.dataset = []
        for i in range(3):
            self.dataset.extend(eval_dataset[i])

        self.eval_loader = torch.utils.data.DataLoader(
            self.dataset,
            # 根据配置文件设置单个批次大小（32）
            batch_size=1,
            # 打乱数据集
            shuffle=False,
            num_workers=1
        )

    # 模型聚合函数
    # weight_accumulator 存储了每个客户端上传参数的变化值
    def model_aggregate(self, signal, weight_accumulator, cnt):
        # 遍历服务器的全局模型
        for name, data in self.global_model.state_dict().items():
            if name in weight_accumulator and cnt[name] > 0:
            # 更新每一次乘以配置文件中的学习率
                update_per_layer = weight_accumulator[name] * self.conf["lambda"] * (1.0 / cnt[name])
                # 累加
                # 添加的差分隐私部分
                if self.conf['dp']:
                    sigma = self.conf['sigma'][signal]
                    if torch.cuda.is_available():
                        noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                    else:
                        noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                    # if update_per_layer.dtype == torch.int64:
                    #     update_per_layer = update_per_layer.to(torch.float64)
                    # update_per_layer = update_per_layer.float()
                    if update_per_layer.dtype == torch.float32:
                    # print(f'update_per_layer.dtype={update_per_layer.dtype}')
                        update_per_layer.add_(noise)

                if data.type() != update_per_layer.type():
                    # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)
                    # 更新SWA模型

    def model_aggregate_urpc(self, signal, weight_accumulator, swa_sign, cnt):
        # 遍历服务器的全局模型
        for name, data in self.global_model.state_dict().items():
            # 更新每一次乘以配置文件中的学习率
            if name in weight_accumulator and cnt[name] > 0:
                update_per_layer = weight_accumulator[name] * self.conf["lambda"] * (1.0 / cnt[name])
                # 累加
                if self.conf['dp']:
                    sigma = self.conf['sigma'][signal]
                    if torch.cuda.is_available():
                        noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                    else:
                        noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)

                    update_per_layer.add_(noise)
                if data.type() != update_per_layer.type():
                    # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)
                    # 更新SWA模型
                if swa_sign:
                    self.swa_model.update_parameters(self.global_model)

    # 模型评估函数
    def model_eval(self):
        # 开启模型评估模式
        self.global_model.eval()
        # 遍历评估数据集合
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(self.eval_loader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"], self.global_model, classes=14)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(self.eval_loader)

        dice_score = np.mean(metric_list, axis=0)[0]

        mean_hd95 = np.mean(metric_list, axis=0)[1]

        return dice_score, mean_hd95

    def model_eval3(self,swa_sign):
        # 开启模型评估模式

        # 遍历评估数据集合
        model_to_evaluate = self.swa_model if swa_sign else self.global_model
        model_to_evaluate.eval()
        metric_list_d1, metric_list_d2 = 0.0, 0.0
        for i_batch, sampled_batch in enumerate(self.eval_loader):
            metric_i_d1, metric_i_d2 = test_single_volume_ds(
                sampled_batch["image"], sampled_batch["label1"],
                sampled_batch["label2"], model_to_evaluate, classes=self.conf['num_classes'], sign=swa_sign)
            metric_list_d1 += np.array(metric_i_d1)
            metric_list_d2 += np.array(metric_i_d2)
        metric_list_d1, metric_list_d2 = metric_list_d1 / len(self.eval_loader), metric_list_d2 / len(
            self.eval_loader)
        dice1, dice2 = np.mean(metric_list_d1, axis=0)[0], \
                                         np.mean(metric_list_d2, axis=0)[0]

        mean_hd95_d1, mean_hd95_d2 = np.mean(metric_list_d1, axis=0)[1], \
                                     np.mean(metric_list_d2, axis=0)[1]
        mean_hd95 = (mean_hd95_d1 + mean_hd95_d2) / 2
        return dice1, dice2, mean_hd95

class Server_CPS(object):
    # 定义构造函数
    def __init__(self, conf, eval_dataset):
        # 导入配置文件
        self.conf = conf
        # 根据配置文件获取模型
        self.global_model1 = net_factory(self.conf["model_name"], in_chns=1, class_num=self.conf['num_classes'])
        self.global_model2 = net_factory(self.conf["model_name"], in_chns=1, class_num=self.conf['num_classes'])

        # 生成测试集合加载器
        self.dataset = []
        for i in range(3):
            self.dataset.extend(eval_dataset[i])

        self.eval_loader = torch.utils.data.DataLoader(
            self.dataset,
            # 根据配置文件设置单个批次大小（32）
            batch_size=1,
            # 打乱数据集
            shuffle=False,
            num_workers=1
        )

    # 模型聚合函数
    # weight_accumulator 存储了每个客户端上传参数的变化值
    def model_aggregate(self, signal,weight_accumulator1, weight_accumulator2, cnt1, cnt2):
        # 遍历服务器的全局模型
        for name, data in self.global_model1.state_dict().items():
            # 更新每一次乘以配置文件中的学习率
            if name in weight_accumulator1 and cnt1[name] > 0:
                update_per_layer = weight_accumulator1[name] * self.conf["lambda"] * (1.0 / cnt1[name])
                # 累加
                if self.conf['dp']:
                    sigma = self.conf['sigma'][signal]
                    if torch.cuda.is_available():
                        noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                    else:
                        noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                    update_per_layer.add_(noise)
                if data.type() != update_per_layer.type():
                    # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)
                    # 更新SWA模型

        for name, data in self.global_model2.state_dict().items():
            # 更新每一次乘以配置文件中的学习率
            if name in weight_accumulator2 and cnt2[name] > 0:
                update_per_layer = weight_accumulator2[name] * self.conf["lambda"] * (1.0 / cnt2[name])
            # 累加
                if self.conf['dp']:
                    sigma = self.conf['sigma'][signal]
                    if torch.cuda.is_available():
                        noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                    else:
                        noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                    update_per_layer.add_(noise)
                if data.type() != update_per_layer.type():
                    # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)
                    # 更新SWA模型

    # 模型评估函数
    def model_eval(self):
        # 开启模型评估模式
        self.global_model1.eval()
        self.global_model2.eval()
        # 遍历评估数据集合
        metric_list1 = 0.0
        for i_batch, sampled_batch in enumerate(self.eval_loader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"], self.global_model1, classes=14)
            metric_list1 += np.array(metric_i)
        metric_list1 = metric_list1 / len(self.eval_loader)

        dice_score1 = np.mean(metric_list1, axis=0)[0]

        mean_hd951 = np.mean(metric_list1, axis=0)[1]

        metric_list2 = 0.0
        for i_batch, sampled_batch in enumerate(self.eval_loader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"], self.global_model2, classes=14)
            metric_list2 += np.array(metric_i)
        metric_list2 = metric_list2 / len(self.eval_loader)

        dice_score2 = np.mean(metric_list2, axis=0)[0]

        mean_hd952 = np.mean(metric_list2, axis=0)[1]

        return dice_score1, mean_hd951, dice_score2, mean_hd952

class Server_Only(object):
    # 定义构造函数
    def __init__(self, conf, eval_dataset):
        # 导入配置文件
        self.conf = conf
        # 根据配置文件获取模型
        self.global_model = net_factory(self.conf["model_name"], in_chns=1, class_num=self.conf['num_classes'])


        # 生成测试集合加载器
        self.dataset = []

        self.dataset.extend(eval_dataset[0])

        self.eval_loader = torch.utils.data.DataLoader(
            self.dataset,
            # 根据配置文件设置单个批次大小（32）
            batch_size=1,
            # 打乱数据集
            shuffle=False,
            num_workers=1
        )

    # 模型聚合函数
    # weight_accumulator 存储了每个客户端上传参数的变化值
    def model_aggregate(self, weight_accumulator):
        # 遍历服务器的全局模型
        for name, data in self.global_model.state_dict().items():
            # 更新每一次乘以配置文件中的学习率
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            # 累加
            if data.type() != update_per_layer.type():
                # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
                # 更新SWA模型

    def model_aggregate_urpc(self, weight_accumulator):
        # 遍历服务器的全局模型
        for name, data in self.global_model.state_dict().items():
            # 更新每一次乘以配置文件中的学习率
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            # 累加
            if data.type() != update_per_layer.type():
                # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
                # 更新SWA模型
            # if swa_sign:
            #     self.swa_model.update_parameters(self.global_model)

    # 模型评估函数
    def model_eval(self):
        # 开启模型评估模式
        self.global_model.eval()
        # 遍历评估数据集合
        metric_list = 0.0
        for i_batch, sampled_batch in enumerate(self.eval_loader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"], self.global_model, classes=14)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(self.eval_loader)

        dice_score = np.mean(metric_list, axis=0)[0]

        mean_hd95 = np.mean(metric_list, axis=0)[1]

        return dice_score, mean_hd95

    def model_eval3(self,swa_sign):
        # 开启模型评估模式
        if self.conf['use_swa']:
            self.swa_model = AveragedModel(self.global_model)
        else:
            self.swa_model = None
        # 遍历评估数据集合
        model_to_evaluate = self.swa_model if swa_sign else self.global_model
        model_to_evaluate.eval()
        metric_list_d1, metric_list_d2 = 0.0, 0.0
        for i_batch, sampled_batch in enumerate(self.eval_loader):
            metric_i_d1, metric_i_d2 = test_single_volume_ds(
                sampled_batch["image"], sampled_batch["label1"],
                sampled_batch["label2"], model_to_evaluate, classes=self.conf['num_classes'], sign=swa_sign)
            metric_list_d1 += np.array(metric_i_d1)
            metric_list_d2 += np.array(metric_i_d2)
        metric_list_d1, metric_list_d2 = metric_list_d1 / len(self.eval_loader), metric_list_d2 / len(
            self.eval_loader)
        dice1, dice2 = np.mean(metric_list_d1, axis=0)[0], \
                                         np.mean(metric_list_d2, axis=0)[0]

        mean_hd95_d1, mean_hd95_d2 = np.mean(metric_list_d1, axis=0)[1], \
                                     np.mean(metric_list_d2, axis=0)[1]
        mean_hd95 = (mean_hd95_d1 + mean_hd95_d2) / 2
        return dice1, dice2, mean_hd95

class Server_CPS_Only(object):
    # 定义构造函数
    def __init__(self, conf, eval_dataset):
        # 导入配置文件
        self.conf = conf
        # 根据配置文件获取模型
        self.global_model1 = net_factory(self.conf["model_name"], in_chns=1, class_num=self.conf['num_classes'])
        self.global_model2 = net_factory(self.conf["model_name"], in_chns=1, class_num=self.conf['num_classes'])

        # 生成测试集合加载器
        self.dataset = []
        self.dataset.extend(eval_dataset[0])

        self.eval_loader = torch.utils.data.DataLoader(
            self.dataset,
            # 根据配置文件设置单个批次大小（32）
            batch_size=1,
            # 打乱数据集
            shuffle=False,
            num_workers=1
        )

    # 模型聚合函数
    # weight_accumulator 存储了每个客户端上传参数的变化值
    def model_aggregate(self, weight_accumulator1, weight_accumulator2):
        # 遍历服务器的全局模型
        for name, data in self.global_model1.state_dict().items():
            # 更新每一次乘以配置文件中的学习率
            update_per_layer = weight_accumulator1[name] * self.conf["lambda"]
            # 累加
            if data.type() != update_per_layer.type():
                # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
                # 更新SWA模型

        for name, data in self.global_model2.state_dict().items():
            # 更新每一次乘以配置文件中的学习率
            update_per_layer = weight_accumulator2[name] * self.conf["lambda"]
            # 累加
            if data.type() != update_per_layer.type():
                # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
                # 更新SWA模型

    # 模型评估函数
    def model_eval(self):
        # 开启模型评估模式
        self.global_model1.eval()
        self.global_model2.eval()
        # 遍历评估数据集合
        metric_list1 = 0.0
        for i_batch, sampled_batch in enumerate(self.eval_loader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"], self.global_model1, classes=14)
            metric_list1 += np.array(metric_i)
        metric_list1 = metric_list1 / len(self.eval_loader)

        dice_score1 = np.mean(metric_list1, axis=0)[0]

        mean_hd951 = np.mean(metric_list1, axis=0)[1]

        metric_list2 = 0.0
        for i_batch, sampled_batch in enumerate(self.eval_loader):
            metric_i = test_single_volume(
                sampled_batch["image"], sampled_batch["label"], self.global_model2, classes=14)
            metric_list2 += np.array(metric_i)
        metric_list2 = metric_list2 / len(self.eval_loader)

        dice_score2 = np.mean(metric_list2, axis=0)[0]

        mean_hd952 = np.mean(metric_list2, axis=0)[1]

        return dice_score1, mean_hd951, dice_score2, mean_hd952