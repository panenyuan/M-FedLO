import argparse
import json
import os.path
import random
import time
from dataloaders import dataset1, dataset, dataset_all,dataset_all1
import MT_Client,EM_Client,CPS_Client,Light_urpc_Client,ICT_Client
from server import *

def MT_main():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/mt.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset1.get_dataset(
        conf["type"], conf["sign"],conf["patch_size"])
    # 启动服务器
    server = Server(conf, eval_datasets)
    epoch_signal = conf["global_epochs"] / 6
    # 定义客户端列表
    clients = []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(MT_Client.Client_train(conf, server.global_model, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        # for c in candidates:
        #     print(c.client_id)
        # 累计权重
        weight_accumulator = {}
        cnt = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
            cnt[name] = 0
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model.train()
        if e<epoch_signal:signal=0
        else:signal=1
        for c in candidates:
            diff = c.local_train(signal, server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                if name in diff:
                    weight_accumulator[name].add_(diff[name])
                    cnt[name] += 1
        # 模型参数聚合
        server.model_aggregate(signal, weight_accumulator, cnt)
        # 模型评估
    dice, hd95 = server.model_eval()
    print(f'dice={dice},hd95={hd95}')
    save_best = r'../model/FL_model/MT/safety-2'
    # r'../model/FL_model/MT/ALL'
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path = os.path.join(save_best, 'model.pth')
    torch.save(server.global_model.state_dict(), model_path)

def MT_main_Only():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/mt.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset_all1.get_dataset(
        conf["type"], conf["sign"],conf["patch_size"])
    # 启动服务器
    server = Server_Only(conf, eval_datasets)
    # 定义客户端列表
    clients = []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(MT_Client.Client_train(conf, server.global_model, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        # for c in candidates:
        #     print(c.client_id)
        # 累计权重
        weight_accumulator = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model.train()
        for c in candidates:
            diff = c.local_train(server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
        # 模型参数聚合
        server.model_aggregate(weight_accumulator)
        # 模型评估
    dice, hd95 = server.model_eval()
    print(f'dice={dice},hd95={hd95}')
    save_best = r'../model/FL_model/MT/ALL4'
    # r'../model/FL_model/MT/ALL'
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path = os.path.join(save_best, 'model.pth')
    torch.save(server.global_model.state_dict(), model_path)

def EM_main():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/em.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset1.get_dataset(
        conf["type"], conf["sign"],conf["patch_size"])
    # train_datasets, eval_datasets = dataset_all1.get_dataset(
    #     conf["type"], conf["sign"], conf["patch_size"])
    # 启动服务器
    server = Server(conf, eval_datasets)
    epoch_signal = conf["global_epochs"] / 6
    # 定义客户端列表
    clients = []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(EM_Client.Client_train(conf, server.global_model, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        weight_accumulator = {}
        cnt = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
            cnt[name] = 0
        # 遍历选中的客户端，每个客户端本地进行训练
        if e < epoch_signal: signal = 0
        else: signal = 1
        server.global_model.train()
        for c in candidates:
            diff = c.local_train(signal, server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                if name in diff:
                    weight_accumulator[name].add_(diff[name])
                    cnt[name] += 1
        # 模型参数聚合
        server.model_aggregate(signal, weight_accumulator, cnt)
        dice, hd95 = server.model_eval()
        print(f'dice={dice},hd95={hd95}')
    save_best = r'../model/FL_model/EM/safety-2'
    # r'../model/FL_model/EM/3_client'ALL
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path = os.path.join(save_best, 'model.pth')
    torch.save(server.global_model.state_dict(), model_path)

def EM_main_only():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/em.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset_all1.get_dataset(
        conf["type"], conf["sign"],conf["patch_size"])
    # train_datasets, eval_datasets = dataset_all1.get_dataset(
    #     conf["type"], conf["sign"], conf["patch_size"])
    # 启动服务器
    server = Server_Only(conf, eval_datasets)
    # 定义客户端列表
    clients = []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(EM_Client.Client_train(conf, server.global_model, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])

        weight_accumulator = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model.train()
        for c in candidates:
            diff = c.local_train(server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
        # 模型参数聚合
        server.model_aggregate(weight_accumulator)
    dice, hd95 = server.model_eval()
    print(f'dice={dice},hd95={hd95}')
    save_best = r'../model/FL_model/EM/ALL4'
    # r'../model/FL_model/EM/3_client'ALL
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path = os.path.join(save_best, 'model.pth')
    torch.save(server.global_model.state_dict(), model_path)

def ICT_main():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/ict.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset1.get_dataset(
        conf["type"], conf["sign"],conf["patch_size"])

    server = Server(conf, eval_datasets)
    epoch_signal = conf["global_epochs"] / 6
    # 定义客户端列表
    clients = []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(ICT_Client.Client_train(conf, server.global_model, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        # for c in candidates:
        #     print(c.client_id)
        # 累计权重
        weight_accumulator = {}
        cnt = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
            cnt[name] = 0
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model.train()
        if e<epoch_signal:signal=0
        else:signal=1
        for c in candidates:
            diff = c.local_train(signal, server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                if name in diff:
                    weight_accumulator[name].add_(diff[name])
                    cnt[name] += 1
        # 模型参数聚合
        server.model_aggregate(signal, weight_accumulator, cnt)
        dice, hd95 = server.model_eval()
        print(f'dice={dice},hd95={hd95}')
    save_best = r'../model/FL_model/ICT/safety-2'
    # r'../model/FL_model/EM/3_client'ALL
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path = os.path.join(save_best, 'model.pth')
    torch.save(server.global_model.state_dict(), model_path)

def ICT_main_Only():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/ict.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset_all1.get_dataset(
        conf["type"], conf["sign"],conf["patch_size"])
    # 启动服务器
    server = Server_Only(conf, eval_datasets)
    # 定义客户端列表
    clients = []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(ICT_Client.Client_train(conf, server.global_model, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        # for c in candidates:
        #     print(c.client_id)
        # 累计权重
        weight_accumulator = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model.train()
        for c in candidates:
            diff = c.local_train(server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
        # 模型参数聚合c
        server.model_aggregate(weight_accumulator)
        # 模型评估
    dice, hd95 = server.model_eval()
    print(f'dice={dice},hd95={hd95}')
    save_best = r'../model/FL_model/ICT/ALL31'
    # r'../model/FL_model/MT/ALL'
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path = os.path.join(save_best, 'model.pth')
    torch.save(server.global_model.state_dict(), model_path)

def CPS_main():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/cps.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset1.get_dataset(
        conf["type"], conf["sign"],conf["patch_size"])
    # train_datasets, eval_datasets = dataset_all1.get_dataset(
    #     conf["type"], conf["sign"], conf["patch_size"])
    # 启动服务器
    server = Server_CPS(conf, eval_datasets)
    epoch_signal = conf["global_epochs"] / 6
    # 定义客户端列表
    clients = []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(CPS_Client.Client_train(conf, server.global_model1, server.global_model2, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])
        # print("select clients is: ")
        # for c in candidates:
        #     print(c.client_id)
        # 累计权重
        weight_accumulator1, weight_accumulator2 = {}, {}
        cnt1,cnt2 = {}, {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model1.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator1[name] = torch.zeros_like(params)
            cnt1[name] = 0
        for name, params in server.global_model2.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator2[name] = torch.zeros_like(params)
            cnt2[name] = 0
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model1.train()
        server.global_model2.train()
        if e<epoch_signal:signal=0
        else:signal=1
        for c in candidates:
            diff1, diff2 = c.local_train(signal, server.global_model1, server.global_model2)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model1.state_dict().items():
                if name in diff1:
                    weight_accumulator1[name].add_(diff1[name])
                    cnt1[name] += 1
            for name, params in server.global_model2.state_dict().items():
                if name in diff2:
                    weight_accumulator2[name].add_(diff2[name])
                    cnt2[name] += 1
        # 模型参数聚合
        server.model_aggregate(signal, weight_accumulator1, weight_accumulator2, cnt1, cnt2)
    dice1, hd951, dice2, hd952 = server.model_eval()
    print(f'dice1={dice2},hd951={hd952},dice2={dice2},hd952={hd952}')
    save_best = r'../model/FL_model/CPS/safety-2'
    # r'../model/FL_model/EM/3_client'ALL
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path1 = os.path.join(save_best, 'model1.pth')
    torch.save(server.global_model1.state_dict(), model_path1)
    model_path2 = os.path.join(save_best, 'model2.pth')
    torch.save(server.global_model2.state_dict(), model_path2)

def CPS_main_only():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/cps.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset_all1.get_dataset(
        conf["type"], conf["sign"],conf["patch_size"])
    # train_datasets, eval_datasets = dataset_all1.get_dataset(
    #     conf["type"], conf["sign"], conf["patch_size"])
    # 启动服务器
    server = Server_CPS_Only(conf, eval_datasets)
    # 定义客户端列表
    clients = []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(CPS_Client.Client_train(conf, server.global_model1, server.global_model2, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])
        weight_accumulator1, weight_accumulator2 = {}, {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model1.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator1[name] = torch.zeros_like(params)
        for name, params in server.global_model2.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator2[name] = torch.zeros_like(params)
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model1.train()
        server.global_model2.train()
        for c in candidates:
            diff1, diff2 = c.local_train(server.global_model1, server.global_model2)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model1.state_dict().items():
                weight_accumulator1[name].add_(diff1[name])
            for name, params in server.global_model2.state_dict().items():
                weight_accumulator2[name].add_(diff2[name])
        # 模型参数聚合
        server.model_aggregate(weight_accumulator1, weight_accumulator2)
    dice1, hd951, dice2, hd952 = server.model_eval()
    print(f'dice1={dice2},hd951={hd952},dice2={dice2},hd952={hd952}')
    save_best = r'../model/FL_model/CPS/ALL3'
    # r'../model/FL_model/EM/3_client'ALL
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path1 = os.path.join(save_best, 'model1.pth')
    torch.save(server.global_model1.state_dict(), model_path1)
    model_path2 = os.path.join(save_best, 'model2.pth')
    torch.save(server.global_model2.state_dict(), model_path2)

def Light_URPC_main():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/light_urpc.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset.get_dataset(
        conf["type"], conf["sign"], conf["patch_size"])

    # 启动服务器
    server = Server(conf, eval_datasets)
    epoch_signal = conf["global_epochs"] / 6
    # 定义客户端列表,旧模型
    clients= []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(Light_urpc_Client.Client_train(conf, server.global_model, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    swa_sign = 0
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d -------------------------------------------" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        if (e == 0 and conf['old_nets_pool']) or (swa_sign and e == conf['swa_start']): candidates = clients
        else: candidates = random.sample(clients, conf["k"])
        weight_accumulator = {}
        cnt = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
            cnt[name] = 0
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model.train()
        if e<epoch_signal:signal=0
        else:signal=1
        for c in candidates:
            diff = c.light_local_train(signal, server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                if name in diff:
                    weight_accumulator[name].add_(diff[name])
                    cnt[name] += 1
        # 模型参数聚合
        server.model_aggregate_urpc(signal, weight_accumulator, swa_sign, cnt)

        dice1, dice2, hd95 = server.model_eval3(swa_sign)
        print(f'dice1={dice1},dice2={dice2},hd95={hd95}')
        if conf['use_swa'] and e >= conf['swa_start']-1:
            swa_sign = 1
    save_best = r'../model/FL_model/Light_URPC/safety-2'
    # r'../model/FL_model/EM/3_client' ALL  3_client_swa
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path = os.path.join(save_best, 'model.pth')
    torch.save(server.global_model.state_dict(), model_path)
    if conf['use_swa']:
        swamodel_path = os.path.join(save_best, 'swa_model.pth')
        torch.save(server.swa_model.state_dict(), swamodel_path)

def Light_URPC_main_only():
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', default='./utils/light_urpc.json', dest='conf')
    # 获取所有参数
    args = parser.parse_args()
    # 读取配置文件，指定编码格式为utf-8
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    # 获取数据集，加载描述信息
    train_datasets, eval_datasets = dataset_all.get_dataset(
        conf["type"], conf["sign"], conf["patch_size"])

    # 启动服务器
    server = Server_Only(conf, eval_datasets)
    # 定义客户端列表,旧模型
    clients= []
    # 创建10个客户端到列表中
    for c in range(conf["no_models"]):
        print(f'train={len(train_datasets[c])},test={len(eval_datasets[c])}')
        clients.append(Light_urpc_Client.Client_train(conf, server.global_model, train_datasets[c], eval_datasets[c], c))
    # 全局模型训练
    swa_sign = 0
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d -------------------------------------------" % e)
        # 每次训练从clients列表中随机抽取k个进行训练
        candidates = random.sample(clients, conf["k"])
        weight_accumulator = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
        # 遍历选中的客户端，每个客户端本地进行训练
        server.global_model.train()
        for c in candidates:
            diff = c.light_local_train_only(server.global_model)
            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
        # 模型参数聚合
        server.model_aggregate_urpc(weight_accumulator)

        dice1, dice2, hd95 = server.model_eval3(swa_sign)
        print(f'dice1={dice1},dice2={dice2},hd95={hd95}')
        if conf['use_swa'] and e >= conf['swa_start']-1:
            swa_sign = 1
    save_best = r'../model/FL_model/Light_URPC/ALL11'
    # r'../model/FL_model/EM/3_client' ALL  3_client_swa
    if not os.path.exists(save_best): os.makedirs(save_best)
    model_path = os.path.join(save_best, 'model.pth')
    torch.save(server.global_model.state_dict(), model_path)
    if conf['use_swa']:
        swamodel_path = os.path.join(save_best, 'swa_model.pth')
        torch.save(server.swa_model.state_dict(), swamodel_path)

if __name__ == '__main__':
    start_time = time.time()

    # MT_main()
    # MT_main_Only()
    ICT_main()
    # ICT_main_Only()
    Light_URPC_main()
    EM_main()
    CPS_main()
    # CPS_main_only()
    # EM_main_only()
    end_time = time.time()
    run_time = end_time - start_time
    print(f'run_time={run_time}s,{run_time / 3600}H')