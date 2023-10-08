import argparse
import os
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import cv2
from copy import copy
from networks.net_factory import net_factory
from torch.optim.swa_utils import AveragedModel

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/huaxi_img/cross_FL', help='Name of Experiment')
parser.add_argument('--data_path', type=str,
                    default='../data/huaxi_img', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='FL_model/Light_URPC/818-6', help='experiment_name')

# FL_model/URPC/3_client
parser.add_argument('--model', type=str,
                    default='light_unet_urpc', help='model_name')
parser.add_argument('--use_swa', type=int,  default=0,
                    help='use or not')
parser.add_argument('--num_classes', type=list,  default=[8,7],
                    help='output channel of network')
parser.add_argument('--labeled_num', type=str, default='49',
                    help='labeled data')

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean() # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()# 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))

def Array_index(sign,prediction,label):
    index_num = []
    ignore_labels = [255]
    imgPredict = torch.tensor(prediction)
    imgLabel = torch.tensor(label)
    metric = SegmentationMetric(sign)  # 3表示有3个分类，有几个分类就填几, 0也是1个分类
    hist = metric.addBatch(imgPredict, imgLabel, ignore_labels)
    pa = metric.pixelAccuracy()
    # cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    index_num.append(pa)
    # index_num.append(cpa)
    index_num.append(mpa)
    # index_num.append(IoU)
    index_num.append(mIoU)
    return index_num,IoU

def calculate_metric_percase(num,case,i,pred, gt):
    result=[]
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    try:dice = metric.binary.dc(pred, gt)
    except:
        print(f'num,dice，文件名={num},{case},类别={i}')
        dice = 0
    try:asd = metric.binary.asd(pred, gt)
    except:
        print(f'num,asd，文件名={num},{case},类别={i}')
        asd = 0
    try:hd95 = metric.binary.hd95(pred, gt)
    except:
        print(f'num,hd95，文件名={num},{case},类别={i}')
        hd95 = 0

    result.append(dice)
    result.append(asd)
    result.append(hd95)
    return result

def onetoone(original_image,num):
    # original_image = original_image.cpu().numpy()
    if type(original_image) == torch.Tensor:
        original_image = original_image.cpu().numpy()
    original_image = original_image.astype(np.uint8)
    ret, binary = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    bin_clo = cv2.dilate(binary, kernel2, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=4)
    gray4 = copy(original_image)
    for i in range(1, num_labels):
        maskin = labels == i
        gray4[:, :][maskin] = i
    gray4 = np.where(gray4 > num, 0, gray4)
    return gray4

def Agg_label(label1,label2):
    indices = np.where((label1 != 0) & (label2 != 0))
    r1= {}
    for i in range(len(indices[0])):
        y = indices[0][i]
        x = indices[1][i]
        pixel1 = label1[y][x]
        pixel2 = label2[y][x]
        # print(f'pixel1={pixel1},pixel2={pixel2}')
        r1[(x, y)] = (pixel1, pixel2)

    return r1

def ronghe(label1,label2):
    # array, counts = np.unique(label1, return_counts=True)
    # print(array,counts)
    copy_label1 = copy(label1)
    for j in range(2, 8):
        copy_label1[label1 == j] = 2 * j - 1

    '椎间盘变化'
    copy_label2 = copy(label2)
    for i in range(1, 7):
        copy_label2[label2 == i] = 2 * i

    label = copy_label1 + copy_label2
    return label

def toone(value,min_value,max_value):
    return (value-min_value)/(max_value-min_value)

def test_single_volume_alone(case, net, test_save_path, FLAGS, sign):
    all_image = np.zeros([5, 256, 256], dtype='uint8')
    h5f = h5py.File(FLAGS.data_path + "/data/FL/FL_slices3/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label1 = h5f['label1'][:]
    label2 = h5f['label2'][:]

    if sign ==3:
        image = cv2.copyMakeBorder(image, 12, 20, 12, 20, cv2.BORDER_CONSTANT, value=0)
        label1 = cv2.copyMakeBorder(label1, 12, 20, 12, 20, cv2.BORDER_CONSTANT, value=0)
        label2 = cv2.copyMakeBorder(label2, 12, 20, 12, 20, cv2.BORDER_CONSTANT, value=0)
    elif sign ==2:
        image = cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=0)
        label1 = cv2.copyMakeBorder(label1, 16, 16, 16, 16,  cv2.BORDER_CONSTANT, value=0)
        label2 = cv2.copyMakeBorder(label2, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=0)

    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (256 / x, 256 / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    if FLAGS.use_swa:
        net = AveragedModel(net)
    net.eval()
    with torch.no_grad():
        d1, d2 = net(input)
        output_d1 = d1[:, :8, :, :]
        output_d2 = d2[:, :7, :, :]
        out_d1 = torch.argmax(torch.softmax(
            output_d1, dim=1), dim=1).squeeze(0)
        out_d2 = torch.argmax(torch.softmax(
            output_d2, dim=1), dim=1).squeeze(0)
        # out_d1 = onetoone(out_d1,7)
        # out_d2 = onetoone(out_d2,6)

        prediction_d1 = out_d1.cpu().detach().numpy()
        prediction_d2 = out_d2.cpu().detach().numpy()

        r1 = Agg_label(prediction_d1,prediction_d2)
        p1 = copy(prediction_d1)
        p2 = copy(prediction_d2)
        # print(f'case={case},{len(r1)}')
        for pos, pixel in r1.items():

            v1 = output_d1[:, pixel[0] - 1:pixel[0], pos[1], pos[0]].item()
            max_v1,min_v1 = torch.max(output_d1[:, pixel[0] - 1:pixel[0], :, :]).item() ,torch.min(
                output_d1[:, pixel[0] - 1:pixel[0], :, :]).item()
            v1 = toone(v1,min_v1,max_v1)

            v2 = output_d2[:, pixel[1] - 1:pixel[1], pos[1], pos[0]].item()
            max_v2, min_v2 = torch.max(output_d2[:, pixel[1] - 1:pixel[1], :, :]).item(), torch.min(
                output_d2[:, pixel[1] - 1:pixel[1], :, :]).item()

            v2 = toone(v2, min_v2, max_v2)
            # 算概率，占比例
            if v1>v2:
                # 椎间盘的值变成椎块的
                # print(f'椎间盘1：{p2[pos[1]][pos[0]]}')
                p2[pos[1]][pos[0]] = 0
                # print(f'椎间盘2：{p2[pos[1]][pos[0]]}')
            else:
                # print(f'椎块1：{p1[pos[1]][pos[0]]}')
                p1[pos[1]][pos[0]] = 0
                # print(f'椎块2：{p1[pos[1]][pos[0]]}')

    prediction = ronghe(p1,p2)
    label = ronghe(label1,label2)

    index_d1,Iou_d1 = Array_index(8,prediction_d1,label1)
    index_d2,Iou_d2 = Array_index(7,prediction_d2,label2)

    index, Iou = Array_index(14, prediction, label)

    metric_list_d1, metric_list_d2 ,metric_list= [], [] ,[]
    for i in range(0, FLAGS.num_classes[0]):

        metric_list_d1.append(calculate_metric_percase(8,case,i,
            prediction_d1 == i, label1 == i))
        metric_list_d1[i].append(Iou_d1[i].item())
    for i in range(0, FLAGS.num_classes[1]):
        metric_list_d2.append(calculate_metric_percase(7,case,i,
            prediction_d2 == i, label2 == i))
        metric_list_d2[i].append(Iou_d2[i].item())
    for i in range(0, sum(FLAGS.num_classes)-1):
        metric_list.append(calculate_metric_percase(14,case,i,
            prediction == i, label == i))
        metric_list[i].append(Iou[i].item())

    all_image[0, :, :] = np.array(image)
    all_image[1, :, :] = np.array(prediction)
    all_image[2, :, :] = np.array(label)
    all_image[3, :, :] = np.array(p1)
    all_image[4, :, :] = np.array(p2)

    all_image_itk = sitk.GetImageFromArray(all_image.astype(np.float32))
    all_image_itk.SetSpacing((3, 1, 10))
    sitk.WriteImage(all_image_itk, test_save_path + case + ".nii.gz")

    return metric_list_d1, metric_list_d2, metric_list, index_d1, index_d2, index

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test1.list', 'r') as f:
        image_list1 = f.readlines()
    image_list1 = sorted([itMT.replace('\n', '').split(".")[0]
                          for itMT in image_list1])
    with open(FLAGS.root_path + '/test2.list', 'r') as f:
        image_list2 = f.readlines()
    image_list2 = sorted([itMT.replace('\n', '').split(".")[0]
                          for itMT in image_list2])
    with open(FLAGS.root_path + '/test3.list', 'r') as f:
        image_list3 = f.readlines()
    image_list3 = sorted([itMT.replace('\n', '').split(".")[0]
                          for itMT in image_list3])
    snapshot_path = f"../model/{FLAGS.exp}"
    test_save_path = f"../model/{FLAGS.exp}/{FLAGS.exp}_predictions/"
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    if FLAGS.use_swa:
        net = AveragedModel(net)
    net.eval()
    save_mode_path = os.path.join(
        snapshot_path, 'model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    score_d1 = [[0. for i in range(4)] for i in range(FLAGS.num_classes[0])]
    score_d2 = [[0. for i in range(4)] for i in range(FLAGS.num_classes[1])]
    score = [[0. for i in range(4)] for i in range(14)]
    score_d1, score_d2,score = np.array(score_d1), np.array(score_d2),np.array(score)
    score_index_d1 = np.array([0., 0., 0.])
    score_index_d2 = np.array([0., 0., 0.])
    score_index = np.array([0., 0., 0.])
    for case in tqdm(image_list1):
        metic_d1, metic_d2, metic,index_d1, index_d2,index = test_single_volume_alone(case, net, test_save_path, FLAGS,1)
        metic_d1,metic_d2,metic = np.array(metic_d1),np.array(metic_d2),np.array(metic)
        index_d1, index_d2 = np.array(index_d1), np.array(index_d2)
        score_d1 += metic_d1
        score_d2 += metic_d2
        score += metic
        score_index_d1 += index_d1
        score_index_d2 += index_d2
        score_index += index
    for case in tqdm(image_list2):
        metic_d1, metic_d2, metic,index_d1, index_d2,index = test_single_volume_alone(case, net, test_save_path, FLAGS,2)
        metic_d1,metic_d2,metic = np.array(metic_d1),np.array(metic_d2),np.array(metic)
        index_d1, index_d2 = np.array(index_d1), np.array(index_d2)
        score_d1 += metic_d1
        score_d2 += metic_d2
        score += metic
        score_index_d1 += index_d1
        score_index_d2 += index_d2
        score_index += index
    for case in tqdm(image_list3):
        metic_d1, metic_d2, metic,index_d1, index_d2,index = test_single_volume_alone(case, net, test_save_path, FLAGS,3)
        metic_d1,metic_d2,metic = np.array(metic_d1),np.array(metic_d2),np.array(metic)
        index_d1, index_d2 = np.array(index_d1), np.array(index_d2)
        score_d1 += metic_d1
        score_d2 += metic_d2
        score += metic
        score_index_d1 += index_d1
        score_index_d2 += index_d2
        score_index += index

    length = len(image_list1) + len(image_list2) + len(image_list3)
    # length = len(image_list1)
    score_d1 /= length
    score_d2 /= length
    score /= length
    score_index_d1 /= length
    score_index_d2 /= length
    score_index /= length
    print(f'PA,MPA,Miou:{score_index_d1},\n,{score_index_d2},\n{score_index}')
    return score_d1, score_d2, score,score_index_d1, score_index_d2,score_index

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    count = [0, 0, 0]
    metric_d1,metric_d2,metric,score1,score2,score = Inference(FLAGS)

    path = f"../model/{FLAGS.exp}"
    with open(path + "/result.txt", 'w') as f:
        f.writelines(str(f'd1:PA,MPA,Miou:') + '\n')
        f.write(str(score1) + '\n')
        f.writelines(str(f'd2:PA,MPA,Miou:') + '\n')
        f.write(str(score2) + '\n')
        f.writelines(str(f'dice,asd,hd95,miou:') + '\n')
        f.write(str(metric.sum(axis=0) / 14) + '\n')
        f.write(str(metric_d1.sum(axis=0) / (FLAGS.num_classes[0])) + '\n')
        f.write(str(metric_d2.sum(axis=0) / (FLAGS.num_classes[1])) + '\n')
        f.writelines(str(f'dice,asd,hd95,miou（list）:') + '\n')
        f.write(str(metric))
        f.writelines(str(f'dice1,asd1,hd951,miou1:') + '\n')
        # f.writelines(str(metric_d1.sum(axis=0) / (FLAGS.num_classes[0])) + '\n')
        f.write(str(metric_d1) + '\n')
        f.writelines(str(f'dice2,asd2,hd952,miou2:') + '\n')
        f.write(str(metric_d2))
    print(metric_d1,metric_d2,metric)
    print(f'dice,asd,hd95:')
    print(metric_d1.sum(axis=0) / (FLAGS.num_classes[0]))
    print(metric_d2.sum(axis=0) / (FLAGS.num_classes[1]))
    print(metric.sum(axis=0) / 14)