import argparse
import os
import shutil

import h5py
# import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import cv2
# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/huaxi_img/cross_FL', help='Name of Experiment')
parser.add_argument('--data_path', type=str,
                    default='../data/huaxi_img', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='FL_model/ICT/safety-3', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--flag', type=int,
                    default='1', help='IS or not urpc(1,4)')
parser.add_argument('--num_classes', type=int,  default=14,
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
        # rMTove classes from unlabeled pixels in gt image and predict
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

def calculate_metric_percase(case,i,pred, gt):
    result=[]
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    try:dice = metric.binary.dc(pred, gt)
    except:
        print(f'dice，文件名={case},类别={i}')
        dice = 0
    try:asd = metric.binary.asd(pred, gt)
    except:
        print(f'asd，文件名={case},类别={i}')
        asd = 0
    try:hd95 = metric.binary.hd95(pred, gt)
    except:
        print(f'hd95，文件名={case},类别={i}')
        hd95 = 0
    result.append(dice)
    result.append(asd)
    result.append(hd95)
    return result

def test_single_volume(case, net, test_save_path, FLAGS,sign):
    all_image = np.zeros([3, 256, 256], dtype='uint8')
    h5f = h5py.File(FLAGS.data_path + f"/data/FL/FL_slices/{case}.h5", 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    if sign ==3:
        image = cv2.copyMakeBorder(image, 12, 20, 12, 20, cv2.BORDER_CONSTANT, value=0)
        label = cv2.copyMakeBorder(label, 12, 20, 12, 20, cv2.BORDER_CONSTANT, value=0)
    elif sign ==2:
        image = cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=0)
        label = cv2.copyMakeBorder(label, 16, 16, 16, 16,  cv2.BORDER_CONSTANT, value=0)

    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (256 / x, 256 / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        if FLAGS.flag == 4:
            out_main, _, _, _ = net(input)
        else:
            out_main = net(input)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / 256, y / 256), order=0)
        prediction = pred

    index_num = []
    ignore_labels = [255]
    imgPredict = torch.tensor(prediction)
    imgLabel = torch.tensor(label)
    metric = SegmentationMetric(14)  # 3表示有3个分类，有几个分类就填几, 0也是1个分类
    hist = metric.addBatch(imgPredict, imgLabel, ignore_labels)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    index_num.append(pa)
    # index_num.append(cpa)
    index_num.append(mpa)
    # index_num.append(IoU)
    index_num.append(mIoU)

    result=[]
    for i in range(FLAGS.num_classes):
        metric = calculate_metric_percase(case,i,prediction == i, label == i)
        result.append(metric)
    all_image[0, :, :] = np.array(image)
    all_image[1, :, :] = np.array(prediction)
    all_image[2, :, :] = np.array(label)

    all_image_itk = sitk.GetImageFromArray(all_image.astype(np.float32))
    all_image_itk.SetSpacing((3, 1, 10))
    sitk.WriteImage(all_image_itk, test_save_path + case + ".nii.gz")

    return result, index_num

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
    save_mode_path = os.path.join(
        snapshot_path, 'model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print(f"init weight from {save_mode_path}")
    net.eval()

    # first_total,second_total,third_total, = 0.0

    score = [[0. for i in range(3)] for i in range(FLAGS.num_classes)]
    score = np.array(score)
    score1 = np.array([0., 0., 0.])

    for case in tqdm(image_list1):
        result_metic,result_index = test_single_volume(case, net, test_save_path, FLAGS,1)
        result_metic=np.array(result_metic)
        result_index = np.array(result_index)
        score += result_metic
        score1 += result_index
    for case in tqdm(image_list2):
        result_metic,result_index = test_single_volume(case, net, test_save_path, FLAGS,2)
        result_metic=np.array(result_metic)
        result_index = np.array(result_index)
        score += result_metic
        score1 += result_index
    for case in tqdm(image_list3):
        result_metic,result_index = test_single_volume(case, net, test_save_path, FLAGS,3)
        result_metic=np.array(result_metic)
        result_index = np.array(result_index)
        score += result_metic
        score1 += result_index

    length = len(image_list1)+len(image_list2)+len(image_list3)
    avg_metric = score / length
    score1 /= length
    print(f'PA,MPA,Miou:{score1}')
    return avg_metric, score1

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric,score1 = Inference(FLAGS)

    path = f"../model/{FLAGS.exp}"
    # 先用总模型
    with open(path + "/result.txt", 'w') as f:
        f.writelines(str(f'PA,MPA,Miou:') + '\n')
        f.write(str(score1) + '\n')
        f.writelines(str(f'dice,asd,hd95:') + '\n')
        f.writelines(str(metric.sum(axis=0) / (FLAGS.num_classes)) + '\n')
        f.write(str(metric))
    print(metric)
    print(f'dice,asd,hd95:')
    print(metric.sum(axis=0) / (FLAGS.num_classes))