import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from torch.optim.swa_utils import AveragedModel

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()

    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label1,label2, net, classes, patch_size=[256, 256], sign=0):
    image, label1,label2 = image.squeeze(0).cpu().detach(
    ).numpy(), label1.squeeze(0).cpu().detach().numpy(), \
                           label2.squeeze(0).cpu().detach().numpy()
    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    if sign:
        net = AveragedModel(net)
    net.eval()
    with torch.no_grad():
        output_d1,output_d2= net(input)
        output_main_d1 = output_d1[:, :8, :, :]
        output_main_d2 = output_d2[:, :7, :, :]
        out_d1 = torch.argmax(torch.softmax(
            output_main_d1, dim=1), dim=1).squeeze(0)
        out_d2 = torch.argmax(torch.softmax(
            output_main_d2, dim=1), dim=1).squeeze(0)
        out_d1 = out_d1.cpu().detach().numpy()
        out_d2 = out_d2.cpu().detach().numpy()
        pred_d1 = zoom(out_d1, (x / patch_size[0], y / patch_size[1]), order=0)
        pred_d2 = zoom(out_d2, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction_d1 = pred_d1
        prediction_d2 = pred_d2
    metric_list_d1,metric_list_d2 = [],[]
    for i in range(1, classes[0]):
        metric_list_d1.append(calculate_metric_percase(
            prediction_d1 == i, label1 == i))
    for i in range(1, classes[1]):
        metric_list_d2.append(calculate_metric_percase(
            prediction_d2 == i, label2 == i))
    return metric_list_d1,metric_list_d2