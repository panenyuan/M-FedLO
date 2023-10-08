from networks.unet import UNet, Light_UNet_URPC
import torch
from utils import losses, ramps
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import math

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "light_unet_urpc":
        net = Light_UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * ramps.sigmoid_rampup(epoch, 240.0)

def prepare(out1,out2):
    for i in range(2):
        diff = out1[:, (i*6+1):(i*6+2), :, :] - out2[:, (i*5+1):(i*5+2), :, :]
        out1[:, (i * 6 + 1):(i * 6 + 2), :, :] = diff

    for i in range(2,7):
        diff = out1[:, i:i+1, :, :] - out2[:, i-1:i, :, :]-out2[:, i:i+1, :, :]
        out1[:, i:i + 1, :, :] = diff

    for i in range(1,7):
        diff = out2[:, i:i+1, :, :] - out1[:, i+1:i+2, :, :] - out1[:, i:i+1, :, :]
        out2[:, i:i + 1, :, :] = diff
    return out1,out2

# def LOSS(conf, output, num_classes, label_batch, epoch):
#     ce_loss = CrossEntropyLoss()
#     dice_loss = losses.DiceLoss(num_classes)
#     kl_distance = nn.KLDivLoss(reduction='none')
#
#     outputs = output[:,:num_classes,:,:]
#     outputs_aux1 = output[:,num_classes:num_classes*2,:,:]
#     outputs_aux2 = output[:,num_classes*2:num_classes*3,:,:]
#     outputs_aux3 = output[:,num_classes*3:num_classes*4,:,:]
#
#     outputs_soft = torch.softmax(outputs, dim=1)
#     # print(outputs_soft)
#     outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
#     outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
#     outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
#     loss_ce = ce_loss(outputs[:conf['labeled_bs']],
#                       label_batch[:conf['labeled_bs']][:].long())
#     loss_ce_aux1 = ce_loss(outputs_aux1[:conf['labeled_bs']],
#                            label_batch[:conf['labeled_bs']][:].long())
#     loss_ce_aux2 = ce_loss(outputs_aux2[:conf['labeled_bs']],
#                            label_batch[:conf['labeled_bs']][:].long())
#     loss_ce_aux3 = ce_loss(outputs_aux3[:conf['labeled_bs']],
#                            label_batch[:conf['labeled_bs']][:].long())
#
#     loss_dice = dice_loss(outputs_soft[:conf['labeled_bs']],
#                           label_batch[:conf['labeled_bs']].unsqueeze(1))
#     loss_dice_aux1 = dice_loss(outputs_aux1_soft[:conf['labeled_bs']],
#                                label_batch[:conf['labeled_bs']].unsqueeze(1))
#     loss_dice_aux2 = dice_loss(outputs_aux2_soft[:conf['labeled_bs']],
#                                label_batch[:conf['labeled_bs']].unsqueeze(1))
#     loss_dice_aux3 = dice_loss( outputs_aux3_soft[:conf['labeled_bs']],
#                                 label_batch[:conf['labeled_bs']].unsqueeze(1))
#     # print(f'dice:',loss_dice,loss_dice_aux1,loss_dice_aux2,loss_dice_aux3)
#     supervised_loss = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3 +
#                        loss_dice + loss_dice_aux1 + loss_dice_aux2 + loss_dice_aux3) / 8
#
#     preds = (outputs_soft + outputs_aux1_soft +
#              outputs_aux2_soft + outputs_aux3_soft) / 4
#
#     variance_main = torch.sum(kl_distance(
#         torch.log(outputs_soft[conf['labeled_bs']:]), preds[conf['labeled_bs']:]), dim=1, keepdim=True)
#     # print(f'inf:',torch.isinf(variance_main).any())
#     # print(f'nan:',torch.isnan(variance_main).any())
#     exp_variance_main = torch.exp(-variance_main)
#
#     variance_aux1 = torch.sum(kl_distance(
#         torch.log(outputs_aux1_soft[conf['labeled_bs']:]), preds[conf['labeled_bs']:]), dim=1, keepdim=True)
#     exp_variance_aux1 = torch.exp(-variance_aux1)
#
#     variance_aux2 = torch.sum(kl_distance(
#         torch.log(outputs_aux2_soft[conf['labeled_bs']:]), preds[conf['labeled_bs']:]), dim=1, keepdim=True)
#     exp_variance_aux2 = torch.exp(-variance_aux2)
#
#     variance_aux3 = torch.sum(kl_distance(
#         torch.log(outputs_aux3_soft[conf['labeled_bs']:]), preds[conf['labeled_bs']:]), dim=1, keepdim=True)
#     exp_variance_aux3 = torch.exp(-variance_aux3)
#
#     consistency_weight = get_current_consistency_weight(epoch)
#     consistency_dist_main = (preds[conf['labeled_bs']:] - outputs_soft[conf['labeled_bs']:]) ** 2
#
#     consistency_loss_main = torch.mean(
#         consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)
#     # print(f'consistency_dist_main={consistency_dist_main},exp_variance_main={exp_variance_main},torch.mean(exp_variance_main)={torch.mean(exp_variance_main)},torch.mean(variance_main)={torch.mean(variance_main)}')
#     consistency_dist_aux1 = (preds[conf['labeled_bs']:] - outputs_aux1_soft[conf['labeled_bs']:]) ** 2
#     consistency_loss_aux1 = torch.mean(
#         consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)
#
#     consistency_dist_aux2 = (preds[conf['labeled_bs']:] - outputs_aux2_soft[conf['labeled_bs']:]) ** 2
#     consistency_loss_aux2 = torch.mean(
#         consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)
#
#     consistency_dist_aux3 = (preds[conf['labeled_bs']:] - outputs_aux3_soft[conf['labeled_bs']:]) ** 2
#     consistency_loss_aux3 = torch.mean(
#         consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)
#
#     consistency_loss = (consistency_loss_main + consistency_loss_aux1 +
#                         consistency_loss_aux2 + consistency_loss_aux3) / 4
#     # print(consistency_loss_main,consistency_loss_aux1,consistency_loss_aux2,consistency_loss_aux3)
#     # print(supervised_loss,consistency_loss,consistency_weight)
#     loss = supervised_loss + consistency_weight * consistency_loss
#     return loss

def Light_LOSS(conf, output, num_classes, label_batch, epoch):
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    kl_distance = nn.KLDivLoss(reduction='none')

    outputs = output[:, :num_classes, :, :]
    outputs_aux1 = output[:, num_classes:num_classes * 2, :, :]


    outputs_soft = torch.softmax(outputs, dim=1)
    # print(outputs_soft)
    outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)

    loss_ce = ce_loss(outputs[:conf['labeled_bs']],
                      label_batch[:conf['labeled_bs']][:].long())
    loss_ce_aux1 = ce_loss(outputs_aux1[:conf['labeled_bs']],
                           label_batch[:conf['labeled_bs']][:].long())

    loss_dice = dice_loss(outputs_soft[:conf['labeled_bs']],
                          label_batch[:conf['labeled_bs']].unsqueeze(1))
    loss_dice_aux1 = dice_loss(outputs_aux1_soft[:conf['labeled_bs']],
                               label_batch[:conf['labeled_bs']].unsqueeze(1))

    # print(f'dice:',loss_dice,loss_dice_aux1,loss_dice_aux2,loss_dice_aux3)
    supervised_loss = (loss_ce + loss_ce_aux1 +loss_dice + loss_dice_aux1 ) / 4

    preds = (outputs_soft + outputs_aux1_soft ) / 2

    variance_main = torch.sum(kl_distance(
        torch.log(outputs_soft[conf['labeled_bs']:]), preds[conf['labeled_bs']:]), dim=1, keepdim=True)
    exp_variance_main = torch.exp(-variance_main)
    # Ds
    variance_aux1 = torch.sum(kl_distance(
        torch.log(outputs_aux1_soft[conf['labeled_bs']:]), preds[conf['labeled_bs']:]), dim=1, keepdim=True)
    # wDs
    exp_variance_aux1 = torch.exp(-variance_aux1)

    consistency_weight = get_current_consistency_weight(epoch)
    consistency_dist_main = (preds[conf['labeled_bs']:] - outputs_soft[conf['labeled_bs']:]) ** 2

    consistency_loss_main = torch.mean(
        consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)
    # print(f'consistency_dist_main={consistency_dist_main},exp_variance_main={exp_variance_main},torch.mean(exp_variance_main)={torch.mean(exp_variance_main)},torch.mean(variance_main)={torch.mean(variance_main)}')
    consistency_dist_aux1 = (preds[conf['labeled_bs']:] - outputs_aux1_soft[conf['labeled_bs']:]) ** 2
    consistency_loss_aux1 = torch.mean(
        consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

    consistency_loss = (consistency_loss_main + consistency_loss_aux1) / 2
    # print(consistency_loss_main,consistency_loss_aux1,consistency_loss_aux2,consistency_loss_aux3)
    # print(supervised_loss,consistency_loss,consistency_weight)
    loss = supervised_loss + consistency_weight * consistency_loss
    return loss

def New_Loss(conf, output, global_output, num_classes, label_batch, epoch):
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    kl_distance = nn.KLDivLoss(reduction='none')

    outputs = output[:, :num_classes, :, :]
    outputs_aux1 = output[:, num_classes:num_classes * 2, :, :]

    outputs_soft = torch.softmax(outputs, dim=1)
    global_output_soft = torch.softmax(global_output[:, :num_classes, :, :], dim=1)
    global_aux1_output_soft = torch.softmax(global_output[:, num_classes:num_classes * 2, :, :], dim=1)


    # print(outputs_soft)
    outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)

    loss_ce = ce_loss(outputs[:conf['labeled_bs']],
                      label_batch[:conf['labeled_bs']][:].long())
    loss_ce_aux1 = ce_loss(outputs_aux1[:conf['labeled_bs']],
                           label_batch[:conf['labeled_bs']][:].long())

    loss_dice = dice_loss(outputs_soft[:conf['labeled_bs']],
                          label_batch[:conf['labeled_bs']].unsqueeze(1))
    loss_dice_aux1 = dice_loss(outputs_aux1_soft[:conf['labeled_bs']],
                               label_batch[:conf['labeled_bs']].unsqueeze(1))

    # print(f'dice:',loss_dice,loss_dice_aux1,loss_dice_aux2,loss_dice_aux3)
    supervised_loss = (loss_ce + loss_ce_aux1 + loss_dice + loss_dice_aux1) / 4

    preds = 0.7*outputs_soft + 0.3*outputs_aux1_soft
    preds_global = 0.7 * global_output_soft + 0.3 * global_aux1_output_soft

    variance_main = torch.sum(kl_distance(
        torch.log(outputs_soft[conf['labeled_bs']:]), preds[conf['labeled_bs']:]), dim=1, keepdim=True)

    exp_variance_main = torch.exp(-variance_main)

    variance_aux1 = torch.sum(kl_distance(
        torch.log(outputs_aux1_soft[conf['labeled_bs']:]), preds[conf['labeled_bs']:]), dim=1, keepdim=True)
    exp_variance_aux1 = torch.exp(-variance_aux1)

    consistency_weight = get_current_consistency_weight(epoch)
    consistency_dist_main = (preds[conf['labeled_bs']:] - outputs_soft[conf['labeled_bs']:]) ** 2

    consistency_loss_main = torch.mean(
        consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)
    # print(f'consistency_dist_main={consistency_dist_main},exp_variance_main={exp_variance_main},torch.mean(exp_variance_main)={torch.mean(exp_variance_main)},torch.mean(variance_main)={torch.mean(variance_main)}')
    consistency_dist_aux1 = (preds[conf['labeled_bs']:] - outputs_aux1_soft[conf['labeled_bs']:]) ** 2
    consistency_loss_aux1 = torch.mean(
        consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

    pseudo_l1 = torch.argmax(preds_global.detach(), dim=1, keepdim=False)
    # pseudo_l2 = torch.argmax(outputs_soft.detach(), dim=1, keepdim=False)
    pseudo_supervision1 = ce_loss(preds[conf['labeled_bs']:], pseudo_l1)/conf['temperature']

    consistency_loss = (consistency_loss_main + consistency_loss_aux1 + pseudo_supervision1) / 3

    loss = supervised_loss + consistency_weight * consistency_loss
    return loss

def model_norm(model_1, model_2):
	squared_sum = 0
	for name, layer in model_1.named_parameters():
	#	print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
		squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
	return math.sqrt(squared_sum)