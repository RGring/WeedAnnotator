import torch
import numpy as np
# Code snippets taken from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/functional.py
def intersection(pr, gt):
    inter = torch.sum(gt * pr)
    return inter.item()

def union(pr, gt, intersect=None, eps=1e-7):
    if not intersection:
        intersect = intersection(pr, gt)
    uni = torch.sum(gt) + torch.sum(pr) - intersect + eps
    return uni.item()

def mIoU_per_batch(pr, gt, ignore_channels=None):
    pr_argmax = torch.argmax(pr, dim=1)
    for i in range(pr.size(1)):
        pr[:, i, :, :] = (pr_argmax == i).type(torch.uint8)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    iou_list = []
    for i in range(pr.size(1)):
        inter_now = intersection(pr[:, i, :, :], gt[:, i, :, :])
        union_now = union(pr[:, i, :, :], gt[:, i, :, :], inter_now)
        iou = inter_now/union_now
        if inter_now == 0 and union_now < 1:
            iou = 1.0
        iou_list.append(iou)
    return np.mean(iou_list)

def mIoU_per_image(pr, gt, ignore_channels=None):
    pr_argmax = torch.argmax(pr, dim=1)
    for i in range(pr.size(1)):
        pr[:, i, :, :] = (pr_argmax == i).type(torch.uint8)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    iou_list = []
    for j in range(batch_size):
        iou_list_temp = []
        for i in range(pr.size(1)):
            inter_now = intersection(pr[j, i, :, :], gt[j, i, :, :])
            union_now = union(pr[j, i, :, :], gt[j, i, :, :], inter_now)
            iou = inter_now / union_now
            if inter_now == 0 and union_now < 1:
                iou = 1.0
            iou_list_temp.append(iou)
        iou_list.append(np.mean(iou_list_temp))
    return np.mean(iou_list)



# def iou_per_batch(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
#     """Calculate Intersection over Union between ground truth and prediction
#     Args:
#         pr (torch.Tensor): predicted tensor
#         gt (torch.Tensor):  ground truth tensor
#         eps (float): epsilon to avoid zero division
#         threshold: threshold for outputs binarization
#     Returns:
#         float: IoU (Jaccard) score
#     """
#     print(pr)
#     pr = _threshold(pr, threshold=threshold)
#     pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
#
#     intersect = intersection(pr, gt)
#     uni = union(pr, gt, eps=eps, intersect=intersect)
#     return (intersect + eps) / uni
#
# def iou_per_image(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
#     pr = _threshold(pr, threshold=threshold)
#     pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
#
#     batch_size = pr.size(0)
#     iou_sum = 0
#
#     for i in range(batch_size):
#         iou_sum += iou_per_batch(pr, gt, eps=eps)
#     return iou_sum/batch_size

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x
