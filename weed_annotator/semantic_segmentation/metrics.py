import torch
import numpy as np
# Code snippets taken from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/functional.py
def _intersection(pr, gt):
    inter = torch.sum(gt * pr, dim=[0, 2, 3])
    return inter

def _union(pr, gt, intersect=None):
    if intersect is not None:
        intersect = _intersection(pr, gt)
    uni = torch.sum(gt, dim=[0, 2, 3]) + torch.sum(pr, dim=[0, 2, 3]) - intersect
    return uni

def non_overlapping_target(pr, overlapping_target):
    if torch.sum(overlapping_target) == (overlapping_target.size(0) * overlapping_target.size(2) * overlapping_target.size(3)):
        return overlapping_target
    pr = pr*10
    target = pr * overlapping_target + overlapping_target
    target = arg_max(target)
    return target

def arg_max(pr):
    pr_argmax = torch.argmax(pr, dim=1)
    for i in range(pr.size(1)):
        pr[:, i, :, :] = (pr_argmax == i).type(torch.uint8)
    return pr

def inter_union_per_class(pr_labels, gt):
    inter_per_class = []
    union_per_class = []
    inter_per_class = _intersection(pr_labels, gt)
    union_per_class = _union(pr_labels, gt, inter_per_class)
    return inter_per_class.cpu().numpy(), union_per_class.cpu().numpy()

def mIoU_per_batch(pr_labels, gt, ignore_channels=None, eps=1e-7):
    pr_labels, gt = _take_channels(pr_labels, gt, ignore_channels=ignore_channels)
    iou_list = []
    inter_per_class = _intersection(pr_labels, gt)
    union_per_class = _union(pr_labels, gt, inter_per_class)
    iou_per_class = (inter_per_class + eps)/(union_per_class + eps)
    return torch.mean(iou_per_class)

def mIoU_per_image(pr_labels, gt, ignore_channels=None, eps=1e-7):
    pr_labels, gt = _take_channels(pr_labels, gt, ignore_channels=ignore_channels)
    miou_list = []
    for j in range(batch_size):
        iou_list_temp = []
        inter_per_class = _intersection(pr_labels[j, :, :, :], gt[j, :, :, :])
        union_per_class = _union(pr_labels[j, :, :, :], gt[j, :, :, :], inter_per_class)
        iou_per_class = (inter_per_class + eps) / (union_per_class + eps)
        miou_list.append(torch.mean(iou_per_class))
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
