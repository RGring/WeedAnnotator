import numpy as np
import cv2


def iou(source, target):
    i = intersection(source, target)
    u = union(source, target)
    iou = i / u
    return iou


def intersection(source, target):
    return np.sum(np.logical_and(source, target)) + 1


def union(source, target):
    return np.sum(np.logical_or(source, target)) + 1


def accuracy(pred_mask, gt_mask):
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    num_pix_total = gt_mask.shape[0] * gt_mask.shape[1]
    correct_pix = pred_mask == gt_mask
    num_correct_pix = np.sum(correct_pix)
    return num_correct_pix / num_pix_total


def bf_score(pred_mask, gt_mask):
    thresh = int(0.0075 * np.linalg.norm(pred_mask.shape))

    pred_mask = pred_mask.astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)

    contours, hierarchy = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_source = []
    for cnt in contours:
        for p in cnt:
            cnt_source.append(p[0].tolist())
    contours, hierarchy = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_target = []
    for cnt in contours:
        for p in cnt:
            cnt_target.append(p[0].tolist())

    precision = boundary_precision_recall(
        cnt_target, cnt_source, thresh)  # Precision

    recall = boundary_precision_recall(
        cnt_source, cnt_target, thresh)  # Recall

    try:
        f1 = 2 * recall * precision / (recall + precision)  # F1 score
    except:
        f1 = 0
    return f1

def boundary_precision_recall(contours_a, contours_b, threshold):
    boundary_overlap = 0
    try:
        for b in contours_b:
            # find the nearest distance
            for a in contours_a:
                dist = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
                if dist < threshold * threshold:
                    boundary_overlap = boundary_overlap + 1
                    break

        precision_recall = boundary_overlap / len(contours_b)
    except Exception as e:
        precision_recall = 0

    return precision_recall


def get_tp_fn_fp(source, gt_annotations, gt_label, iou_thresh):
    contours, hierarchy = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get ground_truth_mask
    gt_mask = np.zeros(source.shape, dtype=np.uint8)
    num_leaves = 0
    for polyon_ann in gt_annotations.polygon_list:
        if polyon_ann.get_label() == gt_label:
            polygon_pts = polyon_ann.get_polygon_points_as_array()
            cv2.fillPoly(gt_mask, [polygon_pts], 1)
            num_leaves += 1

    TP_n = 0
    FP_n = 0
    tp_mask = np.zeros(source.shape, dtype=np.uint8)
    tp_gt_mask = np.zeros(source.shape, dtype=np.uint8)
    for cnt in contours:
        gt_tps_temp = gt_mask.copy()
        num_tp_leaves = num_leaves
        object = cv2.fillPoly(np.zeros(source.shape, dtype=np.uint8), [cnt], 1)
        gt_tps = gt_mask
        max_iou = intersection(object, gt_tps_temp) / (union(object, gt_tps_temp) + 1)
        # Find best overlap
        for polyon_ann in gt_annotations.polygon_list:
            gt_tps_temp = gt_tps.copy()
            polygon_pts = polyon_ann.get_polygon_points_as_array()
            gt_tps_temp = cv2.fillPoly(gt_tps_temp, [polygon_pts], 0)
            inter = intersection(object, gt_tps_temp)
            un = union(object, gt_tps_temp) + 1
            iou_temp = inter / un
            if iou_temp > max_iou:
                num_tp_leaves -= 1
                max_iou = iou_temp
                gt_tps = gt_tps_temp
        if max_iou >= iou_thresh:
            tp_mask = cv2.bitwise_or(object, tp_mask)
            tp_gt_mask = cv2.bitwise_or(gt_tps, tp_gt_mask)
            TP_n += num_tp_leaves
        else:
            FP_n += 1

    tp_gt_mask[np.where(tp_gt_mask > 0)] = 1
    tp_mask[np.where(tp_mask > 0)] = 1
    gt_mask[np.where(gt_mask > 0)] = 1
    source[np.where(source > 0)] = 1

    TP = np.sum(tp_gt_mask)
    FN = np.sum(gt_mask) - TP

    # Because of the function fillPoly, it can happen that the masks do not overlap 100%, the difference is negligible
    FP = max(0, int(np.sum(source)) - int(np.sum(tp_mask)))
    FN_n = len(gt_annotations.polygon_list) - TP
    return TP, FN, FP, TP_n, FN_n, FP_n
