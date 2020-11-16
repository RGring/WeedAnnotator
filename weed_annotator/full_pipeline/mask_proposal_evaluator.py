import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import logging

from weed_annotator.annotation_converter.AnnotationConverter import AnnotationConverter
from weed_annotator.full_pipeline import metrics


class MaskProposalsEvaluator:
    def __init__(self, img_folder, weed_label=None):
        self._img_folder = img_folder
        self._gt_annotation_file = f"{self._img_folder}/annotations.xml"
        self._weed_label = weed_label

        # Logging
        self._logger = logging.getLogger('eval_logger')
        self._logger.setLevel(logging.INFO)
        fh = logging.StreamHandler()
        fh_formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(fh_formatter)
        self._logger.addHandler(fh)
        self._debug = False

    def evaluate(self, pred_mask_folder):
        mask_files = glob.glob(f"{pred_mask_folder}/*/*.png")
        mask_files.extend(glob.glob(f"{pred_mask_folder}/*.png"))
        mask_files.sort()

        scores = self._update_metrices({}, None, None, None, None)

        for mask_file in mask_files:
            img_id = mask_file.replace(f"{pred_mask_folder}/", "").replace(".png", "")
            img_file = f"{self._img_folder}/{img_id}.jpg"
            if not os.path.exists(img_file):
                self._logger.error(f"Corresponding image {img_file} does not exist. Abort.")
                return

            # ToDo: imgs only loaded for debugging
            img = cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            pred_mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

            # Ground truth mask
            gt_mask = np.zeros_like(img)
            annotations = AnnotationConverter.read_cvat_by_id(self._gt_annotation_file, os.path.basename(img_file))
            for polyon_ann in annotations.polygon_list:
                if self._weed_label is not None:
                    if polyon_ann.get_label() != self._weed_label:
                        continue
                polygon_pts = polyon_ann.get_polygon_points_as_array()
                cv2.fillPoly(gt_mask, [polygon_pts], (255, 255, 255))

            if self._debug:
                self._plot_mask_on_img(gt_mask, img)
                self._plot_mask_on_img(pred_mask, img)

            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            ret, gt_mask = cv2.threshold(gt_mask, 1, 255, 0)

            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
            ret, pred_mask = cv2.threshold(pred_mask, 1, 255, 0)

            scores = self._update_metrices(scores, gt_mask, annotations, pred_mask, self._weed_label)

        results = self._get_results(scores)
        if self._debug:
            print(results)
        return results

    def _update_metrices(self, scores, gt_mask, annotations, pred_mask, gt_label):
        if not scores:
            # Init scores
            scores = {"fg": {"union": 0,
                             "intersection": 0,
                             "IoU": 0,
                             "BF": 0,
                             "num_files": 0},
                      "bg": {"union": 0,
                             "intersection": 0,
                             "IoU": 0},
                      "accuracy": 0,
                      "num_leave_pix_over_IoU": {"0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0},
                      "num_files": 0}
        else:
            num_leave_pix = np.sum(gt_mask > 0)
            intersection = metrics.intersection(gt_mask, pred_mask)
            union = metrics.union(gt_mask, pred_mask)
            scores["fg"]["intersection"] += intersection
            scores["fg"]["union"] += union

            # Only determine "per image"-scores, when foregroung objects are present
            if num_leave_pix > 0:
                scores["fg"]["IoU"] += intersection / union
                scores["fg"]["BF"] += metrics.bf_score(pred_mask, gt_mask)
                scores["fg"]["num_files"] += 1

            intersection = metrics.intersection(np.bitwise_not(gt_mask), np.bitwise_not(pred_mask))
            union = metrics.union(np.bitwise_not(gt_mask), np.bitwise_not(pred_mask))
            scores["bg"]["intersection"] += intersection
            scores["bg"]["union"] += union
            if num_leave_pix > 0:
                scores["bg"]["IoU"] += intersection / union
            scores["accuracy"] += metrics.accuracy(pred_mask, gt_mask)
            scores["num_files"] += 1

            leave_counter = 0
            # Evaluation for IoU > 0.9
            TP, FN, FP, TP_n, FN_n, FP_n = metrics.get_tp_fn_fp(pred_mask, annotations, gt_label, 0.9)
            scores["num_leave_pix_over_IoU"]["0.9"] += TP - leave_counter
            leave_counter = TP

            # Evaluation for IoU > 0.8
            TP, FN, FP, TP_n, FN_n, FP_n = metrics.get_tp_fn_fp(pred_mask, annotations, gt_label, 0.8)
            scores["num_leave_pix_over_IoU"]["0.8"] += TP - leave_counter
            leave_counter = TP

            # Evaluation for IoU > 0.7
            TP, FN, FP, TP_n, FN_n, FP_n = metrics.get_tp_fn_fp(pred_mask, annotations, gt_label, 0.7)
            scores["num_leave_pix_over_IoU"]["0.7"] += TP - leave_counter
            leave_counter = TP

            # Evaluation for IoU > 0.6
            TP, FN, FP, TP_n, FN_n, FP_n = metrics.get_tp_fn_fp(pred_mask, annotations, gt_label, 0.6)
            scores["num_leave_pix_over_IoU"]["0.6"] += TP - leave_counter
            leave_counter = TP

            # Evaluation for IoU > 0.5
            TP, FN, FP, TP_n, FN_n, FP_n = metrics.get_tp_fn_fp(pred_mask, annotations, gt_label, 0.5)
            scores["num_leave_pix_over_IoU"]["0.5"] += TP - leave_counter
            scores["num_leave_pix_over_IoU"]["0.4"] += FN

        return scores

    def _get_results(self, scores):
        results = {"IoU_per_dataset": {"fg": scores["fg"]["intersection"] / scores["fg"]["union"],
                                       "bg": scores["bg"]["intersection"] / scores["bg"]["union"]},
                   "IoU_per_image": {"fg": scores["fg"]["IoU"] / scores["fg"]["num_files"],
                                     "bg": scores["bg"]["IoU"] / scores["fg"]["num_files"]},
                   "BF_per_image": scores["fg"]["BF"] / scores["fg"]["num_files"],
                   "accuracy": scores["accuracy"] / scores["num_files"],
                   "num_leave_pix_over_IoU": scores["num_leave_pix_over_IoU"]}
        results["mIoU"] = (results["IoU_per_dataset"]["fg"] + results["IoU_per_dataset"]["bg"]) / 2
        return results

    def _plot_mask_on_img(self, mask, img):
        mask_on_img = cv2.addWeighted(mask, 1, img, 0.8, 0)
        plt.imshow(mask_on_img)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-i', '--img_folder', default='sample_data/imgs_val', type=str,
                        help='Corresponding input images of mask proposals')
    parser.add_argument('-m', '--mask_folder', default='sample_data/mask_proposals/post_processed', type=str,
                        help='Generated mask proposals')
    parser.add_argument('-wl', '--weed_label', default='rumex_leaf', type=str,
                        help='Specific label, that is considered as foreground.')

    args = parser.parse_args()

    evaluator = MaskProposalsEvaluator(args.img_folder, args.weed_label)
    evaluator.evaluate(args.mask_folder)
