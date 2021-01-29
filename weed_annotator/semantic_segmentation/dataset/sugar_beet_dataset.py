import cv2
import glob
from logging import getLogger
import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


from weed_annotator.semantic_segmentation.dataset.weed_data_set import WeedDataset

class SugarBeetDataset(WeedDataset):
    """Sugar Beet Dataset. Read images, create masks, apply augmentation and preprocessing transformations.
    Args:
        file_ids (list): list of img files to consider
        labels_to_consider (list): label in annotation that should be considered.
        num_img_split (int): split of images. If 0 whole images are considered (default).
        augmentation (albumentations.Compose): data transfromation pipeline (e.g. flip, scale, etc.)
        skip_background (Boolean): Whether to consider images with background only.
    """

    def __init__(self, file_ids, labels_to_consider, num_img_split=0, augmentation=None, skip_background=False):
        # Label Definition
        self._labelMap = {'soil': {'id': [0, 1, 97]},
                    'crop': {'id': [10000, 10001, 10002]},
                    'weed': {'id': [2]},
                    'dycot': {
                        'id': [20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011]},
                    'grass': {'id': [20100, 20101, 20102, 20103, 20104, 20105]}}

        self._labels_to_consider = ["background"]
        for label in labels_to_consider:
            if label in self._labelMap.keys():
                self._labels_to_consider.append(label)
            else:
                logger.warn(f"Lable \"{label}\" invalid for sugar beet dataset.")
        self.label_dict = {}
        for i, label in enumerate(self._labels_to_consider):
            self.label_dict[i] = label

        # Load img list
        self.num_subimg_splits = num_img_split
        self.image_list = self._get_img_list(file_ids, skip_background)
        self.augmentation = augmentation
        self._last_img_props = None

    def _get_img_list(self, file_ids, skip_background):
        files = []
        file_ids.sort()
        for id in file_ids:
            if skip_background:
                base_path = os.path.dirname(id).replace("/rgb", "")
                img_id = os.path.basename(id)
                mask_file = f"{base_path}/iMapCleaned/{img_id}"
                if not os.path.exists(mask_file):
                    continue
                iMap = _load_iMap(mask_file)
                sum = np.zeros(iMap.shape)
                background_only = True
                ones = np.ones(iMap.shape)
                zeros = np.zeros(iMap.shape)
                for i, label in enumerate(self._labels_to_consider[1:]):
                    for sub_label in self._labelMap[label]['id']:
                        temp =  np.where(np.equal(iMap, sub_label),
                                               ones,
                                               zeros)
                        if np.sum(temp) > 0:
                            background_only = False
                            break
                    if not background_only:
                        break
                if background_only:
                    continue
            if self.num_subimg_splits > 0:
                for i in range(self.num_subimg_splits):
                    for j in range(self.num_subimg_splits):
                        files.append({"img_id": id, "sub_img_id": f"{id}_{i}_{j}", "split_x": i, "split_y": j})
            else:
                files.append({"img_id": id, "sub_img_id": f"{id}_{0}_{0}", "split_x": 0, "split_y": 0})
        return files

    def _create_mask(self, image_path, image):
        base_path = os.path.dirname(image_path).replace("/rgb", "")
        img_id = os.path.basename(image_path)
        iMap = self._load_iMap(f"{base_path}/iMapCleaned/{img_id}")
        masks = np.zeros((iMap.shape[0], iMap.shape[1], len(self._labels_to_consider)))
        sum = np.zeros(iMap.shape)
        ones = np.ones(iMap.shape)
        zeros = np.zeros(iMap.shape)
        for i, label in enumerate(self._labels_to_consider[1:]):
            label_mask = np.zeros(iMap.shape)
            for sub_label in self._labelMap[label]['id']:
                label_mask += np.where(np.equal(iMap, sub_label),
                                       ones,
                                       zeros)
            masks[:, :, i+1] = label_mask
            sum += label_mask
        # Background class
        masks[:, :, 0] = (sum == 0).astype(np.uint8)
        return masks

    def _load_iMap(self, path):
        iMap = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        iMap = np.expand_dims(iMap, axis=-1)
        return iMap[:, :, 0]

