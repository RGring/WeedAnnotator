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

class AerialFarmlandDataset(WeedDataset):
    """Sugar Beet Dataset. Read images, create masks, apply augmentation and preprocessing transformations.
    Args:
        file_ids (list): list of img files to consider
        labels_to_consider (list): label in annotation that should be considered.
        num_img_split (int): split of images. If 0 whole images are considered (default).
        augmentation (albumentations.Compose): data transfromation pipeline (e.g. flip, scale, etc.)
        skip_background (Boolean): Whether to consider images with background only.
    """

    def __init__(self, file_ids, labels_to_consider, num_img_split=0, augmentation=None, skip_background=False):
        self._labels_to_consider = ["background"]
        self._labels_to_consider.extend(labels_to_consider)
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
            base_path = os.path.dirname(id).replace("/train/images/rgb", "")
            img_id = os.path.basename(id)
            if skip_background:
                image = cv2.imread(id)
                mask = self._create_mask(id, image)
                if np.sum(mask[:, :, 1:]) == 0:
                    continue
            if self.num_subimg_splits > 0:
                for i in range(self.num_subimg_splits):
                    for j in range(self.num_subimg_splits):
                        files.append({"img_id": id, "sub_img_id": f"{id}_{i}_{j}", "split_x": i, "split_y": j})
            else:
                files.append({"img_id": id, "sub_img_id": f"{id}_{0}_{0}", "split_x": 0, "split_y": 0})
        return files

    def _create_mask(self, image_path, image):
        base_path = os.path.dirname(image_path)
        label_path = f"{base_path}/../../labels"
        image_id = os.path.basename(image_path)
        masks = np.zeros((image.shape[0], image.shape[1], len(self._labels_to_consider)))
        sum = np.zeros((image.shape[0], image.shape[1]))
        for i, label in enumerate(self._labels_to_consider[1:]):
            mask_path = (f"{label_path}/{label}/{image_id}").replace(".jpg", ".png")
            mask = cv2.imread(mask_path)
            masks[:, :, i+1] = mask[:, :, 0]/255
            sum += masks[:, :, i+1]
        masks[:, :, 0] = (sum == 0).astype(np.uint8)
        return masks


