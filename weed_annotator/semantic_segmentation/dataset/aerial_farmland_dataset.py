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

    def __getitem__(self, index):
        img_props = self.image_list[index]

        # Loading image
        image_path = img_props["img_id"]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_props["img_width"] = image.shape[0]
        img_props["img_height"] = image.shape[1]
        img_props["rotate"] = False

        # Loading mask
        mask = self._create_mask(image_path, image)

        # Loading validation mask
        valid_pixel_mask = cv2.imread(f"{os.path.dirname(image_path)}/../../masks/{os.path.basename(image_path).replace('.jpg', '.png')}")
        valid_farmland_mask = cv2.imread(f"{os.path.dirname(image_path)}/../../boundaries/{os.path.basename(image_path).replace('.jpg', '.png')}")
        valid_mask = (valid_pixel_mask[:, :, 0] / 255) * (valid_farmland_mask[:, :, 0] / 255)
        # Make sure we always have the same aspect ratio.
        if image.shape[0] < image.shape[1]:
            img_props["rotate"] = True
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            valid_mask = cv2.rotate(valid_mask, cv2.ROTATE_90_CLOCKWISE)

        # plt.imshow(image)
        # plt.show()

        # Augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask, mask1=valid_mask)
            image, mask, valid_mask = sample['image'], sample['mask'], sample['mask1']
        mask = torch.transpose(mask, 1, 2)
        mask = torch.transpose(mask, 0, 1)
        mask = mask * valid_mask
        # self._plot_overlay(image, mask, True)
        self._last_img_props = img_props
        return image, mask, valid_mask

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


