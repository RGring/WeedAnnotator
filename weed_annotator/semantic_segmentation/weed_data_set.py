import cv2
import glob
import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from weed_annotator.annotation_converter.AnnotationConverter import AnnotationConverter


class WeedDataset(Dataset):
    """Rumex Dataset. Read images, create masks, apply augmentation and preprocessing transformations.

    Args:
        file_ids (list): list of img files to consider
        labels_to_consider (list): label in annotation that should be considered.
        num_img_split (int): split of images. If 0 whole images are considered (default).
        augmentation (albumentations.Compose): data transfromation pipeline (e.g. flip, scale, etc.)
    """

    def __init__(self, file_ids, labels_to_consider, num_img_split=0, augmentation=None):
        self.num_subimg_splits = num_img_split
        self.image_list = self._get_img_list(file_ids)

        # convert str names to class values on masks
        self._labels_to_consider = ["background"]
        self._labels_to_consider.extend(labels_to_consider)
        self.label_dict = {}
        for i, label in enumerate(self._labels_to_consider):
            self.label_dict[i] = label

        self.augmentation = augmentation
        self._last_img_props = None

    def _get_img_list(self, file_ids):
        files = []
        file_ids.sort()
        for id in file_ids:
            # ToDo: Make optional to only consider foreground masks
            annotation = AnnotationConverter.read_cvat_by_id(f"{os.path.dirname(id)}/annotations.xml",
                                                             os.path.basename(id))
            if len(annotation.get_polygons()) == 0:
                continue
            if self.num_subimg_splits > 0:
                for i in range(self.num_subimg_splits):
                    for j in range(self.num_subimg_splits):
                        files.append({"img_id": id, "sub_img_id": f"{id}_{i}_{j}", "split_x": i, "split_y": j})
            else:
                files.append({"img_id": id, "sub_img_id": f"{id}_{0}_{0}", "split_x": 0, "split_y": 0})
        return files

    def __len__(self):
        return len(self.image_list)

    def get_img_mask_props(self, index):
        image, mask = self[index]
        return image, mask, self._last_img_props

    def __getitem__(self, index):
        img_props = self.image_list[index]
        image_path = img_props["img_id"]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_props["img_width"] = image.shape[0]
        img_props["img_height"] = image.shape[1]
        img_props["rotate"] = False

        mask = self._create_mask(image_path, image)

        # Make sure we always have the same aspect ratio.
        if image.shape[0] < image.shape[1]:
            img_props["rotate"] = True
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        # plt.imshow(cv2.addWeighted(mask.astype(np.uint8)*255, 1, image, 0.8, 0))
        # plt.show()
        if self.num_subimg_splits > 0:
            image = self._get_sub_img(image, img_props["split_x"], img_props["split_y"])
            mask = self._get_sub_img(mask, img_props["split_x"], img_props["split_y"])
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        mask = torch.transpose(mask, 1, 2)
        mask = torch.transpose(mask, 0, 1)
        self._last_img_props = img_props
        return image, mask

    def _create_mask(self, image_path, image):
        class_masks = {}
        for label in self._labels_to_consider[1:]:
            class_masks[label] = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Foreground classes
        annotation = AnnotationConverter.read_cvat_by_id(f"{os.path.dirname(image_path)}/annotations.xml", os.path.basename(image_path))
        if annotation is not None:
            polygons = annotation.get_polygons()
            for pol in polygons:
                label = pol.get_label()
                if pol.get_label() not in self._labels_to_consider:
                    continue
                cv2.fillPoly(class_masks[label], pts=[pol.get_polygon_points_as_array()], color=1)

        # Background class
        masks = np.zeros((image.shape[0], image.shape[1], len(self._labels_to_consider)), dtype=np.uint8)
        sum = np.zeros((image.shape[0], image.shape[1]))
        for i, label in enumerate(self._labels_to_consider[1:]):
            mask_temp = class_masks[label]
            masks[:, :, i+1] = mask_temp
            sum += mask_temp
        masks[:, :, 0] = (sum == 0).astype(np.uint8)
        return masks

    def _get_sub_img(self, img, split_x, split_y):
        w_img, h_img = img.shape[0:2]
        w_subimg = math.floor(w_i/mg / self.num_subimg_splits)
        h_subimg = math.floor(h_img / self.num_subimg_splits)
        return img[split_x * w_subimg:split_x * w_subimg + w_subimg, split_y * h_subimg:split_y * h_subimg + h_subimg,
               :]
