import cv2
import glob
import math
import numpy as np
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from weed_annotator.annotation_converter.AnnotationConverter import AnnotationConverter


class WeedDataset(Dataset):
    """Rumex Dataset. Read images, create masks, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        ann_file (str): path to annotation file
        weed_label (list): label in annotation that should be exclusively be considered. If None, all annotations are considererd (default).
        num_img_split (int): split of images. If 0 whole images are considered (default).
        augmentation (albumentations.Compose): data transfromation pipeline (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(self, images_dir, ann_file, weed_label=None, num_img_split=0, augmentation=None, preprocessing=None):
        self.num_subimg_splits = num_img_split
        self.image_list = self._get_img_list(images_dir)
        if os.path.exists(ann_file):
            self.ann_file = ann_file
        else:
            self.ann_file = None

        # convert str names to class values on masks
        self._weed_label = weed_label

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self._last_img_props = None

    def _get_img_list(self, images_dir):
        files = []
        file_ids = glob.glob(images_dir + '/*.jpg')
        file_ids.sort()
        for id in file_ids:
            if self.num_subimg_splits > 0:
                for i in range(self.num_subimg_splits):
                    for j in range(self.num_subimg_splits):
                        files.append({"img_id": id, "sub_img_id": f"{id}_{i}_{j}", "split_x": i, "split_y": j})
            else:
                files.append({"img_id": id, "sub_img_id": f"{id}_{0}_{0}", "split_x": 0, "split_y": 0})
        return files

    def __len__(self):
        return len(self.image_list)

    def get_img_and_props(self, index):
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

        if self.num_subimg_splits > 0:
            image = self._get_sub_img(image, img_props["split_x"], img_props["split_y"])
            mask = self._get_sub_img(mask, img_props["split_x"], img_props["split_y"])
        mask = np.expand_dims(mask[:, :, 0], axis=2)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        self._last_img_props = img_props
        return image, mask

    def _create_mask(self, image_path, image):
        mask = np.zeros(image.shape, dtype=np.float32)
        if self.ann_file is not None:
            annotation = AnnotationConverter.read_cvat_by_id(self.ann_file, os.path.basename(image_path))
            polygons = annotation.get_polygons()
            for pol in polygons:
                if self._weed_label is not None:
                    if pol.get_label() != self._weed_label:
                        continue
                    cv2.fillPoly(mask, pts=[pol.get_polygon_points_as_array()], color=(1, 1, 1))
        return mask

    def _get_sub_img(self, img, split_x, split_y):
        w_img, h_img = img.shape[0:2]
        w_subimg = math.floor(w_img / self.num_subimg_splits)
        h_subimg = math.floor(h_img / self.num_subimg_splits)
        return img[split_x * w_subimg:split_x * w_subimg + w_subimg, split_y * h_subimg:split_y * h_subimg + h_subimg,
               :]
