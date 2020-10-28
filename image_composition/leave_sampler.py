import albumentations as albu
import glob
import os
import random
import cv2
import numpy as np
import randomcolor
from PIL import ImageColor
import sys

class LeaveSampler:
    def __init__(self, crop_folder, aug_config):
        self._crop_folder = crop_folder
        self._static_augmentations = self._get_static_augmentations(aug_config)
        self._dropout_aug = aug_config["dropout_aug"]
        self._blur = aug_config["blur"]
        self._sharpen = aug_config["blur"]
        self._scale = aug_config["scale"]
        self._max_width = aug_config["max_width"]
        self._min_width = aug_config["min_width"]
        self._momentary_seed = random.Random()

    def _blur_aug(self):
        if not self._blur:
            return  None
        blur_limit=7
        aug = albu.OneOf([albu.MedianBlur(blur_limit=blur_limit, p=1.0), albu.Blur(blur_limit=blur_limit, p=1.0)])
        return aug

    def _get_dropout_aug(self, leave, color):
        color = ImageColor.getrgb(color[0])
        augm = albu.CoarseDropout(min_height=1, max_height=10, min_width=int(leave.shape[1]/2), max_width=leave.shape[1], fill_value=color, p=0.2)
        return augm

    def _get_static_augmentations(self, config):
        augmentations = []
        if config["horizontal_flip"]:
            augmentations.append(albu.HorizontalFlip(p=0.5))
        else:
            augmentations.append(None)
        if config["grid_distortion"]:
            augmentations.append(albu.GridDistortion(distort_limit=0.5, border_mode=cv2.BORDER_REPLICATE, p=1.0))
        else:
            augmentations.append(None)

        if config["hsv_color_shift"]:
            augmentations.append(albu.HueSaturationValue(p=1.0))
        else:
            augmentations.append(None)

        if config["random_brightness_contrast"]:
            augmentations.append(albu.RandomBrightnessContrast(p=1))
        else:
            augmentations.append(None)

        if config["random_shadow"]:
            augmentations.append(albu.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.2))
        else:
            augmentations.append(None)
        return augmentations

    def _get_dynamic_augmentations(self, leave):
        augmentations = []
        downscale = random.choice([True, False])
        if self._scale:
            h, w, c = leave.shape
            scale_max = self._max_width/w - 1
            scale_min = -(1 - self._min_width/w)
            if downscale:
                augmentations.append(albu.RandomScale(scale_limit=[scale_min, 0.0], p=1))
            else:
                augmentations.append(albu.RandomScale(scale_limit=[0.0, scale_max], p=1))
        else:
            augmentations.append(None)

        rand_color = randomcolor.RandomColor()
        color = rand_color.generate(hue="green", count=1)
        if self._dropout_aug:
            augmentations.append(self._get_dropout_aug(leave, color))
        else:
            augmentations.append(None)

        blur_aug = self._blur_aug()
        if blur_aug:
            augmentations.append(blur_aug)
        else:
            if self._sharpen:
                augmentations.append(albu.IAASharpen(p=0.5))
            else:
                augmentations.append(None)
        return augmentations

    def _reset_global_seed(self):
        random.seed(self._momentary_seed.randint(0, sys.maxsize * 2 + 1))

    def sample_leave(self, set_random=None, do_aug=True):
        self._momentary_seed = set_random
        sample_folder = f"{self._crop_folder}/leaves"
        rumex_plant_file = self._momentary_seed.choice(glob.glob(f"{sample_folder}/*_bb.png"))
        leave_bb = cv2.imread(rumex_plant_file, cv2.IMREAD_UNCHANGED)
        leave_bb = cv2.cvtColor(leave_bb, cv2.COLOR_BGR2RGB)

        leave = cv2.imread(rumex_plant_file.replace("_bb.", "."), cv2.IMREAD_UNCHANGED)
        leave = cv2.cvtColor(leave, cv2.COLOR_BGR2RGB)

        self._reset_global_seed()
        augm_list = self._get_dynamic_augmentations(leave)
        augm_list.extend(self._static_augmentations)
        for aug in augm_list:
            self._reset_global_seed()
            if aug and do_aug:
                augm = albu.Compose([aug], additional_targets={"image_bb" : "image"})
                augmented_images = augm(image=leave, image_bb=leave_bb)
                leave = augmented_images["image"]
                leave_bb = augmented_images["image_bb"]

        leave_bb_aug = cv2.cvtColor(leave_bb, cv2.COLOR_RGB2BGRA)
        leave_aug = cv2.cvtColor(leave, cv2.COLOR_RGB2BGRA)
        leave_aug = self._black_to_white(leave_aug)
        return leave_aug, leave_bb_aug

    def _black_to_white(self, leave_crop):
        gray = cv2.cvtColor(leave_crop, cv2.COLOR_BGRA2GRAY)
        mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
        white = np.zeros(leave_crop.shape, dtype=leave_crop.dtype)
        black_to_white = cv2.bitwise_or(white, leave_crop, mask=mask)
        return black_to_white
