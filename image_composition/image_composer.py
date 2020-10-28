import glob
import random
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from annotation_converter.AnnotationConverter import AnnotationConverter
from annotation_converter.Annotation import Annotation
from annotation_converter.Polygon import Polygon
import albumentations as albu
from image_composition.leave_sampler import LeaveSampler
import sys

SEED = 43
class ImageComposer:
    def __init__(self, config):
        self._collage_ident = config["folders"]["ident"]

        # Folder parameters
        self._output_path = self._init_out_folder(config)
        self._path_to_background_imgs = config["folders"]["background_folder"]
        self._path_to_rumex_crops = config["folders"]["leave_crops_folder"]
        self._non_rumex_crops_folder = config["folders"]["non_leave_crops_folder"]
        self._ann_file = f"{config['folders']['ann_folder']}/{self._collage_ident}/annotations.xml"

        # Collage parameters
        self._label = config["collage_properties"]["label"]
        self._blending_probs = {}
        for blending_method, blending_prob in config["blending_probabilities"].items():
            self._blending_probs[blending_method] = blending_prob

        self._add_shape_distractor = config["image_augmentations"]["shape_distractor"]
        self._add_weed_distractor = config["image_augmentations"]["weed_distractor"]
        self._add_grass_distractor = config["image_augmentations"]["grass_distractor"]

        self._whole_plants_prob = config["collage_properties"]["whole_plants_prob"]
        self._maximum_distance_to_root_center = config["collage_properties"]["maximum_distance_to_root_center"]
        self._maximum_number_of_leaves = config["collage_properties"]["maximum_number_of_leaves"]
        self._minimum_number_of_leaves = config["collage_properties"]["minimum_number_of_leaves"]

        # Objects
        self._seed_random = random.Random()
        self._seed_random.seed(SEED)
        self._class_random = random.Random()
        self._leave_sampler = LeaveSampler(self._path_to_rumex_crops, config["leave_augmentations"])

    def _init_out_folder(self, config):
        # init save path
        out_folder = config["folders"]["out_folder"]
        ident = config["folders"]["ident"]
        img_path = f"{out_folder}/{ident}"
        os.makedirs(img_path, exist_ok=True)

        # save config
        with open(f"{img_path}/confg.json", "w") as write_file:
            json.dump(config, write_file)
        return img_path

    def _gaussian_blur_edges(self, foreground, image_composition):
        blurred_img_1 = cv2.GaussianBlur(image_composition, (3, 3), 0)
        blurred_img_2 = cv2.GaussianBlur(image_composition, (7, 7), 0)
        mask_1 = np.zeros(foreground.shape, np.uint8)
        mask_2 = np.zeros(foreground.shape, np.uint8)
        gray = cv2.cvtColor(foreground, cv2.COLOR_BGRA2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(mask_1, contours, -1, (255, 255, 255, 255), 10)
        cv2.drawContours(mask_2, contours, -1, (255, 255, 255, 255), 5)
        output = np.where(mask_1 == np.array([255, 255, 255, 255]), blurred_img_1, image_composition)
        output = np.where(mask_2 == np.array([255, 255, 255, 255]), blurred_img_2, output)
        return output

    def _poisson_mix_normal(self, foreground, foreground_bb, background):
        background = self._poisson_merge(foreground_bb, background, cv2.MIXED_CLONE)
        output = self._poisson_merge(foreground, background)
        return output

    def _poisson_merge(self, foreground, background, merge_mode=cv2.NORMAL_CLONE):
        background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGRA2BGR)

        gray = cv2.cvtColor(foreground, cv2.COLOR_BGRA2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Getting object center
        x, y, w, h = cv2.boundingRect(thresh)
        center = (int(x + w / 2), int(y + h / 2))

        # Clone seamlessly.
        output = cv2.seamlessClone(foreground, background, thresh, center, merge_mode)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
        return output

    def _overlay_imgs(self, foreground, background, erode=True):
        mask_grey = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask_grey, 1, 255, 0)
        if erode:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        fg = cv2.bitwise_or(foreground, foreground, mask=mask)
        mask = cv2.bitwise_not(mask)
        bk = cv2.bitwise_or(background, background, mask=mask)
        overlay = cv2.bitwise_or(fg, bk)
        return overlay

    def _position_leave(self, foreground, bckgr_size, x_plant, y_plant, center_offset, angle):
        img_center_x = int(bckgr_size[0] / 2)
        img_center_y = int(bckgr_size[1] / 2)

        # Shifting leave by center_offset
        top = int(img_center_y - foreground.shape[1] / 2)
        if top < 0:
            foreground = foreground[:, -top:]
            top = 0
        left = int(img_center_x - foreground.shape[0] - center_offset)
        if left < 0:
            foreground = foreground[-left:, :]
            left = 0
        output = np.zeros((bckgr_size[0], bckgr_size[1], 4), dtype="uint8")
        output[left:left + foreground.shape[0], top:top + foreground.shape[1]] = foreground

        # Rotate leave around plant center and move plant center to (x_plant, y_plant)
        rot_mat = cv2.getRotationMatrix2D((img_center_y, img_center_x), angle, 1.0)
        rot_mat[0, 2] += x_plant
        rot_mat[1, 2] += y_plant
        output = cv2.warpAffine(output, rot_mat, output.shape[1::-1], flags=cv2.INTER_LINEAR)
        return output

    def _get_random_root_center(self, img_size):
        scale_factor = 0.6
        x_plant = self._class_random.randint(0, int(img_size[0] / 2 * scale_factor)) * self._class_random.choice(
            [1, -1])
        y_plant = self._class_random.randint(0, int(img_size[1] / 2 * scale_factor)) * self._class_random.choice(
            [1, -1])
        return x_plant, y_plant

    def _add_arti_plant_composition(self, image_composition):
        img_size = image_composition.shape[0:2]
        # Position of "tap root"
        x_plant, y_plant = self._get_random_root_center(img_size)
        polygons = []

        number_of_leaves = self._class_random.randint(self._minimum_number_of_leaves, self._maximum_number_of_leaves)
        for i_leaf in range(number_of_leaves):
            # Load leave crop
            leave, leave_bb = self._leave_sampler.sample_leave(self._class_random)
            center_offset = self._class_random.randint(0, self._maximum_distance_to_root_center)
            angle = self._class_random.randint(0, 360)
            leave_img = self._position_leave(leave, (img_size), x_plant, y_plant, center_offset, angle)
            polygon = self._get_polygon(leave_img)
            if polygon is not None and len(polygon) > 0:
                polygons.extend(polygon)

            if "poisson_bb" in list(self._blending_probs.keys()):
                leave_img_bb = self._position_leave(leave_bb, (img_size), x_plant, y_plant, center_offset, angle)
            else:
                leave_img_bb = None

            # Blending
            image_composition = self._blend(leave_img, leave_img_bb, image_composition)

        return image_composition, polygons

    def _get_polygon(self, foreground):
        gray = cv2.cvtColor(foreground, cv2.COLOR_BGRA2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        polygons = []
        for cnt in contours:
            polygon = Polygon(self._label)
            polygon.set_polygon_points_as_array(cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True))
            polygons.append(polygon)
        return polygons

    def _blend(self, foreground, foreground_bb, background):
        mode = \
            self._class_random.choices(list(self._blending_probs.keys()),
                                       weights=list(self._blending_probs.values()), k=1)[0]
        if mode == "raw":
            output = self._overlay_imgs(foreground, background)
        elif mode == "raw_gaussian":
            output = self._overlay_imgs(foreground, background)
            output = self._gaussian_blur_edges(foreground, output)
        elif mode == "poisson" or (mode == "poisson_bb" and foreground_bb is None):
            output = self._poisson_merge(foreground, background)
        elif mode == "poisson_bb" and foreground_bb is not None:
            output = self._poisson_mix_normal(foreground, foreground_bb, background)
        elif mode == "hsv":
            output = self._hsv_merge(foreground, background)
        elif mode == "hsv_gaussian":
            output = self._hsv_merge(foreground, background)
            output = self._gaussian_blur_edges(foreground, output)
        return output

    def _add_plant(self, image_composition, plant_file, resize_range=[1.0, 1.0]):
        img_size = image_composition.shape[0:2]
        plant = cv2.imread(plant_file.replace("_bb.", "."), cv2.IMREAD_UNCHANGED)
        plant = cv2.cvtColor(plant, cv2.COLOR_BGR2BGRA)

        # Random resize of plant
        f = self._class_random.uniform(resize_range[0], resize_range[1])
        plant = cv2.resize(plant, (int(plant.shape[1] * f), int(plant.shape[0] * f)),
                           interpolation=cv2.INTER_CUBIC)

        # Position of plant (ToDo: Make sure that it is not on top of the rumex plants?)
        x_plant, y_plant = self._get_random_root_center(img_size)
        center_offset = 0
        angle = self._class_random.randint(0, 360)
        plant_img = self._position_leave(plant, (img_size), x_plant, y_plant, center_offset, angle)

        # Get polygon
        polygons = self._get_polygon(plant_img)
        if "poisson_bb" in list(self._blending_probs.keys()):
            plant_bb = cv2.cvtColor(cv2.imread(plant_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
            plant_img_bb = self._position_leave(plant_bb, (img_size), x_plant, y_plant, center_offset, angle)
        else:
            plant_img_bb = None

        # Blending
        image_composition = self._blend(plant_img, plant_img_bb, image_composition)

        return image_composition, polygons

    def _add_rumex_plant(self, image_composition):
        if self._class_random.uniform(0, 1) > self._whole_plants_prob:
            image_composition, polygons = self._add_arti_plant_composition(image_composition)
            pass
        else:
            rumex_plant_file = self._class_random.choice(
                glob.glob(f"{self._path_to_rumex_crops}/plants/*_bb.png"))
            image_composition, polygons = self._add_plant(image_composition, rumex_plant_file, [1.0, 1.0])
        return image_composition, polygons

    def _add_non_rumex_plant(self, image_composition, plant_mode="weed", scale_range=[0.5, 2.0]):
        if plant_mode == "weed":
            folder = f"{self._non_rumex_crops_folder}/weeds"
        elif plant_mode == "grass":
            folder = f"{self._non_rumex_crops_folder}/grass"
        else:
            return image_composition
        non_rumex_crop_files = glob.glob(f"{folder}/**/*.png", recursive=True)
        non_rumex_file = self._class_random.choice(non_rumex_crop_files)
        image_composition, polygon = self._add_plant(image_composition, non_rumex_file, scale_range)
        return image_composition

    def _add_shape_distractor_object(self, image_composition):
        img_size = image_composition.shape[0:2]
        x_plant, y_plant = self._get_random_root_center(img_size)
        background = cv2.imread(self._class_random.choice(glob.glob(f"{self._path_to_background_imgs}/*.jpg")),
                                cv2.IMREAD_UNCHANGED)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        if background.shape[0] != img_size[0]:
            background = np.swapaxes(background, 0, 1)
        leave, leave_bb = self._leave_sampler.sample_leave(self._class_random, False)
        center_offset = self._class_random.randint(0, 300)
        angle = self._class_random.randint(0, 360)
        leave_img = self._position_leave(leave, (img_size), x_plant, y_plant, center_offset, angle)
        distractor_leave = self._cut_shape(background, leave_img)

        if self._blending_probs["poisson_bb"] > 0:
            leave_img_bb = self._position_leave(leave_bb, (img_size), x_plant, y_plant, center_offset, angle)
            distractor_leave_bb = self._cut_shape(background, leave_img_bb)
        else:
            distractor_leave_bb = None

        # Blending
        image_composition = self._blend(distractor_leave, distractor_leave_bb, image_composition)
        return image_composition

    def _cut_shape(self, img, shape_img):
        gray = cv2.cvtColor(shape_img, cv2.COLOR_BGRA2GRAY)
        mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
        cut_out_shape = cv2.bitwise_or(img, img, mask=mask)
        return cut_out_shape

    def _reset_class_seed(self):
        seed = self._seed_random.randint(0, sys.maxsize * 2 + 1)
        self._class_random.seed(seed)

    def generate_img_composition(self):
        self._reset_class_seed()
        # Select Background image
        background = cv2.imread(self._class_random.choice(glob.glob(f"{self._path_to_background_imgs}/*.jpg")),
                                cv2.IMREAD_UNCHANGED)
        augm = albu.Compose([albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=[-180, 180], p=1.0)])
        background = augm(image=background)["image"]
        image_composition = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        # Add shape distractor
        self._reset_class_seed()
        if self._add_shape_distractor:
            number_of_distr = self._class_random.randint(0, 20)
            for i in range(number_of_distr):
                image_composition = self._add_shape_distractor_object(image_composition)

        # Add non-Rumex plants
        self._reset_class_seed()
        if self._add_weed_distractor:
            number_non_rumex_plants = self._class_random.randint(0, 5)
            for i in range(number_non_rumex_plants):
                image_composition = self._add_non_rumex_plant(image_composition, plant_mode="weed")

        # Add Rumex plants
        self._reset_class_seed()
        polygon_list = []
        number_rumex_plants = self._class_random.randint(1, 3)
        for i in range(number_rumex_plants):
            image_composition, polygons = self._add_rumex_plant(image_composition)
            if polygons is not None:
                polygon_list.extend(polygons)

        # Add grass
        self._reset_class_seed()
        if self._add_grass_distractor:
            number_of_grass = self._class_random.randint(0, 40)
            for i in range(number_of_grass):
                image_composition = self._add_non_rumex_plant(image_composition, plant_mode="grass",
                                                              scale_range=[0.2, 1.0])

        return image_composition, polygon_list

    def show_img_composition(self, image_composition):
        image_composition = cv2.cvtColor(image_composition, cv2.COLOR_BGR2RGB)
        plt.imshow(image_composition)
        plt.show()

    def save_datapoint(self, image_composition, polygon_list):
        # Save compositions with different blending methods
        id = len(glob.glob(f"{self._output_path}/*.jpg"))
        img_name = f"img_fake_{id}.jpg"
        cv2.imwrite(f"{self._output_path}/{img_name}", image_composition)
        annotation = Annotation(img_name, image_composition.shape[0], image_composition.shape[1], None, polygon_list)
        AnnotationConverter.extend_cvat(annotation, self._ann_file)