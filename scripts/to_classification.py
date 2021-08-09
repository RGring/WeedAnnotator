import re
import requests
import cv2
import numpy as np
import glob
import os
import time
import random
import matplotlib.pyplot as plt
from weed_annotator.annotation_converter.AnnotationConverter import AnnotationConverter


def scale_to(image, dim):
    w, h, _ = image.shape
    if w > h:
        h_targ = dim
        w_targ = int(w * dim / h)
    else:
        w_targ = dim
        h_targ = int(h * dim / w)
    image = cv2.resize(image, (h_targ, w_targ))
    return image

def center_crop(image, dim):
    w, h, _ = image.shape
    crop_h = dim[0] if dim[0] < h else h
    crop_w = dim[1] if dim[1] < w else w
    mid_x, mid_y = int(h/2), int(w/2)
    ch2, cw2 = int(crop_h/2), int(crop_w/2)
    crop_img = image[mid_y-cw2:mid_y+cw2, mid_x-ch2:mid_x+ch2]
    return crop_img

def annotated_rumex_resize_to(size):
    img_files = glob.glob("/home/rog/data/sugar_beet/iMapCleaned/*.png")
    out_folder = "/home/rog/data/sugar_beet/iMapCleaned_256"
    for img_file in img_files:
        image = cv2.imread(img_file)
        # image = scale_to(image, size)
        # image = center_crop(image, [size, size])
        image = cv2.resize(image, (size, size))
        id = os.path.basename(img_file)
        cv2.imwrite(f"{out_folder}/{id}", image)

def annotated_rumex_grid_crop_resize(grid, isolation_pixel, size):
    output_path = "/home/rog/data/sugar_beet/rgb_crops_3_2"
    base_path = "/home/rog/data/sugar_beet/rgb_temp/rgb_wo_duplicates7"
    # annotation_file = f"{base_path}/annotations.xml"
    img_files = glob.glob(f"{base_path}/*.png")
    for img_file in img_files:
        # img_file = f"{base_path}/imgs/APC_0614_2f9b865cc41f474196e95ca970c6addb.jpg"
        # annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))
        image = cv2.imread(img_file)
        # seg_mask = AnnotationConverter.get_mask(annotation, ["rumex_leaf"], image.shape[0], image.shape[1], (1, 1, 1))
        # if sum(sum(sum(seg_mask))) == 0:
        #     continue
        # plt.imshow(image)
        # plt.show()
        w, h, _ = image.shape
        w_crop = int((w - ((grid[0] - 1) * isolation_pixel[0])) / grid[0])
        h_crop = int((h - ((grid[1] - 1) * isolation_pixel[1])) / grid[1])
        for x_i in range(grid[0]):
            x_offset = isolation_pixel[0] if x_i > 0 else 0
            for y_i in range(grid[1]):
                y_offset = isolation_pixel[1] if y_i > 0 else 0
                crop = image[x_i*w_crop+x_offset*x_i:x_i*w_crop+x_offset*x_i + w_crop, y_i*h_crop+y_offset*y_i: y_i*h_crop + h_crop+y_offset*y_i]
                # mask = seg_mask[x_i*w_crop+x_offset*x_i:x_i*w_crop+x_offset*x_i + w_crop, y_i*h_crop+y_offset*y_i: y_i*h_crop + h_crop+y_offset*y_i]
                crop = scale_to(crop, max(size))
                # mask = scale_to(mask, max(size))
                crop = center_crop(crop, size)
                # mask = center_crop(mask, size)
                class_id = "background"
                # plt.imshow(crop)
                # plt.show()
                # plt.imshow(mask)
                # plt.show()
                # print(mask.sum())
                # if mask.sum() > 200:
                #     class_id = "rumex"
                img_name = os.path.basename(img_file).replace(".jpg", "")
                img_name += f"_{x_i}_{y_i}"
                cv2.imwrite(f"{output_path}/{img_name}.jpg", crop)
        # break

if __name__ == "__main__":
    # annotated_rumex_resize_to(256)
    #annotated_rumex_grid_crop_resize([2, 3], [300, 300], [244, 244])
    annotated_rumex_grid_crop_resize([2, 3], [200, 200], [256, 256])



