import os
import glob
from weed_annotator.annotation_converter.AnnotationConverter import AnnotationConverter
from weed_annotator.annotation_converter.Annotation import Annotation
from weed_annotator.annotation_converter.Polygon import Polygon
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv

def project_mask_on_img(mask, img):
    out = cv2.addWeighted(mask, 1, img, 0.8, 0)
    return out

def rotate_annotation(img, annotation):
    if annotation is not None:
        polygons = annotation.get_polygons()
        new_polygons = []
        for pol in polygons:
            mask = np.zeros(image.shape[0:2], dtype=np.uint8)
            label = pol.get_label()
            pol_points = pol.get_polygon_points_as_array()
            cv2.fillPoly(mask, pts=[pol_points], color=255)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = cnts[0]
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            new_pol_points = cv2.approxPolyDP(cnt, epsilon, True)
            pol = Polygon(label)
            pol.set_polygon_points_as_array(new_pol_points)
            new_polygons.append(pol)
        annotation.polygon_list = new_polygons
    return annotation

if __name__ == "__main__":
    base_folder = "/home/rog/data/data_iphone6/annotated_rumex"
    weed_label = "rumex_leaf"

    img_info = {}
    ann_file = f"{base_folder}/annotations.xml"
    img_list = glob.glob(f"{base_folder}/imgs/*.jpg")
    for img in img_list[0:10]:
        annotation = AnnotationConverter.read_cvat_by_id(ann_file, os.path.basename(img))
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        w, h, _ = image.shape
        if w > h:
            annotation = rotate_annotation(image, annotation)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        mask = np.zeros(image.shape, dtype=np.uint8)
        if annotation is not None:
            polygons = annotation.get_polygons()
            for pol in polygons:
                # if pol.get_label() != weed_label:
                #     continue
                pol.set_label("rumex_leaf")
                cv2.fillPoly(mask, pts=[pol.get_polygon_points_as_array()], color=(100, 100, 100))
        plt.imshow
        # fields = [os.path.basename(img), location[i], date[i]]
        # with open(f'{base_folder}/img_info.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(fields)
        # cv2.imwrite(f"{base_folder}/imgs/{os.path.basename(img)}", image)
        # out = project_mask_on_img(mask, image)
        # plt.imshow(out)
        # plt.show()




