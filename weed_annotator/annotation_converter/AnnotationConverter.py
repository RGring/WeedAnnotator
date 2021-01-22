import glob
import cv2
import json
import os
import numpy as np
import xml.etree.cElementTree as ET
from weed_annotator.annotation_converter.BoundingBox import BoundingBox
from weed_annotator.annotation_converter.Polygon import Polygon
from weed_annotator.annotation_converter.Annotation import Annotation


class AnnotationConverter:
    @staticmethod
    def write_cvat(annotations, annotation_file):
        labels = []
        root = AnnotationConverter._init_cvat([])
        for annotation in annotations:
            image = ET.SubElement(root, "image", name="%s" % (annotation.get_image_name().replace(".png", ".jpg")),
                                  id="0",
                                  width="%s" % annotation.get_img_width(), height="%s" % annotation.get_img_height())
            polygon_list = annotation.get_polygons()
            for poly in polygon_list:
                label = poly.get_label()
                if label not in labels:
                    labels.append(label)
                points = poly.get_polygon_points()
                poly_string = ""
                for x, y in zip(points["x"], points["y"]):
                    poly_string += "%f, %f;" % (x, y)
                poly_string = poly_string[:-1]
                ET.SubElement(image, "polygon", label=label, points=poly_string, occluded="0", z_order="1")
        AnnotationConverter._add_label_to_cvat(root, labels)
        tree = ET.ElementTree(root)
        tree.write(f"{annotation_file}")

    @staticmethod
    def _init_cvat(label_list):
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "flipped").text = "False"
        ET.SubElement(task, "labels")
        root = AnnotationConverter._add_label_to_cvat(root, label_list)
        owner = ET.SubElement(task, "owner")
        ET.SubElement(owner, "username").text = "user"
        ET.SubElement(owner, "email").text = "user@mail.dk"
        return root

    @staticmethod
    def _add_label_to_cvat(root, label_list):
        for se in root:
            if se.tag == "meta":
                for sse in se:
                    if sse.tag == "task":
                        for ssse in sse:
                            if ssse.tag == "labels":
                                existing_labels = []
                                for sssse in ssse:
                                    for ssssse in sssse:
                                        existing_labels.append(ssssse.text)

                                for label in label_list:
                                    if label in existing_labels:
                                        continue
                                    new_label = ET.SubElement(ssse, "label")
                                    ET.SubElement(new_label, "name").text = label
                                    ET.SubElement(new_label, "attributes")
                                return root
    @staticmethod
    def extend_cvat(ann, path_to_annotation_file):
        if not os.path.isfile(path_to_annotation_file):
            # ToDo: Automatically extract all labels in annotations
            os.makedirs(os.path.dirname(path_to_annotation_file), exist_ok=True)
            root = AnnotationConverter._init_cvat([])
        else:
            root = ET.parse(path_to_annotation_file).getroot()
        image = ET.SubElement(root, "image", name="%s" % (ann.get_image_name()), id="0",
                              width="%s" % ann.get_img_width(),
                              height="%s" % ann.get_img_height())
        polygon_anns = ann.get_polygons()
        if polygon_anns:
            labels = []
            for polygon_ann in polygon_anns:
                label = polygon_ann.get_label()
                if label not in labels:
                    labels.append(label)
                polygon_pts = polygon_ann.get_polygon_points_as_array()
                poly_string = ""
                for point in polygon_pts:
                    # Keeping cvat format
                    x = point[0]
                    y = point[1]
                    poly_string += "%f, %f;" % (x, y)
                poly_string = poly_string[:-1]
                ET.SubElement(image, "polygon", label=label, points=poly_string, occluded="0", z_order="1")
            root = AnnotationConverter._add_label_to_cvat(root, labels)
        tree = ET.ElementTree(root)
        tree.write(path_to_annotation_file)

    @staticmethod
    def get_mask(annotation, label_list, width, height, color=(255, 255, 255)):
        mask = np.zeros((width, height, 3), dtype=np.uint8)
        if annotation is not None:
            polygons = annotation.get_polygons()
            for pol in polygons:
                if pol.get_label() not in label_list:
                    continue
                cv2.fillPoly(mask, pts=[pol.get_polygon_points_as_array()], color=color)
        return mask

    @staticmethod
    def read_cvat_all(path_to_annotation_file):
        root = ET.parse(path_to_annotation_file).getroot()
        annotations = []
        for img in root.findall('image'):
            annotation = AnnotationConverter._cvat_to_annotation(img)
            annotations.append(annotation)
        return annotations

    @staticmethod
    def read_cvat_by_id(path_to_annotation_file, img_id):
        try:
            root = ET.parse(path_to_annotation_file).getroot()
            for img in root.findall('image'):
                if img_id == img.attrib["name"]:
                    annotation = AnnotationConverter._cvat_to_annotation(img)
                    return annotation
        except FileNotFoundError:
            return None

    @staticmethod
    def _cvat_to_annotation(img_xml_info):
        img_width = float(img_xml_info.attrib["width"])
        img_height = float(img_xml_info.attrib["height"])
        polygon_list = []
        for pol in img_xml_info.findall("polygon"):
            polygon_ann = Polygon(pol.attrib["label"])
            points_strs = pol.attrib["points"].split(";")
            for points_str in points_strs:
                points_str = points_str.split(",")
                polygon_ann.add_point(int(float(points_str[0])), int(float(points_str[1])))
            polygon_list.append(polygon_ann)
        annotation = Annotation(img_xml_info.attrib["name"], img_width, img_height, None, polygon_list)
        return annotation

    @staticmethod
    def read_from_mask(label, mask_folder):
        mask_paths = glob.glob(mask_folder)
        mask_paths.sort()
        annotations = []
        for mask_path in mask_paths:
            rgb_mask = cv2.imread(mask_path)
            gray_image = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pol_list = []
            for contour in contours:
                points = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
                if points.shape[0] < 15:
                    continue
                polygon = Polygon(label)
                for p in points[:, 0, :]:
                    polygon.add_point(p[0], p[1])
                pol_list.append(polygon)
            img_name = os.path.basename(mask_path)
            annotations.append(Annotation(img_name, rgb_mask.shape[0], rgb_mask.shape[1], None, pol_list))
        return annotations

    @staticmethod
    def mask_to_cvat(mask_folder, annotation_file, label):
        annotations = AnnotationConverter.read_from_mask(label, mask_folder)
        AnnotationConverter.write_cvat(annotations, annotation_file)

    @staticmethod
    def supervisely_to_cvat(path_to_supervisely_annotations, cvat_save_folder):
        annotations = AnnotationConverter.read_from_supervisely(path_to_supervisely_annotations)
        AnnotationConverter.write_cvat(annotations, cvat_save_folder)

    @staticmethod
    def read_from_supervisely(path_to_annotation_folder):
        annotation_files = glob.glob(f"{path_to_annotation_folder}/*json")
        annotations = []
        for annotation_file in annotation_files:
            bb_list = []
            pol_list = []
            with open(annotation_file) as json_file:
                supervisely_ann = json.load(json_file)

            img_width = supervisely_ann["size"]["width"]
            img_height = supervisely_ann["size"]["height"]

            for object in supervisely_ann["objects"]:
                if object["geometryType"] == "polygon":
                    pol = Polygon(object["classTitle"])
                    for p in object["points"]["exterior"]:
                        pol.add_point(p[0], p[1])
                    pol_list.append(pol)
                if object["geometryType"] == "rectangle":
                    left = object["points"]["exterior"][0][0]
                    top = object["points"]["exterior"][0][1]
                    right = object["points"]["exterior"][1][0]
                    bottom = object["points"]["exterior"][1][1]
                    bb = BoundingBox(object["classTitle"], left, top, (right - left), (bottom - top))
                    bb_list.append(bb)
                    pass
            img_name = os.path.basename(annotation_file).replace(".json", "")
            annotations.append(Annotation(img_name, img_width, img_height, bb_list, pol_list))

        return annotations
