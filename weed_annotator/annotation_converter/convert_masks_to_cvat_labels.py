from weed_annotator.annotation_converter.AnnotationConverter import AnnotationConverter

if __name__ == "__main__":
    mask_folder = "/home/rog/Documents/courses/02506_AIA/AIA-project/rumex_data/out/*.png"
    ann_file = "/home/rog/Documents/courses/02506_AIA/AIA-project/rumex_data/predictions.xml"
    AnnotationConverter.mask_to_cvat(mask_folder, ann_file, "rumex_leaf")
