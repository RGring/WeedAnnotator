import glob
import shutil
import random
import os

def load_data_subset(subset_file):
    data_path = os.path.dirname(subset_file)
    with open(f'{subset_file}', 'r+') as f:
        list_imgs = [line.replace("\n", "") for line in f.readlines()]

    return list_imgs

if __name__ == "__main__":

    #Aerial Farmland
    img_folder = "/home/rog/data/aerial_farmland/val/images/rgb"
    copy_folder = "/media/rog/SONY_16/test_data_network_similarity/aerial_farmland/sample_imgs"
    img_pahts = glob.glob(f"{img_folder}/*.jpg")

    #Grassland Europe
    img_folder = "/home/rog/data/grassland_imgs/rumex_classification/scraped"
    copy_folder = "/media/rog/SONY_16/test_data_network_similarity/grassland_europe/sample_imgs"
    img_pahts = load_data_subset(f"{img_folder}/val_subset_1.0_default.txt")
    img_pahts = [f"{img_folder}/{img_id}" for img_id in img_pahts]

    num_imgs = 4

    imgs = random.choices(img_pahts, k=num_imgs)
    for img in imgs:
        shutil.copyfile(img, f"{copy_folder}/{os.path.basename(img)}")

