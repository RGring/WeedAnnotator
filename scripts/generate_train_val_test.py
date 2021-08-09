import glob
import random
from pathlib import Path
import os
import math

if __name__ == "__main__":
    mode = "balanced"
    train_ratio = 1.0
    val_ratio = 0.0
    perc = 0.01
    path_to_dataset = "/home/rog/data/aerial_farmland/train/images"
    image_extension = "jpg"

    # Get folders in dataset == classes
    class_dirs = [os.path.basename(x[0]) for x in os.walk(path_to_dataset) if x[0] != path_to_dataset]
    class_dirs = ["rgb"]

    minimum_num = math.inf
    if mode == "balanced":
        for class_dir in class_dirs:
            img_num = len(glob.glob(f"{path_to_dataset}/{class_dir}/*.{image_extension}"))
            if img_num < minimum_num:
                minimum_num = img_num



    train = []
    val = []
    test = []
    for class_dir in class_dirs:
        # Img files
        img_files = glob.glob(f"{path_to_dataset}/{class_dir}/*.{image_extension}")
        img_files = [img_file.replace(f"{path_to_dataset}/", "") for img_file in img_files]

        # balancing classes
        if mode == "balanced":
            img_files = random.choices(img_files, k=minimum_num)
        # subset
        k = int(len(img_files)*perc)
        img_files = random.choices(img_files, k=k)

        random.shuffle(img_files)

        # adding images to splits
        train_ind = int(round(train_ratio*len(img_files)))
        val_ind = train_ind + int(round(val_ratio*len(img_files)))
        train.extend(img_files[:train_ind])
        val.extend(img_files[train_ind:val_ind])
        test.extend(img_files[val_ind:])


    with open(f'{path_to_dataset}/train_subset_{perc}_{mode}_fold2.txt', 'w') as f:
        for item in train:
            f.write("%s\n" % item)
    with open(f'{path_to_dataset}/val_subset_{perc}_{mode}.txt', 'w') as f:
        for item in val:
            f.write("%s\n" % item)
    with open(f'{path_to_dataset}/test_subset_{perc}_{mode}.txt', 'w') as f:
        for item in test:
            f.write("%s\n" % item)

    # print(len(train))
    # print(len(val))
    # print(len(val))

