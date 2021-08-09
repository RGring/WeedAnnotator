import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import random
import math

def plot_examples(num_sampels):
    folder = "/home/rog/data/aerial_farmland/train"
    img_ids = glob.glob(f"{folder}/images/rgb/*.jpg")
    img_ids = [img_id.replace(f"{folder}/images/rgb/", "") for img_id in img_ids]

    label_names = ["cloud_shadow", "double_plant", "planter_skip", "standing_water", "waterway", "weed_cluster"]
    colors = [
        (100, 0, 0),
        (0, 100, 0),
        (0, 0, 100),
        (100, 0, 100),
        (0, 100, 100),
        (100, 100, 0)
    ]
    # Plot sample images
    for img_id in random.choices(img_ids, k=num_sampels):
        rgb = cv2.imread(f"{folder}/images/rgb/{img_id}")
        mask = cv2.imread(f"{folder}/masks/{img_id.replace('.jpg', '.png')}")
        # if len((np.where(mask[:, :, 0] == 0))[0]) == 0:
        #     continue
        bound = cv2.imread(f"{folder}/boundaries/{img_id.replace('.jpg', '.png')}")
        if len((np.where(bound[:, :, 0] == 0))[0]) == 0:
            continue
        label = np.zeros(rgb.shape)
        for i, label_name in enumerate(label_names[1:]):
            label_temp = cv2.imread(f"{folder}/labels/{label_name}/{img_id.replace('.jpg', '.png')}")
            label_temp = label_temp[:, :, 0]
            label[np.where(label_temp == 255)] = colors[i]
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4)
        ax1.imshow(rgb)
        ax2.imshow(mask)
        ax3.imshow(bound)
        ax4.imshow(label)
        plt.show()
    pass


if __name__ == "__main__":
    # plot_examples(50)
    folder = "/home/rog/data/aerial_farmland/val"
    img_ids = glob.glob(f"{folder}/images/rgb/*.jpg")
    img_ids = [img_id.replace(f"{folder}/images/rgb/", "") for img_id in img_ids]

    label_names = ["cloud_shadow", "double_plant", "planter_skip", "standing_water", "waterway", "weed_cluster"]

    mode = "default"
    data_perc = 0.5

    pixel_count = np.zeros((len(label_names) + 1, 1))
    class_count = np.zeros((len(label_names), 1))
    classes_per_img = np.zeros((len(label_names), 1))
    img_classes = {}
    for label_name in label_names:
        img_classes[label_name] = []
    for img_id in img_ids:
        class_count_temp = np.zeros((len(label_names), 1))
        for i, label_name in enumerate(label_names):
            label_temp = cv2.imread(f"{folder}/labels/{label_name}/{img_id.replace('.jpg', '.png')}")
            pixel_sum = np.sum(label_temp[:, :, 0] == 255)
            if pixel_sum > 0:
                class_count_temp[i] = pixel_sum
        num_classes = np.sum(class_count_temp != 0) - 1
        if num_classes == 0:
            classes_per_img[num_classes] += 1
        else:
            continue

        if np.sum(class_count_temp) > 0:
            max_ind = np.argmax(class_count_temp)
            class_count[max_ind] += 1
            pixel_count[1:] += class_count_temp
            pixel_count[0] += label_temp.shape[0]*label_temp.shape[1] - np.sum(class_count_temp)
            img_classes[label_names[max_ind]].append(img_id)
        else:
            class_count[0] +=1
    print("Class count")
    print(class_count)
    print("Pixel count")
    print(pixel_count)
    if mode == "balanced":
        min_num_imgs = math.inf
        for img_pathes in img_classes.values():
            num_imgs = len(img_pathes)
            if num_imgs < min_num_imgs:
                min_num_imgs = num_imgs
        for key, img_pathes in img_classes.items():
            img_classes[key] = random.choices(img_pathes, k=min_num_imgs)

    # with open(f'{folder}/images/train_class_subset_{data_perc}_fold1_{mode}.txt', 'w') as f:
    #     for class_id, img_files in img_classes.items():
    #         k = int(len(img_files)*data_perc)
    #         img_files_temp = random.choices(img_files, k=k)
    #         for img in img_files_temp:
    #             f.write("rgb/%s %s\n" % (img, class_id))


    for class_id, img_files in img_classes.items():
        random.shuffle(img_files)

    img_split = int(len(img_files)*data_perc)
    with open(f'{folder}/images/val_class_subset_{data_perc}_fold1_{mode}.txt', 'w') as f:
        for class_id, img_files in img_classes.items():
            img_split = int(len(img_files) * data_perc)
            for img in img_files[:img_split]:
                f.write("rgb/%s %s\n" % (img, class_id))
    with open(f'{folder}/images/val_class_subset_{data_perc}_fold2_{mode}.txt', 'w') as f:
        for class_id, img_files in img_classes.items():
            img_split = int(len(img_files) * data_perc)
            for img in img_files[img_split:]:
                f.write("rgb/%s %s\n" % (img, class_id))

    for class_label, count in zip(label_names, class_count):
        print(f"{class_label}: {count}")
    print(sum(class_count))
    print(classes_per_img)
    print(sum(classes_per_img))




