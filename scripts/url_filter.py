import re
import requests
import cv2
import numpy as np
import glob
import os
import time
import random

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

if __name__ == "__main__":
    output_folder = "/home/rog/data/grassland_imgs/rumex_classification/scraped"
    reg_exp = "https:\/\/bs\.plantnet\.org\/image\/o\/.[a-z0-9]*"
    html_files = glob.glob("/home/rog/data/grassland_imgs/rumex_classification/scraped/*.html") #plantnet_urtica_urens_plant.html"

    for html_file in html_files:
        print(html_file)
        folder_name = os.path.basename(html_file)
        folder_name = folder_name.replace(".html", "").replace("_plant", "").replace("_leaf", "")
        folder_name = f"{output_folder}/{folder_name}"
        os.makedirs(folder_name, exist_ok=True)
        with open(html_file, 'r') as file:
            data = file.read()

        url_list = re.findall(reg_exp, data)

        for url_str in url_list:
            id = url_str.split("/")[-1]
            file_name = f'{folder_name}/{id}.jpg'
            if os.path.exists(file_name):
                continue
            img_data = requests.get(url_str).content
            image = np.asarray(bytearray(img_data), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = scale_to(image, 244)
            image = center_crop(image, [244, 244])
            cv2.imwrite(file_name, image)
            with open(f'{folder_name}/urls.txt', 'a') as f:
                f.write(f"{url_str}\n")
            # time.sleep(random.uniform(0.0, 0.1))



