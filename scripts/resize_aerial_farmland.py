import cv2
import os
import glob

if __name__ == "__main__":
    size=256
    img_files = glob.glob("/zhome/d6/0/152995/data/sugar_beet/iMapCleaned_orig/*.png")
    out_folder = "/zhome/d6/0/152995/data/sugar_beet/iMapCleaned"
    for img_file in img_files:
        id = os.path.basename(img_file)
        out_file = f"{out_folder}/{id}"
        if os.path.exists(out_file):
            continue
        image = cv2.imread(img_file)
        try:
            image = cv2.resize(image, (size, size))
        except:
            print(f"Error at {img_file}. Skipping...")
            continue
        cv2.imwrite(out_file, image)