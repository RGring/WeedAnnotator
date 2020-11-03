import argparse
import glob
import cv2
import os
from post_processing.post_processor import PostProcessor
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s %(message)s')

def mask_on_img(mask, img):
    out = cv2.addWeighted(mask, 1, img, 0.8, 0)
    return out

def post_process_masks(img_folder, mask_folder, output_folder):
    debug = False
    path_to_imgs = img_folder
    mask_files = glob.glob(f"{mask_folder}/*.png")

    if not mask_files:
        logging.warning("Mask folder empty. No post-processing action.")

    for mask_file in mask_files:
        id = os.path.basename(mask_file).replace(".png", "")
        mask_raw = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(f"{path_to_imgs}/{id}.jpg", cv2.IMREAD_UNCHANGED)
        if img is None:
            logging.warning(f"No img with id {id}")

        mask_post_proc = PostProcessor.post_graphcut(img, mask_raw)
        mask_post_proc = PostProcessor.post_open(mask_post_proc)

        cv2.imwrite(f"{output_folder}/{id}.png", mask_post_proc)

        if debug:
            raw_on_img = cv2.cvtColor(mask_on_img(mask_raw, img), cv2.COLOR_BGR2RGB)
            pp_on_img = cv2.cvtColor(mask_on_img(mask_post_proc, img), cv2.COLOR_BGR2RGB)

            f, ax = plt.subplots(1, 2)
            ax[0].imshow(raw_on_img)
            ax[0].set_title("Raw Mask")
            ax[1].imshow(pp_on_img)
            ax[0].set_title(f"Post-processed Mask {id}")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-Processing')
    parser.add_argument('-m', '--mask_folder', default='sample_data/mask_proposals/post_processed', type=str,
                        help='Generated mask proposals')
    parser.add_argument('-i', '--img_folder', default='sample_data/imgs_val', type=str,
                        help='Corresponding input images of mask proposals')
    parser.add_argument('-o', '--output', default='sample_data/mask_proposals', type=str,
                        help='Folder where to save post-processed mask proposals')

    args = parser.parse_args()
    post_process_masks(args.img_folder, args.mask_folder, args.output)

