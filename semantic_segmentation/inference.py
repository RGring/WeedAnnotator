import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch

import logging
logging.basicConfig(format='%(asctime)s %(message)s')

from semantic_segmentation import utils
from semantic_segmentation.weed_data_set import WeedDataset
from semantic_segmentation import aug

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def save_mask(mask, folder, id):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    colored_mask[mask > 0] = (0, 0, 128)

    id = id.replace(".jpg", ".png")
    cv2.imwrite(f"{folder}/{id}", colored_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='sample_data/train_log/sample_training/config.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--model', default='sample_data/train_log/sample_training/recent_model.pth', type=str,
                        help='Checkpoint that will be loaded')
    parser.add_argument('-o', '--output', default='sample_data/mask_proposals', type=str,
                        help='Folder where to save predcitions')

    args = parser.parse_args()

    debug = False
    utils.set_seeds()

    output = args.output

    config = json.load(open(f"{args.config}"))

    encoder = config["arch"]["args"]["encoder"]
    encoder_weights = config["arch"]["args"]["encoder_weights"]

    data_folder = config["data"]["data_folder"]
    infer_split = config["data"]["infer_split"][0]

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    model = torch.load(f"{args.model}")

    infer_dataset = WeedDataset(
        f"{data_folder}/{infer_split}",
        utils.get_ann_file(data_folder, infer_split),
        weed_label=config["data"]["weed_label"],
        augmentation=aug.get_validation_augmentations(config["data"]["aug"]),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    for i in range(len(infer_dataset)):
        image, gt_mask, id = infer_dataset.get_item_and_props(i)

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        if debug:
            visualize(
                image=image.squeeze().swapaxes(0, 1).swapaxes(1, 2),
                ground_truth_mask=gt_mask,
                predicted_mask=pr_mask
            )

        if id["rotate"]:
            pr_mask = cv2.rotate(pr_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pr_mask = cv2.resize(pr_mask, (id["img_height"], id["img_width"]), interpolation=cv2.INTER_LINEAR)
        save_mask(pr_mask, output, os.path.basename(id["img_id"]))
