import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch

from weed_annotator.semantic_segmentation.models.xResnet_encoder import xResnetEncoder
from weed_annotator.semantic_segmentation import utils
from weed_annotator.semantic_segmentation.dataset.weed_data_set import WeedDataset
from weed_annotator.semantic_segmentation.dataset.sugar_beet_dataset import SugarBeetDataset
from weed_annotator.semantic_segmentation import aug

COLORS = [
    (0, 0, 0),
    (100, 0, 0),
    (0, 0, 100),
    (0, 100, 0)
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mask_to_color(pred_mask, skip_background= True):
    num_classes = pred_mask.shape[0]
    img = np.zeros((pred_mask.shape[1], pred_mask.shape[2], 3), dtype=np.uint8)

    for i in range(3):
        if i == 0 and skip_background:
            continue
        img[np.where(pred_mask[i, :, :] == 1)] = COLORS[i]
    return img

def save_mask_on_img(mask, folder, img_props):
    img_file = img_props["img_id"]
    img = cv2.imread(img_file)
    img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    mask_on_img = cv2.addWeighted(mask, 1, img, 0.8, 0)
    cv2.imwrite(f"{folder}/{os.path.basename(img_file)}", mask_on_img)


def inference(config, model_file, input_data, output):

    debug = False
    encoder = config["arch"]["encoder"]
    num_classes = len(config["data"]["labels_to_consider"]) + 1 #Background class
    model = smp.__dict__[config["arch"]["type"]](
        encoder_name=encoder,
        encoder_weights=None,
        classes=num_classes,
        activation=config["training"]["activation"],
    )

    checkpoint = torch.load(f"{model_file}", map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)

    dataset_class = WeedDataset
    if "sugar_beet" in input_data:
        dataset_class = SugarBeetDataset

    infer_dataset = dataset_class(
        utils.load_img_list(input_data),
        labels_to_consider=config["data"]["labels_to_consider"],
        augmentation=aug.get_validation_augmentations(config["data"]["aug"]),
        skip_background=True
    )

    for i in range(len(infer_dataset)):
        image, gt_mask, img_props = infer_dataset.get_img_mask_props(i)

        x_tensor = image.to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)

        pr_argmax = torch.argmax(pr_mask, dim=1)
        for i in range(pr_mask.size(1)):
            pr_mask[:, i, :, :] = (pr_argmax == i).type(torch.uint8)
        pr_mask = (pr_mask.squeeze().cpu().numpy())

        pr_mask = mask_to_color(pr_mask)
        gt_mask = mask_to_color(gt_mask.squeeze().cpu().numpy())

        if img_props["rotate"]:
            pr_mask = cv2.rotate(pr_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt_mask = cv2.rotate(gt_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        save_mask_on_img(pr_mask, output, img_props)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='configs/seg_config.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--model', default='/home/rog/ckp_0.6.pth.tar', type=str,
                        help='Checkpoint that will be loaded')
    parser.add_argument('-i', '--input_data', default='/home/rog/data/sugar_beet/val_subset_0.01_default.txt', type=str,
                        help='Folder where to save predcitions')
    parser.add_argument('-o', '--output', default='/home/rog/temp', type=str,
                        help='Folder where to save predcitions')

    args = parser.parse_args()

    utils.set_seeds()

    config = json.load(open(args.config))

    inference(config, args.model, args.input_data, args.output)

