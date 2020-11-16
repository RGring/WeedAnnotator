import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch

from weed_annotator.semantic_segmentation import utils
from weed_annotator.semantic_segmentation.weed_data_set import WeedDataset
from weed_annotator.semantic_segmentation import aug

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


def inference(config_file, model_file, input_data, output):

    debug = False

    config = json.load(open(f"{config_file}"))

    encoder = config["arch"]["args"]["encoder"]
    encoder_weights = config["arch"]["args"]["encoder_weights"]

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    model = smp.PSPNet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=1,
        activation=config["training"]["activation"],
    )
    model.load_state_dict(torch.load(f"{model_file}")["model_state_dict"])
    model.to(DEVICE)

    infer_dataset = WeedDataset(
        [input_data],
        weed_label=config["data"]["weed_label"],
        augmentation=aug.get_validation_augmentations(config["data"]["aug"]),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    for i in range(len(infer_dataset)):
        image, gt_mask, id = infer_dataset.get_img_and_props(i)

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='sample_data/train_log/sample_training/config.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--model', default='sample_data/train_log/sample_training/checkpoints/best.pth', type=str,
                        help='Checkpoint that will be loaded')
    parser.add_argument('-i', '--input_data', default='sample_data/imgs_val', type=str,
                        help='Folder where to save predcitions')
    parser.add_argument('-o', '--output', default='sample_data/mask_proposals/raw', type=str,
                        help='Folder where to save predcitions')

    args = parser.parse_args()

    utils.set_seeds()

    inference(args.config, args.model, args.input_data, args.output)

