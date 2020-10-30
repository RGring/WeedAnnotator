import json
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os

import logging
logging.basicConfig(format='%(asctime)s %(message)s')

from semantic_segmentation.weed_data_set import WeedDataset
from semantic_segmentation import aug
from semantic_segmentation.losses import LovaszLoss
from semantic_segmentation import utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(config, logger):
    log_path = f"{config['logging_path']}/{config['train_ident']}"
    os.makedirs(log_path, exist_ok=True)

    # Saving config
    with open(f"{log_path}/config.json", 'w') as f:
        json.dump(config, f)

    # Extracting relevant config params
    encoder = config["arch"]["args"]["encoder"]
    encoder_weights = config["arch"]["args"]["encoder_weights"]

    # ToDo Make possible to train on several folders/annotation files
    data_folder = config["data"]["data_folder"]
    train_split = config["data"]["train_split"][0]
    val_split = config["data"]["val_split"][0]

    # Create model
    model = smp.PSPNet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=1,
        activation=config["training"]["activation"],
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = WeedDataset(
        f"{data_folder}/{train_split}",
        f"{data_folder}/{train_split}/annotations.xml",
        weed_label=config["data"]["weed_label"],
        augmentation=aug.get_training_augmentations(config["data"]["aug"]),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    val_dataset = WeedDataset(
        f"{data_folder}/{val_split}",
        f"{data_folder}/{val_split}/annotations.xml",
        weed_label=config["data"]["weed_label"],
        augmentation=aug.get_validation_augmentations(config["data"]["aug"]),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    loss = LovaszLoss()

    # ToDo: Compute IoU over whole dataset.
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001, weight_decay=0.00001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_iou_score = 0

    for i in range(config["training"]["epochs"]):
        logger.info(f'Epoch: {i}')

        train_logs = train_epoch.run(train_loader)
        if config["training"]["val"] and i % config["training"]["val_every_epoch"] == 0:
            valid_logs = valid_epoch.run(valid_loader)
            torch.save(model, f"{log_path}/recent_model.pth")

            if valid_logs["iou_score"] > max_iou_score:
                max_iou_score = valid_logs["iou_score"]
                logger.info(f"New best model with IoU: {max_iou_score}!")
                torch.save(model, f"{log_path}/best_model.pth")


if __name__ == "__main__":
    # create logger
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)

    # Setting seed for reproducability
    utils.set_seeds()

    # Loading config
    config = json.load(open("configs/seg_config.json"))

    # train
    train(config, logger)
