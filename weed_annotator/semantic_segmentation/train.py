import argparse
import json
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os

from weed_annotator.semantic_segmentation.weed_data_set import WeedDataset
from weed_annotator.semantic_segmentation.losses import LovaszLoss
from weed_annotator.semantic_segmentation import utils, aug
from catalyst import dl, metrics

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        # model train/valid step
        x, y = batch
        y_hat = self.model(x)

        loss = LovaszLoss()(y_hat, y)
        iou = metrics.iou(y_hat, y, threshold=0.5, activation=None)
        self.batch_metrics.update(
            {"loss": loss, "iou": iou}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

def train(config):
    log_path = f"{config['logging_path']}/{config['train_ident']}"
    os.makedirs(log_path, exist_ok=True)

    # Saving config
    with open(f"{log_path}/config.json", 'w') as f:
        json.dump(config, f)

    # Extracting relevant config params
    encoder = config["arch"]["args"]["encoder"]
    encoder_weights = config["arch"]["args"]["encoder_weights"]

    # ToDo Make possible to train on several folders/annotation files
    train_data = config["data"]["train_data"]
    val_data = config["data"]["val_data"]

    # Create model
    model = smp.PSPNet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=1,
        activation=config["training"]["activation"],
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = WeedDataset(
        train_data,
        weed_label=config["data"]["weed_label"],
        augmentation=aug.get_training_augmentations(config["data"]["aug"]),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    val_dataset = WeedDataset(
        val_data,
        weed_label=config["data"]["weed_label"],
        augmentation=aug.get_validation_augmentations(config["data"]["aug"]),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=1)
    valid_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=1)
    loaders = {"train": train_loader, "valid": valid_loader}

    lr  = config["training"]["lr"]
    trainable_params = [{'params': filter(lambda p: p.requires_grad, model.decoder.parameters())},
                        {'params': filter(lambda p: p.requires_grad, model.encoder.parameters()),
                         'lr': lr / 10}]

    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=lr*0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)

    # model training
    runner = CustomRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        logdir=log_path,
        num_epochs=config["training"]["epochs"],
        verbose=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='configs/seg_config.json', type=str,
                        help='The config the model has been trained with.')
    args = parser.parse_args()

    # Setting seed for reproducability
    utils.set_seeds()

    # Loading config
    config = json.load(open(args.config))

    # train
    train(config)
