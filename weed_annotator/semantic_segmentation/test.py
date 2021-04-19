import argparse
import cv2
import json
import math
import os
import shutil
import torch
import wandb
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from weed_annotator.semantic_segmentation.dataset.weed_data_set import WeedDataset
from weed_annotator.semantic_segmentation.dataset.sugar_beet_dataset import SugarBeetDataset
from weed_annotator.semantic_segmentation.dataset.aerial_farmland_dataset import AerialFarmlandDataset
from weed_annotator.semantic_segmentation import optimizer, metrics, utils, aug, losses
from weed_annotator.semantic_segmentation.models.xResnet_encoder import xResnetEncoder
import matplotlib.pyplot as plt
import numpy as np
import apex

#os.environ['WANDB_MODE'] = 'dryrun'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["LANG"] = "C.UTF-8"
os.environ["LANGUAGE"] = "C.UTF-8"


def train_network(config):
    global learnable_class_weights
    # Setting seed for reproducability
    utils.set_seeds()

    logger, tb_writer, dump_checkpoints = utils.init_logging(config)

    # Data + Augmentations
    dataset_class = WeedDataset
    test_data = config['data']['val_data'].replace("val_subset", "test_subset")
    if "sugar_beet" in config['data']['train_data']:
        dataset_class = SugarBeetDataset
    elif "aerial_farmland" in config['data']['train_data']:
        dataset_class = AerialFarmlandDataset
        test_data = config['data']['val_data'].replace("val_subset", "test_subset")

    val_dataset = dataset_class(
        utils.load_img_list(test_data),
        labels_to_consider=config["data"]["labels_to_consider"],
        augmentation=aug.get_validation_augmentations(config["data"]["aug"]),
        skip_background=config["data"]["skip_background"]
    )

    valid_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                              num_workers=config["training"]["num_workers"])

    logger.info(f"Building data done with TEST: {len(val_dataset)} images loaded.")

    # Model and pretrained weights
    pretrained_weights = config["arch"]["pretrained_weights"]
    if pretrained_weights == "imagenet":
        encoder_weights = pretrained_weights
    else:
        encoder_weights = None
    num_classes = len(config["data"]["labels_to_consider"]) + 1  # + Background class

    model = smp.__dict__[config["arch"]["type"]](
        encoder_name=config["arch"]["encoder"],
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=config["training"]["activation"],
    )

    model.to(DEVICE)

    # Loss
    weights = None
    if config["training"]["class_weights"]:
        if len(config["training"]["class_weights"]) != len(val_dataset.label_dict):
            logger.error("The number of class weights must fit the number of labels.")
            return -1
        else:
            inv_weights = [1/num for num in config["training"]["class_weights"]]
            weights = torch.tensor(inv_weights, dtype=torch.float32)
            weights = weights / weights.sum()
            weights = weights.cuda()

    loss_func = config["training"]["loss_func"]
    if loss_func == "lovasz":
        criterion = losses.LovaszLoss()
    elif loss_func == "dice":
        criterion = smp.utils.losses.DiceLoss()
    elif loss_func == "weighted_dice":
        if weights != None:
            learnable_class_weights = None
            criterion = losses.WeightedDiceLoss(weight=weights)
        else:
            criterion = losses.WeightedDiceLoss()
            learnable_class_weights = [torch.zeros((1,), requires_grad=True, device=DEVICE) for i in range(num_classes)]
            trainable_params.append({"params": learnable_class_weights})
    elif loss_func == "jaccard":
        criterion = smp.utils.losses.JaccardLoss()
    elif loss_func == "cross_entropy":
        if config["training"]["class_weights"]:
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    elif loss_func == "BCEWithLogitsLoss":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss_func == "focal_loss":
        criterion = losses.FocalLoss()

    # Resume
    to_restore = {"epoch": 0, "best_iou": 0.0}
    utils.restart_from_checkpoint(
        f'{config["logging"]["log_path"]}/{config["identifier"]}/checkpoints/checkpoint-best.pth.tar',
        run_variables=to_restore,
        state_dict=model
    )
    start_epoch = to_restore["epoch"]
    best_iou = to_restore["best_iou"]

    val_loss, val_iou = val(valid_loader, model, criterion, logger, config)

    logger.info(f"Testing finished with mean IoU of {val_iou}.")


def val(valid_loader, model, criterion, logger, config):
    losses = utils.AverageMeter()
    if config["training"]["metric"] == "iou_per_batch":
        mean_ious = utils.AverageMeter()
    elif config["training"]["metric"] == "iou_per_ds":
        class_labels = valid_loader.dataset.label_dict.values()
        inter_per_class = np.zeros(len(class_labels))
        union_per_class = np.zeros(len(class_labels))

    model.eval()
    with torch.no_grad():
        for i, (inp, target, valid_mask) in enumerate(valid_loader):
            inp = inp.to(DEVICE)
            target = target.to(DEVICE).float()
            output = model(inp)

            output_valid = output.clone()
            # output_valid[~valid_mask.unsqueeze(1).expand_as(output_valid).bool()] = -INF_FP16

            # Remove overlapping targets according to Agriculture-Vision Paper
            final_label_predictions = metrics.arg_max(output_valid.clone())
            target = metrics.non_overlapping_target(final_label_predictions, target)

            # Loss
            if config["training"]["loss_func"] in ["cross_entropy", "focal_loss"]:
                loss = criterion(output_valid, torch.argmax(target, dim=1).long())
            elif config["training"]["loss_func"] == "weighted_dice":
                loss = criterion(output_valid, target, weight=learnable_class_weights)
            else:
                loss = criterion(output_valid, target)
            if config["training"]["metric"] == "iou_per_batch":
                miou = metrics.mIoU_per_batch(final_label_predictions, target)
                mean_ious.update(miou.item(), inp.size(0))
            elif config["training"]["metric"] == "iou_per_ds":
                inter_per_class_now, union_per_class_now = metrics.inter_union_per_class(final_label_predictions, target)
                inter_per_class += inter_per_class_now
                union_per_class += union_per_class_now

            # update stats
            losses.update(loss.item(), inp.size(0))

    # Logging stats
    if config["training"]["metric"] == "iou_per_batch":
        mean_iou = mean_ious.avg
    elif config["training"]["metric"] == "iou_per_ds":
        eps = 1e-7
        inter_per_class = np.add(inter_per_class, eps)
        union_per_class = np.add(union_per_class, eps)
        iou_per_class = inter_per_class / union_per_class
        mean_iou = np.mean(iou_per_class)

    logger.info(f"Val\t"
                f"Loss {losses.avg:.3f}\t"
                f"mIoU {mean_iou:.3f}\t")

    if config["training"]["metric"] == "iou_per_ds":
        str = ""
        for class_label, iou in zip(class_labels, iou_per_class):
            str += f"{class_label}: {iou:.3f}, \t"
        logger.info(str)
    return losses.avg, mean_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-c', '--config', default='configs/seg_config_aerial_farmland.json', type=str,
                        help='Training config, that specifies the training settings.')
    args = parser.parse_args()

    # Loading config
    config = json.load(open(args.config))

    # train
    train_network(config)