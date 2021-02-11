import argparse
import cv2
import json
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


# ToDo: Possibly extend this function. Depending on pretrained models that should be loaded.
def load_weights(model, path_to_weights):
    state_dict = torch.load(path_to_weights, map_location=torch.device(DEVICE))
    state_dict = state_dict["state_dict"]
    state_dict_enc = {}
    for k1, (k2, v) in zip(model.encoder.state_dict().keys(), state_dict.items()):
        if "backbone" in k2:
            state_dict_enc[k1] = v
    state_dict_enc["fc.bias"] = 0
    state_dict_enc["fc.weight"] = 0
    msg = model.encoder.load_state_dict(state_dict_enc, strict=False)


def train_network_with_wandb(config):
    log_path = f"{config['logging']['log_path']}/{config['identifier']}/wandb"
    os.makedirs(log_path, exist_ok=True)
    if config["hyper_param_optim"]["enabled"]:
        run = wandb.init(config=get_sweep_config(config), dir=log_path, resume=True,
                         settings=wandb.Settings(show_emoji=False))
        sweep_config = wandb.config
        config = update_config_from_sweep_config(config, sweep_config)
    else:
        run = wandb.init(project=config["logging"]["project_name"], name=config["identifier"], config=config,
                         dir=log_path)

    train_network(config)
    run.finish()


def train_network(config):
    # Setting seed for reproducability
    utils.set_seeds()

    logger, tb_writer, dump_checkpoints = utils.init_logging(config)

    # Data + Augmentations
    dataset_class = WeedDataset
    if "sugar_beet" in config['data']['train_data']:
        dataset_class = SugarBeetDataset
    elif "aerial_farmland" in config['data']['train_data']:
        dataset_class = AerialFarmlandDataset

    train_dataset = dataset_class(
        utils.load_img_list(f"{config['data']['train_data']}"),
        labels_to_consider=config["data"]["labels_to_consider"],
        augmentation=aug.get_training_augmentations(config["data"]["aug"]),
        skip_background=config["data"]["skip_background"]
    )

    val_dataset = dataset_class(
        utils.load_img_list(f"{config['data']['val_data']}"),
        labels_to_consider=config["data"]["labels_to_consider"],
        augmentation=aug.get_validation_augmentations(config["data"]["aug"]),
        skip_background=config["data"]["skip_background"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                              num_workers=config["training"]["num_workers"])
    valid_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                              num_workers=config["training"]["num_workers"])

    logger.info(f"Building data done with TRAIN: {len(train_dataset)}, VAL: {len(val_dataset)} images loaded.")

    # Model and pretrained weights
    pretrained_weights = config["arch"]["pretrained_weights"]
    if pretrained_weights == "imagenet":
        encoder_weights = pretrained_weights
    else:
        encoder_weights = None
    num_classes = len(config["data"]["labels_to_consider"]) + 1  # Background class

    if config["arch"]["type"] == "BiSeNetV1":
        # ToDo
        model = BiSeNetV1(num_classes)
        trainable_params = model.parameters()
    else:
        model = smp.__dict__[config["arch"]["type"]](
            encoder_name=config["arch"]["encoder"],
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=config["training"]["activation"],
        )

        if encoder_weights == None and pretrained_weights != "":
            load_weights(model, pretrained_weights)
        trainable_params = [{'params': filter(lambda p: p.requires_grad, model.decoder.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.segmentation_head.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.encoder.parameters()),
                             'lr': config["training"]["lr"] / 10}]

        if config["training"]["optimization"]["lr_encoder"] == 0.0:
            logger.info("Freezing encoder parameters.")
            for param in model.encoder.parameters():
                param.requires_grad = False
    model.to(DEVICE)
    if tb_writer is None:
        wandb.watch(model)
    logger.info(f"Building model done with pretrained_weights: {pretrained_weights}")

    # Optimization
    optim = optimizer.get_optimizer(model, trainable_params, config)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optim, config["training"]["optimization"]["decay_epochs"], gamma=config["training"]["optimization"]["decay_gamma"])

    if config["training"]["use_fp16"]:
        model, optim = apex.amp.initialize(model, optim, opt_level="O1")

    # Loss
    loss_func = config["training"]["loss_func"]
    if loss_func == "lovasz":
        criterion = losses.LovaszLoss()
    elif loss_func == "dice":
        criterion = smp.utils.losses.DiceLoss()
    elif loss_func == "jaccard":
        criterion = smp.utils.losses.JaccardLoss()

    # Resume
    to_restore = {"epoch": 0, "best_iou": 0.0}
    if config["logging"]["resume"]:
        utils.restart_from_checkpoint(
            config["logging"]["resume"],
            run_variables=to_restore,
            state_dict=model,
            optimizer=optim,
            # scheduler=scheduler,
        )
    start_epoch = to_restore["epoch"]
    best_iou = to_restore["best_iou"]

    for epoch in range(start_epoch, config["training"]["epochs"]):
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # train
        scores = train(train_loader, model, optim, criterion, epoch, logger, tb_writer, config["training"]["use_fp16"])

        # save_checkpoint
        save_dict = {
            "config": {
                "arch": config["arch"]["type"],
                "encoder": config["arch"]["encoder"],
                "activation": config["training"]["activation"],
                "input_size": [int(config["data"]["aug"]["input_width"]), int(config["data"]["aug"]["input_height"])],
                "num_classes": num_classes,
            },
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
            # "scheduler": scheduler.state_dict(),
            "best_iou": best_iou
        }
        torch.save(save_dict, os.path.join(dump_checkpoints, "checkpoint-recent.pth.tar"))
        if epoch % config["logging"]["save_ckp_every"] == 0:
            shutil.copyfile(
                os.path.join(dump_checkpoints, "checkpoint-recent.pth.tar"),
                os.path.join(dump_checkpoints, "ckp-" + str(epoch) + ".pth.tar"),
            )

        # val
        if (epoch + 1) % config["training"]["val_every_epoch"] == 0:
            val_loss, val_iou = val(valid_loader, model, criterion, (epoch + 1) * len(train_loader), logger, tb_writer,
                                    config["logging"]["log_images"])
            if val_iou > best_iou:
                best_iou = val_iou
                shutil.copyfile(
                    os.path.join(dump_checkpoints, "checkpoint-recent.pth.tar"),
                    os.path.join(dump_checkpoints, "checkpoint-best.pth.tar"),
                )

        # scheduler.step()
    logger.info(f"Training finished with best mean IoU of {best_iou}.")


def train(train_loader, model, optimizer, criterion, epoch, logger, writer, use_fp16):
    losses = utils.AverageMeter()
    mean_ious = utils.AverageMeter()

    model.train()
    for iter_epoch, (inp, target) in enumerate(train_loader):
        iteration = epoch * len(train_loader) + iter_epoch
        inp = inp.to(DEVICE)
        target = target.to(DEVICE)
        output = model(inp)
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        if use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # step
        optimizer.step()

        # ToDo: How to handle target images, that do not contain the foreground class??
        miou = metrics.mIoU_per_batch(output, target.float(), ignore_channels=[0])

        # update stats
        losses.update(loss.item(), inp.size(0))
        mean_ious.update(miou.item(), inp.size(0))

    if writer is None:
        wandb.log({"loss/train": losses.avg}, step=iteration)
        wandb.log({"mIoU/train": mean_ious.avg}, step=iteration)
    else:
        writer.add_scalar("loss/train", losses.avg, iteration)
        writer.add_scalar("mIoU/train", mean_ious.avg, iteration)

    logger.info(f"Train\t"
                f"Epoch {epoch}\t"
                f"Loss {losses.avg:.3f}\t"
                f"mIoU {mean_ious.avg:.3f}\t"
                f"lr_enc {optimizer.param_groups[2]['lr']}, lr_dec {optimizer.param_groups[0]['lr']}")
    return losses.avg, mean_ious.avg


def val(valid_loader, model, criterion, iteration, logger, writer, log_images=False):
    losses = utils.AverageMeter()
    mean_ious = utils.AverageMeter()

    model.eval()
    log_images = []
    with torch.no_grad():
        for i, (inp, target) in enumerate(valid_loader):
            inp = inp.to(DEVICE)
            target = target.to(DEVICE)
            output = model(inp)
            loss = criterion(output, target)
            miou = metrics.mIoU_per_batch(output, target.float(), ignore_channels=[0])

            # update stats
            losses.update(loss.item(), inp.size(0))
            mean_ious.update(miou.item(), inp.size(0))

            # Logging images
            if log_images:
                index = 1
                img_upload_size = (300, 300)
                image = inp[index, :, :, :].cpu().numpy()
                image = np.swapaxes(image, 0, 1)
                image = np.swapaxes(image, 1, 2)
                image = cv2.resize(image, img_upload_size)

                target = target[index, :, :, :]
                gt_mask = torch.argmax(target, dim=0)
                gt_mask = gt_mask.cpu().numpy().astype(np.uint8)
                gt_mask = cv2.resize(gt_mask, img_upload_size)

                output = output[index, :, :, :]
                pred_mask = torch.argmax(output, dim=0)
                pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
                pred_mask = cv2.resize(pred_mask, img_upload_size)
                if writer is None:
                    wandb_image = wandb.Image(image, masks={
                        "predictions": {
                            "mask_data": pred_mask,
                            "class_labels": valid_loader.dataset.label_dict
                        },
                        "ground_truth": {
                            "mask_data": gt_mask,
                            "class_labels": valid_loader.dataset.label_dict
                        },
                    })
                    log_images.append(wandb_image)
    if writer is None:
        wandb.log({"loss/val": losses.avg}, step=iteration)
        wandb.log({"mIoU/val": mean_ious.avg}, step=iteration)
        if log_images:
            wandb.log({"predictions": log_images}, step=iteration)
    else:
        writer.add_scalar("loss/val", losses.avg, iteration)
        writer.add_scalar("mIoU/val", mean_ious.avg, iteration)

    logger.info(f"Val\t"
                f"Loss {losses.avg:.3f}\t"
                f"mIoU {mean_ious.avg:.3f}\t")
    return losses.avg, mean_ious.avg


def update_config_from_sweep_config(config, sweep_config):
    for k, v in config["hyper_param_optim"]["parameters"].items():
        if not v["enabled"]:
            continue
        update_val = config
        for ident in v["nested_ident"][:-1]:
            update_val = update_val[ident]
        update_val[v["nested_ident"][-1]] = sweep_config[k]
    return config


def get_sweep_config(config):
    unnested_config = {}
    for k, v in config["hyper_param_optim"]["parameters"].items():
        if not v["enabled"]:
            continue
        value_in_nested_config = config
        for ident in v["nested_ident"]:
            value_in_nested_config = value_in_nested_config[ident]
        unnested_config[k] = value_in_nested_config
    return unnested_config


def get_sweep_id(config):
    sweep_config = config["hyper_param_optim"]["settings"]
    sweep_config["parameters"] = {}
    for k, v in config["hyper_param_optim"]["parameters"].items():
        sweep_config["parameters"][k] = {}
        sweep_config["parameters"][k]["values"] = v["values"]
    sweep_id = wandb.sweep(sweep_config, project=config["logging"]["project_name"])
    return sweep_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-c', '--config', default='configs/seg_config.json', type=str,
                        help='Training config, that specifies the training settings.')
    args = parser.parse_args()

    # Loading config
    config = json.load(open(args.config))

    # train
    if config["logging"]["tool"] == "wandb":
        if config["hyper_param_optim"]["enabled"]:
            if config["hyper_param_optim"]["resume"] != "":
                sweep_id = config["hyper_param_optim"]["resume"]
                wandb.agent(sweep_id, project=config["logging"]["project_name"],
                            function=lambda: train_network_with_wandb(config))
            else:
                sweep_id = get_sweep_id(config)
                wandb.agent(sweep_id, function=lambda: train_network_with_wandb(config))
        else:
            train_network_with_wandb(config)
    else:
        train_network(config)