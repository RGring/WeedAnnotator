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
from weed_annotator.semantic_segmentation.models.xResnet_encoder_concat_pool import xResnetEncoder
import matplotlib.pyplot as plt
import numpy as np
#import apex

os.environ['WANDB_MODE'] = 'dryrun'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["LANG"] = "C.UTF-8"
os.environ["LANGUAGE"] = "C.UTF-8"
INF_FP16 = 2 ** 15

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
    global learnable_class_weights
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
    num_classes = len(config["data"]["labels_to_consider"]) + 1  # + Background class

    if config["arch"]["type"] == "BiSeNetV1":
        # ToDo
        model = BiSeNetV1(num_classes)
        trainable_params = model.parameters()
    else:
        encoder_ident = config["arch"]["encoder"]
        encoder_depth = len(smp.encoders.encoders[encoder_ident]["params"]["out_channels"]) - 1
        if "xresnet" in config["arch"]["encoder"] or "xcpresnet" in config["arch"]["encoder"]:
            smp.encoders.encoders[encoder_ident]["params"]["input_size"] = (config["data"]["aug"]["input_width"], config["data"]["aug"]["input_height"])
        decoder_channels = (512, 256, 128, 64, 32, 16)[-encoder_depth:]
        model = smp.__dict__[config["arch"]["type"]](
            encoder_name=encoder_ident,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=config["training"]["activation"],
        )

        if encoder_weights == None and pretrained_weights != "":
            load_weights(model, pretrained_weights)
        trainable_params = [{'params': filter(lambda p: p.requires_grad, model.decoder.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.segmentation_head.parameters())},
                            {'params': filter(lambda p: p.requires_grad, model.encoder.parameters()),
                             'lr': config["training"]["optimization"]["lr"] / 10}]

        if config["training"]["optimization"]["lr_encoder"] == 0.0:
            logger.info("Freezing encoder parameters.")
            for param in model.encoder.parameters():
                param.requires_grad = False
    model.to(DEVICE)
    if tb_writer is None:
        wandb.watch(model)
    logger.info(f"Building model done with pretrained_weights: {pretrained_weights}")

    # Loss
    if config["training"]["class_weights"]:
        if len(config["training"]["class_weights"]) != len(train_dataset.label_dict):
            logger.error("The number of class weights must fit the number of labels.")
            return -1
        else:
            weights = torch.tensor(config["training"]["class_weights"], dtype=torch.float32)
            weights = weights / weights.sum()
            weights = weights.cuda()

    loss_func = config["training"]["loss_func"]
    if loss_func == "lovasz":
        criterion = losses.LovaszLoss()
    elif loss_func == "dice":
        criterion = smp.utils.losses.DiceLoss()
    elif loss_func == "weighted_dice":
        if config["training"]["class_weights"]:
            criterion = losses.WeightedDiceLoss(weight=weights)
        else:
            criterion = losses.WeightedDiceLoss()
            learnable_class_weights = [torch.zeros((1,), requires_grad=True) for i in range(num_classes)]
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

    # Optimization
    optim = optimizer.get_optimizer(model, trainable_params, config)
    lr_scheduler = optimizer.get_lr_schedule(config["training"]["optimization"]["lr_schedule_mode"],
                                             config["training"]["optimization"]["lr"],
                                             config["training"]["optimization"]["final_lr"],
                                             config["training"]["epochs"],
                                             train_loader,
                                             config["training"]["optimization"]["start_warmup"],
                                             config["training"]["optimization"]["warmup_epochs"]
                                             )
    if config["training"]["use_fp16"]:
        model, optim = apex.amp.initialize(model, optim, opt_level="O1")

    # Resume
    to_restore = {"epoch": 0, "best_iou": 0.0}
    if config["logging"]["resume"]:
        utils.restart_from_checkpoint(
            config["logging"]["resume"],
            run_variables=to_restore,
            state_dict=model,
            optimizer=optim,
        )
    start_epoch = to_restore["epoch"]
    best_iou = to_restore["best_iou"]

    for epoch in range(start_epoch, config["training"]["epochs"]):
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # train
        scores = train(train_loader, model, optim, criterion, epoch, lr_scheduler, logger, tb_writer, config)

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
                                    config)
            if val_iou > best_iou:
                best_iou = val_iou
                shutil.copyfile(
                    os.path.join(dump_checkpoints, "checkpoint-recent.pth.tar"),
                    os.path.join(dump_checkpoints, "checkpoint-best.pth.tar"),
                )

    logger.info(f"Training finished with best mean IoU of {best_iou}.")


def train(train_loader, model, optimizer, criterion, epoch, lr_scheduler, logger, writer, config):
    losses = utils.AverageMeter()

    if config["training"]["metric"] == "iou_per_batch":
        mean_ious = utils.AverageMeter()
    elif config["training"]["metric"] == "iou_per_ds":
        class_labels = train_loader.dataset.label_dict.values()
        inter_per_class = np.zeros(len(class_labels))
        union_per_class = np.zeros(len(class_labels))

    model.train()
    for iter_epoch, (inp, target, valid_mask) in enumerate(train_loader):
        iteration = epoch * len(train_loader) + iter_epoch
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[iteration]
        optimizer.param_groups[2]["lr"] /= 10

        inp = inp.to(DEVICE)
        target = target.to(DEVICE)
        output = model(inp)
        output_valid = output
        # output_valid[~valid_mask.unsqueeze(1).expand_as(output_valid).bool()] = -INF_FP16

        # Remove overlapping targets according to Agriculture-Vision Paper
        final_label_predictions = metrics.arg_max(output_valid.clone())
        target = metrics.non_overlapping_target(final_label_predictions, target)

        # Loss
        if config["training"]["loss_func"] == "cross_entropy":
            loss = criterion(output, torch.argmax(target, dim=1).long())
        elif config["training"]["loss_func"] == "weighted_dice":
            loss = criterion(output, target, weight=learnable_class_weights)
        else:
            loss = criterion(output_valid, target)

        # compute the gradients
        optimizer.zero_grad()
        if config["training"]["use_fp16"]:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # step
        optimizer.step()

        # update stats
        if config["training"]["metric"] == "iou_per_batch":
            miou = metrics.mIoU_per_batch(final_label_predictions.data, target.float().data)
            mean_ious.update(miou.item(), inp.size(0))
        elif config["training"]["metric"] == "iou_per_ds":
            inter_per_class_now, union_per_class_now = metrics.inter_union_per_class(final_label_predictions.data, target.float().data)
            inter_per_class += inter_per_class_now
            union_per_class += union_per_class_now
        losses.update(loss.item(), inp.size(0))

    # Final mean IoU of epoch
    if config["training"]["metric"] == "iou_per_batch":
        mean_iou = mean_ious.avg
    elif config["training"]["metric"] == "iou_per_ds":
        eps = 1e-7
        inter_per_class = np.add(inter_per_class, eps)
        union_per_class = np.add(union_per_class, eps)
        iou_per_class = inter_per_class / union_per_class
        mean_iou = np.mean(iou_per_class)

    # Logging
    if writer is None:
        wandb.log({"loss/train": losses.avg}, step=iteration)
        wandb.log({"mIoU/train": mean_iou}, step=iteration)
    else:
        writer.add_scalar("loss/train", losses.avg, iteration)
        writer.add_scalar("mIoU/train", mean_iou, iteration)

    logger.info(f"Train\t"
                f"Epoch {epoch}\t"
                f"Loss {losses.avg:.3f}\t"
                f"mIoU {mean_iou:.3f}\t"
                f"lr_enc {optimizer.param_groups[2]['lr']:.5f}, lr_dec {optimizer.param_groups[0]['lr']:.5f}")
    if config["training"]["metric"] == "iou_per_ds":
        str = ""
        for class_label, iou in zip(class_labels, iou_per_class):
            str += f"{class_label}: {iou:.3f}, \t"
        logger.info(str)
    return losses.avg, mean_iou


def val(valid_loader, model, criterion, iteration, logger, writer, config):
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
            target = target.to(DEVICE)
            output = model(inp)
            output_valid = output.clone()
            output_valid[~valid_mask.unsqueeze(1).expand_as(output_valid).bool()] = -INF_FP16

            # Remove overlapping targets according to Agriculture-Vision Paper
            final_label_predictions = metrics.arg_max(output_valid)
            target = metrics.non_overlapping_target(final_label_predictions, target)

            # Loss
            if config["training"]["loss_func"] == "cross_entropy":
                loss = criterion(output, torch.argmax(target, dim=1).long())
            elif config["training"]["loss_func"] == "weighted_dice":
                loss = criterion(output, target, weight=learnable_class_weights)
            else:
                loss = criterion(output_valid, target)
            if config["training"]["metric"] == "iou_per_batch":
                miou = metrics.mIoU_per_batch(final_label_predictions, target.float())
                mean_ious.update(miou.item(), inp.size(0))
            elif config["training"]["metric"] == "iou_per_ds":
                inter_per_class_now, union_per_class_now = metrics.inter_union_per_class(final_label_predictions, target.float())
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

    if writer is None:
        wandb.log({"loss/val": losses.avg}, step=iteration)
        wandb.log({"mIoU/val": mean_iou}, step=iteration)
        if config["logging"]["log_images"]:
            log_images = utils.get_wandb_img(4, inp, target, final_label_predictions, valid_loader.dataset.label_dict)
            wandb.log({"predictions": log_images}, step=iteration)
    else:
        writer.add_scalar("loss/val", losses.avg, iteration)
        writer.add_scalar("mIoU/val", mean_iou, iteration)

    logger.info(f"Val\t"
                f"Loss {losses.avg:.3f}\t"
                f"mIoU {mean_iou:.3f}\t")

    if config["training"]["metric"] == "iou_per_ds":
        str = ""
        for class_label, iou in zip(class_labels, iou_per_class):
            str += f"{class_label}: {iou:.3f}, \t"
        logger.info(str)
    return losses.avg, mean_iou


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
    parser.add_argument('-c', '--config', default='configs/seg_config_aerial_farmland.json', type=str,
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