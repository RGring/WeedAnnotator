import torch
import torch_optimizer as optim
import numpy as np
import math

def get_optimizer(model, trainable_params, config):
    opt = config["training"]["optimization"]["optimizer"]
    lr_base = config["training"]["optimization"]["lr"]
    wd = config["training"]["optimization"]["weight_decay"]
    if opt == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr_base,
            weight_decay=wd)
    elif opt == "adamW":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr_base,
            weight_decay=wd)
    elif opt == "sdg":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr_base,
            momentum=0.9,
            weight_decay=wd,
        )
    elif opt == "ranger":
        optimizer = optim.Ranger(
            trainable_params,
            lr=lr_base,
            betas=(.95, .99),
            eps=1e-4,
            weight_decay=wd,
        )
    return optimizer

def get_lr_schedule(lr_schedule_mode, lr, final_lr, epochs, train_loader, start_warmup, warmup_epochs):
    num_batches = len(train_loader)
    warmup_lr_schedule = np.linspace(start_warmup, lr, num_batches * warmup_epochs)
    iters = np.arange(num_batches * (epochs - warmup_epochs))
    if lr_schedule_mode == "cos":
        cosine_lr_schedule = np.array([final_lr +
                                       0.5 * (lr - final_lr) * (1 + math.cos(math.pi * t / (num_batches * (epochs - warmup_epochs))))
                                       for t in iters])
    elif lr_schedule_mode == "cos_cycle":
        cosine_lr_schedule = np.array([lr - t / ((num_batches * (epochs -warmup_epochs))) * (lr - final_lr) -
                                       t / ((len(train_loader) * (epochs - warmup_epochs))) *
                                       (lr - final_lr - t / ((num_batches * (epochs -warmup_epochs))) * (lr - final_lr)) *
                                       (1 + math.sin(math.pi * t / (num_batches * (int(epochs / 5) - warmup_epochs))))
                                       for t in iters])

    lr_scheduler = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    return lr_scheduler

