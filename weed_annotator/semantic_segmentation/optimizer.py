import torch
import torch_optimizer as optim

def get_optimizer(model, trainable_params, config):
    opt = config["training"]["optimization"]["optimizer"]
    lr_base = config["training"]["optimization"]["lr_decoder"]
    wd = config["training"]["optimization"]["weight_decay"]
    if opt == "adam":
        optimizer = torch.optim.Adam(
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

