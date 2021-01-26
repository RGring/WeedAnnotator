import albumentations as albu
from albumentations.pytorch import ToTensorV2



def get_training_augmentations(aug_config):
    train_augments = []
    if aug_config["p_flip"] > 0:
        train_augments.append(albu.HorizontalFlip(p=aug_config["p_flip"]))
        train_augments.append(albu.VerticalFlip(p=aug_config["p_flip"]))
    if aug_config["p_motion_blur"] > 0:
        train_augments.append(albu.MotionBlur(blur_limit=[13, 21], p=0.3))
    if aug_config["p_color_jitter"] > 0:
        jv = aug_config["jitter_values"]
        train_augments.append(albu.ColorJitter(jv[0], jv[1], jv[2], jv[3], p=aug_config["p_color_jitter"]))
    if aug_config["p_to_gray"] > 0:
        train_augments.append(albu.ToGray(p=aug_config["p_to_gray"]))
    train_augments.append(albu.ShiftScaleRotate(scale_limit=[aug_config["scale_min"], aug_config["scale_max"]], rotate_limit=[-aug_config["rot_deg"], aug_config["rot_deg"]], p=1.0))
    train_augments.append(albu.Resize(aug_config["input_width"], aug_config["input_height"], always_apply=True))
    train_augments.append(albu.Normalize(mean=aug_config["mean"], std=aug_config["std"]))
    train_augments.append(ToTensorV2())
    return albu.Compose(train_augments)


def get_validation_augmentations(aug_config):
    val_augments = []
    val_augments.append(albu.Resize(aug_config["input_height"], aug_config["input_width"], always_apply=True))
    val_augments.append(albu.Normalize(mean=aug_config["mean"], std=aug_config["std"]))
    val_augments.append(ToTensorV2())
    return albu.Compose(val_augments)


# def to_tensor(x, **kwargs):
#     return x.transpose(2, 0, 1).astype('float32')
#
#
# def get_preprocessing(preprocessing_fn):
#     """Construct preprocessing transform
#
#     Args:
#         preprocessing_fn (callbale): data normalization function
#             (can be specific for each pretrained neural network)
#     Return:
#         transform: albumentations.Compose
#
#     """
#
#     _transform = [
#         albu.Lambda(image=preprocessing_fn),
#         albu.Lambda(image=to_tensor, mask=to_tensor),
#     ]
#     return albu.Compose(_transform)
