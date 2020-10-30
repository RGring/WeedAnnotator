import albumentations as albu


def get_training_augmentations(aug_config):
    train_augments = []
    train_augments.append(albu.Resize(aug_config["input_height"], aug_config["input_width"]))
    train_augments.append(albu.HorizontalFlip(p=0.5))
    train_augments.append(albu.VerticalFlip(p=0.5))
    train_augments.append(albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=[-180, 180]))

    if aug_config["blur"]:
        train_augments.append(albu.motion_blur(blur_limit=[7, 21]))
    return albu.Compose(train_augments)


def get_validation_augmentations(aug_config):
    val_augments = []
    val_augments.append(albu.Resize(aug_config["input_height"], aug_config["input_width"], always_apply=True))
    return albu.Compose(val_augments)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
