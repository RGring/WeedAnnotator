from torch.utils.data import DataLoader
import json
from weed_annotator.semantic_segmentation.dataset.sugar_beet_dataset import SugarBeetDataset
from weed_annotator.semantic_segmentation.dataset.aerial_farmland_dataset import AerialFarmlandDataset
from weed_annotator.semantic_segmentation import utils, aug
import matplotlib.pyplot as plt


def main():
    config = json.load(open("configs/seg_config_aerial_farmland.json"))
    # build data
    # Data + Augmentations
    dataset_class = AerialFarmlandDataset
    train_dataset = dataset_class(
        utils.load_img_list(f"{config['data']['train_data']}"),
        labels_to_consider=config["data"]["labels_to_consider"],
        augmentation=aug.get_training_augmentations(config["data"]["aug"]),
    )

    val_dataset = dataset_class(
        utils.load_img_list(f"{config['data']['val_data']}"),
        labels_to_consider=config["data"]["labels_to_consider"],
        augmentation=aug.get_validation_augmentations(config["data"]["aug"]),
    )

    test = train_dataset[0]
    print(test[0].size(0))
    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=config["training"]["num_workers"])


    count = 0
    for (images, targets, valid_mask) in loader:
        ind = 2 # random.randint(0, bs)
        for image in images:
            image = image.permute(1, 2, 0)
            plt.imshow(image.numpy())
            plt.show()

        count += 1
        if count == 20:
            break

if __name__ == "__main__":
    main()