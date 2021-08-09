
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == "__main__":
    splits = ["scraped"]

    mean = 0.
    std = 0.
    nb_samples = 0
    for split in splits:
        data_path ="/home/rog/data/grassland_imgs/rumex_classification"

        # Train dataset
        dataset = datasets.ImageFolder(f"{data_path}/{split}", transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(dataset,
                                batch_size=10,
                                num_workers=0,
                                shuffle=False)

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean)
    print(std)