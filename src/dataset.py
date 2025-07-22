from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_transforms(image_size=224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # For RGB images
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    return train_transform, test_transform


def get_dataloaders(data_dir="data", batch_size=32, image_size=224):
    train_transform, test_transform = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
