from src.dataset import get_dataloaders

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir="data", batch_size=32
    )

    print("Classes:", class_names)
    for images, labels in train_loader:
        print("Batch shape:", images.shape)
        break
