import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


valid_split = 0.2
batch_size = 64
path = 'input/flowers'

# define the transforms...
# resize, convert to tensors, ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


dataset = datasets.ImageFolder(path, transform=transform)

dataset_size = len(dataset)
print(f"Total number of images: {dataset_size}")

valid_size = int(valid_split*dataset_size)
train_size = len(dataset) - valid_size


train_data, valid_data = torch.utils.data.random_split(
    dataset, [train_size, valid_size]
)

print(f"Total training images: {len(train_data)}")
print(f"Total valid_images: {len(valid_data)}")

# training and validation data loaders
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=4
)
valid_loader = DataLoader(
    valid_data, batch_size=batch_size, shuffle=False, num_workers=4
)