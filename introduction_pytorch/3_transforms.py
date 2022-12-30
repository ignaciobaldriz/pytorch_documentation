import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# ----------------------------------------- #

# Apply transform and target_transform
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# Inspct lambda function
zeros = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(0), value=1)
zeros