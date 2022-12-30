import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn, optim

# --------------------------------------------------------- #

# Import model
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Generate a random image
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Forward pass
prediction = model(data)
prediction

# Loss & Backward pass
loss = (prediction - labels).sum()
loss.backward()
loss

# Optimizer (Sthocastic Gradient Descent)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim

optim.step()

# --------------------------------------------------------- #

# Finetuning a pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Replace last layer: "(fc): Linear(in_features=512, out_features=1000, bias=True)""
print(model.fc)
model.fc = nn.Linear(512, 10)
print(model.fc)

# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)