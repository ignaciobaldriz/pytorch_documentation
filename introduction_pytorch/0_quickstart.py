import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ------------------------------------------------ #

# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Inspect what is inside each data loader
X_batches = []
y_batches = []

for batch, (X, y) in enumerate(test_dataloader):

    print(f"Batch: {batch}")

    print(f"Shape of X [N, C, H, W]: {X.shape}")
    X_batches.append(X)

    print(f"Shape of y: {y.shape} {y.dtype}")
    y_batches.append(y)


# ------------------------------------------------ #


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):

    def __init__ (self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Save the  model to device
model = NeuralNetwork().to(device)
print(model)

# Define loss metric and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Define training loop
def train(dataloader, model, loss_fn, optimizer):
    
    # Get the size of the dataset and set the model to train mode
    size = len(dataloader.dataset)
    model.train()

    # Send X, y to device
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

    # Comput prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Define testing loop
def test(dataloader, model, loss_fn):

    # Get the size of the dataset, n of batches and set the model to test mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    # Set loss and gradients to zero
    test_loss, correct = 0, 0
    with torch.no_grad():
        # Predict on device, calculate loss
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Run the training and testing functions for each epoch
epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# ------------------------------------------------ #

# Save the model
torch.save(model.state_dict(), "models/model.pth")
print("Saved PyTorch Model State to model.pth")

# Load the model
model_1 = NeuralNetwork()
model_1.load_state_dict(torch.load("models/model.pth"))

# ------------------------------------------------ #

# List of clases to predict
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Set modelo to evaluate mode
model.eval()
x, y = test_data[0][0], test_data[0][1]

# Predict one case
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')