from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ------------------------------------------ #

# Define the Neural Net


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
print(net)


params = list(net.parameters())
print(len(params))
print(params[0].size())


# ------------------------------------------ #

# Inspect and understand the network

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


net.zero_grad()
out.backward(torch.randn(1, 10), retain_graph=True)

target = torch.randn(10)
target = target.view(1, -1)
print(target)
criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# ------------------------------------------ #

# Backpropagation

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward(retain_graph=True)

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# ------------------------------------------ #

# Update the weights

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()