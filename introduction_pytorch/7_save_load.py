import torch
import torchvision.models as models

# ------------------------------------------ #

# Load vgg16 remote architecture with pretrained weights
model_vgg16 = models.vgg16(pretrained=True)
print(model_vgg16.state_dict())

torch.save(model_vgg16.state_dict(), "models/model_vgg16.pth")


# Load local vgg16 architecture
model_local = models.vgg16()
print(model_local.state_dict())
model_local.load_state_dict(torch.load("models/model_vgg16.pth"))
model_local.eval()

# Saving and loading the model with the network structure
torch.save(model_local, "models/model_vgg16_2.pth")
model_local_2 = torch.load("models/model_vgg16_2.pth")