import sys
sys.path.append("src")

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import CNN

transform = transforms.ToTensor()

svhn = datasets.SVHN(
    root="data",
    split="test",
    download=True,
    transform=transform
)

loader = DataLoader(svhn, batch_size=64)

model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

confidences = []

with torch.no_grad():
    for x, _ in loader:
        probs = F.softmax(model(x), dim=1)
        conf, _ = probs.max(dim=1)
        confidences.extend(conf.tolist())

print("Average confidence on SVHN:", sum(confidences) / len(confidences))
