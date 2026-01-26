import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import CNN

transform = transforms.ToTensor()

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(test_data, batch_size=64)

model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
total = 0
confidences = []
correctness = []

with torch.no_grad():
    for x, y in loader:
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        confidences.extend(conf.tolist())
        correctness.extend((pred == y).tolist())

print("Accuracy:", correct / total)
print("Average confidence on CIFAR-10:", sum(confidences) / len(confidences))
