import torch
import torch.nn.functional as F
from models.cnn import SimpleCNN
from data.datasets import get_cifar10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("results/cifar_model.pth"))
model.eval()

loader = get_cifar10(train=False)

confidences, corrects = [], []

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)

        confidences.extend(conf.cpu().numpy())
        corrects.extend((preds == y).cpu().numpy())

print("Collected confidence vs correctness data.")
