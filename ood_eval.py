import torch
import torch.nn.functional as F
from models.cnn import SimpleCNN
from data.datasets import get_svhn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("results/cifar_model.pth"))
model.eval()

loader = get_svhn()
confidences = []

with torch.no_grad():
    for x, _ in loader:
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)
        conf, _ = probs.max(dim=1)
        confidences.extend(conf.cpu().numpy())

print("Mean confidence on SVHN (OOD):", sum(confidences)/len(confidences))
