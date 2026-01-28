import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from models.cnn import SimpleCNN
from data.datasets import get_cifar10, get_svhn
from calibration import expected_calibration_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("results/cifar_model.pth"))
model.eval()

# ---------- CIFAR-10 TEST ----------
confidences, corrects = [], []

loader = get_cifar10(train=False)

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = F.softmax(model(x), dim=1)
        conf, preds = probs.max(dim=1)
        confidences.extend(conf.cpu().numpy())
        corrects.extend((preds == y).cpu().numpy())

confidences = np.array(confidences)
corrects = np.array(corrects)

acc = corrects.mean()
ece = expected_calibration_error(confidences, corrects)

conf_thresh = 0.9
conf_wrong = ((confidences > conf_thresh) & (corrects == 0)).mean()

# Save metrics
with open("results/metrics/cifar_metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"ECE: {ece:.4f}\n")
    f.write(f"Confidently wrong (>0.9): {conf_wrong:.4f}\n")

# ---------- CONFIDENCE HISTOGRAM ----------
plt.hist(confidences[corrects == 1], bins=20, alpha=0.6, label="Correct")
plt.hist(confidences[corrects == 0], bins=20, alpha=0.6, label="Incorrect")
plt.legend()
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Confidence Distribution")
plt.savefig("results/plots/confidence_histogram.png")
plt.close()

# ---------- RELIABILITY DIAGRAM ----------
bins = np.linspace(0, 1, 11)
bin_acc, bin_conf = [], []

for i in range(len(bins) - 1):
    mask = (confidences > bins[i]) & (confidences <= bins[i+1])
    if mask.sum() > 0:
        bin_acc.append(corrects[mask].mean())
        bin_conf.append(confidences[mask].mean())

plt.plot(bin_conf, bin_acc, marker='o')
plt.plot([0,1],[0,1],'--')
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title("Reliability Diagram")
plt.savefig("results/plots/reliability_diagram.png")
plt.close()

# ---------- SVHN OOD ----------
svhn_conf = []

loader = get_svhn()
with torch.no_grad():
    for x, _ in loader:
        x = x.to(device)
        conf, _ = F.softmax(model(x), dim=1).max(dim=1)
        svhn_conf.extend(conf.cpu().numpy())

plt.hist(confidences, bins=20, alpha=0.6, label="CIFAR-10")
plt.hist(svhn_conf, bins=20, alpha=0.6, label="SVHN")
plt.legend()
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("ID vs OOD Confidence")
plt.savefig("results/plots/id_vs_ood_confidence.png")
plt.close()
