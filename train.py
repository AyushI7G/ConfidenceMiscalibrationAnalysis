import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import SimpleCNN
from data.datasets import get_cifar10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_loader = get_cifar10(train=True)

for epoch in range(10):
    model.train()
    correct, total = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch+1} | Train Acc: {correct/total:.4f}")

torch.save(model.state_dict(), "results/cifar_model.pth")
