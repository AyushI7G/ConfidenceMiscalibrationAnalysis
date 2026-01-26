import torch

def expected_calibration_error(confidences, correctness, n_bins=10):
    confidences = torch.tensor(confidences)
    correctness = torch.tensor(correctness, dtype=torch.float)

    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            acc = correctness[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.sum() / len(confidences)) * torch.abs(acc - conf)

    return ece.item()
