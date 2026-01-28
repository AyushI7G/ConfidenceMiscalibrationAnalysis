import numpy as np

def expected_calibration_error(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if mask.sum() > 0:
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            ece += np.abs(acc - avg_conf) * mask.mean()
    return ece
