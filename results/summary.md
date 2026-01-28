# Experimental Results Summary

## In-Distribution Evaluation (CIFAR-10)
The model achieves high training accuracy (≈94%) and moderate test accuracy.
However, calibration analysis reveals a clear mismatch between prediction
confidence and correctness. The Expected Calibration Error (ECE) is non-zero,
and a significant fraction of incorrect predictions are made with high confidence.

The confidence histogram shows substantial overlap between correct and
incorrect predictions at high confidence values. The reliability diagram
deviates from the ideal diagonal, further indicating miscalibration.

## Out-of-Distribution Evaluation (SVHN)
When evaluated on SVHN, a dataset not seen during training, the model assigns
unexpectedly high confidence to its predictions. The mean confidence on SVHN
is comparable to in-distribution confidence, despite severe distribution shift.

This behavior demonstrates that softmax confidence is not a reliable indicator
of correctness under distribution shift.

## Key Takeaway
Standard accuracy metrics fail to capture important reliability failures.
Confidence-aware evaluation is essential for understanding model behavior
in real-world deployment scenarios.
