# When Models Are Confidently Wrong

This project analyzes the relationship between prediction confidence and correctness in deep neural networks.
Models are trained on CIFAR-10 and evaluated under both in-distribution and out-of-distribution (SVHN) settings.

## Key Focus
- Confidence vs correctness
- Calibration and miscalibration
- Confident incorrect predictions
- Distribution shift (OOD detection)

## Datasets
- CIFAR-10 (training and in-distribution testing)
- SVHN (out-of-distribution testing)

## Metrics
- Accuracy
- Confidence distributions
- Expected Calibration Error (ECE)

## Findings
Standard accuracy metrics fail to capture reliability issues, particularly under distribution shift.
