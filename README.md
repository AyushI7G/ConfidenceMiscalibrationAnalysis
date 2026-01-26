Motivation

Modern neural networks often output high confidence scores even when their predictions are incorrect.
Such overconfidence becomes especially problematic under distribution shift, where models encounter data different from their training distribution.

This project focuses on:

Identifying confident failures

Quantifying confidence misalignment

Highlighting reliability risks in deployed ML systems

Datasets

CIFAR-10
Used for training and in-distribution evaluation.

SVHN (Street View House Numbers)
Used as an out-of-distribution dataset to study confidence behavior under distribution shift.

All datasets are downloaded automatically using torchvision.

Experimental Setup

Model: Simple CNN trained using cross-entropy loss

Training Data: CIFAR-10

Evaluation:

CIFAR-10 (in-distribution)

SVHN (out-of-distribution)

Metrics:

Accuracy

Average prediction confidence

Analysis of confident but incorrect predictions

Results (Observed)
Dataset	Accuracy	Average Confidence
CIFAR-10	~63%	~60%
SVHN (OOD)	~10% (near random)	~42%

Despite near-random accuracy on SVHN, the model maintains relatively high confidence, demonstrating severe confidence miscalibration under distribution shift.

Key Observations

Confidence does not reliably reflect correctness under OOD conditions.

Neural networks remain overconfident even when predictions are largely incorrect.

Average confidence drops under shift, but not proportionally to accuracy loss.

Confident failures are common and systematic.
