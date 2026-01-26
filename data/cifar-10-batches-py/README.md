# Confidence Miscalibration Analysis

## Overview

This repository studies **confidence miscalibration and failure modes in machine learning models**.
A convolutional neural network (CNN) is trained on CIFAR-10 and evaluated under **distribution shift** using SVHN to analyze cases where models produce **confident but incorrect predictions**.

---

## Motivation

Prediction confidence is often used as a proxy for model reliability.
However, under distribution shift, neural networks frequently remain overconfident despite poor accuracy, creating hidden failure modes.

This project focuses on understanding and analyzing these failures rather than improving raw accuracy.

---

## Datasets

* **CIFAR-10** — training and in-distribution evaluation
* **SVHN** — out-of-distribution evaluation

All datasets are downloaded automatically via `torchvision`.

---

## Experimental Setup

* Model: Convolutional Neural Network (CNN)
* Training Data: CIFAR-10
* Evaluation Data:

  * CIFAR-10 (in-distribution)
  * SVHN (out-of-distribution)
* Metrics:

  * Accuracy
  * Average prediction confidence

---

## Results

| Dataset    | Accuracy | Average Confidence |
| ---------- | -------- | ------------------ |
| CIFAR-10   | ~63%     | ~60%               |
| SVHN (OOD) | ~10%     | ~42%               |

Despite near-random accuracy on SVHN, the model maintains relatively high confidence, indicating **severe confidence miscalibration**.

---

## Key Observations

* Confidence does not reliably reflect prediction correctness
* Distribution shift leads to systematic overconfidence
* Confident but incorrect predictions are common
* Accuracy alone is insufficient to assess reliability

---

## Repository Structure

```text
confidence-miscalibration-analysis/
├── src/                # Training and evaluation code
├── experiments/        # Out-of-distribution evaluation
├── notebooks/          # Analysis and visualizations
├── data/               # Automatically downloaded datasets
└── README.md
```

---

## How to Run

### Train the model

```bash
python src/train.py
```

### Evaluate on CIFAR-10

```bash
python src/evaluate.py
```

### Evaluate on SVHN (OOD)

```bash
python experiments/svhn_evaluation.py
```

---

## Future Work

* Expected Calibration Error (ECE)
* Temperature scaling
* Reliability diagrams
* Evaluation on corrupted datasets (CIFAR-10-C)

---

## Takeaway

High confidence does not imply high reliability.
This project highlights why confidence calibration is essential for deploying machine learning models in real-world settings.

---c framing), say the word.
