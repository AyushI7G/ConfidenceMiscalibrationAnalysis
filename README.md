# Confidence Calibration in Neural Networks under Distribution Shift

## Overview

A systematic empirical study of prediction confidence, calibration, and failure modes in deep neural networks, focusing on the relationship between model confidence and correctness under in-distribution and out-of-distribution (OOD) conditions.

The project combines controlled experimentation, calibration metrics, and qualitative failure analysis to evaluate the reliability of modern classification models beyond standard accuracy-based evaluation.

---

## Experimental Setup

### Model

* Convolutional Neural Network (CNN) implemented from first principles using PyTorch

### Datasets

* **CIFAR-10**: Training and in-distribution evaluation
* **SVHN**: Out-of-distribution evaluation

### Evaluation Scope

* In-distribution generalization
* Out-of-distribution confidence behavior
* Calibration error and confidence misalignment

---

## Key Focus Areas

* Relationship between prediction confidence and correctness
* Identification of high-confidence incorrect predictions
* Calibration behavior under distribution shift
* Failure modes invisible to accuracy-based metrics
* Limitations of softmax confidence as a reliability measure

---

## Metrics and Analysis

* Classification accuracy
* Expected Calibration Error (ECE)
* Confidence histograms (correct vs incorrect predictions)
* Reliability diagrams
* In-distribution vs out-of-distribution confidence distributions

---

## Failure Modes Analyzed

* Confident misclassification on ambiguous samples
* Overconfidence under severe distribution shift
* Calibration degradation despite high training accuracy
* Overlapping confidence distributions between ID and OOD data

---

## Sample Findings

* High training accuracy does not imply reliable confidence estimates.
* A non-trivial fraction of incorrect predictions are made with high confidence.
* Out-of-distribution samples receive confidence values comparable to in-distribution data.
* Calibration error increases even when accuracy degradation is moderate.

---

## Project Structure

* `models/`: Neural network architectures
* `data/`: Dataset loaders (datasets downloaded automatically)
* `experiments/`: Configuration files for experimental settings
* `results/metrics/`: Quantitative evaluation results
* `results/plots/`: Calibration and confidence visualizations
* `results/summary.md`: Written analysis of findings

---

## Motivation

Machine learning models are increasingly deployed in real-world decision-making systems where incorrect but confident predictions can cause significant harm. While accuracy is the dominant evaluation metric, it fails to capture reliability failures that arise under distribution shift.

This project investigates these hidden failure modes through controlled experimentation and structured analysis, emphasizing the need for confidence-aware evaluation in deployed systems.

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
python analyze_results.py
```

---

## Future Work

* Temperature scaling and post-hoc calibration
* Abstention mechanisms for low-confidence predictions
* Controlled label-noise experiments
* Architecture comparisons (e.g., ResNet variants)
* OOD detection metrics (AUROC)

---

