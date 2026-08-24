# Automation of Pyrometer Data Pre-Processing: Temperature Calibration & Neural Compression

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![University West](https://img.shields.io/badge/University%20West-Master%20Thesis%202026-red.svg)](https://www.hv.se/)

**Author:** Ajay Kumar Avula  
**Institution:** Department of Engineering Science, University West, Trollhättan, Sweden  
**Industrial Partner:** Produktionstekniskt Centrum (PTC)  
**Thesis:** *Automation of Pyrometer Data Pre-processing (Denoising, Calibration, and Compression) for Metal Forming and Additive Manufacturing*  
**Individual Contribution:** ATP-2 Temperature Calibration & ATP-3 Neural Compression

---

## Overview

This repository provides the complete, reproducible Python implementation of the pyrometer data pre-processing pipeline developed in the Master thesis. The pipeline addresses signal calibration, multi-spectral fusion, automated event detection, and high-ratio data compression across two distinct datasets:

1. **NIST AMBench 2018 Benchmark ()**: Standardized open benchmark for initial algorithm development and hyperparameter tuning.
2. **PODFAM Industrial Dataset (10 Files:  to )**: Real-world industrial validation across 10 laser manufacturing runs (800°C to 990°C nominal temperatures).

---

## Repository Structure



---

## Thesis Deliverables & Figure Mapping

| Dataset | Deliverable | Script | Output Figure | Thesis Reference |
| :--- | :--- | :--- | :--- | :--- |
| **PODFAM** | **D1** - Single-File Industrial Pipeline |  |  | **Figure 4.2** & **Table 5.3** |
| **PODFAM** | **D4** - 10-File Multi-Panel Dashboard |  |  | **Figure 4.5** |
| **PODFAM** | **D3** - 10-File Calibration & Complexity Benchmark |  |  | **Figure 5.5** |
| **PODFAM** | **D5** - 12-Combination Accuracy & Sensitivity |  |  | **Figure 5.6** & **Table 5.5** |
| **NIST** | **D1** - Simulated 2-Pyrometer Pipeline |  |  | **Figure 4.1** & **Table 5.1** |
| **NIST** | **D2** - Modular Pipeline Dashboard |  |  | **Figure 4.3** |
| **NIST** | **D3** - NIST Calibration Benchmark |  |  | **Figure 5.1** |
| **NIST** | **D3** - NIST Compression Benchmark |  |  | **Figure 5.2** |
| **NIST** | **D4** - NIST Process Visualisation |  |  | **Figure 4.4** |
| **NIST** | **D5** - NIST Sensitivity & Combinations |  |  | **Figure 5.4** |

---

## Algorithms Implemented

### 1. ATP-2: Temperature Calibration (8 Methods)
* **Classical Algorithms:**
  * Mean Offset Correction
  * Linear Regression (T = a * S + b)
  * Polynomial Regression (Degree 2 & 3)
  * Piecewise Linear Calibration
* **Machine Learning / AI Algorithms:**
  * Support Vector Regression (SVR with RBF kernel)
  * Random Forest Regressor
  * Multi-Layer Perceptron (MLP Neural Network)
  * Gradient Boosting Regressor

### 2. ATP-3: High-Ratio Data Compression
* **Delta Encoding:** Lossless/low-error baseline (CR ~ 1.05x, RMSE ~ 1.4°C).
* **Variational Autoencoder (VAE):** Latent space probabilistic compression (CR = 8x, RMSE ~ 46.2°C).
* **Deep Autoencoder:** Deterministic deep bottleneck compression (CR = 8x, RMSE ~ 17.8°C).

### 3. Automated Event Detection
* Gradient-based thresholding for multi-stage processes:
  * Process Start Marker
  * Peak Temperature Marker
  * Process End / Cooling Marker

---

## Installation & Setup

### 1. Clone the repository:


### 2. Set up virtual environment and install dependencies:


---

## How to Run

### Running PODFAM Industrial Pipeline:


### Running NIST AMBench Development Pipeline:


---

## Key Results Summary

1. **Calibration Accuracy (PODFAM)**:
   * Raw single-colour uncalibrated error: RMSE = 1694.9°C (across 50,000 points).
   * After Linear Regression calibration: RMSE = 317.9°C (an **81.2% error reduction**).
   * Classical Linear Regression outperforms computationally intensive ML models on PODFAM due to monotonic sensor linearity and 1000x lower computational complexity (0.22 ms vs 5209 ms).

2. **Neural Compression Trade-Off**:
   * **Deep Autoencoder** achieves 8x compression with reconstruction error of only 17.8°C.
   * Optimal overall pipeline configuration: **Linear Calibration + Delta Encoding** (RMSE = 318.0°C) for lossless storage, or **Linear Calibration + Deep Autoencoder** (RMSE = 319.2°C) for 8x compressed transmission.

---

## Citation & Academic Notice


