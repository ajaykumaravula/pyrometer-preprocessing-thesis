# Automation of Pyrometer Data Pre-Processing: Temperature Calibration & Neural Compression

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![University West](https://img.shields.io/badge/University%20West-Master%20Thesis%202026-red.svg)](https://www.hv.se/)

**Author:** Ajay Kumar Avula  
**Institution:** Department of Engineering Science, University West, Trollhättan, Sweden  
**Research Collaboration:** Produktionstekniskt Centrum (PTC), Trollhättan, Sweden  
**Thesis:** *Automation of Pyrometer Data Pre-processing (Denoising, Calibration, and Compression) for Metal Forming and Additive Manufacturing*  
**Individual Contribution:** ATP-2 Temperature Calibration & ATP-3 Neural Compression

---

## Overview

This repository provides the complete, reproducible Python implementation of the pyrometer data pre-processing pipeline developed in the Master thesis. The pipeline addresses signal calibration, multi-spectral sensor fusion, automated event detection, and high-ratio neural compression across two complementary datasets:

1. **NIST AMBench 2018 Open Benchmark (`Layer01.mat`)**: Standardized open benchmark from the National Institute of Standards and Technology (NIST) used for algorithm development, hyperparameter selection, and baseline testing.
2. **PODFAM Real Industrial Dataset (10 Files: `001_820.pcd` to `010_990.pcd`)**: Real-world industrial validation across 10 laser manufacturing runs (800°C to 990°C nominal setpoints, 10.8M data points), recorded on the **Aconity TWO** metal additive manufacturing system at **Produktionstekniskt Centrum (PTC)** in Trollhättan, Sweden (provided by Research Engineer Jonas Olsson).

---

## Repository Structure

```
pyrometer-preprocessing-thesis/
│
├── README.md                              # Complete documentation & usage guide
├── requirements.txt                       # Python package dependencies
├── .gitignore                             # Git ignore configuration
│
├── nist_pipeline/                         # 1. NIST AMBench Open Benchmark Pipeline (Development)
│   ├── two_pyrometer_pipeline.py          # D1: Standalone 2-pyrometer pipeline (Figure 4.1 & Table 5.1)
│   ├── pipeline.py                        # D2: Modular pipeline runner (Figure 4.3)
│   ├── calibrate.py                       # ATP-2: Calibration module (8 classical & ML algorithms)
│   ├── compress.py                        # ATP-3: Compression module (Delta, VAE, Deep Autoencoder)
│   ├── d3_calibration.py                  # D3: NIST Calibration benchmark (Figure 5.1)
│   ├── d3_compression.py                  # D3: NIST Compression benchmark (Figure 5.2)
│   ├── d4_visualisation.py                # D4: NIST 5-Panel Visualisation Dashboard (Figure 4.4)
│   └── d5_analysis.py                     # D5: NIST Window sensitivity & combination ranking (Figure 5.4)
│
├── podfam_pipeline/                       # 2. PODFAM Real Industrial Dataset Pipeline (Validation)
│   ├── podfam_pipeline.py                 # D1: Single-file pipeline (Figure 4.2 & Table 5.3)
│   ├── podfam_d4_visualisation.py         # D4: 10-File Visualisation Dashboard (Figure 4.5)
│   ├── podfam_d3_comparison.py            # D3: 10-File Calibration & Complexity Benchmark (Figure 5.5)
│   └── podfam_d5_analysis.py              # D5: 12-Combination Accuracy & Sensitivity Analysis (Figure 5.6 & Table 5.5)
│
└── figures/                               # Publication Figures Matching Thesis Document
    ├── two_pyrometer_result.png           # Figure 4.1 (NIST D1)
    ├── podfam_result.png                  # Figure 4.2 (PODFAM D1)
    ├── pipeline_dashboard.png             # Figure 4.3 (NIST D2)
    ├── d4_dashboard.png                   # Figure 4.4 (NIST D4)
    ├── podfam_d4_dashboard.png            # Figure 4.5 (PODFAM D4)
    ├── d3_calibration_comparison.png      # Figure 5.1 (NIST D3)
    ├── d3_compression_comparison.png      # Figure 5.2 (NIST D3)
    ├── d5_analysis.png                    # Figure 5.4 (NIST D5)
    ├── podfam_d3_comparison.png           # Figure 5.5 (PODFAM D3)
    └── podfam_d5_tradeoff.png             # Figure 5.6 (PODFAM D5)
```

---

## Thesis Deliverables & Figure Mapping

| Dataset | Deliverable | Script | Output Figure | Thesis Reference |
| :--- | :--- | :--- | :--- | :--- |
| **NIST** | **D1** - Standalone 2-Pyrometer Pipeline | `two_pyrometer_pipeline.py` | `two_pyrometer_result.png` | **Figure 4.1** & **Table 5.1** |
| **NIST** | **D2** - Modular Pipeline Dashboard | `pipeline.py` | `pipeline_dashboard.png` | **Figure 4.3** |
| **NIST** | **D3** - NIST Calibration Benchmark | `d3_calibration.py` | `d3_calibration_comparison.png` | **Figure 5.1** |
| **NIST** | **D3** - NIST Compression Benchmark | `d3_compression.py` | `d3_compression_comparison.png` | **Figure 5.2** |
| **NIST** | **D4** - NIST Process Visualisation | `d4_visualisation.py` | `d4_dashboard.png` | **Figure 4.4** |
| **NIST** | **D5** - NIST Sensitivity & Combinations | `d5_analysis.py` | `d5_analysis.png` | **Figure 5.4** |
| **PODFAM** | **D1** - Single-File Industrial Pipeline | `podfam_pipeline.py` | `podfam_result.png` | **Figure 4.2** & **Table 5.3** |
| **PODFAM** | **D4** - 10-File Multi-Panel Dashboard | `podfam_d4_visualisation.py` | `podfam_d4_dashboard.png` | **Figure 4.5** |
| **PODFAM** | **D3** - 10-File Calibration & Complexity Benchmark | `podfam_d3_comparison.py` | `podfam_d3_comparison.png` | **Figure 5.5** |
| **PODFAM** | **D5** - 12-Combination Accuracy & Sensitivity | `podfam_d5_analysis.py` | `podfam_d5_tradeoff.png` | **Figure 5.6** & **Table 5.5** |

---

## Algorithms Implemented

### 1. ATP-2: Temperature Calibration (8 Methods)
* **Classical Regression Algorithms:**
  * Mean Offset Correction
  * Linear Regression ($T = a \cdot S + b$)
  * Polynomial Regression (Degree 2 & 3)
  * Piecewise Linear Calibration
* **Machine Learning / AI Algorithms:**
  * Support Vector Regression (SVR with RBF kernel)
  * Random Forest Regressor
  * Multi-Layer Perceptron (MLP Neural Network)
  * Gradient Boosting Regressor

### 2. ATP-3: High-Ratio Data Compression
* **Delta Encoding:** Lossless / low-error baseline (CR $pprox$ 1.05x, RMSE $pprox$ 1.79°C).
* **Variational Autoencoder (VAE):** Latent space probabilistic neural compression (CR = 16x, RMSE = 45.6°C on PODFAM).
* **Deep Autoencoder:** Deterministic deep bottleneck compression (CR = 16x, RMSE = 17.2°C on PODFAM).

### 3. Automated Event Detection
* Gradient-based thresholding identifying key multi-stage events:
  * Process Start Marker (Green)
  * Peak Temperature Marker (Red)
  * Process End / Cooling Marker (Dark Red)

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/ajaykumaravula/pyrometer-preprocessing-thesis.git
cd pyrometer-preprocessing-thesis

# 2. Set up virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Running NIST AMBench Development Pipeline:
```bash
# Run NIST D1 Standalone Pipeline (Figure 4.1 & Table 5.1)
python nist_pipeline/two_pyrometer_pipeline.py

# Run NIST D2 Modular Pipeline (Figure 4.3)
python nist_pipeline/pipeline.py

# Run NIST D3 Calibration Benchmark (Figure 5.1)
python nist_pipeline/d3_calibration.py

# Run NIST D3 Compression Benchmark (Figure 5.2)
python nist_pipeline/d3_compression.py

# Run NIST D4 Visualisation Dashboard (Figure 4.4)
python nist_pipeline/d4_visualisation.py

# Run NIST D5 Sensitivity Analysis (Figure 5.4)
python nist_pipeline/d5_analysis.py
```

### 2. Running PODFAM Industrial Pipeline:
```bash
# Run PODFAM D1 Pipeline (Figure 4.2 & Table 5.3)
python podfam_pipeline/podfam_pipeline.py

# Run PODFAM D4 10-File Dashboard (Figure 4.5)
python podfam_pipeline/podfam_d4_visualisation.py

# Run PODFAM D3 10-File Benchmark (Figure 5.5)
python podfam_pipeline/podfam_d3_comparison.py

# Run PODFAM D5 Combined Accuracy Analysis (Figure 5.6 & Table 5.5)
python podfam_pipeline/podfam_d5_analysis.py
```

---

## Key Results Summary

1. **Calibration Accuracy (RQ1)**:
   * On PODFAM real industrial data, Linear Regression reduces raw uncalibrated error from **RMSE = 1694.9°C to 317.9°C** (an **81.2% error reduction**).
   * Linear Regression achieves identical accuracy to complex ML models (MLP at 317.1°C) while being over **2,000x faster** (2.5 ms vs 5,209 ms).
2. **Neural Compression Trade-Off (RQ2)**:
   * **Deep Autoencoder** achieves 16x compression with reconstruction error of only **17.2°C** on PODFAM industrial data.
   * Optimal overall pipeline configuration: **Linear Calibration + Delta Encoding** (total RMSE = 316.9°C) for lossless storage, or **Linear Calibration + Deep Autoencoder** (total RMSE = 317.3°C) for 16x compressed transmission.

---

## Citation & Academic Notice

```bibtex
@mastersthesis{avula2026pyrometer,
  author       = {Ajay Kumar Avula},
  title        = {Automation of Pyrometer Data Pre-processing (Calibration and Compression) for Metal Forming and Additive Manufacturing},
  school       = {University West, Department of Engineering Science},
  year         = {2026},
  address      = {Trollhättan, Sweden},
  note         = {In collaboration with Produktionstekniskt Centrum (PTC)}
}
```
