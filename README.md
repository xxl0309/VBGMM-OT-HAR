numpy>=1.20.0
pandas>=1.2.0
scipy>=1.6.0
scikit-learn>=0.24.0
POT>=0.8.0
matplotlib>=3.3.0
seaborn>=0.11.0


# Cross-domain Human Activity Recognition based on VBGMM and Optimal Transport

This repository contains the official Python implementation of our paper: **"Cross-domain Human Activity Recognition based on Variational Bayesian Gaussian Mixture Model and Optimal Transport"**.

Human Activity Recognition (HAR) models often suffer from performance degradation across different domains due to heterogeneous sensor placements and varying users. In this repository, we provide a fine-grained sub-distribution level matching framework that synergizes the Variational Bayesian Gaussian Mixture Model (VBGMM) with structure-preserving Optimal Transport (OT). 


* `data_loader.py`: Script for loading and preprocessing raw signals from 4 public HAR datasets.
* `feature_engineering.py`: Script for sliding-window segmentation and extracting 27-dimensional time-frequency features.
* `main_cm.py`: The **core algorithm** implementation (VBGMM + OT with Fused Gromov-Wasserstein distance) and confusion matrix visualization.
* `evaluation.py`: Script for executing hyperparameter sensitivity analysis.

s
To run the code, please install the required packages using:
```bash
pip install -r requirements.txt

We evaluated our framework on 4 public benchmark datasets. Due to licensing and size constraints, we do not include the raw data in this repository. Please download them from their official sources and place them in an empty data/ folder in the root directory:

DSADS: Daily and Sports Activities Dataset

PAMAP2: PAMAP2 Physical Activity Monitoring

UCI-HAR: Human Activity Recognition Using Smartphones

USC-HAD: USC Human Activity Dataset

Follow these steps to reproduce the cross-domain HAR experiments:
Step 1: Parse Raw Data
python data_loader.py
Step 2: Extract Time-Frequency Features
python feature_engineering.py
Step 3: Run the Main Cross-Domain Alignment
python main_cm.py
