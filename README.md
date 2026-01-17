# Multivariate Time-Series Classification of Daily and Sports Activities (ANN2 Project)

This repository contains the code, experiments, and report materials for the ANN2 course project (Winter Semester 2026) on Human Activity Recognition (HAR) using multivariate wearable-sensor time-series data.

## Project aim
To classify **19 daily and sports activities** from multivariate time-series signals recorded by **5 body-worn sensor units** (accelerometer + gyroscope + magnetometer), and to compare several deep learning architectures under a **subject-wise split** (generalization to unseen people).

## Data
**Dataset:** Daily and Sports Activities (wearable sensors, 8 subjects, 19 classes).  
- Each sample is a multivariate sequence with shape **(T = 125, F = 45)**  
- 45 features = 5 body locations × 9 channels (3 accel + 3 gyro + 3 magnetometer)

**Important:** The raw dataset is **not included** in this repository (size/licensing/privacy constraints depending on how you obtained it).  
To run the notebook, download the dataset separately and set the dataset path in the notebook.
https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

## Models implemented
This project evaluates multiple deep learning models commonly used in HAR:

- **Baseline 1D-CNN**
- **Multi-sensor Fusion CNN (5-branch)** (one branch per body location, feature-level fusion)
- **TCN (Temporal Convolutional Network)**
- **TCN v1 / TCN v2 (tuned variants)**
- **CNN + BiLSTM**

Evaluation includes accuracy, macro-precision/recall/F1, confusion matrices, and top confusions.

## Repository structure
- `ANN2Project.ipynb` — main notebook (training + evaluation + figures/tables)
- `ANN2Project.html` — exported HTML version of the notebook with all outputs
- `report/` — final report PDF/Word (if included)
- `outputs/` — exported figures (accuracy/loss curves, confusion matrices) and CSV summaries  
  (e.g., `model_results_summary.csv`, `Top3_confusions_all_models.csv`)



## Experimental setup (summary)
- **Subject-wise split:** train/val on subjects 1–6, test on subjects 7–8  
- **Validation:** 20% of training (stratified)  
- **Preprocessing:** z-score standardization (fit on train only)  
- **Optimizer:** Adam, multi-class cross-entropy  
- **Metrics:** Accuracy, Macro Precision/Recall/F1, confusion matrices (counts/normalized), top confusions

- **results correspond to the final run**

## How to run
### Option A: Run the notebook (recommended)
1. Clone the repository:
   ```bash
   git clone <YOUR_REPO_URL>
   cd <YOUR_REPO_FOLDER>

