# Automated Heartbeat Classification from ECG Signals

Classifying ECG heartbeats into five AAMI arrhythmia categories
(N, S, V, F, Q) using a 1D Convolutional Neural Network, compared
against Random Forest and XGBoost baselines.

## Dataset
MIT-BIH Arrhythmia Database (pre-segmented version).
Not included in this repo due to size. Place the CSV files in
`ECG_Signals_Dataset/` before running.

## Scripts
- `01_explore_data.py` - load data, class distribution, beat visualisation
- `02_train_cnn.py` - train and evaluate the 1D CNN

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scipy matplotlib seaborn scikit-learn xgboost wfdb streamlit
```

## Note
Educational research project. Not a medical device and not intended
for clinical diagnosis.
