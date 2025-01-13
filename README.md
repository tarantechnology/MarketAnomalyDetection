# README

## Overview

This repository contains all the necessary files and scripts for running, adapting, and testing machine learning models on financial datasets. Below is a description of the files and their respective purposes:
---

### File: `model.py`
- **Description**: Contains the machine learning models.
- **Purpose**:
  - Includes the code for building, training, and evaluating machine learning models on the Bloomberg dataset.
  - Serves as the backbone for generating and fine-tuning models for financial market predictions.\
- **Dependencies**: Requires `FinancialMarketData.csv` as the primary dataset.
  
---

### Folder: `pkl_files/`
- This folder contains all the pre-trained model `.pkl` files.
- Each `.pkl` file is a serialized machine learning model trained on the Bloomberg dataset for financial predictions.
- Ensure this folder is not moved or renamed, as the scripts depend on its relative path.

---

### File: `app.py`
- **Description**: The main code for the Streamlit app.
- **Purpose**: 
  - Serves as the user interface for portfolio allocation advice with ML Model
- **Dependencies**: Requires Streamlit and the `pkl_files/` folder for loading models.

---

### File: `backtest.py`
- **Description**: A script for testing the trained models on a synthetic dataset.
- **Purpose**:
  - Adapts the trained models for use on synthetic datasets to evaluate their generalizability.
  - Provides metrics and performance visualizations for backtesting.
- **Dependencies**: Requires the synthetic dataset and the models from the `pkl_files/` folder.

---

### File: `FinancialMarketData.csv`
- **Description**: The Bloomberg financial market dataset.
- **Purpose**:
  - Acts as the training and testing dataset for the machine learning models.
  - Ensure this file is in the root directory to avoid file path issues in `model.py`.

---

## Instructions
1. Place all the `.pkl` files inside the `pkl_files/` folder.
2. Run `run streamlit app.py` to start the Streamlit application and interact with the models.

---

## Requirements
`pip install -r requirements.txt`

---

## Contact
For any questions or issues, please reach out to tpuvvala@gatech.edu
