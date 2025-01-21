# K-adapt Framework



This repository provides a framework to generate and analyze recommendation scores using various models, calculate oracle metrics, and compare different dynamic-k methods with K-Adapt

## Directory Structure
- `datasets/`: Contains the dataset files for user-item interactions.
- `models/`: Includes the implementation of various recommendation models.
- `methods/`: Houses scripts to compare and evaluate the performance of different methods.
- `cal_methods.py`: Contains calibration methods for PerK.
- `evaluation.py`: Implements evaluation utilities for metrics.
- `oracle_calculator.py`: Script to calculate oracle values for the dataset.
- `run.py`: Main script to train models and generate user-item interaction scores.
- `train.py`: Script for training the models using the provided datasets.
- `utils_new.py`: Utility functions for data preprocessing and loading.

---

## Workflow

### 1. **Run `run.py`**
This script uses the datasets in the `datasets` folder to train recommendation models listed in the `models` directory. It generates scores for user-item interactions.

**Output**:
- Scores for validation and test sets are saved in the `score_files` directory.

**Command**:
```bash
python run.py
```
---

### 2. **Calculate Oracle Values**
Use `oracle_calculator.py` to calculate the possible oracle values for the generated scores.

**Output**:
- Oracle values are calculated for metrics like Hit Rate, Recall, MRR, NDCG, and F1-Score.

**Command**:
```bash
python oracle_calculator.py
```

---

### 3. **Compare Methods**
Compare the results of all methods using the scripts in the `methods` directory. This step provides a performance comparison across different recommendation models based on evaluation metrics.

**Output**:
- Provides performance comparisons across various recommendation models based on evaluation metrics.

**Command**:
```bash
python methods/<comparison_script>.py
```
