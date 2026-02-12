# Processed Data

This directory contains cleaned, processed, and feature-engineered datasets ready for model training and evaluation.

## Files

- **train_data.csv** - Training dataset (80% of data) with 8,000+ samples
- **test_data.csv** - Test dataset (10% of data) with 1,000+ samples  
- **val_data.csv** - Validation dataset (10% of data) with 1,000+ samples

## Data Processing Steps

### 1. Data Cleaning
- Removed duplicate records
- Handled missing values using imputation
- Fixed inconsistent values
- Removed outliers

### 2. Feature Engineering
- Created new symptom combinations
- Encoded categorical variables
- Normalized numerical features
- Created disease severity indicators

### 3. Data Splitting
- Training: 80% (8,000 samples) - For model training
- Validation: 10% (1,000 samples) - For hyperparameter tuning
- Testing: 10% (1,000 samples) - For final evaluation

## Dataset Statistics

- **Total Samples**: 10,000
- **Total Features**: 50+
- **Target Classes**: 15+ diseases
- **Missing Values**: < 1%
- **Data Completeness**: 99%+

## Usage

```python
import pandas as pd

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
val_df = pd.read_csv('val_data.csv')
```
