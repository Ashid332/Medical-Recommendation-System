# Data Directory

This directory contains datasets for the Medical Recommendation System.

## Folder Structure

```
data/
├── raw/                      # Original, immutable raw data
│   ├── medical_records.csv   # Patient medical history
│   ├── symptoms.csv          # Symptom-disease mapping
│   └── medicines.csv         # Medicine information
├── processed/                # Cleaned and processed data
│   ├── train_data.csv        # Training dataset
│   ├── test_data.csv         # Testing dataset
│   └── val_data.csv          # Validation dataset
├── sample_data.csv           # Sample data for testing
├── data_dictionary.csv       # Feature descriptions
and└── README.md              # This file
```

## Data Description

### Raw Data

**medical_records.csv**
- Contains patient demographic and medical information
- Columns: patient_id, age, gender, medical_history, comorbidities
- Records: 10,000+ patient records

**symptoms.csv**
- Maps symptoms to diseases
- Columns: symptom_id, symptom_name, disease, severity
- Unique symptoms: 200+

**medicines.csv**
- Comprehensive medicine database
- Columns: medicine_id, medicine_name, type, uses, side_effects
- Total medicines: 500+

### Processed Data

**train_data.csv** (80% of data)
- Features: 50+ clinical attributes
- Target: disease_classification (multi-class)
- Samples: ~8000

**test_data.csv** (10% of data)
- Same features as training data
- Samples: ~1000

**val_data.csv** (10% of data)
- Validation dataset for hyperparameter tuning
- Samples: ~1000

## Data Dictionary

See `data_dictionary.csv` for detailed feature descriptions including:
- Feature name
- Data type
- Description
- Values/Range
- Missing value percentage

## Data Pipeline

1. **Raw Data Ingestion**: Data imported from medical databases
2. **Cleaning**: Handle missing values, remove duplicates
3. **Transformation**: Feature engineering and scaling
4. **Splitting**: Train/test/validation split
5. **Validation**: Data quality checks

## Usage

```python
import pandas as pd

# Load training data
train_df = pd.read_csv('data/processed/train_data.csv')

# Load sample data for testing
sample_df = pd.read_csv('data/sample_data.csv')

# Load data dictionary
data_dict = pd.read_csv('data/data_dictionary.csv')
```

## Data Privacy

All data is anonymized and does not contain personally identifiable information (PII).
Data complies with medical privacy regulations (HIPAA).

## Data Size

- Raw data: ~50 MB
- Processed data: ~15 MB
- Sample data: ~500 KB

## Updates

Data is updated monthly with new patient records and medical information.
Last updated: 2026-02-12

## Support

For data-related questions, contact: ashidulislam332@gmail.com
LinkedIn: linkedin.com/in/ashidulislam/
