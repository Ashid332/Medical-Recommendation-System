# Machine Learning Models

This directory contains trained machine learning models for the medical recommendation system.

## Model Artifacts

### 1. medical_model.pkl
- **Type**: RandomForestClassifier
- **Trained on**: Medical dataset with patient symptoms and medical history
- **Features**: 50+ clinical features
- **Classes**: Disease categories (Diabetes, Hypertension, Heart Disease, etc.)
- **Performance**: ~92% accuracy on test set
- **Size**: ~45 MB

### 2. Feature Scaler (scaler.pkl)
- **Type**: StandardScaler
- **Purpose**: Feature normalization
- **Usage**: Apply before making predictions

## Model Loading

```python
import pickle

# Load the trained model
with open('medical_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 92.3% |
| Precision | 91.8% |
| Recall | 92.1% |
| F1-Score | 91.9% |

## Requirements

- scikit-learn >= 1.0
- pandas >= 1.3
- numpy >= 1.21

## Model Retraining

To retrain the model, run:

```bash
python src/train.py
```

To evaluate the model, run:

```bash
python src/evaluate.py
```

## Contact
For questions about model performance or usage: ashidulislam332@gmail.com
LinkedIn: linkedin.com/in/ashidulislam/
