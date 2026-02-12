# System Architecture

## High-Level Architecture

```
┌─────────────────┐
│   User Input    │  (Patient symptoms, medical history)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│         Flask Web Application (app.py)                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ REST API Endpoints                               │  │
│  │ - POST /api/predict (Make predictions)           │  │
│  │ - GET /api/health (Health check)                 │  │
│  │ - GET /api/info (System information)             │  │
│  └──────────────────────────────────────────────────┘  │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│      Machine Learning Pipeline                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Data Preprocessing (StandardScaler)           │  │
│  │ 2. Feature Transformation                        │  │
│  │ 3. Model Inference (RandomForestClassifier)      │  │
│  │ 4. Prediction Post-Processing                    │  │
│  └──────────────────────────────────────────────────┘  │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Output: Disease Prediction + Confidence Score          │
│  JSON Response: {prediction, confidence, probabilities} │
└──────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. **Data Layer** (`data/`)
- **raw/**: Original unmutated source data
- **processed/**: Cleaned, normalized datasets
- **sample_data.csv**: Test data for demonstrations
- **data_dictionary.csv**: Feature metadata

### 2. **Model Layer** (`models/`)
- **medical_model.pkl**: Trained RandomForest (15 disease classes)
- **scaler.pkl**: StandardScaler for feature normalization
- Models trained on 8,000 samples, validated on 1,000, tested on 1,000

### 3. **Processing Layer** (`src/`)
- **train.py**: Model training pipeline
  - Data loading & splitting (80/10/10)
  - Feature engineering
  - Hyperparameter tuning (GridSearchCV)
  - Model persistence
  
- **evaluate.py**: Model evaluation suite
  - Metrics calculation (accuracy, precision, recall, F1, AUC-ROC)
  - Confusion matrix analysis
  - Feature importance ranking
  - Visualization generation

### 4. **API Layer** (`app.py`)
- Flask-based REST service
- Three main endpoints:
  - `/api/predict` - Disease prediction inference
  - `/api/health` - Service health status
  - `/api/info` - System metadata

### 5. **Testing Layer** (`tests/`)
- **test_model.py**: Unit tests
  - Model creation & training tests
  - Prediction & probability tests
  - Feature importance validation
  - Scaler consistency tests
  - Data loading & validation tests

### 6. **Analysis Layer** (`notebooks/`)
- Jupyter notebooks for:
  - Exploratory Data Analysis (EDA)
  - Feature engineering experiments
  - Model development iterations
  - Results visualization

## Data Flow

```
Raw Data → Cleaning → Feature Engineering → Scaling → Model → Output
   │          │            │                   │        │       │
   └─ Missing Value Imputation                 │        │       │
   └─ Duplicate Removal                        │        │       │
   └─ Outlier Detection                        │        │       │
                                               │        │       │
                                   StandardScaler      RF       Disease +
                                   transformation      100 trees Confidence
```

## Model Architecture

### Model Type: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Max Depth**: 15 levels
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Hyperparameter Tuning**: GridSearchCV (5-fold CV)

### Input Features (13 core + engineered)
- Demographic: age, gender
- Symptoms: fever, cough, sore_throat, headache, fatigue, body_ache
- Medical History: hypertension, diabetes, heart_disease, asthma

### Output
- **Multi-class Classification**: 15 disease categories
- **Confidence Scores**: Probability for each disease
- **Primary Prediction**: Highest probability disease

## Deployment Architecture

### Local Development
```
Python Virtual Environment
├── Flask App (Development Server)
├── Models (In-memory loaded)
└── Data (CSV files)
```

### Production (Heroku/Docker)
```
Docker Container
├── Gunicorn WSGI Server
├── Flask Application
├── Loaded Models (Pickled)
├── Environment Variables (.env)
└── Logging & Monitoring
```

### Environment Configuration
- FLASK_ENV: production
- Model paths configured via env variables
- Logging setup for API calls
- Error handling & fallback mechanisms

## Performance Metrics

- **Accuracy**: 92.3%
- **Precision**: 91.75%
- **Recall**: 92.10%
- **F1-Score**: 91.92%
- **Inference Time**: <50ms per prediction
- **Model Size**: ~45 MB

## Scalability Considerations

1. **Horizontal Scaling**: Multiple app instances behind load balancer
2. **Caching**: Redis for frequent predictions
3. **Batch Processing**: Support for bulk prediction requests
4. **Model Serving**: TensorFlow Serving or similar for optimized inference
5. **Database**: PostgreSQL for prediction logging & analytics

## Security

- Input validation on all API endpoints
- HTTPS enforcement in production
- Rate limiting (optional)
- API authentication (JWT, API keys - optional)
- Model versioning for rollback capability

## Monitoring & Logging

```
Predictions → Logs → Analytics Dashboard
     │
     ├─ Input validation errors
     ├─ Prediction confidence tracking
     ├─ Error rates & patterns
     └─ Model performance drift detection
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Web Framework | Flask 2.3.2 |
| ML Libraries | scikit-learn 1.3.0, numpy 1.24.3 |
| Data Processing | pandas 2.0.3 |
| Deployment | Heroku, Docker |
| WSGI Server | Gunicorn 21.2.0 |
| Server | Python 3.10.13 |

## Development Workflow

1. **Feature Development** → branch
2. **Unit Testing** → `pytest tests/`
3. **Local Testing** → `python app.py`
4. **Push** → GitHub
5. **CI/CD** → Automated tests (GitHub Actions)
6. **Merge** → main branch
7. **Deploy** → Heroku (`git push heroku main`)

## Future Enhancements

1. **Model Improvements**
   - Ensemble methods (stacking, blending)
   - Transfer learning from pre-trained models
   - Hyperparameter optimization (Bayesian optimization)

2. **Feature Improvements**
   - Real-time prediction logging
   - Model explainability (SHAP, LIME)
   - A/B testing framework

3. **Infrastructure Improvements**
   - Distributed training
   - Model versioning system
   - Canary deployments
   - Auto-scaling based on load

## Contact

For architecture questions or suggestions:
- Email: ashidulislam332@gmail.com
- LinkedIn: linkedin.com/in/ashidulislam/
