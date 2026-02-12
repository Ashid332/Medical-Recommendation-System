"""
Medical Recommendation System - Model Training Script
This script handles the training of ML models for disease prediction and medicine recommendation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
from datetime import datetime

class MedicalModelTrainer:
    def __init__(self, data_path='data/medical_data.csv'):
        """Initialize the trainer with data path"""
        self.data_path = data_path
        self.models = {}
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and prepare the medical dataset"""
        print(f"Loading data from {self.data_path}...")
        try:
            df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            print("Creating sample dataset for demonstration...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample medical data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        symptoms = ['fever', 'cough', 'fatigue', 'headache', 'body_pain']
        diseases = ['Flu', 'Cold', 'COVID-19', 'Dengue', 'Malaria']
        
        data = {
            'fever': np.random.randint(0, 2, n_samples),
            'cough': np.random.randint(0, 2, n_samples),
            'fatigue': np.random.randint(0, 2, n_samples),
            'headache': np.random.randint(0, 2, n_samples),
            'body_pain': np.random.randint(0, 2, n_samples),
            'disease': np.random.choice(diseases, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save sample data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/medical_data.csv', index=False)
        print("Sample data created and saved to data/medical_data.csv")
        
        return df
    
    def preprocess_data(self, df, target_column='disease'):
        """Preprocess the data for training"""
        print("\\nPreprocessing data...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode categorical variables if any
        for column in X.columns:
            if X[column].dtype == 'object':
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                self.label_encoders[column] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        self.label_encoders['target'] = le_target
        
        print(f"Features shape: {X.shape}")
        print(f"Target classes: {list(le_target.classes_)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple ML models"""
        print("\\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Define models
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10, 
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', 
                C=1.0, 
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\\nTraining {name}...")
            start_time = datetime.now()
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model and results
            self.models[name] = model
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"  Training Time: {training_time:.2f}s")
        
        return results
    
    def save_models(self, output_dir='models'):
        """Save trained models to disk"""
        print(f"\\nSaving models to {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{name.lower()}_model.pkl')
            joblib.dump(model, model_path)
            print(f"  Saved {name} to {model_path}")
        
        # Save label encoders
        encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)
        print(f"  Saved label encoders to {encoders_path}")
        
        print("\\nAll models saved successfully!")
    
    def print_best_model(self, results):
        """Print the best performing model"""
        print("\\n" + "="*60)
        print("BEST MODEL SUMMARY")
        print("="*60)
        
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\\nBest Model: {best_model[0]}")
        print(f"Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"F1-Score: {best_model[1]['f1_score']:.4f}")
        print(f"Cross-Validation Score: {best_model[1]['cv_mean']:.4f}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("MEDICAL RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = MedicalModelTrainer()
    
    # Load data
    df = trainer.load_data()
    
    # Preprocess data
    X, y = trainer.preprocess_data(df)
    
    # Split data
    trainer.split_data(X, y)
    
    # Train models
    results = trainer.train_models()
    
    # Save models
    trainer.save_models()
    
    # Print best model
    trainer.print_best_model(results)
    
    print("\\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\\nModels saved to 'models/' directory")
    print("Use evaluate.py to evaluate model performance in detail")

if __name__ == "__main__":
    main()
