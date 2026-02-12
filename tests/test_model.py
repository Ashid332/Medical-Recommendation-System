import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sys

class TestMedicalModel(unittest.TestCase):
    """Unit tests for Medical Recommendation System"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        np.random.seed(42)
        cls.n_samples = 100
        cls.n_features = 13
        
        # Create synthetic test data
        cls.X = np.random.rand(cls.n_samples, cls.n_features)
        cls.y = np.random.randint(0, 15, cls.n_samples)
        
        # Initialize model and scaler
        cls.scaler = StandardScaler()
        cls.X_scaled = cls.scaler.fit_transform(cls.X)
        
        cls.model = RandomForestClassifier(n_estimators=10, random_state=42)
        cls.model.fit(cls.X_scaled, cls.y)

    def test_model_creation(self):
        """Test model instantiation"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.n_estimators, 10)

    def test_model_training(self):
        """Test model training"""
        train_score = self.model.score(self.X_scaled, self.y)
        self.assertGreater(train_score, 0.5)
        self.assertLessEqual(train_score, 1.0)

    def test_predictions(self):
        """Test model predictions"""
        predictions = self.model.predict(self.X_scaled)
        self.assertEqual(len(predictions), self.n_samples)
        self.assertTrue(all(0 <= p < 15 for p in predictions))

    def test_prediction_probabilities(self):
        """Test prediction probabilities"""
        proba = self.model.predict_proba(self.X_scaled)
        self.assertEqual(proba.shape[0], self.n_samples)
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_feature_importance(self):
        """Test feature importance calculation"""
        importances = self.model.feature_importances_
        self.assertEqual(len(importances), self.n_features)
        self.assertGreater(sum(importances), 0)
        self.assertAlmostEqual(sum(importances), 1.0, places=5)

    def test_scaler_fit_transform(self):
        """Test scaler fit_transform"""
        X_transformed = self.scaler.fit_transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertLess(abs(X_transformed.mean()), 0.1)
        self.assertLess(abs(X_transformed.std() - 1.0), 0.1)

    def test_scaler_transform_consistency(self):
        """Test scaler transform consistency"""
        X_test = np.random.rand(10, self.n_features)
        X_transformed1 = self.scaler.transform(X_test)
        X_transformed2 = self.scaler.transform(X_test)
        np.testing.assert_array_equal(X_transformed1, X_transformed2)

class TestDataValidation(unittest.TestCase):
    """Test data validation and preprocessing"""

    def test_sample_data_loading(self):
        """Test sample data loading"""
        try:
            df = pd.read_csv('data/sample_data.csv')
            self.assertGreater(len(df), 0)
            self.assertIn('fever', df.columns)
        except FileNotFoundError:
            self.skipTest('Sample data file not found')

    def test_missing_values(self):
        """Test handling of missing values"""
        try:
            df = pd.read_csv('data/sample_data.csv')
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            self.assertLess(missing_ratio, 0.05)
        except FileNotFoundError:
            self.skipTest('Sample data file not found')

if __name__ == '__main__':
    unittest.main()
