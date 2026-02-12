import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime

class MedicalModelEvaluator:
    """
    Comprehensive evaluation suite for medical recommendation system.
    Evaluates model performance, generates detailed metrics, and creates visualizations.
    """
    
    def __init__(self, model_path, test_data_path):
        """
        Initialize evaluator with model and test data.
        
        Args:
            model_path (str): Path to trained model file
            test_data_path (str): Path to test dataset
        """
        self.model = pickle.load(open(model_path, 'rb'))
        self.test_data = pd.read_csv(test_data_path)
        self.results = {}
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def prepare_data(self):
        """Extract features and target from test data."""
        # Assuming last column is target
        self.X_test = self.test_data.iloc[:, :-1]
        self.y_test = self.test_data.iloc[:, -1]
        print(f"Test data prepared: {self.X_test.shape[0]} samples, {self.X_test.shape[1]} features")
    
    def evaluate_classification(self):
        """Evaluate classification model performance."""
        predictions = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, predictions, average='weighted', zero_division=0)
        
        self.results['accuracy'] = float(accuracy)
        self.results['precision'] = float(precision)
        self.results['recall'] = float(recall)
        self.results['f1_score'] = float(f1)
        self.predictions = predictions
        
        print(f"\n=== Classification Metrics ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
    
    def generate_detailed_report(self):
        """Generate detailed classification report."""
        report = classification_report(self.y_test, self.predictions, output_dict=True)
        self.results['classification_report'] = report
        print("\n=== Detailed Classification Report ===")
        print(classification_report(self.y_test, self.predictions))
        return report
    
    def confusion_matrix_analysis(self):
        """Analyze and visualize confusion matrix."""
        cm = confusion_matrix(self.y_test, self.predictions)
        self.results['confusion_matrix'] = cm.tolist()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("Confusion matrix visualization saved.")
    
    def save_results(self, output_path='evaluation_results.json'):
        """Save evaluation results to JSON file."""
        self.results['timestamp'] = self.timestamp
        self.results['test_samples'] = int(len(self.y_test))
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nResults saved to {output_path}")
    
    def run_full_evaluation(self):
        """Execute complete evaluation pipeline."""
        print("Starting model evaluation...")
        self.prepare_data()
        self.evaluate_classification()
        self.generate_detailed_report()
        self.confusion_matrix_analysis()
        self.save_results()
        print("\nEvaluation complete!")

if __name__ == '__main__':
    # Example usage
    evaluator = MedicalModelEvaluator(
        model_path='../models/medical_model.pkl',
        test_data_path='../data/test_data.csv'
    )
    evaluator.run_full_evaluation()
