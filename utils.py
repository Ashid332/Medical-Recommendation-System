import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Optional, Tuple

class DataProcessor:
    """Utility class for data processing and cleaning"""
    
    @staticmethod
    def clean_drug_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess drug data"""
        df_clean = df.copy()
        
        # Handle missing values
        df_clean['Drug'] = df_clean['Drug'].fillna('Unknown Drug')
        df_clean['Disease'] = df_clean['Disease'].fillna('Unknown Disease')
        df_clean['Gender'] = df_clean['Gender'].fillna('Unknown')
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
        
        # Extract active ingredients
        df_clean['Active_Ingredient'] = df_clean['Drug'].apply(
            DataProcessor.extract_ingredient
        )
        
        # Extract drug form
        df_clean['Drug_Form'] = df_clean['Drug'].apply(
            DataProcessor.extract_drug_form
        )
        
        # Extract strength
        df_clean['Strength'] = df_clean['Drug'].apply(
            DataProcessor.extract_strength
        )
        
        # Create age groups
        df_clean['Age_Group'] = pd.cut(
            df_clean['Age'],
            bins=[0, 18, 30, 45, 60, 100],
            labels=['0-18', '19-30', '31-45', '46-60', '60+']
        )
        
        return df_clean
    
    @staticmethod
    def extract_ingredient(drug_name: str) -> str:
        """Extract active ingredient from drug name"""
        if not isinstance(drug_name, str):
            return 'Unknown'
        
        # Common patterns for ingredient extraction
        patterns = [
            r'([A-Z][a-z]+)\s+\d',  # Word before number
            r'^([A-Za-z]+)\s+',     # First word
            r'([A-Za-z]+)\s+[A-Z]', # Word before capital
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drug_name)
            if match:
                return match.group(1)
        
        # Check for common ingredients
        common_ingredients = [
            'Retino', 'Benzoyl', 'Clindamycin', 'Isotretinoin',
            'Cetirizine', 'Loratadine', 'Fexofenadine', 'Levocetirizine',
            'Metformin', 'Glimepiride', 'Glipizide', 'Insulin',
            'Azithromycin', 'Doxycycline', 'Minocycline'
        ]
        
        for ingredient in common_ingredients:
            if ingredient.lower() in drug_name.lower():
                return ingredient
        
        return drug_name.split()[0] if drug_name.split() else 'Unknown'
    
    @staticmethod
    def extract_drug_form(drug_name: str) -> str:
        """Extract drug form from drug name"""
        if not isinstance(drug_name, str):
            return 'Unknown'
        
        drug_lower = drug_name.lower()
        
        forms = {
            'tablet': ['tablet', 'tab', 'tabs'],
            'capsule': ['capsule', 'cap', 'caps'],
            'gel': ['gel'],
            'cream': ['cream'],
            'lotion': ['lotion'],
            'injection': ['injection', 'inj'],
            'syrup': ['syrup'],
            'solution': ['solution', 'sol'],
            'ointment': ['ointment', 'oint'],
            'drops': ['drops', 'drop'],
            'spray': ['spray'],
            'powder': ['powder'],
            'suspension': ['suspension', 'susp']
        }
        
        for form, keywords in forms.items():
            for keyword in keywords:
                if keyword in drug_lower:
                    return form.title()
        
        return 'Other'
    
    @staticmethod
    def extract_strength(drug_name: str) -> str:
        """Extract strength from drug name"""
        if not isinstance(drug_name, str):
            return 'Unknown'
        
        # Look for patterns like 20mg, 5%, 500mg, etc.
        patterns = [
            r'(\d+\.?\d*)\s*mg',    # mg strength
            r'(\d+\.?\d*)\s*%',     # percentage
            r'(\d+\.?\d*)\s*gm',    # grams
            r'(\d+\.?\d*)\s*ml',    # ml
            r'(\d+\.?\d*)\s*i[uv]', # IU
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drug_name.lower())
            if match:
                return match.group(1)
        
        return 'Unknown'
    
    @staticmethod
    def create_interaction_matrix(df: pd.DataFrame, 
                                 user_col: str = 'User_ID',
                                 item_col: str = 'Drug',
                                 value_col: Optional[str] = None) -> pd.DataFrame:
        """Create user-item interaction matrix"""
        if value_col is None:
            # Create binary interactions
            interactions = df.groupby([user_col, item_col]).size().reset_index()
            interactions['interaction'] = 1
            value_col = 'interaction'
        else:
            interactions = df[[user_col, item_col, value_col]].copy()
        
        # Create pivot table
        interaction_matrix = interactions.pivot_table(
            index=user_col,
            columns=item_col,
            values=value_col,
            fill_value=0
        )
        
        return interaction_matrix
    
    @staticmethod
    def calculate_disease_statistics(df: pd.DataFrame) -> Dict:
        """Calculate statistics for each disease"""
        stats = {}
        
        for disease in df['Disease'].unique():
            disease_df = df[df['Disease'] == disease]
            
            stats[disease] = {
                'total_drugs': len(disease_df),
                'avg_age': disease_df['Age'].mean(),
                'gender_distribution': disease_df['Gender'].value_counts().to_dict(),
                'common_forms': disease_df['Drug_Form'].value_counts().head(5).to_dict(),
                'common_ingredients': disease_df['Active_Ingredient'].value_counts().head(5).to_dict()
            }
        
        return stats
    
    @staticmethod
    def generate_health_metrics(age: int, gender: str) -> Dict:
        """Generate simulated health metrics for a user"""
        np.random.seed(hash(f"{age}{gender}") % 10000)
        
        # Base values based on age and gender
        if gender.lower() == 'male':
            base_bp = 120 + (age - 30) * 0.5
            base_hr = 72 - (age - 30) * 0.1
        else:
            base_bp = 115 + (age - 30) * 0.4
            base_hr = 75 - (age - 30) * 0.08
        
        # Add some variation
        metrics = {
            'blood_pressure_systolic': max(90, min(180, np.random.normal(base_bp, 10))),
            'blood_pressure_diastolic': max(60, min(120, np.random.normal(base_bp * 0.67, 8))),
            'heart_rate': max(50, min(120, np.random.normal(base_hr, 8))),
            'glucose_level': max(70, min(200, np.random.normal(100 + age * 0.2, 15))),
            'cholesterol_total': max(150, min(300, np.random.normal(200 + age * 0.5, 30))),
            'bmi': max(18, min(35, np.random.normal(22 + age * 0.05, 3)))
        }
        
        # Determine blood pressure category
        bp_sys = metrics['blood_pressure_systolic']
        if bp_sys < 90:
            metrics['blood_pressure_category'] = 'Low'
        elif bp_sys < 120:
            metrics['blood_pressure_category'] = 'Normal'
        elif bp_sys < 140:
            metrics['blood_pressure_category'] = 'Elevated'
        else:
            metrics['blood_pressure_category'] = 'High'
        
        # Determine glucose category
        glucose = metrics['glucose_level']
        if glucose < 100:
            metrics['glucose_category'] = 'Normal'
        elif glucose < 126:
            metrics['glucose_category'] = 'Prediabetes'
        else:
            metrics['glucose_category'] = 'Diabetes'
        
        return metrics


class RecommendationUtils:
    """Utility functions for recommendation system"""
    
    @staticmethod
    def calculate_similarity_score(user_data: Dict, drug_data: pd.Series) -> float:
        """Calculate similarity score between user and drug"""
        score = 0.0
        
        # Age similarity (closer age gets higher score)
        age_diff = abs(user_data.get('age', 30) - drug_data.get('Age', 30))
        age_score = max(0, 1 - (age_diff / 50))  # Normalize
        score += age_score * 0.3
        
        # Gender match
        gender_match = 1.0 if user_data.get('gender', '').lower() == \
            str(drug_data.get('Gender', '')).lower() else 0.0
        score += gender_match * 0.2
        
        # Disease match (if specified)
        if 'disease' in user_data and 'Disease' in drug_data:
            disease_match = 1.0 if user_data['disease'].lower() == \
                str(drug_data['Disease']).lower() else 0.0
            score += disease_match * 0.5
        
        return min(1.0, score)
    
    @staticmethod
    def filter_by_preferences(drugs_df: pd.DataFrame, 
                            preferences: Dict) -> pd.DataFrame:
        """Filter drugs based on user preferences"""
        filtered_df = drugs_df.copy()
        
        # Filter by form if specified
        if 'preferred_forms' in preferences and preferences['preferred_forms']:
            forms = [form.lower() for form in preferences['preferred_forms']]
            filtered_df = filtered_df[
                filtered_df['Drug'].str.lower().str.contains('|'.join(forms))
            ]
        
        # Filter by budget if specified
        if 'budget' in preferences and preferences['budget'] != 'Any':
            # Simplified budget filtering (in real app, you'd have actual prices)
            if preferences['budget'] == 'Low':
                filtered_df = filtered_df.head(len(filtered_df) // 3)
            elif preferences['budget'] == 'Medium':
                mid = len(filtered_df) // 2
                filtered_df = filtered_df.iloc[mid-100:mid+100]
        
        # Filter out allergens if specified
        if 'allergies' in preferences and preferences['allergies']:
            allergies = [allergy.strip().lower() 
                        for allergy in preferences['allergies'].split(',')]
            
            def has_allergen(drug_name):
                drug_lower = str(drug_name).lower()
                return any(allergy in drug_lower for allergy in allergies)
            
            filtered_df = filtered_df[~filtered_df['Drug'].apply(has_allergen)]
        
        return filtered_df
    
    @staticmethod
    def generate_recommendation_explanation(user_data: Dict, 
                                          drug_data: pd.Series,
                                          score: float) -> str:
        """Generate human-readable explanation for recommendation"""
        explanations = []
        
        # Age-based explanation
        age_diff = abs(user_data.get('age', 30) - drug_data.get('Age', 30))
        if age_diff <= 5:
            explanations.append("Well-suited for your age group")
        elif age_diff <= 10:
            explanations.append("Commonly prescribed for your age range")
        
        # Gender-based explanation
        if user_data.get('gender', '').lower() == \
           str(drug_data.get('Gender', '')).lower():
            explanations.append("Specifically formulated for your gender")
        
        # Disease-based explanation
        if 'disease' in user_data and 'Disease' in drug_data:
            if user_data['disease'].lower() == str(drug_data['Disease']).lower():
                explanations.append("Primary treatment for your condition")
        
        # Form-based explanation
        if 'preferred_forms' in user_data and user_data['preferred_forms']:
            drug_lower = str(drug_data['Drug']).lower()
            for form in user_data['preferred_forms']:
                if form.lower() in drug_lower:
                    explanations.append(f"Available in your preferred {form} form")
                    break
        
        if not explanations:
            explanations.append("Recommended based on similar patient profiles")
        
        return " • ".join(explanations)
    
    @staticmethod
    def create_visualization_data(recommendations: pd.DataFrame) -> Dict:
        """Create data for visualization of recommendations"""
        if recommendations.empty:
            return {}
        
        viz_data = {
            'age_distribution': recommendations['Age'].tolist(),
            'gender_distribution': recommendations['Gender'].value_counts().to_dict(),
            'disease_distribution': recommendations['Disease'].value_counts().to_dict(),
            'form_distribution': recommendations['Drug_Form'].value_counts().to_dict()
        }
        
        return viz_data


class DataValidator:
    """Validate input data for the system"""
    
    @staticmethod
    def validate_user_input(user_data: Dict) -> Tuple[bool, List[str]]:
        """Validate user input data"""
        errors = []
        
        # Validate age
        age = user_data.get('age')
        if not isinstance(age, (int, float)) or age < 0 or age > 120:
            errors.append("Age must be between 0 and 120")
        
        # Validate gender
        gender = user_data.get('gender', '').lower()
        valid_genders = ['male', 'female', 'other', 'unknown']
        if gender not in valid_genders:
            errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
        
        # Validate symptoms (if provided)
        if 'symptoms' in user_data:
            symptoms = user_data['symptoms']
            if not isinstance(symptoms, list):
                errors.append("Symptoms must be a list")
            elif len(symptoms) > 50:
                errors.append("Too many symptoms provided (max 50)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_drug_data(drug_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate drug dataset"""
        errors = []
        
        if drug_data.empty:
            errors.append("Drug data is empty")
            return False, errors
        
        required_columns = ['Drug', 'Disease', 'Gender', 'Age']
        missing_columns = [col for col in required_columns 
                          if col not in drug_data.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for null values in required columns
        for col in required_columns:
            if col in drug_data.columns:
                null_count = drug_data[col].isnull().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}' has {null_count} null values")
        
        # Check age values
        if 'Age' in drug_data.columns:
            invalid_ages = drug_data[
                (drug_data['Age'] < 0) | (drug_data['Age'] > 120)
            ].shape[0]
            if invalid_ages > 0:
                errors.append(f"Found {invalid_ages} invalid age values")
        
        return len(errors) == 0, errors


class CacheManager:
    """Simple cache manager for system data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
    
    def set(self, key: str, value, expiry_seconds: int = 3600):
        """Set value in cache with expiry"""
        self.cache[key] = value
        self.cache_expiry[key] = datetime.now() + \
            timedelta(seconds=expiry_seconds)
    
    def get(self, key: str):
        """Get value from cache if not expired"""
        if key not in self.cache:
            return None
        
        if datetime.now() > self.cache_expiry.get(key, datetime.min):
            # Cache expired
            del self.cache[key]
            del self.cache_expiry[key]
            return None
        
        return self.cache[key]
    
    def clear(self, key: Optional[str] = None):
        """Clear cache entries"""
        if key is None:
            self.cache.clear()
            self.cache_expiry.clear()
        elif key in self.cache:
            del self.cache[key]
            del self.cache_expiry[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'expired_entries': sum(
                1 for expiry in self.cache_expiry.values()
                if datetime.now() > expiry
            ),
            'memory_usage': sum(
                len(str(value).encode('utf-8')) 
                for value in self.cache.values()
            )
        }


# Singleton instances
data_processor = DataProcessor()
recommendation_utils = RecommendationUtils()
data_validator = DataValidator()
cache_manager = CacheManager()

if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test data cleaning
    print("\n1. Testing data cleaning...")
    sample_data = pd.DataFrame({
        'Drug': ['Acnetoin 20mg Capsule', 'Cetirizine Tablet', None],
        'Disease': ['Acne', 'Allergy', None],
        'Gender': ['Male', 'Female', 'Unknown'],
        'Age': [23, 30, None]
    })
    
    cleaned = data_processor.clean_drug_data(sample_data)
    print(f"Cleaned data shape: {cleaned.shape}")
    print(f"Extracted ingredients: {cleaned['Active_Ingredient'].tolist()}")
    print(f"Extracted forms: {cleaned['Drug_Form'].tolist()}")
    
    # Test health metrics generation
    print("\n2. Testing health metrics generation...")
    metrics = data_processor.generate_health_metrics(35, 'Male')
    print(f"Health metrics for 35-year-old male:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test recommendation utilities
    print("\n3. Testing recommendation utilities...")
    user_data = {'age': 30, 'gender': 'male', 'disease': 'Acne'}
    drug_data = pd.Series({'Age': 28, 'Gender': 'Male', 'Disease': 'Acne'})
    
    score = recommendation_utils.calculate_similarity_score(user_data, drug_data)
    explanation = recommendation_utils.generate_recommendation_explanation(
        user_data, drug_data, score
    )
    
    print(f"Similarity score: {score:.2f}")
    print(f"Explanation: {explanation}")
    
    # Test data validation
    print("\n4. Testing data validation...")
    test_user_data = {'age': 150, 'gender': 'unknown', 'symptoms': []}
    is_valid, errors = data_validator.validate_user_input(test_user_data)
    print(f"Validation result: {is_valid}")
    if not is_valid:
        print(f"Errors: {errors}")
    
    # Test cache manager
    print("\n5. Testing cache manager...")
    cache_manager.set('test_key', 'test_value', 10)
    cached_value = cache_manager.get('test_key')
    print(f"Cached value: {cached_value}")
    
    stats = cache_manager.get_stats()
    print(f"Cache stats: {stats}")
    
    print("\n✅ All utility functions tested successfully!")