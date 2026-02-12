import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictionModel:
    """Advanced ML model for disease prediction using multiple algorithms"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize disease prediction model
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 
                       'logistic_regression', 'svm', 'knn', 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'gender', 'blood_pressure', 'glucose_level', 
            'heart_rate', 'cholesterol', 'bmi', 'symptoms_count',
            'family_history', 'smoking', 'alcohol', 'exercise'
        ]
        self.disease_mapping = {}
        self.feature_importance = {}
        
    def prepare_data(self, data):
        """Prepare and preprocess data for training"""
        df = data.copy()
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Encode categorical variables
        categorical_cols = ['gender', 'blood_pressure', 'cholesterol', 
                          'family_history', 'smoking', 'alcohol', 'exercise']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Process symptoms
        if 'symptoms' in df.columns:
            df['symptoms_count'] = df['symptoms'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
            # Create binary features for common symptoms
            common_symptoms = ['fever', 'cough', 'headache', 'fatigue', 'nausea', 
                             'rash', 'itching', 'pain', 'swelling', 'dizziness']
            for symptom in common_symptoms:
                df[f'symptom_{symptome}'] = df['symptoms'].str.contains(
                    symptom, case=False, na=False
                ).astype(int)
                self.feature_columns.append(f'symptom_{symptom}')
        
        # Create BMI categories
        df['bmi_category'] = pd.cut(df['bmi'], 
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        le_bmi = LabelEncoder()
        df['bmi_category'] = le_bmi.fit_transform(df['bmi_category'])
        self.label_encoders['bmi_category'] = le_bmi
        self.feature_columns.append('bmi_category')
        
        # Create age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 18, 30, 45, 60, 100],
                                labels=['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])
        le_age = LabelEncoder()
        df['age_group'] = le_age.fit_transform(df['age_group'])
        self.label_encoders['age_group'] = le_age
        self.feature_columns.append('age_group')
        
        # Scale numerical features
        numerical_cols = ['age', 'glucose_level', 'heart_rate', 'bmi', 'symptoms_count']
        if len(df) > 1:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        # Encode target variable
        if 'disease' in df.columns:
            le_target = LabelEncoder()
            df['disease_encoded'] = le_target.fit_transform(df['disease'])
            self.label_encoders['disease'] = le_target
            self.disease_mapping = dict(zip(le_target.transform(le_target.classes_), 
                                          le_target.classes_))
        
        return df
    
    def train(self, X, y, tune_hyperparameters=False):
        """Train the disease prediction model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            if tune_hyperparameters:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                grid_search = GridSearchCV(
                    self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
        
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                subsample=0.8
            )
        
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear'
            )
        
        elif self.model_type == 'svm':
            self.model = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        
        elif self.model_type == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski'
            )
        
        elif self.model_type == 'ensemble':
            from sklearn.ensemble import VotingClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            lr = LogisticRegression(random_state=42)
            
            self.model = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft',
                weights=[2, 1, 1]
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                X.columns, 
                self.model.feature_importances_
            ))
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy_val = accuracy_score(y_test, y_pred)
        
        print(f"Model trained: {self.model_type}")
        print(f"Test Accuracy: {accuracy_val:.2%}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return self.model, accuracy_val
    
    def predict(self, features, return_probabilities=True):
        """Make disease prediction with confidence scores"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare input features
        input_df = pd.DataFrame([features])
        
        # Encode categorical features
        for col, le in self.label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except:
                    # Handle unseen labels by using most common category
                    input_df[col] = le.transform([le.classes_[0]])[0]
        
        # Scale numerical features
        numerical_cols = ['age', 'glucose_level', 'heart_rate', 'bmi', 'symptoms_count']
        available_numerical = [col for col in numerical_cols if col in input_df.columns]
        if available_numerical:
            input_df[available_numerical] = self.scaler.transform(
                input_df[available_numerical]
            )
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        predicted_disease = self.disease_mapping.get(prediction, f"Class_{prediction}")
        
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_df)[0]
            prob_dict = {
                self.disease_mapping.get(i, f"Class_{i}"): prob 
                for i, prob in enumerate(probabilities)
            }
            return predicted_disease, prob_dict
        else:
            return predicted_disease, None
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        accuracy_val = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy_val,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        # Calculate additional metrics if probabilities are available
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score, log_loss
            try:
                metrics['roc_auc'] = roc_auc_score(
                    pd.get_dummies(y_test), y_pred_proba, average='macro'
                )
                metrics['log_loss'] = log_loss(
                    pd.get_dummies(y_test), y_pred_proba
                )
            except:
                pass
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'disease_mapping': self.disease_mapping,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.disease_mapping = model_data['disease_mapping']
        self.feature_importance = model_data.get('feature_importance', {})
        self.model_type = model_data.get('model_type', 'random_forest')
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if not self.feature_importance:
            return {}
        
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return dict(sorted_features)


class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering for personalized medicine recommendations"""
    
    def __init__(self, n_users, n_items, n_factors=50, use_deep_layers=True):
        """
        Initialize NCF model
        
        Args:
            n_users: Number of unique users
            n_items: Number of unique items
            n_factors: Embedding dimension
            use_deep_layers: Whether to use deep neural network layers
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.use_deep_layers = use_deep_layers
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.history = None
    
    def build_gmf_model(self):
        """Build Generalized Matrix Factorization model"""
        # User embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(
            input_dim=self.n_users, 
            output_dim=self.n_factors, 
            name='user_embedding'
        )(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)
        
        # Item embedding
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(
            input_dim=self.n_items, 
            output_dim=self.n_factors, 
            name='item_embedding'
        )(item_input)
        item_vec = Flatten(name='item_flatten')(item_embedding)
        
        # Element-wise product (GMF)
        gmf_layer = tf.keras.layers.multiply([user_vec, item_vec])
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(gmf_layer)
        
        # Create model
        self.model = Model(
            inputs=[user_input, item_input],
            outputs=output,
            name='GMF_Model'
        )
        
        return self.model
    
    def build_mlp_model(self):
        """Build Multi-Layer Perceptron model"""
        # User embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(
            input_dim=self.n_users, 
            output_dim=self.n_factors, 
            name='user_embedding'
        )(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)
        
        # Item embedding
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(
            input_dim=self.n_items, 
            output_dim=self.n_factors, 
            name='item_embedding'
        )(item_input)
        item_vec = Flatten(name='item_flatten')(item_embedding)
        
        # Concatenate embeddings
        concat = Concatenate(name='concat_layer')([user_vec, item_vec])
        
        # Deep layers
        dense1 = Dense(128, activation='relu', name='dense1')(concat)
        dropout1 = Dropout(0.3, name='dropout1')(dense1)
        batch_norm1 = BatchNormalization(name='batch_norm1')(dropout1)
        
        dense2 = Dense(64, activation='relu', name='dense2')(batch_norm1)
        dropout2 = Dropout(0.3, name='dropout2')(dense2)
        batch_norm2 = BatchNormalization(name='batch_norm2')(dropout2)
        
        dense3 = Dense(32, activation='relu', name='dense3')(batch_norm2)
        dropout3 = Dropout(0.2, name='dropout3')(dense3)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(dropout3)
        
        # Create model
        self.model = Model(
            inputs=[user_input, item_input],
            outputs=output,
            name='MLP_Model'
        )
        
        return self.model
    
    def build_neuMF_model(self):
        """Build Neural Matrix Factorization (NeuMF) model - Combines GMF and MLP"""
        # User embedding for GMF
        user_input = Input(shape=(1,), name='user_input')
        user_embedding_gmf = Embedding(
            input_dim=self.n_users, 
            output_dim=self.n_factors, 
            name='user_embedding_gmf'
        )(user_input)
        user_vec_gmf = Flatten(name='user_flatten_gmf')(user_embedding_gmf)
        
        # Item embedding for GMF
        item_input = Input(shape=(1,), name='item_input')
        item_embedding_gmf = Embedding(
            input_dim=self.n_items, 
            output_dim=self.n_factors, 
            name='item_embedding_gmf'
        )(item_input)
        item_vec_gmf = Flatten(name='item_flatten_gmf')(item_embedding_gmf)
        
        # GMF path
        gmf_layer = tf.keras.layers.multiply([user_vec_gmf, item_vec_gmf])
        gmf_layer = Dropout(0.2)(gmf_layer)
        
        # MLP path
        user_embedding_mlp = Embedding(
            input_dim=self.n_users, 
            output_dim=self.n_factors * 2, 
            name='user_embedding_mlp'
        )(user_input)
        user_vec_mlp = Flatten(name='user_flatten_mlp')(user_embedding_mlp)
        
        item_embedding_mlp = Embedding(
            input_dim=self.n_items, 
            output_dim=self.n_factors * 2, 
            name='item_embedding_mlp'
        )(item_input)
        item_vec_mlp = Flatten(name='item_flatten_mlp')(item_embedding_mlp)
        
        mlp_concat = Concatenate(name='mlp_concat')([user_vec_mlp, item_vec_mlp])
        
        # MLP layers
        mlp_dense1 = Dense(128, activation='relu')(mlp_concat)
        mlp_dropout1 = Dropout(0.3)(mlp_dense1)
        mlp_batch1 = BatchNormalization()(mlp_dropout1)
        
        mlp_dense2 = Dense(64, activation='relu')(mlp_batch1)
        mlp_dropout2 = Dropout(0.3)(mlp_dense2)
        mlp_batch2 = BatchNormalization()(mlp_dropout2)
        
        mlp_dense3 = Dense(32, activation='relu')(mlp_batch2)
        mlp_output = Dropout(0.2)(mlp_dense3)
        
        # Concatenate GMF and MLP
        concat_gmf_mlp = Concatenate(name='concat_gmf_mlp')([gmf_layer, mlp_output])
        
        # Final layers
        final_dense1 = Dense(16, activation='relu')(concat_gmf_mlp)
        final_dropout = Dropout(0.2)(final_dense1)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(final_dropout)
        
        # Create model
        self.model = Model(
            inputs=[user_input, item_input],
            outputs=output,
            name='NeuMF_Model'
        )
        
        return self.model
    
    def build_model(self, model_type='neumf'):
        """Build NCF model with specified architecture"""
        if model_type == 'gmf':
            self.model = self.build_gmf_model()
        elif model_type == 'mlp':
            self.model = self.build_mlp_model()
        elif model_type == 'neumf':
            self.model = self.build_neuMF_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from ['gmf', 'mlp', 'neumf']")
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'mse', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model.summary()
        return self.model
    
    def prepare_data(self, interactions_df, test_size=0.2):
        """Prepare data for NCF training"""
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(interactions_df['user_id'].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(interactions_df['item_id'].unique())}
        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        
        # Map IDs to indices
        interactions_df['user_idx'] = interactions_df['user_id'].map(self.user_mapping)
        interactions_df['item_idx'] = interactions_df['item_id'].map(self.item_mapping)
        
        # Normalize ratings to [0, 1] if needed
        if 'rating' in interactions_df.columns:
            scaler = MinMaxScaler()
            interactions_df['rating_norm'] = scaler.fit_transform(
                interactions_df[['rating']]
            ).flatten()
        else:
            # Create binary interactions (implicit feedback)
            interactions_df['rating_norm'] = 1.0
        
        # Split data
        train_df, test_df = train_test_split(
            interactions_df, 
            test_size=test_size, 
            random_state=42,
            stratify=interactions_df['user_id'] if len(interactions_df) > 100 else None
        )
        
        # Create negative samples for training (optional)
        train_df = self._create_negative_samples(train_df, num_negatives=4)
        
        return train_df, test_df
    
    def _create_negative_samples(self, df, num_negatives=4):
        """Create negative samples for implicit feedback"""
        if 'rating' not in df.columns:
            return df
        
        negative_samples = []
        all_items = set(self.item_mapping.keys())
        
        for _, row in df.iterrows():
            user_items = set(df[df['user_id'] == row['user_id']]['item_id'])
            negative_items = list(all_items - user_items)
            
            if len(negative_items) > num_negatives:
                selected_negatives = np.random.choice(
                    negative_items, 
                    num_negatives, 
                    replace=False
                )
            else:
                selected_negatives = negative_items[:num_negatives]
            
            for item in selected_negatives:
                negative_samples.append({
                    'user_id': row['user_id'],
                    'item_id': item,
                    'rating': 0,
                    'rating_norm': 0,
                    'user_idx': row['user_idx'],
                    'item_idx': self.item_mapping[item]
                })
        
        # Combine positive and negative samples
        result_df = pd.concat([df, pd.DataFrame(negative_samples)], ignore_index=True)
        return result_df
    
    def train(self, train_df, val_df=None, epochs=20, batch_size=256, 
              patience=5, learning_rate=0.001):
        """Train the NCF model"""
        if self.model is None:
            self.build_model('neumf')  # Default to NeuMF
        
        # Prepare training data
        X_train = [train_df['user_idx'].values, train_df['item_idx'].values]
        y_train = train_df['rating_norm'].values
        
        # Validation data
        if val_df is not None:
            X_val = [val_df['user_idx'].values, val_df['item_idx'].values]
            y_val = val_df['rating_norm'].values
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        return self.history
    
    def predict(self, user_idx, item_idx):
        """Make prediction for user-item pair"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        prediction = self.model.predict(
            [np.array([user_idx]), np.array([item_idx])],
            verbose=0
        )
        return prediction[0][0]
    
    def recommend(self, user_id, top_n=10, filter_items=None):
        """Generate top N recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if user_id not in self.user_mapping:
            raise ValueError(f"User ID {user_id} not found in training data")
        
        user_idx = self.user_mapping[user_id]
        
        # Get predictions for all items
        item_indices = list(range(self.n_items))
        
        # Filter items if specified
        if filter_items is not None:
            item_indices = [idx for idx in item_indices 
                           if self.reverse_item_mapping[idx] in filter_items]
        
        user_indices = [user_idx] * len(item_indices)
        
        predictions = self.model.predict(
            [np.array(user_indices), np.array(item_indices)],
            batch_size=1024,
            verbose=0
        ).flatten()
        
        # Get top N items
        top_indices = np.argsort(predictions)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'item_id': self.reverse_item_mapping[idx],
                'item_idx': idx,
                'score': float(predictions[idx]),
                'confidence': f"{predictions[idx]:.1%}"
            })
        
        return recommendations
    
    def evaluate(self, test_df, metrics=['rmse', 'mae', 'precision', 'recall', 'ndcg']):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test = [test_df['user_idx'].values, test_df['item_idx'].values]
        y_test = test_df['rating_norm'].values
        
        # Get predictions
        y_pred = self.model.predict(X_test, batch_size=1024, verbose=0).flatten()
        
        evaluation_results = {}
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        if 'rmse' in metrics:
            evaluation_results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        
        if 'mae' in metrics:
            evaluation_results['mae'] = mean_absolute_error(y_test, y_pred)
        
        if 'precision' in metrics or 'recall' in metrics or 'ndcg' in metrics:
            # For ranking metrics, use top-k recommendations
            evaluation_results.update(
                self._calculate_ranking_metrics(test_df, k=10)
            )
        
        return evaluation_results
    
    def _calculate_ranking_metrics(self, test_df, k=10):
        """Calculate ranking metrics (Precision@k, Recall@k, NDCG@k)"""
        metrics = {}
        
        # Group by user
        user_groups = test_df.groupby('user_idx')
        
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_idx, group in user_groups:
            # Get true positive items for this user
            true_items = set(group[group['rating_norm'] > 0.5]['item_idx'])
            
            if len(true_items) == 0:
                continue
            
            # Get recommendations
            recommendations = self.recommend(
                self.reverse_user_mapping[user_idx], 
                top_n=k
            )
            recommended_items = set([rec['item_idx'] for rec in recommendations])
            
            # Calculate metrics
            if len(recommended_items) > 0:
                intersection = true_items & recommended_items
                
                # Precision@k
                precision = len(intersection) / len(recommended_items)
                precisions.append(precision)
                
                # Recall@k
                recall = len(intersection) / len(true_items)
                recalls.append(recall)
                
                # NDCG@k (simplified)
                dcg = sum([1 / np.log2(i + 2) for i in range(len(intersection))])
                idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
        
        if precisions:
            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'recall@{k}'] = np.mean(recalls)
            metrics[f'ndcg@{k}'] = np.mean(ndcgs)
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model and mappings"""
        model_data = {
            'model': self.model,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_factors': self.n_factors,
            'history': self.history
        }
        joblib.dump(model_data, filepath)
        print(f"NCF model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.user_mapping = model_data['user_mapping']
        self.item_mapping = model_data['item_mapping']
        self.reverse_user_mapping = model_data['reverse_user_mapping']
        self.reverse_item_mapping = model_data['reverse_item_mapping']
        self.n_users = model_data['n_users']
        self.n_items = model_data['n_items']
        self.n_factors = model_data['n_factors']
        self.history = model_data.get('history')
        print(f"NCF model loaded from {filepath}")


class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches"""
    
    def __init__(self, drug_data, weight_content=0.4, weight_collab=0.4, weight_popularity=0.2):
        """
        Initialize hybrid recommender
        
        Args:
            drug_data: DataFrame with drug information
            weight_content: Weight for content-based filtering
            weight_collab: Weight for collaborative filtering
            weight_popularity: Weight for popularity-based filtering
        """
        self.drug_data = drug_data.copy()
        self.weight_content = weight_content
        self.weight_collab = weight_collab
        self.weight_popularity = weight_popularity
        
        self.tfidf_vectorizer = None
        self.content_similarity = None
        self.popularity_scores = None
        self.user_item_matrix = None
        
        # Initialize models
        self.content_model = None
        self.collab_model = None
        
    def build_content_based_model(self, use_features=['Drug', 'Disease', 'Gender']):
        """Build content-based filtering model"""
        # Prepare drug descriptions
        drug_features = []
        for _, row in self.drug_data.iterrows():
            features = []
            for feature in use_features:
                if feature in row:
                    features.append(str(row[feature]))
            drug_features.append(' '.join(features))
        
        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(drug_features)
        
        # Calculate similarity matrix using multiple similarity measures
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Additional similarity: Jaccard similarity for categorical features
        if 'Drug_Form' in self.drug_data.columns and 'Active_Ingredient' in self.drug_data.columns:
            jaccard_sim = self._calculate_jaccard_similarity()
            # Combine similarities
            self.content_similarity = 0.7 * cosine_sim + 0.3 * jaccard_sim
        else:
            self.content_similarity = cosine_sim
        
        return self.content_similarity
    
    def _calculate_jaccard_similarity(self):
        """Calculate Jaccard similarity for categorical features"""
        n_drugs = len(self.drug_data)
        jaccard_sim = np.zeros((n_drugs, n_drugs))
        
        # Create feature sets for each drug
        feature_sets = []
        for _, row in self.drug_data.iterrows():
            features = set()
            if 'Drug_Form' in row and pd.notna(row['Drug_Form']):
                features.add(f"form_{row['Drug_Form']}")
            if 'Active_Ingredient' in row and pd.notna(row['Active_Ingredient']):
                features.add(f"ingredient_{row['Active_Ingredient']}")
            if 'Disease' in row and pd.notna(row['Disease']):
                features.add(f"disease_{row['Disease']}")
            feature_sets.append(features)
        
        # Calculate Jaccard similarity
        for i in range(n_drugs):
            for j in range(i, n_drugs):
                intersection = len(feature_sets[i] & feature_sets[j])
                union = len(feature_sets[i] | feature_sets[j])
                similarity = intersection / union if union > 0 else 0
                jaccard_sim[i][j] = similarity
                jaccard_sim[j][i] = similarity
        
        return jaccard_sim
    
    def build_popularity_model(self):
        """Build popularity-based model"""
        if 'rating' in self.drug_data.columns:
            # Use actual ratings
            self.popularity_scores = self.drug_data.groupby('Drug')['rating'].mean()
        else:
            # Simulate popularity based on disease prevalence
            disease_counts = self.drug_data['Disease'].value_counts()
            self.popularity_scores = self.drug_data['Disease'].map(
                lambda x: disease_counts.get(x, 0) / len(self.drug_data)
            )
        
        return self.popularity_scores
    
    def content_based_recommendations(self, drug_idx, top_n=10, min_similarity=0.1):
        """Get content-based recommendations"""
        if self.content_similarity is None:
            self.build_content_based_model()
        
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity[drug_idx]))
        
        # Filter by minimum similarity
        sim_scores = [(i, score) for i, score in sim_scores if score >= min_similarity and i != drug_idx]
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_indices = [i for i, _ in sim_scores[:top_n]]
        top_scores = [score for _, score in sim_scores[:top_n]]
        
        recommendations = []
        for idx, score in zip(top_indices, top_scores):
            recommendations.append({
                'drug_idx': idx,
                'drug_name': self.drug_data.iloc[idx]['Drug'],
                'score': float(score),
                'type': 'content_based',
                'explanation': f"Similar to {self.drug_data.iloc[drug_idx]['Drug']}"
            })
        
        return recommendations
    
    def collaborative_recommendations(self, user_id, top_n=10):
        """Get collaborative filtering recommendations"""
        if self.collab_model is None:
            raise ValueError("Collaborative model not built. Call build_collaborative_model() first.")
        
        try:
            recommendations = self.collab_model.recommend(user_id, top_n=top_n)
        except:
            # Fallback to content-based if collaborative fails
            recommendations = self.content_based_recommendations(0, top_n=top_n)
            for rec in recommendations:
                rec['type'] = 'content_based_fallback'
        
        return recommendations
    
    def popularity_recommendations(self, top_n=10):
        """Get popularity-based recommendations"""
        if self.popularity_scores is None:
            self.build_popularity_model()
        
        # Get top N popular drugs
        if isinstance(self.popularity_scores, pd.Series):
            top_drugs = self.popularity_scores.sort_values(ascending=False).head(top_n)
            
            recommendations = []
            for idx, (drug, score) in enumerate(top_drugs.items()):
                drug_idx = self.drug_data[self.drug_data['Drug'] == drug].index[0]
                recommendations.append({
                    'drug_idx': drug_idx,
                    'drug_name': drug,
                    'score': float(score),
                    'type': 'popularity',
                    'explanation': 'Popular choice for this condition'
                })
        else:
            # Fallback: most frequent diseases
            top_diseases = self.drug_data['Disease'].value_counts().head(top_n)
            recommendations = []
            
            for disease, count in top_diseases.items():
                drug_idx = self.drug_data[self.drug_data['Disease'] == disease].index[0]
                score = count / len(self.drug_data)
                recommendations.append({
                    'drug_idx': drug_idx,
                    'drug_name': self.drug_data.iloc[drug_idx]['Drug'],
                    'score': float(score),
                    'type': 'popularity',
                    'explanation': f'Commonly prescribed for {disease}'
                })
        
        return recommendations
    
    def hybrid_recommendations(self, user_data, top_n=10, 
                             include_content=True, 
                             include_collab=True,
                             include_popularity=True):
        """Generate hybrid recommendations"""
        all_recommendations = {}
        
        # Get content-based recommendations if user has history
        if include_content and 'history' in user_data and user_data['history']:
            last_drug = user_data['history'][-1]
            if last_drug in self.drug_data['Drug'].values:
                drug_idx = self.drug_data[self.drug_data['Drug'] == last_drug].index[0]
                content_recs = self.content_based_recommendations(drug_idx, top_n=top_n * 2)
                for rec in content_recs:
                    drug_idx = rec['drug_idx']
                    if drug_idx not in all_recommendations:
                        all_recommendations[drug_idx] = {
                            'drug_name': rec['drug_name'],
                            'scores': {'content': rec['score'] * self.weight_content},
                            'explanations': [rec.get('explanation', '')]
                        }
                    else:
                        all_recommendations[drug_idx]['scores']['content'] = rec['score'] * self.weight_content
        
        # Get collaborative recommendations
        if include_collab and 'user_id' in user_data:
            try:
                collab_recs = self.collaborative_recommendations(user_data['user_id'], top_n=top_n * 2)
                for rec in collab_recs:
                    if 'drug_idx' in rec:
                        drug_idx = rec['drug_idx']
                        if drug_idx not in all_recommendations:
                            all_recommendations[drug_idx] = {
                                'drug_name': rec['drug_name'],
                                'scores': {'collab': rec['score'] * self.weight_collab},
                                'explanations': [rec.get('explanation', '')]
                            }
                        else:
                            all_recommendations[drug_idx]['scores']['collab'] = rec['score'] * self.weight_collab
            except:
                pass
        
        # Get popularity recommendations
        if include_popularity:
            pop_recs = self.popularity_recommendations(top_n=top_n * 2)
            for rec in pop_recs:
                drug_idx = rec['drug_idx']
                if drug_idx not in all_recommendations:
                    all_recommendations[drug_idx] = {
                        'drug_name': rec['drug_name'],
                        'scores': {'popularity': rec['score'] * self.weight_popularity},
                        'explanations': [rec.get('explanation', '')]
                    }
                else:
                    all_recommendations[drug_idx]['scores']['popularity'] = rec['score'] * self.weight_popularity
        
        # Calculate hybrid scores
        hybrid_recommendations = []
        for drug_idx, data in all_recommendations.items():
            total_score = sum(data['scores'].values())
            explanations = ' | '.join(filter(None, data['explanations']))
            
            hybrid_recommendations.append({
                'drug_idx': drug_idx,
                'drug_name': data['drug_name'],
                'hybrid_score': total_score,
                'component_scores': data['scores'],
                'explanation': explanations,
                'drug_info': self.drug_data.iloc[drug_idx].to_dict()
            })
        
        # Sort by hybrid score and return top N
        hybrid_recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_recommendations[:top_n]
    
    def contextual_filtering(self, recommendations, context):
        """Apply contextual filtering to recommendations"""
        filtered_recs = []
        
        for rec in recommendations:
            drug_info = rec['drug_info']
            score_modifier = 1.0
            
            # Time-based filtering (e.g., morning vs evening medications)
            if 'time_of_day' in context:
                if context['time_of_day'] == 'morning':
                    # Prefer once-daily medications in morning
                    if 'once daily' in str(drug_info.get('Drug', '')).lower():
                        score_modifier *= 1.2
                
                elif context['time_of_day'] == 'evening':
                    # Prefer medications that cause drowsiness in evening
                    if any(word in str(drug_info.get('Drug', '')).lower() 
                           for word in ['night', 'pm', 'sleep']):
                        score_modifier *= 1.2
            
            # Season-based filtering
            if 'season' in context:
                if context['season'] == 'winter':
                    # Prefer medications for cold/flu
                    if any(word in str(drug_info.get('Disease', '')).lower() 
                           for word in ['cold', 'flu', 'cough']):
                        score_modifier *= 1.1
                
                elif context['season'] == 'summer':
                    # Prefer medications for allergies/skin conditions
                    if any(word in str(drug_info.get('Disease', '')).lower() 
                           for word in ['allergy', 'acne', 'skin']):
                        score_modifier *= 1.1
            
            # Age-based filtering
            if 'age' in context:
                age = context['age']
                drug_age = drug_info.get('Age', 30)
                
                if abs(age - drug_age) <= 5:
                    score_modifier *= 1.2
                elif abs(age - drug_age) <= 10:
                    score_modifier *= 1.1
            
            # Update score with context modifier
            rec['hybrid_score'] *= score_modifier
            rec['context_modifier'] = score_modifier
            
            filtered_recs.append(rec)
        
        # Re-sort based on updated scores
        filtered_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return filtered_recs
    
    def save_model(self, filepath):
        """Save hybrid recommender model"""
        model_data = {
            'drug_data': self.drug_data,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'content_similarity': self.content_similarity,
            'popularity_scores': self.popularity_scores,
            'weight_content': self.weight_content,
            'weight_collab': self.weight_collab,
            'weight_popularity': self.weight_popularity
        }
        
        if hasattr(self, 'collab_model'):
            model_data['collab_model'] = self.collab_model
        
        joblib.dump(model_data, filepath)
        print(f"Hybrid recommender saved to {filepath}")
    
    def load_model(self, filepath):
        """Load hybrid recommender model"""
        model_data = joblib.load(filepath)
        
        self.drug_data = model_data['drug_data']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.content_similarity = model_data['content_similarity']
        self.popularity_scores = model_data['popularity_scores']
        self.weight_content = model_data['weight_content']
        self.weight_collab = model_data['weight_collab']
        self.weight_popularity = model_data['weight_popularity']
        
        if 'collab_model' in model_data:
            self.collab_model = model_data['collab_model']
        
        print(f"Hybrid recommender loaded from {filepath}")


class ContentBasedRecommender:
    """Content-Based Filtering for medicine recommendations"""
    
    def __init__(self, drug_data):
        self.drug_data = drug_data.copy()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.feature_matrix = None
        self.similarity_matrix = None
        
    def prepare_features(self):
        """Prepare features for content-based filtering"""
        # Combine multiple features into a single text
        features_text = []
        for _, row in self.drug_data.iterrows():
            text_parts = []
            
            # Drug name and description
            if 'Drug' in row:
                text_parts.append(str(row['Drug']))
            
            # Disease information
            if 'Disease' in row:
                text_parts.append(str(row['Disease']))
            
            # Additional features
            if 'Drug_Form' in row:
                text_parts.append(f"form_{row['Drug_Form']}")
            
            if 'Active_Ingredient' in row:
                text_parts.append(f"ingredient_{row['Active_Ingredient']}")
            
            features_text.append(' '.join(text_parts))
        
        # Create TF-IDF matrix
        self.feature_matrix = self.vectorizer.fit_transform(features_text)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        return self.similarity_matrix
    
    def recommend(self, query, top_n=10):
        """Recommend drugs based on content similarity"""
        if self.feature_matrix is None:
            self.prepare_features()
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with all drugs
        similarities = cosine_similarity(query_vector, self.feature_matrix).flatten()
        
        # Get top N recommendations
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:
                recommendations.append({
                    'drug': self.drug_data.iloc[idx]['Drug'],
                    'disease': self.drug_data.iloc[idx]['Disease'] if 'Disease' in self.drug_data.columns else 'Unknown',
                    'similarity': float(similarities[idx]),
                    'age': self.drug_data.iloc[idx]['Age'] if 'Age' in self.drug_data.columns else None,
                    'gender': self.drug_data.iloc[idx]['Gender'] if 'Gender' in self.drug_data.columns else None
                })
        
        return recommendations


# Utility functions for model training and evaluation
def create_sample_disease_data(n_samples=10000):
    """Create sample disease prediction data"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(10, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48]),
        'blood_pressure': np.random.choice(['Normal', 'High', 'Low'], n_samples, p=[0.6, 0.3, 0.1]),
        'glucose_level': np.random.normal(100, 20, n_samples),
        'heart_rate': np.random.normal(75, 10, n_samples),
        'cholesterol': np.random.choice(['Normal', 'High'], n_samples, p=[0.7, 0.3]),
        'bmi': np.random.normal(25, 4, n_samples),
        'family_history': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'smoking': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'alcohol': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'exercise': np.random.choice(['Regular', 'Occasional', 'None'], n_samples, p=[0.4, 0.4, 0.2])
    }
    
    # Generate symptoms based on conditions
    symptoms_list = []
    for i in range(n_samples):
        age = data['age'][i]
        glucose = data['glucose_level'][i]
        bp = data['blood_pressure'][i]
        
        symptoms = []
        if glucose > 140:
            symptoms.extend(['Increased thirst', 'Frequent urination', 'Fatigue'])
        if bp == 'High':
            symptoms.extend(['Headache', 'Dizziness', 'Chest pain'])
        if age < 30:
            symptoms.extend(['Acne', 'Skin issues'])
        if age > 50:
            symptoms.extend(['Joint pain', 'Fatigue'])
        
        # Add some random symptoms
        all_symptoms = ['Fever', 'Cough', 'Headache', 'Fatigue', 'Nausea', 
                       'Rash', 'Itching', 'Pain', 'Swelling', 'Dizziness',
                       'Shortness of breath', 'Chest pain', 'Abdominal pain']
        num_symptoms = np.random.randint(1, 5)
        random_symptoms = np.random.choice(all_symptoms, num_symptoms, replace=False)
        symptoms.extend(random_symptoms)
        
        symptoms_list.append(','.join(set(symptoms)))  # Remove duplicates
    
    data['symptoms'] = symptoms_list
    
    # Generate disease labels based on features
    diseases = []
    for i in range(n_samples):
        age = data['age'][i]
        glucose = data['glucose_level'][i]
        bp = data['blood_pressure'][i]
        symptoms = data['symptoms'][i]
        
        if glucose > 140 or 'Increased thirst' in symptoms:
            diseases.append('Diabetes')
        elif bp == 'High' or 'Chest pain' in symptoms:
            diseases.append('Hypertension')
        elif 'Acne' in symptoms or 'Rash' in symptoms:
            diseases.append('Acne')
        elif 'Itching' in symptoms or 'Allergy' in symptoms:
            diseases.append('Allergy')
        else:
            diseases.append('General')
    
    data['disease'] = diseases
    
    return pd.DataFrame(data)


def create_sample_interaction_data(n_users=1000, n_items=500):
    """Create sample interaction data for collaborative filtering"""
    np.random.seed(42)
    
    # Generate user-item interactions
    interactions = []
    
    for user_id in range(n_users):
        # Each user interacts with 10-50 items
        n_interactions = np.random.randint(10, 51)
        items = np.random.choice(n_items, n_interactions, replace=False)
        
        for item_id in items:
            # Generate rating (1-5 stars) with some patterns
            base_rating = np.random.normal(3.5, 1.0)
            
            # Add some user-specific bias
            user_bias = np.random.normal(0, 0.5)
            
            # Add some item-specific bias
            item_bias = np.random.normal(0, 0.3)
            
            # Calculate final rating
            rating = max(1, min(5, base_rating + user_bias + item_bias))
            
            interactions.append({
                'user_id': f'user_{user_id}',
                'item_id': f'item_{item_id}',
                'rating': rating,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
            })
    
    return pd.DataFrame(interactions)


def train_disease_model():
    """Train and evaluate disease prediction model"""
    print("Training Disease Prediction Model...")
    
    # Create sample data
    disease_data = create_sample_disease_data(5000)
    
    # Initialize and train model
    model = DiseasePredictionModel(model_type='random_forest')
    prepared_data = model.prepare_data(disease_data)
    
    # Prepare features and target
    X = prepared_data[model.feature_columns]
    y = prepared_data['disease_encoded']
    
    # Train model
    trained_model, accuracy = model.train(X, y, tune_hyperparameters=True)
    
    # Get feature importance
    feature_importance = model.get_feature_importance(top_n=10)
    print("\nTop 10 Important Features:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    # Test prediction
    test_features = {
        'age': 35,
        'gender': 'Male',
        'blood_pressure': 'High',
        'glucose_level': 150,
        'heart_rate': 85,
        'cholesterol': 'High',
        'bmi': 28,
        'family_history': 'Yes',
        'smoking': 'No',
        'alcohol': 'Occasional',
        'exercise': 'Regular',
        'symptoms': 'Headache,Fatigue,Increased thirst'
    }
    
    prediction, probabilities = model.predict(test_features)
    print(f"\nTest Prediction: {prediction}")
    if probabilities:
        print("Probabilities:")
        for disease, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {disease}: {prob:.2%}")
    
    return model


def train_recommendation_model():
    """Train and evaluate recommendation model"""
    print("\nTraining Recommendation Model...")
    
    # Create sample drug data
    drug_data = pd.DataFrame({
        'Drug': [f'Drug_{i}' for i in range(100)],
        'Disease': np.random.choice(['Acne', 'Allergy', 'Diabetes', 'Hypertension'], 100),
        'Drug_Form': np.random.choice(['Tablet', 'Capsule', 'Gel', 'Cream'], 100),
        'Active_Ingredient': np.random.choice(['Ingredient_A', 'Ingredient_B', 'Ingredient_C', 'Ingredient_D'], 100),
        'Age': np.random.randint(15, 70, 100),
        'Gender': np.random.choice(['Male', 'Female'], 100)
    })
    
    # Create sample interactions
    interactions_data = create_sample_interaction_data(500, 100)
    
    # Initialize and train hybrid recommender
    hybrid_model = HybridRecommender(drug_data)
    hybrid_model.build_content_based_model()
    hybrid_model.build_popularity_model()
    
    # Initialize and train NCF model
    n_users = interactions_data['user_id'].nunique()
    n_items = interactions_data['item_id'].nunique()
    
    ncf_model = NeuralCollaborativeFiltering(n_users, n_items, n_factors=32)
    train_df, test_df = ncf_model.prepare_data(interactions_data)
    
    # Train NCF model
    history = ncf_model.train(train_df, test_df, epochs=10, batch_size=128)
    
    # Evaluate NCF model
    evaluation = ncf_model.evaluate(test_df)
    print("NCF Model Evaluation:")
    for metric, value in evaluation.items():
        print(f"  {metric}: {value:.4f}")
    
    # Set collaborative model in hybrid recommender
    hybrid_model.collab_model = ncf_model
    
    # Test hybrid recommendations
    test_user_data = {
        'user_id': 'user_0',
        'age': 30,
        'gender': 'Male',
        'history': ['Drug_1', 'Drug_5'],
        'symptoms': ['Acne', 'Itching']
    }
    
    recommendations = hybrid_model.hybrid_recommendations(
        test_user_data, 
        top_n=5,
        include_content=True,
        include_collab=True,
        include_popularity=True
    )
    
    print("\nHybrid Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['drug_name']} (Score: {rec['hybrid_score']:.3f})")
        print(f"   Explanation: {rec['explanation']}")
    
    return hybrid_model, ncf_model


def main():
    """Main function to train all models"""
    print("=" * 60)
    print("HEALTHCARE ML MODELS TRAINING")
    print("=" * 60)
    
    try:
        # Train disease prediction model
        disease_model = train_disease_model()
        
        # Train recommendation models
        hybrid_model, ncf_model = train_recommendation_model()
        
        # Save models
        disease_model.save_model('models/disease_model.pkl')
        ncf_model.save_model('models/ncf_model.pkl')
        hybrid_model.save_model('models/hybrid_model.pkl')
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Models saved to:")
        print("  - models/disease_model.pkl")
        print("  - models/ncf_model.pkl")
        print("  - models/hybrid_model.pkl")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()