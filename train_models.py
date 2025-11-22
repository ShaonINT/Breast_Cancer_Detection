"""
Breast Cancer Detection - Model Training and Comparison Script
Compares multiple machine learning models to find the best performer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, data_path):
        """Initialize the model comparison class."""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the breast cancer dataset."""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Drop 'id' column as it's not useful for prediction
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Drop unnamed/empty columns (common in CSV files)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Normalize feature names: replace spaces with underscores
        # This fixes issues with LightGBM and other models that don't support spaces in feature names
        df.columns = [col.replace(' ', '_') for col in df.columns]
        
        # Encode target variable: M=Malignant (1), B=Benign (0)
        df['diagnosis'] = self.label_encoder.fit_transform(df['diagnosis'])
        
        # Separate features and target
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check for NaN values and fill them if any exist
        if self.X_train.isnull().sum().sum() > 0:
            print(f"Warning: Found {self.X_train.isnull().sum().sum()} NaN values in training data. Filling with mean.")
            self.X_train = self.X_train.fillna(self.X_train.mean())
        if self.X_test.isnull().sum().sum() > 0:
            print(f"Warning: Found {self.X_test.isnull().sum().sum()} NaN values in test data. Filling with mean.")
            self.X_test = self.X_test.fillna(self.X_train.mean())
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Ensure no NaN values after scaling
        if np.isnan(self.X_train_scaled).sum() > 0 or np.isnan(self.X_test_scaled).sum() > 0:
            print(f"Warning: NaN values found after scaling. Replacing with 0.")
            self.X_train_scaled = np.nan_to_num(self.X_train_scaled, nan=0.0)
            self.X_test_scaled = np.nan_to_num(self.X_test_scaled, nan=0.0)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Number of features: {self.X_train.shape[1]}")
        print(f"Class distribution - Train: {self.y_train.value_counts().to_dict()}")
        print(f"Class distribution - Test: {self.y_test.value_counts().to_dict()}")
        print()
        
        return self
    
    def initialize_models(self):
        """Initialize all models to compare."""
        print("Initializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        print()
        
        return self
    
    def train_and_evaluate_models(self):
        """Train all models and evaluate their performance."""
        print("Training and evaluating models...\n")
        print("=" * 80)
        
        best_score = 0
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 80)
            
            # Choose scaled or unscaled data based on model
            if model_name in ['SVM', 'Neural Network']:
                X_train = self.X_train_scaled
                X_test = self.X_test_scaled
            elif model_name == 'LightGBM':
                # LightGBM has issues with special characters in feature names
                # Convert to numpy arrays to avoid column name issues
                X_train = self.X_train.values
                X_test = self.X_test.values
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # Train model
            model.fit(X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Cross-validation score (use original format for consistency)
            cv_train = X_train if model_name != 'LightGBM' else self.X_train.values
            cv_scores = cross_val_score(model, cv_train, self.y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print results
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"CV Score:  {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            
            # Update best model
            if f1 > best_score:
                best_score = f1
                self.best_model = model
                self.best_model_name = model_name
        
        print("\n" + "=" * 80)
        print(f"\nBest Model: {self.best_model_name} (F1-Score: {best_score:.4f})")
        print()
        
        return self
    
    def print_comparison_table(self):
        """Print a comparison table of all models."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'CV Score':<12}")
        print("-" * 80)
        
        # Sort by F1-score
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {metrics['cv_mean']:<12.4f}")
        
        print("=" * 80)
        print()
    
    def print_best_model_details(self):
        """Print detailed results for the best model."""
        if self.best_model_name:
            print("\n" + "=" * 80)
            print(f"DETAILED RESULTS FOR BEST MODEL: {self.best_model_name}")
            print("=" * 80)
            
            metrics = self.results[self.best_model_name]
            y_pred = metrics['y_pred']
            
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['Benign (B)', 'Malignant (M)']))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            print(f"\nTrue Negatives (TN): {cm[0][0]}")
            print(f"False Positives (FP): {cm[0][1]}")
            print(f"False Negatives (FN): {cm[1][0]}")
            print(f"True Positives (TP): {cm[1][1]}")
            print("=" * 80)
            print()
    
    def save_best_model(self, model_path='best_model.pkl', scaler_path='scaler.pkl'):
        """Save the best model and scaler for deployment."""
        if self.best_model_name:
            print(f"Saving best model ({self.best_model_name})...")
            
            # Determine if we need scaler
            needs_scaler = self.best_model_name in ['SVM', 'Neural Network']
            
            joblib.dump(self.best_model, model_path)
            if needs_scaler:
                joblib.dump(self.scaler, scaler_path)
                print(f"Model saved to: {model_path}")
                print(f"Scaler saved to: {scaler_path}")
            else:
                print(f"Model saved to: {model_path}")
                print("Note: This model does not require a scaler")
            
            # Save model metadata
            metadata = {
                'model_name': self.best_model_name,
                'needs_scaler': needs_scaler,
                'features': list(self.X_train.columns),
                'results': {k: {m: v for m, v in metrics.items() if m != 'model' and m != 'y_pred' and m != 'y_pred_proba'} 
                           for k, metrics in self.results.items()}
            }
            joblib.dump(metadata, 'model_metadata.pkl')
            print(f"Metadata saved to: model_metadata.pkl")
            print()
        
        return self
    
    def run_complete_pipeline(self):
        """Run the complete model comparison pipeline."""
        print("=" * 80)
        print("BREAST CANCER DETECTION - MODEL COMPARISON")
        print("=" * 80)
        print()
        
        self.load_and_preprocess_data()
        self.initialize_models()
        self.train_and_evaluate_models()
        self.print_comparison_table()
        self.print_best_model_details()
        self.save_best_model()
        
        print("=" * 80)
        print("MODEL COMPARISON COMPLETE!")
        print("=" * 80)
        
        return self

if __name__ == "__main__":
    # Initialize and run model comparison
    comparison = ModelComparison('Breast_cancer_dataset.csv')
    comparison.run_complete_pipeline()

