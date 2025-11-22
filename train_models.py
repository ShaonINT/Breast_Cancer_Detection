"""
Breast Cancer Detection - Model Training and Comparison Script
Compares multiple machine learning models to find the best performer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Additional metrics from confusion matrix
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate (Critical for medical diagnosis)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            
            # Composite score: Weighted combination of F1, Accuracy, and low False Negative Rate
            # For medical diagnosis, minimizing false negatives is critical
            composite_score = (0.4 * f1) + (0.3 * accuracy) + (0.3 * (1 - fnr))
            
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
                'confusion_matrix': cm,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'specificity': specificity,
                'fnr': fnr,
                'fpr': fpr,
                'composite_score': composite_score,
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'X_train_format': X_train,
                'uses_scaled': model_name in ['SVM', 'Neural Network']
            }
            
            # Print results
            print(f"Accuracy:     {accuracy:.4f}")
            print(f"Precision:    {precision:.4f}")
            print(f"Recall:       {recall:.4f}")
            print(f"F1-Score:     {f1:.4f}")
            print(f"CV Score:     {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Benign  Malignant")
            print(f"Actual Benign    {tn:4d}      {fp:4d}")
            print(f"      Malignant  {fn:4d}      {tp:4d}")
            print(f"\nFalse Negative Rate (FNR): {fnr:.4f} (Lower is better)")
            print(f"Specificity (TNR):         {specificity:.4f}")
            print(f"Composite Score:           {composite_score:.4f}")
            
            # Update best model based on composite score (considers F1, Accuracy, and FNR)
            if composite_score > best_score:
                best_score = composite_score
                self.best_model = model
                self.best_model_name = model_name
        
        print("\n" + "=" * 80)
        if self.best_model_name:
            best_metrics = self.results[self.best_model_name]
            print(f"\nBest Model: {self.best_model_name}")
            print(f"  F1-Score:       {best_metrics['f1_score']:.4f}")
            print(f"  Accuracy:       {best_metrics['accuracy']:.4f}")
            print(f"  False Neg Rate: {best_metrics['fnr']:.4f}")
            print(f"  Composite Score: {best_metrics['composite_score']:.4f}")
        print()
        
        return self
    
    def print_comparison_table(self):
        """Print a comparison table of all models."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'FNR':<12} {'Composite':<12} {'CV Score':<12}")
        print("-" * 80)
        
        # Sort by composite score (considers F1, Accuracy, and False Negative Rate)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} "
                  f"{metrics['fnr']:<12.4f} {metrics['composite_score']:<12.4f} {metrics['cv_mean']:<12.4f}")
        
        print("=" * 80)
        print("\nNote: Composite Score = 0.4*F1 + 0.3*Accuracy + 0.3*(1-FNR)")
        print("      Lower FNR (False Negative Rate) is critical for medical diagnosis")
        print()
    
    def print_best_model_details(self):
        """Print detailed results for the best model."""
        if self.best_model_name:
            print("\n" + "=" * 80)
            print(f"DETAILED RESULTS FOR BEST MODEL: {self.best_model_name}")
            print("=" * 80)
            
            metrics = self.results[self.best_model_name]
            y_pred = metrics['y_pred']
            cm = metrics['confusion_matrix']
            
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['Benign (B)', 'Malignant (M)']))
            
            print("\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Benign  Malignant")
            print(f"Actual Benign    {metrics['tn']:4d}      {metrics['fp']:4d}")
            print(f"      Malignant  {metrics['fn']:4d}      {metrics['tp']:4d}")
            
            print(f"\nKey Metrics:")
            print(f"  True Negatives (TN):  {metrics['tn']:4d}  (Correctly identified benign)")
            print(f"  False Positives (FP): {metrics['fp']:4d}  (Benign classified as malignant)")
            print(f"  False Negatives (FN): {metrics['fn']:4d}  (Malignant missed - CRITICAL)")
            print(f"  True Positives (TP):  {metrics['tp']:4d}  (Correctly identified malignant)")
            
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:       {metrics['accuracy']:.4f}")
            print(f"  Precision:      {metrics['precision']:.4f}")
            print(f"  Recall:         {metrics['recall']:.4f}  (Sensitivity - ability to catch malignant)")
            print(f"  F1-Score:       {metrics['f1_score']:.4f}")
            print(f"  Specificity:    {metrics['specificity']:.4f}  (Ability to identify benign)")
            print(f"  False Neg Rate: {metrics['fnr']:.4f}  (Lower is better for medical diagnosis)")
            print(f"  False Pos Rate: {metrics['fpr']:.4f}")
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
    
    def tune_best_model(self):
        """Perform hyperparameter tuning on the best model to improve performance."""
        if not self.best_model_name:
            print("No best model selected. Skipping hyperparameter tuning.")
            return self
        
        print("\n" + "=" * 80)
        print(f"HYPERPARAMETER TUNING FOR: {self.best_model_name}")
        print("=" * 80)
        print("\nPerforming grid search to find optimal hyperparameters...")
        print("This may take several minutes...\n")
        
        best_metrics = self.results[self.best_model_name]
        uses_scaled = best_metrics['uses_scaled']
        X_train_tune = self.X_train_scaled if uses_scaled else self.X_train
        
        # Handle LightGBM separately
        if self.best_model_name == 'LightGBM':
            X_train_tune = self.X_train.values
        
        # Define parameter grids for each model type
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'CatBoost': {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100), (200, 100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [300, 500, 1000]
            }
        }
        
        # Get base model and parameter grid
        param_grid = param_grids.get(self.best_model_name)
        if not param_grid:
            print(f"No parameter grid defined for {self.best_model_name}. Skipping tuning.")
            return self
        
        # Create base model with same configuration
        if self.best_model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif self.best_model_name == 'XGBoost':
            base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        elif self.best_model_name == 'CatBoost':
            base_model = cb.CatBoostClassifier(random_state=42, verbose=False)
        elif self.best_model_name == 'LightGBM':
            base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        elif self.best_model_name == 'SVM':
            base_model = SVC(probability=True, random_state=42)
        elif self.best_model_name == 'Neural Network':
            base_model = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
        
        # Perform GridSearchCV with F1-score as primary metric (important for medical diagnosis)
        # Use 3-fold CV to speed up the process while still getting good results
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_tune, self.y_train)
        
        # Get best tuned model
        tuned_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\nBest Parameters Found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV F1-Score: {grid_search.best_score_:.4f}")
        
        # Evaluate tuned model on test set
        X_test_tune = self.X_test_scaled if uses_scaled else self.X_test
        if self.best_model_name == 'LightGBM':
            X_test_tune = self.X_test.values
        
        y_pred_tuned = tuned_model.predict(X_test_tune)
        
        # Calculate metrics for tuned model
        accuracy_tuned = accuracy_score(self.y_test, y_pred_tuned)
        precision_tuned = precision_score(self.y_test, y_pred_tuned)
        recall_tuned = recall_score(self.y_test, y_pred_tuned)
        f1_tuned = f1_score(self.y_test, y_pred_tuned)
        cm_tuned = confusion_matrix(self.y_test, y_pred_tuned)
        tn_tuned, fp_tuned, fn_tuned, tp_tuned = cm_tuned.ravel()
        
        fnr_tuned = fn_tuned / (fn_tuned + tp_tuned) if (fn_tuned + tp_tuned) > 0 else 0
        composite_tuned = (0.4 * f1_tuned) + (0.3 * accuracy_tuned) + (0.3 * (1 - fnr_tuned))
        
        # Compare with original model
        print("\n" + "-" * 80)
        print("COMPARISON: Before vs After Tuning")
        print("-" * 80)
        print(f"{'Metric':<25} {'Before':<15} {'After':<15} {'Improvement':<15}")
        print("-" * 80)
        
        metrics_before = best_metrics
        accuracy_improvement = accuracy_tuned - metrics_before['accuracy']
        f1_improvement = f1_tuned - metrics_before['f1_score']
        fnr_improvement = metrics_before['fnr'] - fnr_tuned  # Lower FNR is better
        composite_improvement = composite_tuned - metrics_before['composite_score']
        
        print(f"{'Accuracy':<25} {metrics_before['accuracy']:<15.4f} {accuracy_tuned:<15.4f} {accuracy_improvement:+.4f}")
        print(f"{'F1-Score':<25} {metrics_before['f1_score']:<15.4f} {f1_tuned:<15.4f} {f1_improvement:+.4f}")
        print(f"{'False Neg Rate':<25} {metrics_before['fnr']:<15.4f} {fnr_tuned:<15.4f} {fnr_improvement:+.4f}")
        print(f"{'Composite Score':<25} {metrics_before['composite_score']:<15.4f} {composite_tuned:<15.4f} {composite_improvement:+.4f}")
        print("-" * 80)
        
        # Decide whether to use tuned model
        if composite_tuned >= metrics_before['composite_score']:
            print(f"\n✓ Tuned model performs better or equal. Updating best model...")
            self.best_model = tuned_model
            self.results[self.best_model_name]['model'] = tuned_model
            self.results[self.best_model_name]['accuracy'] = accuracy_tuned
            self.results[self.best_model_name]['precision'] = precision_tuned
            self.results[self.best_model_name]['recall'] = recall_tuned
            self.results[self.best_model_name]['f1_score'] = f1_tuned
            self.results[self.best_model_name]['confusion_matrix'] = cm_tuned
            self.results[self.best_model_name]['tn'] = tn_tuned
            self.results[self.best_model_name]['fp'] = fp_tuned
            self.results[self.best_model_name]['fn'] = fn_tuned
            self.results[self.best_model_name]['tp'] = tp_tuned
            self.results[self.best_model_name]['fnr'] = fnr_tuned
            self.results[self.best_model_name]['composite_score'] = composite_tuned
            self.results[self.best_model_name]['best_params'] = best_params
            print(f"  Accuracy improvement: {accuracy_improvement:+.4f}")
            print(f"  F1-Score improvement: {f1_improvement:+.4f}")
            print(f"  FNR improvement: {fnr_improvement:+.4f} (Lower is better)")
        else:
            print(f"\n✗ Original model performs better. Keeping original model.")
            print(f"  Tuned model had lower composite score by {abs(composite_improvement):.4f}")
        
        print("=" * 80)
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
        self.tune_best_model()
        self.print_best_model_details()  # Print again after tuning
        self.save_best_model()
        
        print("=" * 80)
        print("MODEL COMPARISON COMPLETE!")
        print("=" * 80)
        
        return self

if __name__ == "__main__":
    # Initialize and run model comparison
    comparison = ModelComparison('Breast_cancer_dataset.csv')
    comparison.run_complete_pipeline()

