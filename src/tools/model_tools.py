"""
Model Training and Evaluation Tools for Agent System
Provides functions for training models, cross-validation, and generating predictions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Any
import joblib

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


def prepare_data(filepath: str, target_column: str, 
                 drop_columns: List[str] = None) -> tuple:
    """
    Prepares data for model training.
    
    Args:
        filepath: Path to CSV file
        target_column: Target variable column name
        drop_columns: Columns to drop
        
    Returns:
        Tuple of (X, y, feature_names, label_encoders)
    """
    df = pd.read_csv(filepath)
    
    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names, label_encoders


def train_model(filepath: str, target_column: str, model_type: str = 'random_forest',
                drop_columns: List[str] = None, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Trains a model and returns performance metrics.
    
    Args:
        filepath: Path to CSV file
        target_column: Target variable column name
        model_type: Type of model ('logistic', 'random_forest', 'xgboost', 'lightgbm')
        drop_columns: Columns to drop
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with model performance metrics
    """
    X, y, feature_names, label_encoders = prepare_data(filepath, target_column, drop_columns)
    
    # Select model
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    }
    
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBClassifier(n_estimators=100, random_state=42, max_depth=6, 
                                          learning_rate=0.1, eval_metric='logloss')
    
    if LGBM_AVAILABLE:
        models['lightgbm'] = LGBMClassifier(n_estimators=100, random_state=42, max_depth=6,
                                             learning_rate=0.1, verbose=-1)
    
    if model_type not in models:
        return {"error": f"Model type '{model_type}' not available"}
    
    model = models[model_type]
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
    
    # Train on full data
    model.fit(X, y)
    train_pred = model.predict(X)
    train_accuracy = accuracy_score(y, train_pred)
    
    # Feature importance (if available)
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
    
    return {
        "model_type": model_type,
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "train_accuracy": float(train_accuracy),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "model_object": model
    }


def train_all_models(filepath: str, target_column: str, 
                     drop_columns: List[str] = None) -> Dict[str, Any]:
    """
    Trains multiple models and compares performance.
    
    Args:
        filepath: Path to CSV file
        target_column: Target variable column name
        drop_columns: Columns to drop
        
    Returns:
        Dictionary with all model results
    """
    model_types = ['logistic', 'random_forest']
    
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    if LGBM_AVAILABLE:
        model_types.append('lightgbm')
    
    results = {}
    all_scores = []
    
    for model_type in model_types:
        print(f"Training {model_type}...")
        result = train_model(filepath, target_column, model_type, drop_columns)
        results[model_type] = result
        all_scores.append({
            "model": model_type,
            "cv_mean": result["cv_mean_accuracy"],
            "cv_std": result["cv_std_accuracy"]
        })
    
    # Find best model
    best_model = max(all_scores, key=lambda x: x['cv_mean'])
    
    return {
        "all_results": results,
        "comparison": all_scores,
        "best_model": best_model['model'],
        "best_cv_score": best_model['cv_mean']
    }


def generate_predictions(model_filepath: str, test_data_filepath: str, 
                        target_column: str, drop_columns: List[str] = None,
                        output_filepath: str = None) -> Dict[str, Any]:
    """
    Generates predictions using a trained model.
    
    Args:
        model_filepath: Path to saved model pickle
        test_data_filepath: Path to test CSV file
        target_column: Target variable column name
        drop_columns: Columns to drop
        output_filepath: Path to save predictions
        
    Returns:
        Dictionary with predictions
    """
    # Load model
    model_results = joblib.load(model_filepath)
    model = model_results['model_object']
    label_encoders = model_results['label_encoders']
    feature_names = model_results['feature_names']
    
    # Load test data
    df = pd.read_csv(test_data_filepath)
    
    # Keep ID column if exists
    id_col = None
    if 'PassengerId' in df.columns:
        id_col = df['PassengerId']
    
    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')
    
    # Handle categorical variables using saved encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    
    # Ensure same features as training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_names]
    
    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    
    # Generate predictions
    predictions = model.predict(X)
    
    # Save predictions
    if output_filepath:
        result_df = pd.DataFrame()
        if id_col is not None:
            result_df['PassengerId'] = id_col
        result_df[target_column] = predictions
        result_df.to_csv(output_filepath, index=False)
    
    return {
        "predictions": predictions.tolist(),
        "n_predictions": len(predictions),
        "output_file": output_filepath
    }


def save_model(model_results: Dict[str, Any], filepath: str) -> str:
    """
    Saves model and related objects to file.
    
    Args:
        model_results: Dictionary containing model and encoders
        filepath: Path to save model
        
    Returns:
        Path to saved model
    """
    joblib.dump(model_results, filepath)
    return filepath
