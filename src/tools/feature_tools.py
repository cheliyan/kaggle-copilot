"""
Feature Engineering Tools for Agent System
Provides functions for creating, testing, and selecting features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any, Tuple


def create_feature(filepath: str, feature_name: str, expression: str, output_path: str = None) -> Dict[str, Any]:
    """
    Creates a new feature using a pandas expression.
    
    Args:
        filepath: Path to CSV file
        feature_name: Name for the new feature
        expression: Pandas eval expression
        output_path: Optional path to save updated dataframe
        
    Returns:
        Dictionary with status and feature info
    """
    df = pd.read_csv(filepath)
    
    try:
        df[feature_name] = df.eval(expression)
        
        if output_path:
            df.to_csv(output_path, index=False)
        
        return {
            "status": "success",
            "feature_name": feature_name,
            "expression": expression,
            "sample_values": df[feature_name].head().tolist(),
            "dtype": str(df[feature_name].dtype)
        }
    except Exception as e:
        return {
            "status": "error",
            "feature_name": feature_name,
            "error": str(e)
        }


def extract_title_from_name(name: str) -> str:
    """
    Extracts title from a name string (e.g., 'Mr.', 'Mrs.', etc.)
    
    Args:
        name: Full name string
        
    Returns:
        Extracted title or 'Unknown'
    """
    if pd.isna(name):
        return 'Unknown'
    
    try:
        title = name.split(',')[1].split('.')[0].strip()
        # Group rare titles
        if title in ['Mr', 'Miss', 'Mrs', 'Master']:
            return title
        elif title in ['Dr', 'Rev', 'Col', 'Major', 'Capt']:
            return 'Officer'
        elif title in ['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona']:
            return 'Royalty'
        else:
            return 'Rare'
    except:
        return 'Unknown'


def engineer_features_titanic(filepath: str, output_path: str) -> Dict[str, Any]:
    """
    Creates Titanic-specific engineered features.
    
    Args:
        filepath: Path to CSV file
        output_path: Path to save updated dataframe
        
    Returns:
        Dictionary with created features
    """
    df = pd.read_csv(filepath)
    created_features = []
    
    # Extract Title from Name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].apply(extract_title_from_name)
        created_features.append('Title')
    
    # Family Size
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        created_features.append('FamilySize')
        
        # Is Alone
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        created_features.append('IsAlone')
    
    # Age bins
    if 'Age' in df.columns:
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                               labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        created_features.append('AgeBin')
        
        # Fill missing ages with median by Title
        if 'Title' in df.columns:
            df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Fare per person
    if 'Fare' in df.columns and 'FamilySize' in df.columns:
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
        df['FarePerPerson'] = df['FarePerPerson'].fillna(0)
        created_features.append('FarePerPerson')
    
    # Cabin deck
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Unknown')
        created_features.append('Deck')
    
    # Embarked - fill missing with mode
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fare - fill missing with median
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    df.to_csv(output_path, index=False)
    
    return {
        "status": "success",
        "created_features": created_features,
        "total_features": len(created_features),
        "output_path": output_path
    }


def test_feature_importance(filepath: str, target_column: str, 
                           feature_columns: List[str] = None,
                           problem_type: str = 'classification') -> Dict[str, Any]:
    """
    Tests feature importance using Random Forest.
    
    Args:
        filepath: Path to CSV file
        target_column: Target variable column name
        feature_columns: List of feature columns (if None, uses all except target)
        problem_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with feature importance scores
    """
    df = pd.read_csv(filepath)
    
    # Separate features and target
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found"}
    
    y = df[target_column]
    
    if feature_columns is None:
        X = df.drop(columns=[target_column])
    else:
        X = df[feature_columns]
    
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
    
    # Train model
    if problem_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    
    model.fit(X, y)
    
    # Get feature importance
    importance_dict = dict(zip(X.columns, model.feature_importances_))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "status": "success",
        "feature_importance": dict(sorted_importance),
        "top_10_features": dict(sorted_importance[:10]),
        "model_type": problem_type
    }


def create_interaction_features(filepath: str, col1: str, col2: str, 
                                output_path: str) -> Dict[str, Any]:
    """
    Creates interaction features between two columns.
    
    Args:
        filepath: Path to CSV file
        col1: First column name
        col2: Second column name
        output_path: Path to save updated dataframe
        
    Returns:
        Dictionary with created interaction features
    """
    df = pd.read_csv(filepath)
    
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"Column not found: {col1} or {col2}"}
    
    feature_name = f"{col1}_{col2}_interaction"
    
    # Handle numeric columns
    if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
        df[feature_name] = df[col1] * df[col2]
    else:
        # Concatenate for categorical
        df[feature_name] = df[col1].astype(str) + "_" + df[col2].astype(str)
    
    df.to_csv(output_path, index=False)
    
    return {
        "status": "success",
        "feature_name": feature_name,
        "sample_values": df[feature_name].head().tolist()
    }
