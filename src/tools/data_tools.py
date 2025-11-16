"""
Data Analysis Tools for Agent System
Provides functions for exploratory data analysis, visualization, and data profiling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')


def analyze_dataframe(filepath: str) -> Dict[str, Any]:
    """
    Analyzes a CSV file and returns comprehensive statistics.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dictionary with shape, dtypes, missing data, and statistics
    """
    df = pd.read_csv(filepath)
    
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
        "numeric_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        "categorical_unique": {col: int(df[col].nunique()) for col in df.select_dtypes(include=['object']).columns}
    }


def detect_outliers(filepath: str, column: str) -> Dict[str, Any]:
    """
    Detects outliers in a numeric column using IQR method.
    
    Args:
        filepath: Path to CSV file
        column: Column name to analyze
        
    Returns:
        Dictionary with outlier information
    """
    df = pd.read_csv(filepath)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        "column": column,
        "outlier_count": len(outliers),
        "outlier_percent": len(outliers) / len(df) * 100,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_indices": outliers.index.tolist()
    }


def correlation_analysis(filepath: str, target_column: str = None) -> Dict[str, Any]:
    """
    Performs correlation analysis on numeric columns.
    
    Args:
        filepath: Path to CSV file
        target_column: Optional target column to focus on
        
    Returns:
        Dictionary with correlation information
    """
    df = pd.read_csv(filepath)
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        return {"error": "No numeric columns found"}
    
    corr_matrix = numeric_df.corr()
    
    result = {
        "correlation_matrix": corr_matrix.to_dict()
    }
    
    if target_column and target_column in corr_matrix.columns:
        target_corr = corr_matrix[target_column].sort_values(ascending=False)
        result["target_correlations"] = target_corr.to_dict()
        result["top_correlations"] = target_corr.head(10).to_dict()
    
    return result


def create_visualizations(filepath: str, output_dir: str, target_column: str = None) -> List[str]:
    """
    Creates comprehensive visualizations for EDA.
    
    Args:
        filepath: Path to CSV file
        output_dir: Directory to save visualizations
        target_column: Optional target column for focused analysis
        
    Returns:
        List of created visualization file paths
    """
    df = pd.read_csv(filepath)
    created_files = []
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Missing data heatmap
    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Data Pattern')
        filepath_out = f"{output_dir}/missing_data_heatmap.png"
        plt.savefig(filepath_out, bbox_inches='tight', dpi=100)
        plt.close()
        created_files.append(filepath_out)
    
    # 2. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Heatmap')
        filepath_out = f"{output_dir}/correlation_heatmap.png"
        plt.savefig(filepath_out, bbox_inches='tight', dpi=100)
        plt.close()
        created_files.append(filepath_out)
    
    # 3. Distribution plots for numeric columns
    numeric_cols = numeric_df.columns.tolist()
    if len(numeric_cols) > 0:
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        # Hide extra subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath_out = f"{output_dir}/distributions.png"
        plt.savefig(filepath_out, bbox_inches='tight', dpi=100)
        plt.close()
        created_files.append(filepath_out)
    
    # 4. Target variable analysis (if provided)
    if target_column and target_column in df.columns:
        plt.figure(figsize=(8, 6))
        if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() < 10:
            df[target_column].value_counts().plot(kind='bar', edgecolor='black')
            plt.title(f'Distribution of {target_column}')
            plt.ylabel('Count')
        else:
            df[target_column].hist(bins=30, edgecolor='black')
            plt.title(f'Distribution of {target_column}')
            plt.ylabel('Frequency')
        plt.xlabel(target_column)
        filepath_out = f"{output_dir}/target_distribution.png"
        plt.savefig(filepath_out, bbox_inches='tight', dpi=100)
        plt.close()
        created_files.append(filepath_out)
    
    return created_files


def generate_data_summary(filepath: str) -> str:
    """
    Generates a human-readable summary of the dataset.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        String summary of the dataset
    """
    df = pd.read_csv(filepath)
    analysis = analyze_dataframe(filepath)
    
    summary = f"""
Dataset Summary:
- Shape: {analysis['shape'][0]} rows × {analysis['shape'][1]} columns
- Columns: {', '.join(analysis['columns'])}

Missing Data:
"""
    
    missing_cols = [col for col, pct in analysis['missing_percent'].items() if pct > 0]
    if missing_cols:
        for col in missing_cols:
            pct = analysis['missing_percent'][col]
            summary += f"  - {col}: {pct:.1f}%\n"
    else:
        summary += "  - No missing data\n"
    
    summary += "\nData Types:\n"
    for col, dtype in analysis['dtypes'].items():
        summary += f"  - {col}: {dtype}\n"
    
    if analysis['categorical_unique']:
        summary += "\nCategorical Features:\n"
        for col, unique_count in analysis['categorical_unique'].items():
            summary += f"  - {col}: {unique_count} unique values\n"
    
    return summary
