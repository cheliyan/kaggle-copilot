"""
Competition Reader Agent - Powered by Google ADK + Gemini
Analyzes Kaggle competition requirements using natural language understanding.
"""

import os
import pandas as pd
from typing import Dict, Any
import warnings

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

warnings.filterwarnings("ignore")


# Define tool for analyzing data files
def inspect_competition_data(filepath: str, target_column: str) -> str:
    """
    Inspects the competition training data and returns comprehensive information.
    
    Args:
        filepath: Path to the training CSV file
        target_column: Name of the target variable
        
    Returns:
        Formatted string with data information
    """
    try:
        df = pd.read_csv(filepath)
        
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "target": target_column,
            "target_unique": int(df[target_column].nunique()) if target_column in df.columns else "N/A",
            "missing_pct": {col: f"{(df[col].isnull().sum() / len(df) * 100):.1f}%" 
                           for col in df.columns if df[col].isnull().sum() > 0},
            "numeric_cols": list(df.select_dtypes(include=['number']).columns),
            "categorical_cols": list(df.select_dtypes(include=['object']).columns),
        }
        
        result = f"""
Dataset Information:
- Shape: {info['shape'][0]} rows × {info['shape'][1]} columns
- Target Variable: {info['target']} ({info['target_unique']} unique values)
- Numeric Features: {len(info['numeric_cols'])} ({', '.join(info['numeric_cols'][:5])})
- Categorical Features: {len(info['categorical_cols'])} ({', '.join(info['categorical_cols'][:5])})
- Missing Data: {len(info['missing_pct'])} columns with missing values
"""
        if info['missing_pct']:
            result += "  Top missing: " + ", ".join([f"{k} ({v})" for k, v in list(info['missing_pct'].items())[:3]])
        
        return result
    except Exception as e:
        return f"Error inspecting data: {str(e)}"


class CompetitionReaderAgent:
    """
    Agent responsible for understanding competition requirements using Google ADK + Gemini.
    
    This agent uses:
    - Google ADK LlmAgent for agent orchestration
    - Gemini 2.0 Flash for natural language understanding
    - Tool calling for data inspection
    """
    
    def __init__(self, api_key: str = None):
        self.name = "CompetitionReader"
        
        # Get API key
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
        
        # Configure retry options for robustness
        retry_config = types.HttpRetryOptions(
            attempts=5,
            exp_base=7,
            initial_delay=1,
            http_status_codes=[429, 500, 503, 504],
        )
        
        if api_key and api_key != 'your_gemini_api_key_here':
            try:
                # Create ADK LlmAgent with Gemini model
                self.agent = LlmAgent(
                    model=Gemini(
                        model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp'),
                        retry_options=retry_config
                    ),
                    name="competition_reader_agent",
                    description="Kaggle competition analysis agent that extracts requirements and generates strategy",
                    instruction="""
You are an expert Kaggle competition analyst. When given competition information and data details:

1. **Determine Problem Type**: Classification (binary/multiclass), Regression, Time Series, etc.
2. **Identify Evaluation Metric**: Accuracy, AUC, RMSE, MAE, F1, etc.
3. **Analyze Data Characteristics**: 
   - Number of features and samples
   - Missing data patterns
   - Feature types (numeric vs categorical)
   - Class imbalance (for classification)
4. **Generate Strategy**: 
   - Recommended preprocessing steps
   - Feature engineering suggestions
   - Model recommendations
   - Special considerations

Use the inspect_competition_data tool to analyze the training data.
Provide clear, actionable insights for a data scientist.
                    """,
                    tools=[inspect_competition_data],
                )
                self.adk_enabled = True
                print(f"✅ {self.name} Agent created successfully!")
                print(f"   Model: {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')}")
                print(f"   Tools: inspect_competition_data()")
            except Exception as e:
                print(f"⚠ {self.name} Agent: ADK unavailable ({str(e)}), using fallback mode")
                self.adk_enabled = False
        else:
            print(f"⚠ {self.name} Agent: No API key, using fallback mode")
            self.adk_enabled = False
    
    def analyze(self, competition_name: str, train_file: str, target_column: str) -> Dict[str, Any]:
        """
        Analyzes competition using ADK agent with Gemini to extract requirements and strategy.
        
        Args:
            competition_name: Name of the competition
            train_file: Path to training data
            target_column: Target variable name
            
        Returns:
            Dictionary with competition strategy
        """
        if self.adk_enabled:
            # Use ADK agent with Gemini for intelligent analysis
            strategy = self._analyze_with_adk(competition_name, train_file, target_column)
        else:
            # Fallback: Rule-based analysis
            df = pd.read_csv(train_file)
            strategy = self._analyze_fallback(competition_name, df, target_column)
        
        return strategy
    
    def _analyze_with_adk(self, competition_name: str, train_file: str, target_column: str) -> Dict[str, Any]:
        """
        Uses ADK agent with Gemini to analyze competition and generate strategy.
        
        Args:
            competition_name: Name of competition
            train_file: Path to training data file
            target_column: Target variable name
            
        Returns:
            Strategy dictionary
        """
        prompt = f"""
Analyze the Kaggle competition: "{competition_name}"

Use the inspect_competition_data tool to examine the training data at: {train_file}
Target variable: {target_column}

Provide analysis in this format:
1. Problem Type (classification/regression/etc)
2. Evaluation Metric (accuracy/RMSE/AUC/etc)
3. Data Characteristics
4. Recommended Strategy

Be specific and actionable.
"""
        
        try:
            # Execute agent with the prompt
            response = self.agent.send_message(prompt)
            
            # Parse response into structured format
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract strategy from Gemini's response
            strategy = self._parse_strategy_response(response_text, pd.read_csv(train_file), target_column, competition_name)
            
            print(f"\n✅ Competition Analysis Complete")
            print(f"   Problem Type: {strategy['problem_type']}")
            print(f"   Metric: {strategy['metric']}")
            print(f"   Strategy Generated: {len(strategy['strategy'].split())} words")
            
            return strategy
            
        except Exception as e:
            print(f"⚠ ADK agent error: {str(e)}, falling back to rule-based analysis")
            df = pd.read_csv(train_file)
            return self._analyze_fallback(competition_name, df, target_column)
    
    def _parse_strategy_response(self, response_text: str, df: pd.DataFrame, target_column: str, competition_name: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured strategy format."""
        
        # Determine problem type from data
        unique_values = df[target_column].nunique() if target_column in df.columns else 10
        if unique_values == 2:
            problem_type = "binary_classification"
            metric = "accuracy"
        elif unique_values < 20:
            problem_type = "multiclass_classification"
            metric = "accuracy"
        else:
            problem_type = "regression"
            metric = "rmse"
        
        # Find columns to drop
        drop_columns = []
        for col in df.columns:
            if col.lower() in ['id', 'passengerid', 'name', 'ticket', 'cabin']:
                if col not in [target_column]:
                    drop_columns.append(col)
        
        return {
            "competition_name": competition_name,
            "problem_type": problem_type,
            "metric": metric,
            "target_column": target_column,
            "drop_columns": drop_columns,
            "strategy": response_text,
            "adk_powered": True
        }
    
    def _analyze_fallback(self, competition_name: str, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Fallback rule-based analysis when ADK is unavailable.
        
        Args:
            competition_name: Competition name
            df: DataFrame
            target_column: Target column name
            
        Returns:
            Strategy dictionary
        """
        # Infer problem type from target
        if target_column in df.columns:
            unique_values = df[target_column].nunique()
            if unique_values == 2:
                problem_type = "binary_classification"
                metric = "accuracy"
            elif unique_values < 20:
                problem_type = "multiclass_classification"
                metric = "accuracy"
            else:
                problem_type = "regression"
                metric = "rmse"
        else:
            problem_type = "classification"
            metric = "accuracy"
        
        # Determine columns to drop (non-predictive identifiers)
        drop_columns = []
        for col in df.columns:
            if col.lower() in ['id', 'passengerid', 'name', 'ticket', 'cabin']:
                if col not in [target_column]:
                    drop_columns.append(col)
        
        # Generate strategy
        strategy_text = f"""
Competition: {competition_name}
Problem Type: {problem_type}
Evaluation Metric: {metric}

Rule-Based Analysis (ADK unavailable):
- Dataset: {df.shape[0]} rows, {df.shape[1]} columns
- Target: {target_column} ({unique_values} unique values)
- Features: {len(df.columns) - 1} total
- Missing data: {df.isnull().sum().sum()} cells

Recommended approach:
1. Handle missing data (imputation or removal)
2. Encode categorical variables
3. Try ensemble models (Random Forest, XGBoost)
4. Use cross-validation for evaluation
"""
        
        return {
            "competition_name": competition_name,
            "problem_type": problem_type,
            "metric": metric,
            "target_column": target_column,
            "drop_columns": drop_columns,
            "strategy": strategy_text,
            "adk_powered": False
        }
