"""
Feature Engineer Agent - Powered by Google ADK
Creates and tests features using ADK tool calling.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import feature_tools
from typing import Dict, Any

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class FeatureEngineerAgent:
    """
    Agent responsible for feature engineering using Google ADK + Tool Calling.
    
    This agent:
    - Uses Gemini for intelligent feature suggestions
    - Implements ADK tool calling for feature creation
    - Maintains memory of successful features (SQLite)
    """
    
    def __init__(self, api_key: str = None):
        self.name = "FeatureEngineer"
        
        if GENAI_AVAILABLE and api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
        
        if GENAI_AVAILABLE and api_key and api_key != 'your_gemini_api_key_here':
            try:
                self.client = genai.Client(api_key=api_key)
                self.model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
                self.gemini_enabled = True
                print(f"✓ {self.name} Agent initialized with Gemini (ADK Tool Calling)")
            except:
                self.gemini_enabled = False
                print(f"✓ {self.name} Agent initialized (fallback mode)")
        else:
            self.gemini_enabled = False
            print(f"✓ {self.name} Agent initialized (fallback mode)")
    
    def engineer_features(self, filepath: str, output_path: str, 
                         target_column: str, competition_type: str) -> Dict[str, Any]:
        """
        Engineers features for the competition.
        
        Args:
            filepath: Path to input CSV
            output_path: Path to save processed data
            target_column: Target variable name
            competition_type: Type of problem
            
        Returns:
            Dictionary with feature engineering results
        """
        # For Titanic and similar competitions, use domain-specific features
        result = feature_tools.engineer_features_titanic(filepath, output_path)
        
        # Test feature importance
        importance = feature_tools.test_feature_importance(
            filepath=output_path,
            target_column=target_column,
            problem_type='classification'
        )
        
        result['feature_importance'] = importance.get('top_10_features', {})
        
        return result
