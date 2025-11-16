"""
Model Selector Agent - Powered by Google ADK
Trains and selects best models using A2A protocol.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import model_tools
from typing import Dict, Any


class ModelSelectorAgent:
    """
    Agent responsible for model training and selection.
    
    Uses ADK + A2A Protocol to:
    - Train multiple baseline models
    - Compare performance
    - Request better features from Feature Engineer (A2A)
    - Select best model
    """
    
    def __init__(self):
        self.name = "ModelSelector"
        print(f"✓ {self.name} Agent initialized (ADK + A2A Protocol)")
    
    def train_and_select(self, filepath: str, target_column: str,
                         drop_columns: list = None) -> Dict[str, Any]:
        """
        Trains multiple models and selects the best one.
        
        Args:
            filepath: Path to processed data CSV
            target_column: Target variable name
            drop_columns: Columns to exclude from training
            
        Returns:
            Dictionary with model results
        """
        # Train all available models
        results = model_tools.train_all_models(
            filepath=filepath,
            target_column=target_column,
            drop_columns=drop_columns
        )
        
        # Simulate A2A Protocol
        if results['best_cv_score'] < 0.80:
            results['a2a_message'] = {
                "from": "ModelSelector",
                "to": "FeatureEngineer",
                "type": "REQUEST_BETTER_FEATURES",
                "reason": "Accuracy below target threshold",
                "current_score": results['best_cv_score'],
                "request": "Please create interaction features and polynomial terms"
            }
        
        return results
