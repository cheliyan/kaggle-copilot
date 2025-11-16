"""
Data Explorer Agent
Performs autonomous exploratory data analysis using tool functions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import data_tools
from typing import Dict, Any


class DataExplorerAgent:
    """
    Agent responsible for exploratory data analysis.
    
    Uses tool functions to:
    - Analyze data structure and statistics
    - Detect missing data patterns
    - Create visualizations
    - Identify outliers and correlations
    """
    
    def __init__(self):
        self.name = "DataExplorer"
        print(f"✓ {self.name} Agent initialized")
    
    def explore(self, filepath: str, target_column: str, output_dir: str) -> Dict[str, Any]:
        """
        Performs comprehensive EDA on the dataset.
        
        Args:
            filepath: Path to CSV file
            target_column: Target variable name
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with EDA results
        """
        # Use data analysis tools
        analysis = data_tools.analyze_dataframe(filepath)
        
        # Generate visualizations
        visualizations = data_tools.create_visualizations(
            filepath, output_dir, target_column
        )
        
        # Correlation analysis
        correlations = data_tools.correlation_analysis(filepath, target_column)
        
        # Generate summary
        summary = data_tools.generate_data_summary(filepath)
        
        return {
            **analysis,
            "visualizations": visualizations,
            "correlations": correlations.get("target_correlations", {}),
            "summary": summary
        }
