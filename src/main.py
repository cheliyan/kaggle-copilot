"""
Kaggle Competition Co-Pilot: Main Orchestrator
Coordinates all agents to autonomously handle Kaggle competitions.
Powered by Google ADK + Gemini AI
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.competition_reader import CompetitionReaderAgent
from agents.data_explorer import DataExplorerAgent
from agents.feature_engineer import FeatureEngineerAgent
from agents.model_selector import ModelSelectorAgent
from agents.report_generator import ReportGeneratorAgent


class KaggleOrchestrator:
    """
    Main orchestrator that coordinates all specialized agents.
    Implements the multi-agent workflow for autonomous competition handling.
    """
    
    def __init__(self, competition_name: str, data_dir: str, output_dir: str):
        """
        Initialize the orchestrator with competition details.
        
        Args:
            competition_name: Name of the competition
            data_dir: Directory containing competition data
            output_dir: Directory for outputs
        """
        self.competition_name = competition_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output subdirectories
        self.notebooks_dir = self.output_dir / "notebooks"
        self.submissions_dir = self.output_dir / "submissions"
        self.reports_dir = self.output_dir / "reports"
        self.figures_dir = self.output_dir / "figures"
        self.models_dir = Path("models")
        
        for dir_path in [self.notebooks_dir, self.submissions_dir, 
                         self.reports_dir, self.figures_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents
        print("🤖 Initializing Agent System...")
        self.competition_reader = CompetitionReaderAgent()
        self.data_explorer = DataExplorerAgent()
        self.feature_engineer = FeatureEngineerAgent()
        self.model_selector = ModelSelectorAgent()
        self.report_generator = ReportGeneratorAgent()
        
        # State management
        self.state = {
            "competition": competition_name,
            "start_time": datetime.now(),
            "phases_completed": []
        }
        
        print("✓ Agent System Initialized\n")
    
    def run_competition(self, train_file: str, test_file: str = None, 
                       target_column: str = None):
        """
        Orchestrates the full competition workflow.
        
        Args:
            train_file: Path to training data CSV
            test_file: Optional path to test data CSV
            target_column: Target variable column name
        """
        print("=" * 80)
        print(f"🎯 KAGGLE COMPETITION CO-PILOT")
        print(f"Competition: {self.competition_name}")
        print(f"Start Time: {self.state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # Phase 1: Competition Analysis
        print("📖 PHASE 1: Competition Analysis")
        print("-" * 80)
        strategy = self.competition_reader.analyze(
            competition_name=self.competition_name,
            train_file=train_file,
            target_column=target_column
        )
        self.state["strategy"] = strategy
        self.state["phases_completed"].append("competition_analysis")
        print(f"✓ Problem Type: {strategy['problem_type']}")
        print(f"✓ Evaluation Metric: {strategy['metric']}")
        print(f"✓ Target Column: {strategy['target_column']}")
        print()
        
        # Phase 2: Data Exploration
        print("🔍 PHASE 2: Data Exploration")
        print("-" * 80)
        eda_results = self.data_explorer.explore(
            filepath=train_file,
            target_column=strategy['target_column'],
            output_dir=str(self.figures_dir)
        )
        self.state["eda_results"] = eda_results
        self.state["phases_completed"].append("data_exploration")
        print(f"✓ Dataset: {eda_results['shape'][0]} rows × {eda_results['shape'][1]} columns")
        print(f"✓ Missing Data: {sum(1 for v in eda_results['missing_percent'].values() if v > 0)} columns")
        print(f"✓ Visualizations: {len(eda_results['visualizations'])} created")
        print()
        
        # Phase 3: Feature Engineering
        print("⚙️ PHASE 3: Feature Engineering")
        print("-" * 80)
        
        # Create processed data file
        processed_file = str(self.data_dir / f"{self.competition_name}_processed.csv")
        
        features_result = self.feature_engineer.engineer_features(
            filepath=train_file,
            output_path=processed_file,
            target_column=strategy['target_column'],
            competition_type=strategy['problem_type']
        )
        self.state["features"] = features_result
        self.state["processed_file"] = processed_file
        self.state["phases_completed"].append("feature_engineering")
        print(f"✓ Features Created: {features_result['total_features']}")
        print(f"✓ Top Features: {', '.join(list(features_result['feature_importance'].keys())[:5])}")
        print()
        
        # Phase 4: Model Selection
        print("🤖 PHASE 4: Model Selection & Training")
        print("-" * 80)
        
        # Define columns to drop (non-predictive)
        drop_cols = strategy.get('drop_columns', [])
        
        model_results = self.model_selector.train_and_select(
            filepath=processed_file,
            target_column=strategy['target_column'],
            drop_columns=drop_cols
        )
        self.state["model_results"] = model_results
        self.state["phases_completed"].append("model_selection")
        
        # Display results
        print("\nCross-Validation Results:")
        print(f"{'Model':<20} {'Accuracy':<12} {'Std Dev':<10}")
        print("-" * 42)
        for result in model_results['comparison']:
            print(f"{result['model']:<20} {result['cv_mean']:<12.4f} {result['cv_std']:<10.4f}")
        print()
        print(f"✓ Best Model: {model_results['best_model']} ({model_results['best_cv_score']:.4f} accuracy)")
        print()
        
        # Check if we need feature improvement (A2A simulation)
        if model_results['best_cv_score'] < 0.80:
            print("💡 A2A Protocol: Model accuracy below target, requesting feature improvements...")
            # In a real system, this would trigger Feature Engineer agent
            print("   (Feature Engineer would create additional features here)")
            print()
        
        # Save best model
        best_model_data = model_results['all_results'][model_results['best_model']]
        model_path = str(self.models_dir / f"{self.competition_name}_best_model.pkl")
        
        # Prepare data for saving
        from tools import model_tools
        X, y, feature_names, label_encoders = model_tools.prepare_data(
            processed_file, strategy['target_column'], drop_cols
        )
        save_data = {
            'model_object': best_model_data['model_object'],
            'label_encoders': label_encoders,
            'feature_names': feature_names
        }
        model_tools.save_model(save_data, model_path)
        print(f"✓ Model saved: {model_path}")
        print()
        
        # Phase 5: Generate Predictions (if test file provided)
        if test_file and os.path.exists(test_file):
            print("📊 PHASE 5: Generating Predictions")
            print("-" * 80)
            
            submission_file = str(self.submissions_dir / f"{self.competition_name}_submission.csv")
            
            pred_result = model_tools.generate_predictions(
                model_filepath=model_path,
                test_data_filepath=test_file,
                target_column=strategy['target_column'],
                drop_columns=drop_cols,
                output_filepath=submission_file
            )
            
            print(f"✓ Predictions generated: {pred_result['n_predictions']} rows")
            print(f"✓ Submission file: {submission_file}")
            print()
        
        # Phase 6: Report Generation
        print("📝 PHASE 6: Report Generation")
        print("-" * 80)
        
        report_files = self.report_generator.generate_reports(
            strategy=strategy,
            eda_results=eda_results,
            features=features_result,
            models=model_results,
            output_dir=str(self.reports_dir),
            notebooks_dir=str(self.notebooks_dir)
        )
        
        self.state["report_files"] = report_files
        self.state["phases_completed"].append("report_generation")
        
        for report_type, filepath in report_files.items():
            print(f"✓ {report_type}: {filepath}")
        print()
        
        # Final Summary
        end_time = datetime.now()
        duration = (end_time - self.state['start_time']).total_seconds() / 60
        
        print("=" * 80)
        print("🎉 COMPETITION CO-PILOT COMPLETE")
        print("=" * 80)
        print(f"Duration: {duration:.1f} minutes")
        print(f"Phases Completed: {len(self.state['phases_completed'])}/6")
        print()
        print("📦 DELIVERABLES:")
        print(f"  • EDA Notebook: {self.notebooks_dir}/eda_notebook.ipynb")
        print(f"  • Features Notebook: {self.notebooks_dir}/features_notebook.ipynb")
        print(f"  • Models Report: {self.reports_dir}/model_comparison.md")
        if test_file:
            print(f"  • Submission File: {self.submissions_dir}/{self.competition_name}_submission.csv")
        print(f"  • Executive Summary: {self.reports_dir}/executive_summary.md")
        print()
        print("🚀 Ready for human review and advanced optimization!")
        print("=" * 80)


def main():
    """
    Main entry point for the Kaggle Co-Pilot system.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Kaggle Competition Co-Pilot")
    parser.add_argument('--competition', type=str, required=True, 
                       help='Competition name')
    parser.add_argument('--train', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--test', type=str, default=None,
                       help='Path to test data CSV')
    parser.add_argument('--target', type=str, required=True,
                       help='Target column name')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = KaggleOrchestrator(
        competition_name=args.competition,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run competition pipeline
    orchestrator.run_competition(
        train_file=args.train,
        test_file=args.test,
        target_column=args.target
    )


if __name__ == "__main__":
    main()
