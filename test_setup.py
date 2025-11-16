"""
Test script to verify Kaggle Co-Pilot setup
Run this before executing the main system
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (need 3.11+)")
        return False

def check_dependencies():
    """Check required packages"""
    required = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'xgboost', 'lightgbm', 'nbformat', 'joblib'
    ]
    
    optional = ['google.genai']
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('.', '/'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    print("\nOptional (for full Gemini features):")
    for package in optional:
        try:
            __import__(package.replace('.', '/'))
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠ {package} (install with: pip install google-genai)")
    
    return len(missing) == 0

def check_env_file():
    """Check .env configuration"""
    env_path = Path('.env')
    if env_path.exists():
        print(f"✓ .env file exists")
        
        # Check if API key is configured
        with open(env_path) as f:
            content = f.read()
            if 'your_gemini_api_key_here' in content or 'GOOGLE_API_KEY=' not in content:
                print("  ⚠ API key not configured (Gemini features will use fallback mode)")
                print("  ℹ Get API key from: https://aistudio.google.com/app/apikey")
            else:
                print("  ✓ API key configured")
        return True
    else:
        print(f"✗ .env file missing")
        print(f"  Run: cp .env.example .env")
        return False

def check_data_files():
    """Check if data files exist"""
    train_path = Path('data/raw/train.csv')
    test_path = Path('data/raw/test.csv')
    
    if train_path.exists():
        print(f"✓ train.csv found")
        has_train = True
    else:
        print(f"✗ train.csv missing")
        has_train = False
    
    if test_path.exists():
        print(f"✓ test.csv found")
        has_test = True
    else:
        print(f"⚠ test.csv missing (optional)")
        has_test = False
    
    if not has_train:
        print("  ℹ Download from: https://www.kaggle.com/c/titanic/data")
    
    return has_train

def check_directories():
    """Check required directories"""
    required_dirs = [
        'data/raw',
        'outputs/notebooks',
        'outputs/submissions',
        'outputs/reports',
        'outputs/figures',
        'models'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (will be created)")
            path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return all_exist

def main():
    """Run all checks"""
    print("=" * 60)
    print("Kaggle Competition Co-Pilot - Setup Verification")
    print("=" * 60)
    print()
    
    print("1. Python Version:")
    python_ok = check_python_version()
    print()
    
    print("2. Dependencies:")
    deps_ok = check_dependencies()
    print()
    
    print("3. Environment Configuration:")
    env_ok = check_env_file()
    print()
    
    print("4. Data Files:")
    data_ok = check_data_files()
    print()
    
    print("5. Directory Structure:")
    dirs_ok = check_directories()
    print()
    
    print("=" * 60)
    if python_ok and deps_ok and data_ok:
        print("✓ Setup Complete! Ready to run:")
        print()
        print("python src/main.py \\")
        print("  --competition titanic \\")
        print("  --train data/raw/train.csv \\")
        print("  --test data/raw/test.csv \\")
        print("  --target Survived")
    else:
        print("✗ Setup Incomplete. Please fix the issues above.")
        print()
        if not deps_ok:
            print("Install dependencies: pip install -r requirements.txt")
        if not env_ok:
            print("Configure environment: cp .env.example .env")
        if not data_ok:
            print("Download data: https://www.kaggle.com/c/titanic/data")
    print("=" * 60)

if __name__ == "__main__":
    main()
