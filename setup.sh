# Quick Setup Script for Kaggle Co-Pilot

echo "================================================"
echo "Kaggle Competition Co-Pilot Setup"
echo "================================================"
echo

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p outputs/notebooks
mkdir -p outputs/submissions
mkdir -p outputs/reports
mkdir -p outputs/figures
mkdir -p models

# Download Titanic dataset (if kaggle CLI is available)
echo
echo "Attempting to download Titanic dataset..."
if command -v kaggle &> /dev/null
then
    echo "Kaggle CLI found. Downloading..."
    kaggle competitions download -c titanic -p data/raw/
    cd data/raw && unzip -o titanic.zip && rm titanic.zip && cd ../..
    echo "✓ Titanic dataset downloaded"
else
    echo "⚠ Kaggle CLI not found. Please download manually:"
    echo "  1. Go to: https://www.kaggle.com/c/titanic/data"
    echo "  2. Download train.csv and test.csv"
    echo "  3. Place them in data/raw/"
fi

# Check for .env file
echo
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo "⚠ IMPORTANT: Edit .env and add your GOOGLE_API_KEY"
else
    echo "✓ .env file already exists"
fi

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo
echo "Next steps:"
echo "1. Edit .env and add your Google API key (get from: https://aistudio.google.com/app/apikey)"
echo "2. Ensure train.csv and test.csv are in data/raw/"
echo "3. Run: python src/main.py --competition titanic --train data/raw/train.csv --test data/raw/test.csv --target Survived"
echo
