# Model Comparison Report

## Overview
Trained and evaluated 4 baseline models using 5-fold cross-validation.

## Results

| Model | Mean Accuracy | Std Dev |
|-------|--------------|---------|
| logistic | 0.7924 | 0.0160 |
| random_forest | 0.8249 | 0.0316 |
| xgboost | 0.8305 | 0.0249 |
| lightgbm | 0.8395 | 0.0256 |

## Best Model: lightgbm
- **Cross-Validation Accuracy**: 0.8395
- **Standard Deviation**: 0.0256

## Feature Importance (Top 10)
- **FarePerPerson**: 402.0000
- **Age**: 363.0000
- **Fare**: 298.0000
- **Deck**: 67.0000
- **Title**: 66.0000
- **FamilySize**: 59.0000
- **AgeBin**: 52.0000
- **Sex**: 49.0000
- **Embarked**: 47.0000
- **Pclass**: 34.0000
