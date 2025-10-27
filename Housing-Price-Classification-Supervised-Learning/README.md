# Supervised Learning for Housing Price Prediction

A comprehensive machine learning project demonstrating supervised learning techniques for housing price classification. The project explores data preprocessing, feature engineering, and multiple classification algorithms including KNN, Decision Trees, Random Forests, and SVM with various hyperparameter tuning strategies.

## Overview

This project analyzes housing data to classify properties into economic, regular, and luxury categories based on features like crime rates, average rooms, property taxes, and other socio-economic factors.

## Dataset

- **Source**: Boston Housing dataset (506 samples, 14 features)
- **Target Variable**: MEDV (Median home value)
- **Classification**: Transformed into 3 categories (economic, regular, luxury) based on price quintiles

## Technical Approach

### Data Preprocessing
- Missing value handling using multiple techniques (dropping, mean imputation, random sampling)
- Outlier detection and analysis using IQR method
- Feature standardization using StandardScaler
- Correlation analysis to identify key predictive features

### Classification Algorithms
- **KNN**: k=4 neighbors with grid search optimization
- **Decision Trees**: Entropy-based with parameter tuning
- **Random Forests**: Ensemble method with bootstrap sampling
- **Support Vector Machines**: Both RBF and linear kernels with hyperparameter optimization

### Model Evaluation
- Train/Validation/Test split (70%/15%/15%)
- Cross-validation for hyperparameter tuning
- Accuracy, precision, recall, and F1-score metrics
- Confusion matrix visualization

## Results

| Model | Test Accuracy | Best Parameters |
|-------|-------------|----------------|
| KNN | 85.37% | k=4, uniform weights |
| Decision Tree | 82.93% | Entropy, min_samples_split=10 |
| Random Forest | 85.37% | 100-250 trees, optimized |
| SVM (RBF) | 85.37% | C=5, gamma='auto' |
| SVM (Linear) | 80.49% | C=0.5, scale |

**Key Findings**:
- LSTAT (lower status population %) was the most predictive single feature
- RBF kernel SVM and Random Forest achieved the best performance
- Ensemble methods outperformed individual decision trees
- Grid search and random search both found similar optimal parameters

## Technical Implementation

Built using scikit-learn for all ML algorithms with:
- Data preprocessing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Model selection: GridSearchCV, RandomizedSearchCV
- Evaluation metrics: Classification reports and confusion matrices

## Key Concepts Demonstrated

- Missing data handling strategies and trade-offs
- Feature scaling and normalization approaches
- Supervised learning workflow: train/validation/test splits
- Hyperparameter tuning techniques (grid vs random search)
- Ensemble methods: bagging vs boosting
- Model comparison and selection strategies

