# Iris Flower Classification

**Intern:** Annika Dubey
**Platform:** CodeZoner Virtual Internship
**Duration:** 4 Weeks

## Problem Statement
Given measurements of iris flowers (sepal length, sepal width, petal length,
petal width), classify them into one of three species:
Setosa, Versicolor, or Virginica.

## Dataset
- Source: UCI Machine Learning Repository / Kaggle
- 150 rows, 5 columns
- 3 balanced classes (50 samples each)
- No missing values

## Approach
1. Data Exploration — understand distributions and patterns
2. Data Cleaning — handle nulls, duplicates, fix types
3. Feature Engineering — create new features, encode, scale, split
4. Model Building — KNN, Decision Tree, SVM (Week 2)
5. Evaluation — accuracy, confusion matrix (Week 3)
6. Export model using joblib/pickle (Week 4)

## Tech Stack
Python, scikit-learn, pandas, seaborn, matplotlib, Jupyter

## Folder Structure
- /src → Jupyter notebooks
- /data → raw and processed datasets
- /docs → project plan and documentation

## Visualizations

### Species Distribution
![Species Count](docs/countplot.png)

### Pairplot
![Pairplot](docs/pairplot.png)

## What's Been Built (Week 1 & 2)

### Week 1 — Data Pipeline
-  Data exploration and EDA
-  Data cleaning (nulls, duplicates, types)
-  Feature engineering (new features, encoding, scaling)

### Week 2 — ML Models
-  KNN Classifier (accuracy: ~97%)
-  Decision Tree Classifier (accuracy: ~97%)
-  SVM Classifier (accuracy: ~97%)
-  Input validation & error handling
-  Full testing suite

## How to Run
1. Clone the repo
2. Install dependencies: `pip install pandas scikit-learn seaborn matplotlib jupyter`
3. Open Jupyter: `jupyter notebook`
4. Run notebooks in order: exploration → data_cleaning → feature_engineering → models