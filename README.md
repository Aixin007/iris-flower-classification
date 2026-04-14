# 🌸 Iris Flower Classification

A machine learning web app that classifies iris flowers into three species — **Setosa, Versicolor, and Virginica** — based on sepal and petal measurements, built as part of a 4-week AI/ML internship.

**Intern:** Annika Dubey  
**Platform:** CodeZoner Virtual Internship  
**Duration:** 4 Weeks  

## 🚀 Live Demo
👉 **[Click here to try the app!](https://frmcovbmwyvrxxp7bdfdbh.streamlit.app)**

---

## Problem Statement
Given measurements of iris flowers (sepal length, sepal width, petal length, petal width), classify them into one of three species: Setosa, Versicolor, or Virginica.

## Dataset
- Source: UCI Machine Learning Repository / Kaggle
- 150 rows, 5 columns
- 3 balanced classes (50 samples each)
- No missing values

---

## 📸 Screenshots

### Species Distribution
![Species Count](docs/countplot.png)

### Pairplot
![Pairplot](docs/pairplot.png)

### Confusion Matrices
![Confusion Matrices](docs/confusion_matrices.png)

### Cross Validation Results
![Cross Validation](docs/cross_validation.png)

---

## ✅ Features Built

### Week 1 — Data Pipeline
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data cleaning (nulls, duplicates, types)
- ✅ Feature engineering (new features, encoding, scaling)

### Week 2 — ML Models
- ✅ KNN Classifier (~97% accuracy)
- ✅ Decision Tree Classifier (~97% accuracy)
- ✅ SVM Classifier (~98% accuracy)
- ✅ Input validation & error handling
- ✅ Full testing suite

### Week 3 — Advanced Features
- ✅ Confusion matrices for all 3 models
- ✅ 5-fold cross validation
- ✅ KNN hyperparameter tuning (best K finder)
- ✅ Performance & memory optimization
- ✅ Professional prediction UI with confidence scores
- ✅ User testing & feedback fixes

### Week 4 — Deployment
- ✅ Live web app on Streamlit Cloud
- ✅ Interactive sliders for real-time predictions
- ✅ Confidence scores and flower images per prediction
- ✅ Mobile responsive UI

---

## 📊 Model Results

| Model | Test Accuracy | CV Mean |
|-------|--------------|---------|
| KNN | ~97% | ~97% |
| Decision Tree | ~97% | ~95% |
| SVM | ~98% | ~97% |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| ML Library | scikit-learn |
| Data | pandas, numpy |
| Visualization | seaborn, matplotlib |
| Web App | Streamlit |
| Model Export | joblib |
| Notebook | Jupyter |
| Version Control | Git & GitHub |

---

## ⚙️ How to Run Locally

1. Clone the repo:  https://github.com/Aixin007/iris-flower-classification.git

2. Install dependencies: pip install pandas scikit-learn seaborn matplotlib jupyter joblib streamlit

3. Run the web app: streamlit run src/app.py

4. Run prediction script: cd src
                          python predict.py

5. Or explore the notebooks: ---

## 📁 Folder Structure

iris-flower-classification/
├── src/                  → Notebooks and scripts
│   ├── exploration.ipynb
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   ├── knn_model.ipynb
│   ├── decision_tree_model.ipynb
│   ├── svm_model.ipynb
│   ├── confusion_matrix.ipynb
│   ├── cross_validation.ipynb
│   ├── optimization.ipynb
│   ├── predict.py
│   └── app.py
├── data/
│   ├── iris.csv          → Raw dataset
│   └── processed/        → Cleaned dataset
├── docs/                 → Screenshots and charts
├── requirements.txt
└── README.md

---

## 👩‍💻 User Testing

**Tester:** Mum  
**Platform:** Edge (Streamlit Cloud)  

**Feedback:** "Sounds really calming and a venture for part-time explorers!"  
**Suggestion:** Show a picture of the flower species when prediction appears  
**Fix Applied:** ✅ Random flower image now shows with every prediction  

---

## 👩‍💻 Author
**Annika Dubey**  
CodeZoner AI/ML Virtual Internship  
GitHub: [Aixin007](https://github.com/Aixin007)

