# ============================================
# Iris Flower Prediction Script
# Clean, professional UI in terminal
# Handles loading, empty & error states
# ============================================

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import sys

# --- LOADING STATE ---
def show_loading(message, duration=1):
    print(f"\n⏳ {message}", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" Done! ✅")

# --- EMPTY STATE ---
def show_empty_state():
    print("\n" + "="*50)
    print("  📭 No input provided!")
    print("  Please enter all 4 flower measurements.")
    print("="*50)

# --- ERROR STATE ---
def show_error(message):
    print("\n" + "="*50)
    print(f"  ❌ Error: {message}")
    print("  Please try again with valid measurements.")
    print("="*50)

# --- SUCCESS STATE ---
def show_result(prediction, confidence_note):
    species_emoji = {
        'Iris-setosa':     '🌸',
        'Iris-versicolor': '🌺',
        'Iris-virginica':  '🌷'
    }
    emoji = species_emoji.get(prediction, '🌿')
    print("\n" + "="*50)
    print(f"  {emoji} Predicted Species: {prediction}")
    print(f"  📊 {confidence_note}")
    print("="*50)

# --- TRAIN MODEL ---
def load_model():
    show_loading("Loading dataset")
    df = pd.read_csv('../data/processed/iris_cleaned.csv')
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    show_loading("Training SVM model")
    model = SVC(kernel='rbf', random_state=42, probability=True)
    model.fit(X_train, y_train)
    return model

# --- VALIDATE INPUT ---
RANGES = {
    'Sepal Length': (4.0, 8.0),
    'Sepal Width':  (2.0, 5.0),
    'Petal Length': (1.0, 7.0),
    'Petal Width':  (0.1, 2.6),
}

def validate(values):
    for i, (name, (low, high)) in enumerate(RANGES.items()):
        if not isinstance(values[i], (int, float)):
            return False, f"{name} must be a number"
        if not (low <= values[i] <= high):
            return False, f"{name} ({values[i]}) out of range [{low}-{high}]"
    return True, None

# --- MAIN PREDICTION FUNCTION ---
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    inputs = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Empty state check
    if any(v is None for v in inputs):
        show_empty_state()
        return
    
    # Validation
    is_valid, error = validate(inputs)
    if not is_valid:
        show_error(error)
        return
    
    # Prediction
    show_loading("Analyzing flower measurements")
    cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    df_input = pd.DataFrame([inputs], columns=cols)
    
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    confidence = max(proba) * 100
    
    show_result(prediction, f"Model confidence: {confidence:.1f}%")

# --- RUN ---
model = load_model()

print("\n" + "🌸" * 20)
print("   IRIS FLOWER CLASSIFIER")
print("🌸" * 20)

# Test all states
print("\n📋 RUNNING DEMO PREDICTIONS:")
predict_flower(5.1, 3.5, 1.4, 0.2)   # Setosa
predict_flower(6.2, 2.9, 4.3, 1.3)   # Versicolor  
predict_flower(7.3, 2.9, 6.3, 1.8)   # Virginica
predict_flower(None, 3.5, 1.4, 0.2)  # Empty state
predict_flower(99,  3.5, 1.4, 0.2)   # Error state