# ============================================
# Iris Flower Prediction Script
# Fixed based on user testing feedback:
# - Added sepal/petal explanation
# - Added example measurements
# - Added clearer instructions
# - Added input guide with ranges
# ============================================

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time

# --- STATES ---
def show_loading(message, duration=1):
    print(f"\n⏳ {message}", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" Done! ✅")

def show_empty_state():
    print("\n" + "="*55)
    print("  📭 No input provided!")
    print("  Please enter all 4 flower measurements.")
    print("  Use the example values below to get started.")
    print("="*55)

def show_error(message):
    print("\n" + "="*55)
    print(f"  ❌ Error: {message}")
    print("  Please try again with valid measurements.")
    print("="*55)

def show_result(prediction, confidence):
    species_info = {
        'Iris-setosa':     ('🌸', 'Small flowers, clearly separate from others'),
        'Iris-versicolor': ('🌺', 'Medium flowers, moderate petal size'),
        'Iris-virginica':  ('🌷', 'Large flowers, biggest petals'),
    }
    emoji, desc = species_info.get(prediction, ('🌿', ''))
    print("\n" + "="*55)
    print(f"  {emoji}  Predicted Species : {prediction}")
    print(f"  📊 Model Confidence  : {confidence:.1f}%")
    print(f"  📝 Description       : {desc}")
    print("="*55)

# --- TOOLTIP: explain measurements ---
def show_help():
    print("""
╔══════════════════════════════════════════════════╗
║           IRIS MEASUREMENT GUIDE                 ║
╠══════════════════════════════════════════════════╣
║  🌿 SEPAL  = The outer leaf-like parts           ║
║  🌸 PETAL  = The inner colourful flower parts    ║
╠══════════════════════════════════════════════════╣
║  Measurement        Valid Range                  ║
║  ─────────────────────────────────────────────  ║
║  Sepal Length (cm)  4.0  → 8.0                  ║
║  Sepal Width  (cm)  2.0  → 5.0                  ║
║  Petal Length (cm)  1.0  → 7.0                  ║
║  Petal Width  (cm)  0.1  → 2.6                  ║
╠══════════════════════════════════════════════════╣
║  EXAMPLE VALUES:                                 ║
║  Setosa     → [5.1, 3.5, 1.4, 0.2]             ║
║  Versicolor → [6.2, 2.9, 4.3, 1.3]             ║
║  Virginica  → [7.3, 2.9, 6.3, 1.8]             ║
╚══════════════════════════════════════════════════╝
""")

# --- VALIDATION ---
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
            return False, f"{name} value {values[i]} is out of range [{low} - {high}]"
    return True, None

# --- LOAD MODEL ---
def load_model():
    show_loading("Loading dataset")
    df = pd.read_csv('../data/processed/iris_cleaned.csv')
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    show_loading("Training SVM model")
    model = SVC(kernel='rbf', random_state=42, probability=True)
    model.fit(X_train, y_train)
    print("\n🎉 System ready!")
    return model

# --- PREDICT ---
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    inputs = [sepal_length, sepal_width, petal_length, petal_width]
    if any(v is None for v in inputs):
        show_empty_state()
        return
    is_valid, error = validate(inputs)
    if not is_valid:
        show_error(error)
        return
    show_loading("Analyzing flower measurements")
    cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    df_input = pd.DataFrame([inputs], columns=cols)
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    confidence = max(proba) * 100
    show_result(prediction, confidence)

# --- RUN ---
model = load_model()

print("\n" + "🌸" * 22)
print("      IRIS FLOWER CLASSIFIER v2.0")
print("🌸" * 22)

show_help()

print("📋 RUNNING DEMO PREDICTIONS:")
predict_flower(5.1, 3.5, 1.4, 0.2)
predict_flower(6.2, 2.9, 4.3, 1.3)
predict_flower(7.3, 2.9, 6.3, 1.8)
predict_flower(None, 3.5, 1.4, 0.2)
predict_flower(99,  3.5, 1.4, 0.2)