# ============================================
# Iris Flower Classifier — Streamlit Web App
# Interactive web interface for predictions
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import os
# Real iris flower image URLs per species
import random

SPECIES_IMAGES = {
    'Iris-setosa': [
        "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
    ],
    'Iris-versicolor': [
        "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/2/27/Blue_Flag%2C_Ottawa.jpg",
    ],
    'Iris-virginica': [
        "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/f/f8/Iris_virginica_2.jpg",
    ],
}

# --- Page config ---
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# --- Load and train model ---
@st.cache_resource
def load_model():
    df = pd.read_csv('data/processed/iris_cleaned.csv')
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', random_state=42, probability=True)
    model.fit(X_train, y_train)
    return model

model = load_model()

# --- UI ---
st.title("🌸 Iris Flower Classifier")
st.markdown("Enter flower measurements below to predict the species!")

st.markdown("---")

# Input sliders
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 5.0, 3.5, 0.1)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.6, 0.2, 0.1)

st.markdown("---")

# Predict button
if st.button("🔍 Predict Species", use_container_width=True):
    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    )

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = max(proba) * 100

    species_info = {
        'Iris-setosa':     ('🌸', '#FFB6C1', 'Small flowers, clearly separate from others'),
        'Iris-versicolor': ('🌺', '#FFD700', 'Medium flowers, moderate petal size'),
        'Iris-virginica':  ('🌷', '#98FB98', 'Large flowers, biggest petals'),
    }

    emoji, color, desc = species_info[prediction]

    st.success(f"{emoji} **Predicted Species: {prediction}**")
    st.info(f"📊 Model Confidence: **{confidence:.1f}%**")
    st.caption(f"📝 {desc}")

    # Show a random image of the predicted species
    image_url = random.choice(SPECIES_IMAGES[prediction])
    st.image(image_url, 
         caption=f"This is what {prediction} looks like!", 
         width=300)

    # Confidence bar
    st.markdown("**Confidence per species:**")
    for i, species in enumerate(model.classes_):
        st.progress(int(proba[i] * 100), text=f"{species}: {proba[i]*100:.1f}%")

st.markdown("---")
st.caption("Built by Annika Dubey | CodeZoner ML Internship")