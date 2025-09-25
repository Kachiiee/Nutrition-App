
import streamlit as st
import numpy as np
import json

# Load model weights
with open("nutrition_model_weights.json", "r") as f:
    model_data = json.load(f)

weights = np.array(model_data["weights"])
bias = model_data["bias"]

st.title("Nutrition Predictor App")

# User inputs
protein = st.number_input("Protein (g)", min_value=0.0)
carbs = st.number_input("Carbohydrates (g)", min_value=0.0)
fat = st.number_input("Fat (g)", min_value=0.0)
sugar = st.number_input("Sugar (g)", min_value=0.0)
sodium = st.number_input("Sodium (mg)", min_value=0.0)

if st.button("Predict"):
    features = np.array([protein, carbs, fat, sugar, sodium])
    prediction = np.dot(weights[:5], features) + bias
    st.success(f"Predicted Calories: {prediction:.2f}")
