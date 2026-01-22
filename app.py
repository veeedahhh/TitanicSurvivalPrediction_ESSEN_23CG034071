# app.py

import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model/titanic_survival_model.pkl")

st.title("Titanic Survival Prediction System")
st.markdown("Enter passenger details below:")

# --- User Inputs ---
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)

# --- Preprocess input ---
sex_encoded = 1 if sex == "male" else 0
input_data = np.array([[pclass, sex_encoded, age, sibsp, fare]])  # 2D array for model

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]  # No scaling needed for Random Forest
    if prediction == 1:
        st.success("Passenger would have Survived ✅")
    else:
        st.error("Passenger would NOT have Survived ❌")
