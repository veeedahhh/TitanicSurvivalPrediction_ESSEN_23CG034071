# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model & scaler
model = joblib.load("model/titanic_survival_model.pkl")
# Remove or comment out this line
# scaler = joblib.load("model/scaler.pkl")

# Remove the scaling step
# input_data_scaled = scaler.transform(input_data)
# Instead, just use the raw input
input_data_for_model = input_data  # For Random Forest
prediction = model.predict(input_data_for_model)[0]

st.title("Titanic Survival Prediction System")

st.markdown("Enter passenger details below:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1,2,3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)

# Preprocess input
sex_encoded = 1 if sex == "male" else 0
input_data = np.array([[pclass, sex_encoded, age, sibsp, fare]])
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]  # raw input
    if prediction == 1:
        st.success("Passenger would have Survived ✅")
    else:
        st.error("Passenger would NOT have Survived ❌")
