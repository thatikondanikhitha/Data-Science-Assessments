import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Logistic Regression Prediction App")
st.write("Predict the probability of the target class using the trained model.")

# Load model
model = joblib.load('logistic_model.pkl')

# User inputs
age = st.number_input("Enter Age", min_value=1, max_value=100, value=25)
salary = st.number_input("Enter Salary", min_value=1000, max_value=100000, value=50000)
gender = st.selectbox("Select Gender", ["Male", "Female"])

# Prepare input
input_data = pd.DataFrame({
    'Age': [age],
    'Salary': [salary],
    'Gender': [gender]
})

# Prediction
prediction_class = model.predict(input_data)[0]
prediction_prob = model.predict_proba(input_data)[:, 1][0]

# Display results
st.subheader("Prediction Results")
st.write(f"Predicted Class: {prediction_class}")
st.write(f"Probability of Positive Class: {prediction_prob:.2f}")
