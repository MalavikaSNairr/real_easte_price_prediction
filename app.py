import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model/linear_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
st.title("Bangalore Real Estate Price Predictor")
st.write("This app estimates property prices based on basic features.")

# User input fields
sqft = st.number_input("Total Square Feet (e.g., 1000)", min_value=100, max_value=10000, step=10)
bath = st.slider("Number of Bathrooms", 1, 5, 2)
bhk = st.slider("Number of Bedrooms (BHK)", 1, 5, 2)

# Predict on button click
if st.button("Predict Price"):
    input_data = np.array([[sqft, bath, bhk]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹{predicted_price:,.0f}")
