import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI Title
st.title("🏠 House Price Prediction")

# Description
st.write("Enter house details below to predict the estimated price.")

# Input section
st.subheader("Enter House Details")

area = st.number_input("Area (sq ft)", min_value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0)

# Prediction button
if st.button("Predict Price"):

    # Input validation
    if area > 0 and bedrooms > 0 and bathrooms > 0:
        prediction = model.predict([[area, bedrooms, bathrooms]])
        
        # Output
        st.success(f"Estimated Price: ₹ {prediction[0]:,.0f}")
    else:
        st.error("Please enter valid values (greater than 0)")
