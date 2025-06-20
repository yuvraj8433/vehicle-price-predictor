import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load model and preprocessor
model = tf.keras.models.load_model("model/price_model.h5")
preprocessor = joblib.load("model/preprocessor.pkl")

# USD to INR conversion rate (approx)
USD_TO_INR = 83.0

# Page configuration
st.set_page_config(page_title="Vehicle Price Predictor", layout="wide", page_icon="🚘")

st.markdown("<h1 style='text-align: center; color: #00adb5;'>🚗 Vehicle Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter vehicle details to estimate its market price in Indian Rupees 🇮🇳</p>", unsafe_allow_html=True)
st.markdown("---")

# Main input and output container
with st.container():
    col1, col2 = st.columns([1.5, 1])

    with col1:
        with st.form("vehicle_form"):
            st.subheader("🔧 Vehicle Specifications")

            c1, c2 = st.columns(2)

            with c1:
                year = st.number_input("Year of Manufacture", 1990, 2025, step=1)
                cylinders = st.number_input("Engine Cylinders", 2, 16, step=1)
                mileage = st.number_input("Mileage (in miles)", 0.0)
                doors = st.number_input("No. of Doors", 2, 6, step=1)

            with c2:
                make = st.text_input("Make (e.g., Toyota)")
                model_name = st.text_input("Model (e.g., Corolla)")
                fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"])
                transmission = st.selectbox("Transmission Type", ["Automatic", "Manual", "Other"])
                body = st.selectbox("Body Style", ["Sedan", "SUV", "Hatchback", "Pickup Truck", "Coupe", "Van", "Other"])
                drivetrain = st.selectbox("Drivetrain", ["Front-wheel Drive", "Rear-wheel Drive", "All-wheel Drive", "Four-wheel Drive", "Other"])

            submitted = st.form_submit_button("🚀 Predict Price")

    with col2:
        st.subheader("📈 Predicted Price")
        if submitted:
            # Create input dictionary (include required but hidden columns)
            input_data = {
                "year": [year],
                "cylinders": [cylinders],
                "mileage": [mileage],
                "doors": [doors],
                "make": [make],
                "model": [model_name],
                "fuel": [fuel],
                "transmission": [transmission],
                "body": [body],
                "drivetrain": [drivetrain],

                # Add missing columns with default/empty values
                "trim": [""],
                "exterior_color": [""],
                "interior_color": [""]
            }

            input_df = pd.DataFrame(input_data)

            try:
                # Preprocess and predict
                input_processed = preprocessor.transform(input_df)
                prediction = model.predict(input_processed)
                price_usd = round(prediction[0][0], 2)
                price_inr = round(price_usd * USD_TO_INR, 2)

                st.markdown(f"""
                    <div style='padding: 1.5rem; background-color: #222831; border-radius: 12px; text-align: center;'>
                        <h2 style='color: #00ffcc;'>₹ {price_inr}</h2>
                        <p style='color: #eeeeee;'>Estimated Market Price (INR)</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error("Something went wrong during prediction.")
                st.exception(e)
