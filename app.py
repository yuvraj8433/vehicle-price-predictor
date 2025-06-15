import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load saved model and preprocessor
model = tf.keras.models.load_model("model/price_model.h5")
preprocessor = joblib.load("model/preprocessor.pkl")

st.title("🚗 Vehicle Price Predictor")
st.write("Fill the vehicle details below to predict its price.")

# Input form
with st.form("vehicle_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
        cylinders = st.number_input("No. of Cylinders", min_value=2, max_value=16, step=1)
        mileage = st.number_input("Mileage (in miles)", min_value=0.0)
        doors = st.number_input("No. of Doors", min_value=2, max_value=6, step=1)

    with col2:
        make = st.text_input("Make (e.g. Toyota, BMW)")
        model_name = st.text_input("Model (e.g. Corolla, X5)")
        fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"])
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Other"])
        trim = st.text_input("Trim (e.g. LX, Sport, Base)")
        body = st.selectbox("Body Style", ["Sedan", "SUV", "Hatchback", "Pickup Truck", "Coupe", "Van", "Other"])
        exterior_color = st.text_input("Exterior Color")
        interior_color = st.text_input("Interior Color")
        drivetrain = st.selectbox("Drivetrain", ["Front-wheel Drive", "Rear-wheel Drive", "All-wheel Drive", "Four-wheel Drive", "Other"])

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Build input data
        input_dict = {
            "year": [year],
            "cylinders": [cylinders],
            "mileage": [mileage],
            "doors": [doors],
            "make": [make],
            "model": [model_name],
            "fuel": [fuel],
            "transmission": [transmission],
            "trim": [trim],
            "body": [body],
            "exterior_color": [exterior_color],
            "interior_color": [interior_color],
            "drivetrain": [drivetrain]
        }

        input_df = pd.DataFrame(input_dict)

        # Preprocess and predict
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)
        predicted_price = round(prediction[0][0], 2)

        st.success(f"💰 Predicted Vehicle Price: **${predicted_price}**")
