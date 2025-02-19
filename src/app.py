import streamlit as st
import requests
import numpy as np
import cv2

API_URL = "https://crop-prediction-1-6nny.onrender.com"

st.title("Crop Prediction & Soil Analysis")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an option", ["Crop Prediction", "Soil Classification", "Fertilizer Recommendation"])

if option == "Crop Prediction":
    st.header("Crop Prediction")
    
    N = st.number_input("Nitrogen (N)", min_value=0.0, step=0.1)
    P = st.number_input("Phosphorous (P)", min_value=0.0, step=0.1)
    K = st.number_input("Potassium (K)", min_value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", step=0.1)
    humidity = st.number_input("Humidity (%)", step=0.1)
    ph = st.number_input("Soil pH", step=0.1)
    rainfall = st.number_input("Rainfall (mm)", step=0.1)
    
    if st.button("Predict Crop"):
        payload = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }
        response = requests.post(f"{API_URL}/prediction", json=payload)
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error("Failed to get prediction")

elif option == "Soil Classification":
    st.header("Soil Classification")
    uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        if st.button("Classify Soil"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/soil_classification", files=files)
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error("Failed to classify soil")

elif option == "Fertilizer Recommendation":
    st.header("Fertilizer Recommendation")
    
    temperature = st.number_input("Temperature (°C)", step=0.1)
    humidity = st.number_input("Humidity (%)", step=0.1)
    moisture = st.number_input("Moisture (%)", step=0.1)
    soil_type = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clayey", "Black", "Red"])
    crop_type = st.selectbox("Crop Type", ["Sugarcane", "Cotton", "Millets", "Pulses", "Paddy", "Wheat", "Barley", "Oil seeds", "Tobacco", "Ground Nuts", "Maize"])
    nitrogen = st.number_input("Nitrogen (N)", step=0.1)
    potassium = st.number_input("Potassium (K)", step=0.1)
    phosphorous = st.number_input("Phosphorous (P)", step=0.1)
    
    if st.button("Recommend Fertilizer"):
        payload = {
            "Temparature": temperature,
            "Humidity": humidity,
            "Moisture": moisture,
            "Soil_Type": soil_type,
            "Crop_Type": crop_type,
            "Nitrogen": nitrogen,
            "Potassium": potassium,
            "Phosphorous": phosphorous
        }
        response = requests.post(f"{API_URL}/fertilizerReccommendation", json=payload)
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error("Failed to get recommendation")

st.sidebar.write("Made with ❤️ using Streamlit")
