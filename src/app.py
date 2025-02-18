import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

st.title("Crop Prediction & Soil Analysis")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an option", ["Crop Prediction", "Soil Classification", "Fertilizer Recommendation"])

# Load Models
with open('models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('models/scalar.pkl', 'rb') as f:
    scalar = pickle.load(f)
soil_classifier = tf.keras.models.load_model('models/soilClassification.keras')
with open('models/label_encoder_soil_classification.pkl', 'rb') as f:
    label_encoder_sc = pickle.load(f)
with open('models/ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)
with open('models/fertilizerLabelencoder.pkl', 'rb') as f:
    label_encoder_fc = pickle.load(f)
with open('models/fertilizerScalar.pkl', 'rb') as f:
    scalar_sc = pickle.load(f)
with open('models/random_forest_fc.pkl', 'rb') as f:
    rnd_clf = pickle.load(f)

def preprocess_image(image):
    img_array = np.array(image) / 255.0
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

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
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        scaled_data = scalar.transform(input_data)
        probabilities = model.predict_proba(scaled_data)
        top_indices = probabilities[0].argsort()[-4:][::-1]
        top_predictions = [(label_encoder.inverse_transform([idx])[0], probabilities[0][idx]) for idx in top_indices]
        results = [{"class": pred[0], "probability": pred[1]} for pred in top_predictions]
        st.json(results)

elif option == "Soil Classification":
    st.header("Soil Classification")
    uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        if st.button("Classify Soil"):
            file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preprocessed_image = preprocess_image(img)
            prediction = soil_classifier.predict(preprocessed_image)
            decoded_prediction = label_encoder_sc.inverse_transform([np.argmax(prediction)])
            st.json({'soil': decoded_prediction[0]})

elif option == "Fertilizer Recommendation":
    st.header("Fertilizer Recommendation")
    
    temperature = st.number_input("Temperature (°C)", step=0.1)
    humidity = st.number_input("Humidity (%)", step=0.1)
    moisture = st.number_input("Moisture (%)", step=0.1)
    soil_type = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clayey", "Black", "Red"])
    crop_type = st.selectbox("Crop Type", ["Sugarcane", "Cotton", "Millets", "Pulses", "Paddy","Wheat","Barley", "Oil seeds","Tobacco","Ground Nuts","Maize"])
    nitrogen = st.number_input("Nitrogen (N)", step=0.1)
    potassium = st.number_input("Potassium (K)", step=0.1)
    phosphorous = st.number_input("Phosphorous (P)", step=0.1)
    
    if st.button("Recommend Fertilizer"):
        input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]],
                                  columns=["Temparature", "Humidity ", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous"])
        encoded_data = ohe.transform(input_data[['Soil Type', 'Crop Type']]).toarray()
        encoded_feature_names = ohe.get_feature_names_out(['Soil Type', 'Crop Type'])
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)
        new_input = pd.concat([input_data, encoded_df], axis=1).drop(['Soil Type', 'Crop Type'], axis=1)
        scaled_data = scalar_sc.transform(new_input)
        prediction = rnd_clf.predict(scaled_data)
        decoded_prediction = label_encoder_fc.inverse_transform(prediction)
        st.json({"prediction": decoded_prediction[0]})

st.sidebar.write("Made with ❤️ using Streamlit")