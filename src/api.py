from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pickle
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

app = FastAPI()

with open('models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('models/scalar.pkl', 'rb') as f:
    scalar = pickle.load(f)
    
soil_classifier = load_model('models/soilClassification.keras')

with open('models/label_encoder_soil_classification.pkl', 'rb') as f:
    label_encoder_sc = pickle.load(f)
    
def preprocess_image(image):
    img_array = np.array(image) / 255.0
    img_array = tf.image.resize(img_array, (224, 224))  
    img_array = tf.expand_dims(img_array, axis=0)  
    return img_array

class Values(BaseModel):
    N : float
    P : float
    K : float
    temperature : float
    humidity : float
    ph : float
    rainfall : float

class Values()

@app.post('/prediction')
def prediction(values:Values):
    try:
        input_data = pd.DataFrame([{
            "N": values.N,
            "P": values.P,
            "K": values.K,
            "temperature": values.temperature,
            "humidity": values.humidity,
            "ph": values.ph,
            "rainfall": values.rainfall
        }])
        scaled_data = scalar.transform(input_data)
        prediction = model.predict(scaled_data)
        decoded_prediction = label_encoder.inverse_transform(prediction)
        
        return {"prediction": decoded_prediction[0]}
    except KeyError as e:
        return {"error": f"Missing key in input data: {str(e)}"}, 400
    
    except ValueError as e:
        return {"error": f"Value error: {str(e)}"}, 400
    
    except AttributeError as e:
        return {"error": f"Attribute error: {str(e)}. Ensure model and scalar are properly loaded."}, 500
    
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}, 500


@app.post('/soil_classification')
async def soil_classification(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        np_image = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(
                content={"error": "Failed to decode the image."},
                status_code=400,
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        preprocessed_image = preprocess_image(img)
        
        prediction = soil_classifier.predict(preprocessed_image)
        
        decoded_prediction = label_encoder_sc.inverse_transform([np.argmax(prediction)])
        return{'soil': decoded_prediction[0]}
        
        
    except Exception as e:
        pass


@app.post("/fertilizerReccommendation")
def fertilizerReccommendation()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Prediction API"}