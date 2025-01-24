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
    
with open('models/ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)
    
with open('models/fertilizerLabelencoder.pkl', 'rb') as f:
    label_encoder_fc = pickle.load(f)

with open('models/fertilizerScalar.pkl', 'rb') as f:
    scalar_sc = pickle.load(f) 
    
with open('models/random_forest.pkl', 'rb') as f:
    rnd_clf = pickle.load(f)   

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

class Fertilizer_Values(BaseModel):
    Temparature : float
    Humidity	: float
    Moisture    : float
    Soil_Type   : str
    Crop_Type   : str
    Nitrogen    : float
    Potassium   : float
    Phosphorous : float
    
    
    
    
    
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
def fertilizerReccommendation(fc_values: Fertilizer_Values):
    try:
       
        input_data = pd.DataFrame([{
            "Temparature": fc_values.Temparature,
            "Humidity ": fc_values.Humidity,
            "Moisture": fc_values.Moisture,
            "Soil Type": fc_values.Soil_Type,
            "Crop Type": fc_values.Crop_Type,
            "Nitrogen": fc_values.Nitrogen,
            "Potassium": fc_values.Potassium,
            "Phosphorous": fc_values.Phosphorous
        }])

     
        try:
            encoded_data = ohe.transform(input_data[['Soil Type', 'Crop Type']]).toarray()
            encoded_feature_names = ohe.get_feature_names_out(['Soil Type', 'Crop Type'])
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)
        except Exception as encoding_error:
            return {"error": f"Encoding failed: {str(encoding_error)}"}, 400

 
        new_input = pd.concat([input_data, encoded_df], axis=1)
        new_input.drop(['Soil Type', 'Crop Type'], axis=1, inplace=True)
        
        try:
            scaled_data = scalar_sc.transform(new_input)
        except Exception as scaling_error:
            return {"error": f"Scaling failed: {str(scaling_error)}"}, 400


        try:
            prediction = rnd_clf.predict(scaled_data)
            decoded_prediction = label_encoder_fc.inverse_transform(prediction)
        except Exception as prediction_error:
            return {"error": f"Prediction failed: {str(prediction_error)}"}, 400

        return {"prediction": decoded_prediction[0]}

    except Exception as e:
        
        return {"error": f"An unexpected error occurred: {str(e)}"}, 500


@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Prediction API"}