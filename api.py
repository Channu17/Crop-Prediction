from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

with open('models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('models/scalar.pkl', 'rb') as f:
    scalar = pickle.load(f)
    

class Values(BaseModel):
    N : float
    P : float
    K : float
    temperature : float
    humidity : float
    ph : float
    rainfall : float

@app.post('/prediction')
def prediction(values:Values):
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