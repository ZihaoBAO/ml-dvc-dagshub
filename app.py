from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

class InputData(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(input_data: InputData):
    if len(input_data.features) != 64:
        return {"error": "Input must contain 64 features."}
    y = model.predict([input_data.features])
    return {"prediction": y.tolist()}
