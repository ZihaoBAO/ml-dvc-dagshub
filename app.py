from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/predict")
def predict(x: float):
    import numpy as np
    y = model.predict(np.array([[x]]))
    return {"prediction": y.tolist()}
