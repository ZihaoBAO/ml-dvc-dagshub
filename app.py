from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/predict")
def predict(x: float):
    y = model.predict([[x]])
    return {"prediction": y.tolist()}
