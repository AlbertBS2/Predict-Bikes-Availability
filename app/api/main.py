from fastapi import FastAPI
import joblib
from app.api.api_utils import preprocess_input_streamlit


model_path = "app/model/trained_rf.joblib"

app = FastAPI()

# Load the pre-trained model
model= joblib.load(model_path)

@app.get("/")
def index():
    return {"message": "Welcome to the Bike Availability Prediction API"}

@app.post("/predict")
def predict(data: dict) -> dict:

    # Preprocess the input data
    input_data = preprocess_input_streamlit(data)
    
    # Make prediction
    prediction = model.predict(input_data)
    predict_proba = model.predict_proba(input_data)
    
    return {
        "prediction": prediction.tolist(),
        "predict_proba": predict_proba.tolist()
    }


