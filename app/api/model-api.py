from fastapi import FastAPI
import joblib
from app.api.api_utils import preprocess_input_streamlit


model_path = "app/model/rf_model.joblib"

app = FastAPI()

# Load the pre-trained model
model_package = joblib.load(model_path)
model = model_package['model']
scaler_stats = model_package['scaler_stats']

@app.get("/")
def index():
    return {"message": "Welcome to the Bike Availability Prediction API"}

@app.post("/predict")
def predict(data: dict) -> dict:

    # Preprocess the input data
    input_data = preprocess_input_streamlit(data, scaler_stats)
    
    # Make prediction
    prediction = model.predict(input_data)
    predict_proba = model.predict_proba(input_data)
    
    return {
        "prediction": prediction.tolist(),
        "predict_proba": predict_proba.tolist()
    }


