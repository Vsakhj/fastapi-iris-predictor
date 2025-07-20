# 1. Import libraries
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 2. Initialize the FastAPI app
app = FastAPI(title="Iris Species Predictor API")

# 3. Load the saved model artifacts
model = joblib.load('iris_model.joblib')
target_names = joblib.load('iris_target_names.joblib')

# 4. Define the request body structure using Pydantic
# This provides data validation for your API.
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 5. Define the prediction endpoint
@app.post("/predict", tags=["Predictions"])
async def predict_species(features: IrisFeatures):
    """
    Receives Iris flower measurements and returns the predicted species.
    """
    # Convert input data to a NumPy array for the model
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    # Make a prediction
    prediction_index = model.predict(input_data)[0]
    predicted_species = target_names[prediction_index]
    
    # Return the prediction
    return {"predicted_species": predicted_species}

# A simple root endpoint for checking if the API is running
@app.get("/", tags=["Status"])
async def read_root():
    return {"status": "API is running successfully!"}