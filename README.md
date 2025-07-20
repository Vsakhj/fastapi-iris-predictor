This project is a machine learning-powered REST API built to solve a real-world challenge: automating flower species identification from petal and sepal measurements. It uses a Logistic Regression model trained on the Iris dataset and serves predictions via a high-performance API developed with FastAPI.

This project was created as part of the AI Engineering Internship assignment.

Features
ML Model: A scikit-learn Logistic Regression model trained to classify three species of Iris flowers.

FastAPI Backend: A high-performance, asynchronous API to serve the model's predictions.

Data Validation: Uses Pydantic to ensure that all incoming requests have the correct data types and structure.

Interactive Docs: Automatically generates interactive API documentation using Swagger UI, available at the /docs endpoint.

Tech Stack
Python 3.8+

FastAPI: For building the API.

Uvicorn: As the ASGI server to run the application.

Scikit-learn: For training the machine learning model.

Pandas & NumPy: For data manipulation.

Joblib: For saving and loading the trained model.

Setup and Installation
Follow these steps to run the project locally.

1. Clone the repository:

Bash

git clone https://github.com/[YOUR_GITHUB_USERNAME]/[YOUR_REPOSITORY_NAME].git
cd [YOUR_REPOSITORY_NAME]
2. Create and activate a virtual environment:

Bash

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies:

Bash

pip install -r requirements.txt
4. Train the model:
This script will train the Logistic Regression model and save it as iris_model.joblib.

Bash

python model.py
5. Start the API server:

Bash

uvicorn main:app --reload
The server will be running at http://127.0.0.1:8000.

API Usage
The API has one main endpoint for predictions.

Predict Flower Species
Endpoint: POST /predict

Description: Receives sepal and petal measurements and returns the predicted Iris species.

Request Body:

JSON

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Success Response (200 OK):

JSON

{
  "predicted_species": "setosa"
}
Interactive Testing
For easy testing, navigate to http://127.0.0.1:8000/docs in your browser to access the interactive Swagger UI documentation.
