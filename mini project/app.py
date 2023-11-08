from flask import Flask, render_template, request
import joblib
from joblib import load
import pandas as pd
import os

img = os.path.join('static', 'Images')

app = Flask(__name__)

# Load the trained logistic regression model
model = load('random_forest_model (1).joblib')

# Assuming these are the columns/features in your model
numeric_columns = ['BPrev', 'B_Age', 'B_Height', 'B_Weight', 'RPrev', 'R_Age', 'R_Height', 'R_Weight', 'winby']
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {}
        for col in numeric_columns:
            data[col] = float(request.form[col])
        
        # Debug: Print received data
        print("Received Data:", data)

        # Create a DataFrame from the received data
        input_data = pd.DataFrame([data])

        # Debug: Check the input data
        print("Input Data:", input_data)

        # Use the loaded model for prediction
        prediction = model.predict(input_data)

        # Debug: Print model prediction
        print("Model Prediction (Raw):", prediction)

        # Convert prediction to readable string
        result = "Blue" if prediction == 1 else "Red"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
