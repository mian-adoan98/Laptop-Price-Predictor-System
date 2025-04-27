## Laptop Price Predictor Application: LLP-App

import flask 
import joblib
import numpy as np 
from flask import Flask 
from flask import render_template, request

# Create a Flask application 
llp_app = Flask(__name__)

# Import the model for making predictions
model = joblib.load("D:\Projectwork Platform\MEP-Machine-Learning\Laptop_Price_Prediction\models\linear_model1.pkl")

# Render the template
@llp_app.route("/home")
def home():
    return render_template("home.html")

# Predict the price value 
@llp_app.route("/predict", method = ["POST"])
def predict():
    if request.form == "POST":
        ram_size = float(request.form["ram_size"])

        # Reshape input for prediction
        feature = np.array([ram_size])
        predictions = model.predict(feature)

        return render_template('home.html', prediction_text = f"Predicted Price: {predictions:.2f} EUR")
# Main function: run environment of the application 
if __name__ == "__main__": 
    # Run the llp app on web browser
    llp_app.run(debug=True)