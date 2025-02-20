import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the model, label encoder, and scaler
model = tf.keras.models.load_model("model/air_quality_gru_model.h5")
label_encoder = joblib.load('model/label_encoder.pkl')  # Might not be needed anymore
scaler = joblib.load('model/scaler.pkl')

label_mapping = {
    "Class 1 (Good Air Quality)": 0,
    "Class 2 (Moderate Air Quality)": 1,
    "Class 3 (Unhealthy Air Quality)": 2,
    "Class 4 (Very Unhealthy Air Quality)": 3,
    "Class 5 (Hazardous Air Quality)": 4,
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the data value from the form
        data_value = float(request.form['data_value'])
        
        # Preprocess input (add dummy second feature)
        processed_input = preprocess_input(data_value)
        
        # Make prediction
        prediction = model.predict(processed_input)
        predicted_class = np.argmax(prediction, axis=-1)
        
        # Get readable class label
        class_label = list(label_mapping.keys())[predicted_class[0]]
        
        return render_template('index.html', result=class_label)
    
    return render_template('index.html', result=None)

def preprocess_input(data_value):
    # Add a dummy second feature (e.g., setting it to 0)
    # This makes it a 2D input that matches the scaler's expectations
    processed_input = np.array([[data_value, 0]])  # Second feature is 0
    
    # Normalize the input using the saved scaler
    scaled_input = scaler.transform(processed_input)
    
    # Reshape for GRU model
    return scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
