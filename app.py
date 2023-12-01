from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn import preprocessing
import os

app = Flask(__name__)

# Define the absolute path to the model file
def load_model():
    model_path = os.path.join("models", "dt_model.pkl")
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model

# Define the label encoder for 'Gender'
le_Sex = preprocessing.LabelEncoder()
le_Sex.fit(['Female', 'Male'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = float(request.form['age'])
    gender = request.form['gender']
    salary = float(request.form['salary'])

    # Encode gender using the LabelEncoder
    gender_encoded = le_Sex.transform([gender])[0]

    # Create a NumPy array with the input values
    input_data = np.array([[age, gender_encoded, salary]])

    #Load Model
    model = load_model()

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the prediction
    result = "Will Buy a Car" if prediction[0] > 0.5 else "Will Not Buy a Car"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

