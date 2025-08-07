from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import json

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load the model and columns
with open('src/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('src/model_columns.json', 'r') as f:
    model_columns = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Create a dataframe from the input data
    input_data = pd.DataFrame([data])

    # Reindex the input_data to match the model's columns
    # This will add missing columns (the one-hot encoded ones) and fill them with 0
    query = pd.DataFrame(columns=model_columns)
    query = pd.concat([query, input_data], ignore_index=True, sort=False)
    query.fillna(0, inplace=True)

    # Ensure the order of columns is the same as in the training data
    query = query[model_columns]

    # Make prediction
    prediction = model.predict(query)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
