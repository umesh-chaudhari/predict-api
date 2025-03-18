from flask import Flask, request, jsonify
import os
import pickle
import extract_v3
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load classifiers from pickle file
with open('classifiers.pkl', 'rb') as f:
    classifiers = pickle.load(f)

# File to store the random values dictionary
R_VALUES_FILE = 'values.pkl'


# Function to load random values from file if it exists
def load_values():
    if os.path.exists(R_VALUES_FILE):
        with open(R_VALUES_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


# Function to save random values to file
def save_random_values(data):
    with open(R_VALUES_FILE, 'wb') as f:
        pickle.dump(data, f)


# Load random values from the file on server startup
values_store = load_values()


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract features using your custom extract_v3 function
    raw_features = extract_v3.start(file.filename)

    feature_sets = [
        [raw_features[0], raw_features[6]],  # Emotional Stability
        [raw_features[2], raw_features[5]],  # Mental Energy or Will Power
        [raw_features[2], raw_features[1]],  # Modesty
        [raw_features[3], raw_features[4]],  # Personal Harmony and Flexibility
        [raw_features[6], raw_features[1]],  # Lack of Discipline
        [raw_features[2], raw_features[3]],  # Poor Concentration
        [raw_features[2], raw_features[4]],  # Non Communicativeness
        [raw_features[3], raw_features[4]]  # Social Isolation
    ]

    predictions = []
    for clf, features in zip(classifiers, feature_sets):
        prediction = clf.predict([features])[0]
        predictions.append(prediction)

    # Check if the image has been processed before and return stored random values if available
    if file.filename in values_store:
        return jsonify({"predictions": predictions, "random_values": values_store[file.filename]})

    # Generate random values based on predictions (between 0-35 for 0, and 36-70 for 1)
    values = []
    for prediction in predictions:
        if prediction == 0:
            random_value = random.randint(0, 35)
        elif prediction == 1:
            random_value = random.randint(36, 70)
        else:
            random_value = random.randint(71, 98)
        values.append(random_value)

    # Store random values and save to file for persistence
    values_store[file.filename] = values
    save_random_values(values_store)  # Save the updated dictionary to the file

    return jsonify({"predictions": predictions, "random_values": values})


@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        with open('metrics.pkl', 'rb') as fi:
            data = pickle.load(fi)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "Metrics file not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
