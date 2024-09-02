from flask import Flask, request, jsonify
import os
import pickle
import extract_v3
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load classifiers from pickle file
with open('classifiers.pkl', 'rb') as f:
    classifiers = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400


    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
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

    return jsonify(predictions)

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
