from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "rf_model.pkl"

# Endpoint to train the model
@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    
    # Get hyperparameters from request or set defaults
    n_estimators = data.get('n_estimators', 100)
    max_depth = data.get('max_depth', None)
    random_state = data.get('random_state', 42)

    # Load iris data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X, y)

    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return jsonify({"message": "Model trained and saved successfully."})


# Endpoint for inference
@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet."}), 400

    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing input features."}), 400

    # Load the trained model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict([features])
    return jsonify({"prediction": int(prediction[0])})


# Endpoint to get hyperparameters
@app.route('/hyperparameters', methods=['GET'])
def get_hyperparameters():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet."}), 400

    # Load the trained model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Get model hyperparameters directly from the model object
    params = model.get_params()

    return jsonify(params)



if __name__ == '__main__':
    app.run(debug=True)
