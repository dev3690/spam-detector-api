from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer when app starts
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Home route
@app.route('/')
def home():
    return jsonify({
        "service": "SMS Spam Detector",
        "description": "Send a text message and get spam or ham prediction",
        "how_to_use": "Send POST request to /predict with JSON body: {message: your text}",
        "example_input": {"message": "You won a free prize! Call now!"},
        "example_output": {"message": "You won a free prize! Call now!", "prediction": "spam", "confidence": 0.87}
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get message from request
        data = request.get_json()

        # Check if message exists
        if not data or 'message' not in data:
            return jsonify({"error": "Please send a message field in your request"}), 400

        message = data['message']

        # Check if message is empty
        if not message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400

        # Transform and predict
        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]

        # Get label and confidence
        label = "spam" if prediction == 1 else "ham"
        confidence = round(float(max(probability)), 4)

        return jsonify({
            "message": message,
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check route
@app.route('/health')
def health():
    return jsonify({"status": "running"})

if __name__ == '__main__':
    app.run(debug=False)
