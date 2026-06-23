from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer when app starts
model = None
vectorizer = None

def load_models():
    global model, vectorizer
    try:
        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

# Home route
@app.route('/')
def home():
    return jsonify({
        "service": "SMS Spam Detector API",
        "status": "running",
        "description": "Send any SMS text message and get spam or ham prediction",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "body": {"message": "your text here"}
        },
        "example": {
            "input": {"message": "You won a free prize! Call now!"},
            "output": {"message": "You won a free prize! Call now!", "prediction": "spam", "confidence": 0.87}
        }
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({"error": "Please provide a message field"}), 400

        message = data['message']

        if not message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400

        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]

        label = "spam" if prediction == 1 else "ham"
        confidence = round(float(max(probability)), 4)

        return jsonify({
            "message": message,
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
