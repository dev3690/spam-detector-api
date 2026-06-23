# SMS Spam Detector API

A simple REST API that predicts if an SMS message is spam or ham.

## Files:
- app.py - Main Flask application
- spam_model.pkl - Trained Logistic Regression model
- vectorizer.pkl - Trained TF-IDF vectorizer
- requirements.txt - Python dependencies
- Procfile - For deployment

## How to Use:

### Home Page:
GET https://your-url.onrender.com/

### Predict:
POST https://your-url.onrender.com/predict

Body:
{
    "message": "Your SMS text here"
}

### Health Check:
GET https://your-url.onrender.com/health
