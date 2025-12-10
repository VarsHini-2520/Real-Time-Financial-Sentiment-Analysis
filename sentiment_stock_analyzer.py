# ...existing code...
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# determine project base and models folder (works when you run the script from anywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL1_PATH = os.path.join(MODELS_DIR, r"C:\Users\VARSHINI.M\Downloads\Sentiment Analysis for Financial News.h5"
")
MODEL2_PATH = os.path.join(MODELS_DIR, r"C:\Users\VARSHINI.M\Downloads\tweet_sentiment.h5"
")

# helper to load model with a clear message
def safe_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(path)

# Try to load models/tokenizer; if missing, raise readable error
try:
    model1 = safe_load_model(MODEL1_PATH)
    model2 = safe_load_model(MODEL2_PATH)
except Exception as e:
    # do not crash silently; print to console and re-raise so you see a clear message on startup
    print("ERROR loading models:", e)
    raise

# ...existing code...
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

    pred1 = model1.predict(padded)
    pred2 = model2.predict(padded)

    avg_pred = (pred1 + pred2) / 2.0

    if avg_pred.shape[1] == 3:
        class_labels = ["Bearish 🔴", "Neutral 🟡", "Bullish 🟢"]
        predicted_class = np.argmax(avg_pred[0])
        sentiment = class_labels[predicted_class]
        chart_data = {
            "negative": float(avg_pred[0][0]),
            "neutral": float(avg_pred[0][1]),
            "positive": float(avg_pred[0][2])
        }
    else:
        sentiment = "Bullish 🟢" if avg_pred[0][0] >= 0.5 else "Bearish 🔴"
        chart_data = {
            "positive": float(avg_pred[0][0]),
            "neutral": 0.0,
            "negative": float(1 - avg_pred[0][0])
        }

    return sentiment, chart_data

# ...existing code...
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text") or request.values.get("text")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        sentiment, chart_data = predict_sentiment(text)
        return jsonify({"sentiment": sentiment, "chart": chart_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
# ...existing code...