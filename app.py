# app.py
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

# ✅ Import TensorFlow safely
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

app = Flask(__name__)

# --- File paths ---
CSV_PATH = "1_Daily_minimum_temps.csv"
MODEL_PATH = "model_lstm.h5"
SCALER_PATH = "scaler.pkl"
WINDOW_SIZE = 30

# --- Load model & scaler safely ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ File model_lstm.h5 tidak ditemukan! Jalankan dulu model_lstm.ipynb untuk membuat model.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("❌ File scaler.pkl tidak ditemukan! Jalankan dulu model_lstm.ipynb untuk membuat scaler.")

# 🔧 FIX — load tanpa compile agar tidak error
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# --- Function for preparing data and predictions ---
def prepare_data_and_prediction():
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    # Bersihkan data dari karakter non-numerik
    df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
    df = df.dropna(subset=['Temp'])

    # Ambil data suhu dan tanggal
    dates = df.index.astype(str).tolist()
    temps = df['Temp'].tolist()

    # Normalisasi
    series = df['Temp'].values.reshape(-1, 1)
    series_scaled = scaler.transform(series)

    # Buat sequence
    X_all, idxs = [], []
    for i in range(len(series_scaled) - WINDOW_SIZE):
        X_all.append(series_scaled[i:i + WINDOW_SIZE])
        idxs.append(df.index[i + WINDOW_SIZE])

    X_all = np.array(X_all).reshape((-1, WINDOW_SIZE, 1))
    preds_scaled = model.predict(X_all, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()

    # Tanggal prediksi
    pred_dates = [d.strftime('%Y-%m-%d') for d in idxs]
    pred_values = preds.tolist()

    # Prediksi suhu hari berikutnya
    last_window = series_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    next_scaled = model.predict(last_window)
    next_value = float(scaler.inverse_transform(next_scaled).flatten()[0])
    next_date = (df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')

    # Hitung RMSE
    from sklearn.metrics import mean_squared_error
    actual_for_preds = df['Temp'].values[WINDOW_SIZE:]
    rmse = float(np.sqrt(mean_squared_error(actual_for_preds, preds)))

    return {
        "dates": dates,
        "temps": temps,
        "pred_dates": pred_dates,
        "pred_values": pred_values,
        "next_date": next_date,
        "next_value": round(next_value, 3),
        "rmse": round(rmse, 4)
    }

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    payload = prepare_data_and_prediction()
    return jsonify(payload)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
