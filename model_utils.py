import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import streamlit as st

def extract_features_from_files(directory):
    try:
        csv_files = os.listdir(directory)
        st.write("📂 Files in ZIP:", csv_files)  # Debug output

        acc_file = next((f for f in csv_files if "accelerometer" in f.lower()), None)
        gyro_file = next((f for f in csv_files if "gyroscope" in f.lower()), None)
        gps_file = next((f for f in csv_files if "location" in f.lower()), None)

        if not acc_file or not gyro_file or not gps_file:
            st.error("❌ Required sensor files not found. Please ensure filenames include 'accelerometer', 'gyroscope', and 'location'.")
            return pd.DataFrame()

        acc = pd.read_csv(os.path.join(directory, acc_file))
        gyro = pd.read_csv(os.path.join(directory, gyro_file))
        gps = pd.read_csv(os.path.join(directory, gps_file))

        # Basic Feature Extraction (can be extended)
        features = pd.DataFrame({
            'mean_speed': [gps['speed'].mean()],
            'std_accel_y': [acc['accel_y'].std()],
            'mean_accel_y': [acc['accel_y'].mean()],
            'ms_accel_y': [(acc['accel_y'] ** 2).mean()],
            'gyro_x_std': [gyro['gyro_x'].std()],
            'gyro_y_std': [gyro['gyro_y'].std()],
            'peak2peak_accel_y': [acc['accel_y'].max() - acc['accel_y'].min()],
            'elevation_change': [gps['altitude'].max() - gps['altitude'].min()],
        })

        return features

    except Exception as e:
        st.error(f"❌ Feature extraction error: {e}")
        return pd.DataFrame()

def predict_with_model(model, features, label_encoder):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    y_pred_encoded = model.predict(X_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    features['Prediction'] = y_pred
    return y_pred, features
