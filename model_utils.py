# model_utils.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def extract_features_from_files(directory):
    try:
        csv_files = os.listdir(directory)
        acc_file = [f for f in csv_files if "accelerometer" in f.lower()][0]
        gyro_file = [f for f in csv_files if "gyroscope" in f.lower()][0]
        gps_file = [f for f in csv_files if "location" in f.lower()][0]

        acc = pd.read_csv(os.path.join(directory, acc_file))
        gyro = pd.read_csv(os.path.join(directory, gyro_file))
        gps = pd.read_csv(os.path.join(directory, gps_file))

        # Normalize all column names (lowercase, no spaces)
        acc.columns = acc.columns.str.strip().str.lower()
        gyro.columns = gyro.columns.str.strip().str.lower()
        gps.columns = gps.columns.str.strip().str.lower()

        # Rename common column alternatives for robustness
        if 'accel_y' not in acc.columns and 'y' in acc.columns:
            acc = acc.rename(columns={'y': 'accel_y'})
        if 'gyro_x' not in gyro.columns and 'x' in gyro.columns:
            gyro = gyro.rename(columns={'x': 'gyro_x'})
        if 'gyro_y' not in gyro.columns and 'y' in gyro.columns:
            gyro = gyro.rename(columns={'y': 'gyro_y'})
        if 'altitude' not in gps.columns and 'alt' in gps.columns:
            gps = gps.rename(columns={'alt': 'altitude'})

        # Check for required columns
        required_acc_cols = ['accel_y']
        required_gyro_cols = ['gyro_x', 'gyro_y']
        required_gps_cols = ['speed', 'altitude']
        for col in required_acc_cols + required_gyro_cols + required_gps_cols:
            if col not in pd.concat([acc, gyro, gps], axis=1).columns:
                raise ValueError(f"Missing required column: {col}")

        # Feature extraction
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
        print("Feature extraction error:", e)
        return pd.DataFrame()

def predict_with_model(model, features, label_encoder):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    y_pred_encoded = model.predict(X_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    features['Prediction'] = y_pred
    return y_pred, features
