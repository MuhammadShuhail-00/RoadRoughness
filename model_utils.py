# model_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

def merge_and_extract_features(accel, gyro, gps):
    # Preprocess timestamps
    for df in [accel, gyro, gps]:
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if df["timestamp"].max() > 1e15:
            df["timestamp"] = df["timestamp"] / 1_000_000
        df.sort_values("timestamp", inplace=True)
        df.drop(columns=[col for col in df.columns if "elapsed" in col.lower()], inplace=True)

    # Merge sensor data
    merged = pd.merge_asof(accel, gyro, on="timestamp", direction="nearest", tolerance=50)
    merged = pd.merge_asof(merged, gps, on="timestamp", direction="nearest", tolerance=200)
    merged.dropna(inplace=True)
    merged.insert(0, "readable_time", pd.to_datetime(merged["timestamp"], unit='ms') + timedelta(hours=8))

    # Rename for consistency
    df = merged.rename(columns={
        'x_x': 'accel_x',
        'y_x': 'accel_y',
        'z_x': 'accel_z',
        'x_y': 'gyro_x',
        'y_y': 'gyro_y',
        'z_y': 'gyro_z'
    }).dropna().sort_values("timestamp").reset_index(drop=True)

    # Segment-wise feature extraction
    segments = []
    window_size = 200
    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i:i+window_size]
        if window['speed'].mean() >= 5:
            y_filtered = window['accel_y'].rolling(window=5, center=True).mean().bfill().ffill()
            features = {
                'start_time': window['timestamp'].iloc[0],
                'end_time': window['timestamp'].iloc[-1],
                'mean_accel_y': y_filtered.mean(),
                'std_accel_y': y_filtered.std(),
                'rms_accel_y': np.sqrt(np.mean(y_filtered**2)),
                'peak2peak_accel_y': y_filtered.max() - y_filtered.min(),
                'mean_speed': window['speed'].mean() * 3.6,
                'elevation_change': window['altitude'].iloc[-1] - window['altitude'].iloc[0] if 'altitude' in window else 0,
                'gyro_y_std': window['gyro_y'].std(),
                'gyro_x_std': window['gyro_x'].std(),
                'latitude': window['latitude'].mean(),
                'longitude': window['longitude'].mean()
            }
            segments.append(features)

    return pd.DataFrame(segments)


def predict_with_model(model, features_df, label_encoder):
    features_df_cleaned = features_df.drop(columns=["start_time", "end_time", "latitude", "longitude", "readable_time"], errors="ignore")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df_cleaned)
    y_pred_encoded = model.predict(X_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    features_df["Prediction"] = y_pred
    return y_pred, features_df
