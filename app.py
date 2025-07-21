import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
import joblib
from datetime import timedelta
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("iri_rf_classifier_cleaned.pkl")

st.set_page_config(page_title="Road Roughness Detection", layout="wide")

st.title("🚗 Road Surface Roughness Detection")
st.markdown("Upload a **.zip file** containing Accelerometer.csv, Gyroscope.csv, and Location.csv files.")

uploaded_zip = st.file_uploader("Upload your raw sensor .zip file", type="zip")

# === Helper: Feature Extraction ===
def extract_features(df):
    window_size = 200
    segments = []

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
                'gyro_x_std': window['gyro_x'].std()
            }
            segments.append(features)

    features_df = pd.DataFrame(segments)
    features_df["estimated_iri"] = (
        2.0 * features_df["rms_accel_y"] +
        0.03 * features_df["mean_speed"] +
        0.1
    )
    return features_df

# === Helper: Assign labels ===
def final_label(row):
    iri = row["estimated_iri"]
    if iri < 2.0:
        return "Smooth"
    elif iri < 3.5:
        return "Fair"
    else:
        return "Rough"

# === Main Processing ===
if uploaded_zip:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        try:
            accel = pd.read_csv(os.path.join(temp_dir, "Accelerometer.csv"))
            gyro = pd.read_csv(os.path.join(temp_dir, "Gyroscope.csv"))
            gps = pd.read_csv(os.path.join(temp_dir, "Location.csv"))
        except:
            st.error("❌ Make sure your ZIP file contains Accelerometer.csv, Gyroscope.csv, and Location.csv.")
            st.stop()

        for df in [accel, gyro, gps]:
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            if df["timestamp"].max() > 1e15:
                df["timestamp"] = df["timestamp"] / 1_000_000
            df.sort_values("timestamp", inplace=True)
            df.drop(columns=[col for col in df.columns if "elapsed" in col.lower()], inplace=True)

        merged = pd.merge_asof(accel, gyro, on="timestamp", direction="nearest", tolerance=50)
        merged = pd.merge_asof(merged, gps, on="timestamp", direction="nearest", tolerance=200)
        merged.dropna(inplace=True)
        merged.insert(0, "readable_time", pd.to_datetime(merged["timestamp"], unit='ms') + timedelta(hours=8))

        merged.rename(columns={
            'x_x': 'accel_x', 'y_x': 'accel_y', 'z_x': 'accel_z',
            'x_y': 'gyro_x',  'y_y': 'gyro_y',  'z_y': 'gyro_z'
        }, inplace=True)

        st.success(f"✅ Merged {len(merged)} rows from sensor data.")

        features_df = extract_features(merged)
        if features_df.empty:
            st.warning("⚠️ No valid road segments detected (speed too low or short session).")
            st.stop()

        features_df["label"] = features_df.apply(final_label, axis=1)
        X = features_df[['mean_accel_y', 'std_accel_y', 'rms_accel_y', 'peak2peak_accel_y',
                         'mean_speed', 'elevation_change', 'gyro_y_std', 'gyro_x_std']]
        features_df["prediction"] = model.predict(X)

        # === Buttons ===
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Show Pie Chart"):
                st.subheader("Road Condition Distribution")
                fig, ax = plt.subplots()
                features_df["prediction"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                st.pyplot(fig)

        with col2:
            if st.button("🧾 Show Prediction Table"):
                st.subheader("Segment-wise Prediction")
                st.dataframe(features_df[["readable_time", "mean_speed", "prediction"]])

        with col3:
            if st.button("📄 Show Raw Merged Data"):
                st.subheader("Raw Sensor Merged Data")
                st.dataframe(merged.head(500))

        # === Map Display ===
        st.subheader("🗺️ Map View of Road Segments")
        label_color = {"Smooth": "green", "Fair": "blue", "Rough": "red"}
        center_lat = merged["latitude"].median()
        center_lon = merged["longitude"].median()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB dark_matter")

        for _, row in features_df.iterrows():
            segment = merged[(merged["timestamp"] >= row["start_time"]) & (merged["timestamp"] <= row["end_time"])]
            coords = list(zip(segment["latitude"], segment["longitude"]))
            for lat, lon in coords:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=2,
                    color=label_color.get(row["prediction"], "gray"),
                    fill=True,
                    fill_opacity=0.8
                ).add_to(m)

        st_data = st_folium(m, width=1100, height=500)
