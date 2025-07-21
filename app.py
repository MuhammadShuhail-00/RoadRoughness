import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import tempfile
import os
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Road Roughness Detector", layout="wide")
st.title("🚗 Road Surface Roughness Classification App")

# Load trained Random Forest model
model = joblib.load("iri_rf_classifier_cleaned.pkl")

# Define preprocessing + feature extraction function
def preprocess_and_extract_features(accel_df, gyro_df, gps_df):
    # Standardize timestamp and sort
    for df in [accel_df, gyro_df, gps_df]:
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if df["timestamp"].max() > 1e15:
            df["timestamp"] = df["timestamp"] / 1_000_000
        df.sort_values("timestamp", inplace=True)
        df.drop(columns=[col for col in df.columns if "elapsed" in col.lower()], inplace=True)

    # Merge 3 sensors
    df = pd.merge_asof(accel_df, gyro_df, on="timestamp", direction="nearest", tolerance=50)
    df = pd.merge_asof(df, gps_df, on="timestamp", direction="nearest", tolerance=200)
    df.dropna(inplace=True)
    df.insert(0, "readable_time", pd.to_datetime(df["timestamp"], unit='ms') + timedelta(hours=8))

    # Rename for consistency
    df.rename(columns={
        'x_x': 'accel_x',
        'y_x': 'accel_y',
        'z_x': 'accel_z',
        'x_y': 'gyro_x',
        'y_y': 'gyro_y',
        'z_y': 'gyro_z'
    }, inplace=True)

    # Segmenting
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
                'gyro_x_std': window['gyro_x'].std(),
                'latitude': window['latitude'].iloc[-1],
                'longitude': window['longitude'].iloc[-1]
            }
            segments.append(features)

    features_df = pd.DataFrame(segments)
    features_df["estimated_iri"] = (
        2.0 * features_df["rms_accel_y"] +
        0.03 * features_df["mean_speed"] +
        0.1
    )

    # Labeling based on IRI thresholds
    def label_segment(row):
        iri = row["estimated_iri"]
        if iri <= 2.0:
            return "Smooth"
        elif iri <= 3.5:
            return "Fair"
        else:
            return "Rough"
    
    features_df["label"] = features_df.apply(label_segment, axis=1)
    return df, features_df

# ========================
# Sidebar Upload
# ========================
with st.sidebar:
    st.header("📂 Upload Raw Sensor Data (ZIP)")
    uploaded_zip = st.file_uploader("Upload ZIP containing Accelerometer.csv, Gyroscope.csv, Location.csv", type="zip")

# ========================
# Process if File Uploaded
# ========================
if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        filenames = os.listdir(tmpdir)
        accel = pd.read_csv(os.path.join(tmpdir, [f for f in filenames if "Accelerometer" in f][0]))
        gyro = pd.read_csv(os.path.join(tmpdir, [f for f in filenames if "Gyroscope" in f][0]))
        gps = pd.read_csv(os.path.join(tmpdir, [f for f in filenames if "Location" in f][0]))

        # Extract
        raw_df, features_df = preprocess_and_extract_features(accel, gyro, gps)

        # Predict
        X_pred = features_df.drop(columns=["label", "start_time", "end_time", "estimated_iri", "latitude", "longitude"], errors="ignore")
        features_df["prediction"] = model.predict(X_pred)

        # ========================
        # Section: Data Table
        # ========================
        with st.expander("📊 View Extracted Segment Data"):
            st.dataframe(features_df[["readable_time", "mean_speed", "prediction"]])

        # ========================
        # Section: Pie Chart
        # ========================
        with st.expander("🧮 Class Distribution Pie Chart"):
            fig, ax = plt.subplots()
            features_df["prediction"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Predicted Road Surface Distribution")
            st.pyplot(fig)

        # ========================
        # Section: Map
        # ========================
        st.subheader("🗺️ Map View of Road Segments")
        m = folium.Map(location=[features_df["latitude"].mean(), features_df["longitude"].mean()], zoom_start=15, tiles="CartoDB dark_matter")
        marker_cluster = MarkerCluster().add_to(m)

        color_map = {"Smooth": "green", "Fair": "blue", "Rough": "red"}

        for i, row in features_df.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=color_map.get(row["prediction"], "gray"),
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['readable_time']}: {row['prediction']}"
            ).add_to(marker_cluster)

        st_folium(m, width=1100, height=500)

else:
    st.info("Please upload a ZIP file containing the 3 sensor CSVs.")
