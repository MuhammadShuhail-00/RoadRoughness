
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
from sklearn.ensemble import RandomForestClassifier

# Load model and scaler
model = joblib.load("iri_rf_classifier_cleaned.pkl")

# Set page config
st.set_page_config(page_title="Road Roughness Detection App", layout="wide")
st.title("📱 Road Surface Roughness Detection Prototype")

# Upload zip file containing raw data
zip_file = st.file_uploader("Upload ZIP file containing Accelerometer, Gyroscope, and Location CSVs", type="zip")

if zip_file:
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)

        # Find sensor folder
        sensor_folder = None
        for root, dirs, files in os.walk(tmpdirname):
            if all(file in files for file in ["Accelerometer.csv", "Gyroscope.csv", "Location.csv"]):
                sensor_folder = root
                break

        if sensor_folder:
            try:
                accel = pd.read_csv(os.path.join(sensor_folder, "Accelerometer.csv"))
                gyro = pd.read_csv(os.path.join(sensor_folder, "Gyroscope.csv"))
                gps = pd.read_csv(os.path.join(sensor_folder, "Location.csv"))

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

                # Rename columns
                merged.rename(columns={
                    'x_x': 'accel_x', 'y_x': 'accel_y', 'z_x': 'accel_z',
                    'x_y': 'gyro_x', 'y_y': 'gyro_y', 'z_y': 'gyro_z'
                }, inplace=True)

                window_size = 200
                segments = []

                for i in range(0, len(merged) - window_size, window_size):
                    window = merged.iloc[i:i+window_size]
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

                def final_label(row):
                    iri = row["estimated_iri"]
                    if iri < 2.0:
                        return "Smooth"
                    elif iri < 3.5:
                        return "Fair"
                    else:
                        return "Rough"

                features_df["label"] = features_df.apply(final_label, axis=1)

                # Merge time and GPS for map view
                gps_lookup = merged[["timestamp", "readable_time", "latitude", "longitude"]].drop_duplicates(subset="timestamp")
                features_df = features_df.merge(gps_lookup, left_on="start_time", right_on="timestamp", how="left")

                st.success(f"✅ Extracted {len(features_df)} segments")

                # Display summary
                st.subheader("🧾 Segment Summary")
                st.dataframe(features_df[["readable_time", "mean_speed", "rms_accel_y", "estimated_iri", "label"]])

                # Map view
                st.subheader("🗺️ Map View of Road Segments")
                color_map = {"Smooth": "green", "Fair": "blue", "Rough": "red"}
                gps_map = folium.Map(location=[features_df["latitude"].mean(), features_df["longitude"].mean()], zoom_start=16)
                marker_cluster = MarkerCluster().add_to(gps_map)

                for _, row in features_df.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=4,
                        color=color_map.get(row["label"], "gray"),
                        fill=True,
                        fill_opacity=0.8,
                        popup=f"Label: {row['label']}\nIRI: {row['estimated_iri']:.2f}"
                    ).add_to(marker_cluster)

                st_folium(gps_map, width=1000, height=600)

            except Exception as e:
                st.error(f"❌ Error processing files: {e}")
        else:
            st.error("❌ ZIP must contain a folder with Accelerometer.csv, Gyroscope.csv, and Location.csv")
