import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import tempfile
import os
import joblib
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from datetime import timedelta

# Load model and scaler
model = joblib.load("iri_rf_classifier_cleaned.pkl")

# UI layout
st.set_page_config(layout="wide")
st.title("🚗 Road Surface Roughness Detection App")
st.markdown("Upload a ZIP file containing Accelerometer.csv, Gyroscope.csv, and Location.csv")

uploaded_zip = st.file_uploader("Upload Sensor Data (ZIP format)", type="zip")

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Detect folder (in case it's nested like 'sensor data/')
        all_files = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                all_files.append(os.path.join(root, file))

        accel_file = next((f for f in all_files if "accelerometer" in f.lower()), None)
        gyro_file = next((f for f in all_files if "gyroscope" in f.lower()), None)
        gps_file = next((f for f in all_files if "location" in f.lower()), None)

        if accel_file and gyro_file and gps_file:
            accel = pd.read_csv(accel_file)
            gyro = pd.read_csv(gyro_file)
            gps = pd.read_csv(gps_file)

            # Preprocess timestamps
            for df in [accel, gyro, gps]:
                df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                if df["timestamp"].max() > 1e15:
                    df["timestamp"] /= 1_000_000
                df.sort_values("timestamp", inplace=True)
                df.drop(columns=[col for col in df.columns if "elapsed" in col.lower()], inplace=True)

            # Merge
            df = pd.merge_asof(accel, gyro, on="timestamp", direction="nearest", tolerance=50)
            df = pd.merge_asof(df, gps, on="timestamp", direction="nearest", tolerance=200)
            df.dropna(inplace=True)
            df.insert(0, "readable_time", pd.to_datetime(df["timestamp"], unit='ms') + timedelta(hours=8))

            # Rename columns
            df.rename(columns={
                'x_x': 'accel_x',
                'y_x': 'accel_y',
                'z_x': 'accel_z',
                'x_y': 'gyro_x',
                'y_y': 'gyro_y',
                'z_y': 'gyro_z'
            }, inplace=True)

            st.success(f"Merged and cleaned {len(df)} rows of data.")

            ### Feature Extraction
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
                        'gyro_x_std': window['gyro_x'].std()
                    }
                    segments.append(features)

            features_df = pd.DataFrame(segments)
            features_df["estimated_iri"] = (
                2.0 * features_df["rms_accel_y"] +
                0.03 * features_df["mean_speed"] +
                0.1
            )

            # Label assignment
            def final_label(row):
                if row["mean_speed"] < 5:
                    return "Idle"
                if row["gyro_y_std"] > 0.4:
                    return "Turning"
                if row["peak2peak_accel_y"] > 10 and row["mean_speed"] > 30:
                    return "Sharp Accel/Brake"
                if (
                    8 < row["mean_speed"] < 22 and
                    row["peak2peak_accel_y"] > 8.0 and
                    row["rms_accel_y"] > 1.5 and
                    row["std_accel_y"] < 1.5
                ):
                    return "Speed Bump"
                iri = row["estimated_iri"]
                if iri < 2.0:
                    return "Smooth"
                elif iri < 3.5:
                    return "Fair"
                else:
                    return "Rough"

            features_df["label"] = features_df.apply(final_label, axis=1)
            features_df = features_df[~features_df["label"].isin(["Idle", "Turning", "Sharp Accel/Brake"])]
            features_df["readable_time"] = pd.to_datetime(features_df["start_time"], unit='ms') + timedelta(hours=8)

            # Load scaler and predict
            scaler = joblib.load("iri_svm_scaler.pkl")
            X_pred = features_df.drop(columns=["label", "start_time", "end_time", "estimated_iri", "readable_time"], errors="ignore")
            X_scaled = scaler.transform(X_pred)
            features_df["prediction"] = model.predict(X_scaled)

            ### Display
            st.subheader("📊 Segment-wise Prediction")
            st.dataframe(features_df[["readable_time", "mean_speed", "prediction"]])

            st.subheader("📈 Prediction Distribution")
            fig, ax = plt.subplots()
            features_df["prediction"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

            # Display Map
            if "latitude" in df.columns and "longitude" in df.columns:
                st.subheader("🗺️ Map View of Road Segments")
                m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()],
                               zoom_start=15, tiles="CartoDB dark_matter")
                color_map = {"Smooth": "green", "Fair": "blue", "Rough": "red"}

                for _, row in features_df.iterrows():
                    segment_df = df[df["timestamp"] >= row["start_time"]]
                    if not segment_df.empty:
                        lat = segment_df["latitude"].iloc[0]
                        lon = segment_df["longitude"].iloc[0]
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=4,
                            color=color_map.get(row["prediction"], "gray"),
                            fill=True,
                            fill_opacity=0.9,
                            popup=row["prediction"]
                        ).add_to(m)

                st_folium(m, width=1000, height=600)
        else:
            st.error("The ZIP must contain 3 files: Accelerometer.csv, Gyroscope.csv, and Location.csv.")
