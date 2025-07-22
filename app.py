
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import tempfile
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.preprocessing import LabelEncoder

# Load models and label encoder
label_encoder = joblib.load("label_encoder.pkl")
models = {
    "SVM 4": joblib.load("svm4_model.pkl"),
    "RF 1": joblib.load("rf1_model.pkl"),
    "XGB 2": joblib.load("xgb2_model.pkl")
}

# Optional fix to handle unseen label error
int_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}

st.set_page_config(page_title="Road Roughness Detection", layout="wide")
st.title("🚗 Road Surface Roughness Detection App")

# Sidebar model selection
st.sidebar.title("Model Selector")
model_name = st.sidebar.selectbox("Choose a model to use", list(models.keys()))
model = models[model_name]

uploaded_file = st.file_uploader("Upload ZIP file with Accelerometer, Gyroscope, and Location CSVs", type="zip")

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        csv_files = [os.path.join(tmpdir, file) for file in os.listdir(tmpdir) if file.endswith(".csv")]
        location_file = [f for f in csv_files if "location" in f.lower()][0]
        sensor_file = [f for f in csv_files if "feature" in f.lower()][0]

        # Load data
        df_features = pd.read_csv(sensor_file)
        df_location = pd.read_csv(location_file)

        drop_cols = ["label", "readable_time", "start_time", "end_time", "estimated_iri"]
        X = df_features.drop(columns=[col for col in drop_cols if col in df_features.columns], errors="ignore")

        # Predict
        y_pred_encoded = model.predict(X)
        y_pred = [int_to_label.get(i, str(i)) for i in y_pred_encoded]
        df_features["Prediction"] = y_pred

        # Display pie chart
        st.subheader("🧾 Prediction Summary")
        pred_count = df_features["Prediction"].value_counts().reset_index()
        pred_count.columns = ["Label", "Count"]
        st.dataframe(pred_count)

        st.subheader("📍 Map View of Segments")
        df_map = pd.merge(df_features, df_location, left_index=True, right_index=True)

        m = folium.Map(location=[df_map["latitude"].mean(), df_map["longitude"].mean()], zoom_start=16, control_scale=True)
        marker_cluster = MarkerCluster().add_to(m)

        color_map = {"Smooth": "green", "Fair": "orange", "Rough": "red"}
        for i, row in df_map.iterrows():
            folium.CircleMarker(
                location=(row["latitude"], row["longitude"]),
                radius=4,
                color=color_map.get(row["Prediction"], "blue"),
                fill=True,
                fill_color=color_map.get(row["Prediction"], "blue"),
                fill_opacity=0.8,
                popup=f"{row['Prediction']}"
            ).add_to(marker_cluster)

        st_data = st_folium(m, width=700, height=500)
