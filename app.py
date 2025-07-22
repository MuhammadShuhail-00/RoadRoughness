import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from model_utils import extract_features_from_files, predict_with_model

# Load models
svm_model = joblib.load("models/iri_svm4.pkl")
rf_model = joblib.load("models/iri_rf1.pkl")
xgb_model = joblib.load("models/iri_xgb2.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# App title
st.title("🚗 Road Surface Roughness Detection App")
st.markdown("Upload your accelerometer, gyroscope, and location data in a ZIP file.")

# Sidebar
st.sidebar.title("Model Selector")
model_name = st.sidebar.radio("Choose model:", ("SVM 4", "Random Forest 1", "XGBoost 2"))

# File uploader
uploaded_zip = st.file_uploader("Upload ZIP file with Accelerometer, Gyroscope, and Location CSVs", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        tmpdir = "temp_zip"
        zip_ref.extractall(tmpdir)

    # List CSVs from extracted zip
    csv_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".csv")]

    try:
        accel_file = [f for f in csv_files if "accel" in f.lower()][0]
        gyro_file = [f for f in csv_files if "gyro" in f.lower()][0]
        gps_file = [f for f in csv_files if "location" in f.lower() or "gps" in f.lower()][0]
    except IndexError:
        st.error("❌ ZIP must contain CSVs with 'accel', 'gyro', and 'location' in the filename.")
        st.stop()

    # Extract features
    st.info("🔄 Extracting features from sensor data...")
    df_features = extract_features_from_files(accel_file, gyro_file, gps_file)

    # Load selected model
    if model_name == "SVM 4":
        model = svm_model
    elif model_name == "Random Forest 1":
        model = rf_model
    elif model_name == "XGBoost 2":
        model = xgb_model

    # Predict
    st.success(f"✅ Using {model_name} to classify road condition...")
    y_pred_encoded = model.predict(df_features.drop(columns=["readable_time"], errors="ignore"))
    try:
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    except Exception as e:
        st.warning("⚠️ Could not decode labels. Showing encoded predictions instead.")
        y_pred = y_pred_encoded

    df_features["Predicted Road Condition"] = y_pred

    # Show output
    st.subheader("📊 Prediction Results")
    st.dataframe(df_features[["readable_time", "Predicted Road Condition"]] if "readable_time" in df_features.columns else df_features[["Predicted Road Condition"]])

    # Summary counts
    st.write("### 🔍 Summary")
    st.write(df_features["Predicted Road Condition"].value_counts())

    # Clean up temp files
    for f in csv_files:
        os.remove(f)
    os.rmdir(tmpdir)
