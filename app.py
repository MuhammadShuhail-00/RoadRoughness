# app.py
import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import joblib
from model_utils import merge_and_extract_features, predict_with_model

from sklearn.preprocessing import LabelEncoder

# Load label encoder and models
label_encoder = joblib.load("models/label_encoder.pkl")
MODELS = {
    "SVM (Model 4)": joblib.load("models/svm4_model.pkl"),
    "Random Forest (Model 1)": joblib.load("models/rf1_model.pkl"),
    "XGBoost (Model 2)": joblib.load("models/xgb2_model.pkl")
}

# Streamlit UI setup
st.set_page_config(page_title="Road Surface Roughness Detection", layout="wide")
st.title("Road Surface Roughness Detection")

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select a model", list(MODELS.keys()))
model = MODELS[model_choice]

# Upload ZIP file
st.markdown("Upload ZIP file with Accelerometer, Gyroscope, and Location CSVs")
uploaded_zip = st.file_uploader("Upload your road session ZIP file", type="zip")

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        def find_file(keywords):
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    fname = file.lower()
                    if all(k in fname for k in keywords):
                        return os.path.join(root, file)
            return None

        accel_path = find_file(["accelerometer"])
        gyro_path = find_file(["gyroscope"])
        gps_path = find_file(["location"])

        if not accel_path or not gyro_path or not gps_path:
            st.error("❌ Required files not found: Accelerometer, Gyroscope, Location CSVs.")
        else:
            try:
                accel = pd.read_csv(accel_path)
                gyro = pd.read_csv(gyro_path)
                gps = pd.read_csv(gps_path)

                df_features = merge_and_extract_features(accel, gyro, gps)

                if df_features.empty:
                    st.error("❌ No features extracted. Please check data quality and file format.")
                else:
                    predictions, df_pred = predict_with_model(model, df_features, label_encoder)
                    st.success("✅ Prediction completed.")

                    st.subheader("Prediction Results")
                    st.dataframe(df_pred)

                    st.subheader("Roughness Summary")
                    summary = df_pred['Prediction'].value_counts().reset_index()
                    summary.columns = ['Surface Condition', 'Segment Count']
                    st.dataframe(summary)

                    st.subheader("Roughness Distribution")
                    st.bar_chart(summary.set_index('Surface Condition'))

            except Exception as e:
                st.error(f"❌ Feature extraction or prediction error: {e}")
