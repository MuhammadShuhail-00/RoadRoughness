# app.py
import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from model_utils import extract_features_from_files, predict_with_model


# Load label encoder
label_encoder = joblib.load("models/label_encoder.pkl")

# Load models
MODELS = {
    "SVM (Model 4)": joblib.load("models/svm4_model.pkl"),
    "Random Forest (Model 1)": joblib.load("models/rf1_model.pkl"),
    "XGBoost (Model 2)": joblib.load("models/xgb2_model.pkl")
}

# Streamlit UI
st.set_page_config(page_title="Road Surface Roughness Detection", layout="wide")
st.title("Road Surface Roughness Detection")

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select a model", list(MODELS.keys()))
model = MODELS[model_choice]

# File uploader
st.markdown("Upload ZIP file with Accelerometer, Gyroscope, and Location CSVs")
uploaded_zip = st.file_uploader("Upload your road session ZIP file", type="zip")

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Extract features from the extracted files
        try:
            df_features = extract_features_from_files(tmpdir)

            if df_features.empty:
                st.error("❌ No features extracted. Check file naming and content.")
            else:
                # Predict and display results
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
            st.error(f"❌ Error processing file: {e}")
