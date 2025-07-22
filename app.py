import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import joblib
from model_utils import extract_features, merge_sensor_data
from sklearn.preprocessing import LabelEncoder

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

# Upload ZIP
st.markdown("Upload ZIP file with Accelerometer, Gyroscope, and Location CSVs")
zip_file = st.file_uploader("Upload your road session ZIP file", type="zip")

if zip_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        try:
            df_merged = merge_sensor_data(tmpdir)

            if df_merged.empty:
                st.error("❌ Merged data is empty. Check file content.")
            else:
                df_features = extract_features(df_merged)

                if df_features.empty:
                    st.error("❌ Feature extraction failed. Check sensor data.")
                else:
                    # Predict
                    y_pred_encoded = model.predict(df_features.drop(columns=['start_time', 'end_time', 'latitude', 'longitude']))
                    y_pred = label_encoder.inverse_transform(y_pred_encoded)
                    df_features['Prediction'] = y_pred

                    st.success("✅ Prediction completed.")

                    # Show table
                    st.subheader("Prediction Results")
                    st.dataframe(df_features[['start_time', 'end_time', 'latitude', 'longitude', 'Prediction']])

                    # Summary
                    st.subheader("Roughness Summary")
                    summary = df_features['Prediction'].value_counts().reset_index()
                    summary.columns = ['Surface Condition', 'Segment Count']
                    st.dataframe(summary)

                    # Chart
                    st.subheader("Roughness Distribution")
                    st.bar_chart(summary.set_index('Surface Condition'))

        except Exception as e:
            st.error(f"❌ Feature extraction or prediction error: {e}")
