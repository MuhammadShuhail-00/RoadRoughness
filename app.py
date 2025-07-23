# app.py
import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import joblib
from model_utils import extract_features_from_zip
from sklearn.preprocessing import LabelEncoder

# Load models
MODELS = {
    "XGBoost (Model 2)": joblib.load("models/xgb2_model.pkl"),
    "Random Forest (Model 1)": joblib.load("models/rf1_model.pkl"),
    "SVM (Model 4)": joblib.load("models/svm4_model.pkl")
}

# Load label encoder (only used for XGBoost)
label_encoder = joblib.load("models/label_encoder.pkl")

# UI Setup
st.set_page_config(page_title="Road Surface Roughness Detection", layout="wide")
st.title("Road Surface Roughness Detection App")

# Sidebar options
model_choice = st.sidebar.selectbox("Select a model", list(MODELS.keys()))
show_map = st.sidebar.checkbox("Show Map", value=True)
show_pie = st.sidebar.checkbox("Show Pie Chart", value=True)
show_bar = st.sidebar.checkbox("Show Bar Chart", value=True)
show_true = st.sidebar.checkbox("Show IRI & True Label", value=True)

model = MODELS[model_choice]

# File upload
st.markdown("Upload ZIP file containing Accelerometer, Gyroscope, and Location CSVs")
uploaded_zip = st.file_uploader("Upload your ZIP file", type="zip")

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        try:
            df_features = extract_features_from_zip(tmpdir)

            if df_features.empty:
                st.error("❌ No valid features extracted.")
            else:
                # Prediction
                X_input = df_features.drop(columns=["start_time", "end_time", "latitude", "longitude"], errors="ignore")

                if "XGBoost" in model_choice:
                    y_pred_encoded = model.predict(X_input)
                    y_pred = label_encoder.inverse_transform(y_pred_encoded)
                else:
                    y_pred = model.predict(X_input)

                df_features['Prediction'] = y_pred

                # --- Add IRI estimation and label ---
                df_features["estimated_iri"] = 5.132 * df_features["rms_accel_y"] + 1.112

                def iri_to_label(iri):
                    if iri <= 2.0:
                        return "Smooth"
                    elif iri <= 3.5:
                        return "Fair"
                    else:
                        return "Rough"

                df_features["True_Label"] = df_features["estimated_iri"].apply(iri_to_label)

                st.success("✅ Prediction completed.")

                st.subheader("Prediction Results")
                if show_true:
                    st.dataframe(df_features)
                else:
                    st.dataframe(df_features.drop(columns=["True_Label", "estimated_iri"], errors="ignore"))

                # Pie Chart
                if show_pie:
                    st.subheader("Road Condition Distribution (Pie Chart)")
                    pie_data = df_features['Prediction'].value_counts()
                    st.pyplot(pie_data.plot.pie(autopct='%1.1f%%', figsize=(5, 5), title="Predicted Conditions").get_figure())

                # Bar Chart
                if show_bar:
                    st.subheader("Road Condition Summary (Bar Chart)")
                    bar_data = df_features['Prediction'].value_counts().reset_index()
                    bar_data.columns = ['Surface Condition', 'Segment Count']
                    st.bar_chart(bar_data.set_index("Surface Condition"))

                # Map View
                if show_map and "latitude" in df_features.columns and "longitude" in df_features.columns:
                    import folium
                    from streamlit_folium import st_folium

                    st.subheader("Map View of Predicted Segments")

                    color_map = {
                        "Smooth": "green",
                        "Fair": "blue",
                        "Rough": "red"
                    }

                    m = folium.Map(location=[
                        df_features["latitude"].mean(),
                        df_features["longitude"].mean()
                    ], zoom_start=15, tiles="CartoDB positron")

                    for _, row in df_features.iterrows():
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=4,
                            color=color_map.get(row["Prediction"], "gray"),
                            fill=True,
                            fill_opacity=0.8,
                            tooltip=folium.Tooltip(row["Prediction"])
                        ).add_to(m)

                    st_folium(m, height=500, width=1100)

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
