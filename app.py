import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
import joblib
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(layout="wide")
st.title("Road Surface Roughness Detection")

# Load models
svm_model = joblib.load("models/iri_svm4.pkl")
rf_model = joblib.load("models/iri_rf1.pkl")
xgb_model = joblib.load("models/iri_xgb2.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Dropdown to select model
model_name = st.sidebar.selectbox("Select Model", ("SVM (Best)", "Random Forest (RF1)", "XGBoost (XGB2)"))
model_dict = {
    "SVM (Best)": svm_model,
    "Random Forest (RF1)": rf_model,
    "XGBoost (XGB2)": xgb_model
}
selected_model = model_dict[model_name]

st.sidebar.markdown("---")
st.sidebar.title("Upload ZIP file with Accelerometer, Gyroscope, and Location CSVs")

uploaded_file = st.sidebar.file_uploader("Upload ZIP", type="zip")

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        csv_files = [os.path.join(tmpdir, file) for file in os.listdir(tmpdir) if file.endswith(".csv")]
        feature_file = [f for f in csv_files if "feature" in f.lower()]

        if feature_file:
            df = pd.read_csv(feature_file[0])
            st.success("✅ Feature file successfully loaded!")

            # Preprocess input features
            drop_cols = ["label", "readable_time", "start_time", "end_time", "estimated_iri"]
            X = df.drop(columns=[col for col in drop_cols if col in df.columns])

            # Make predictions
            y_pred_encoded = selected_model.predict(X)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            df['Predicted Label'] = y_pred

            # Show prediction table
            st.subheader("Prediction Table")
            st.dataframe(df[['readable_time', 'Predicted Label']])

            # Plot pie chart
            st.subheader("Prediction Summary")
            summary = df['Predicted Label'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(summary, labels=summary.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

            # Map road segments
            if 'latitude' in df.columns and 'longitude' in df.columns:
                st.subheader("Map View of Road Segments")
                map_center = [df['latitude'].mean(), df['longitude'].mean()]
                m = folium.Map(location=map_center, zoom_start=16)
                color_dict = {"Smooth": "green", "Fair": "orange", "Rough": "red"}

                for _, row in df.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=4,
                        color=color_dict.get(row['Predicted Label'], "blue"),
                        fill=True,
                        fill_color=color_dict.get(row['Predicted Label'], "blue"),
                        fill_opacity=0.7,
                        popup=f"{row['readable_time']} - {row['Predicted Label']}"
                    ).add_to(m)

                st_data = st_folium(m, width=900, height=500)
            else:
                st.warning("Location data not available for mapping.")
        else:
            st.error("No feature CSV found in ZIP. Make sure the file includes 'feature' in the name.")
