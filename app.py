import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import joblib
import pydeck as pdk
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
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        try:
            df_features = extract_features_from_zip(tmpdir)

            if df_features.empty:
                st.error("❌ No features extracted. Check file naming and content.")
            else:
                X_input = df_features.drop(columns=["start_time", "end_time", "latitude", "longitude"])

                # Predict
                if "XGBoost" in model_choice:
                    y_pred_encoded = model.predict(X_input)
                    y_pred = label_encoder.inverse_transform(y_pred_encoded)
                else:
                    y_pred = model.predict(X_input)

                df_features["Prediction"] = y_pred

                st.success("✅ Prediction completed.")
                st.subheader("Prediction Results")
                st.dataframe(df_features)

                # Summary
                st.subheader("Roughness Summary")
                summary = df_features["Prediction"].value_counts().reset_index()
                summary.columns = ["Surface Condition", "Segment Count"]
                st.dataframe(summary)

                # Chart toggle
                chart_type = st.sidebar.radio("Choose Chart Type", ["Bar Chart", "Pie Chart"])
                st.subheader("Roughness Distribution")
                if chart_type == "Bar Chart":
                    st.bar_chart(summary.set_index("Surface Condition"))
                else:
                    st.plotly_chart(
                        pd.DataFrame({
                            "labels": summary["Surface Condition"],
                            "values": summary["Segment Count"]
                        }).plot.pie(y="values", labels=summary["Surface Condition"], autopct="%1.1f%%", legend=False).figure,
                        use_container_width=True
                    )

                # Map Visualization
                st.subheader("Map of Segments")
                color_map = {"Smooth": [0, 255, 0], "Fair": [255, 255, 0], "Rough": [255, 0, 0]}
                df_features["color"] = df_features["Prediction"].map(color_map)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_features,
                    get_position="[longitude, latitude]",
                    get_color="color",
                    get_radius=30,
                    pickable=True
                )

                view_state = pdk.ViewState(
                    latitude=df_features["latitude"].mean(),
                    longitude=df_features["longitude"].mean(),
                    zoom=14,
                    pitch=0
                )

                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

        except Exception as e:
            st.error(f"❌ Feature extraction or prediction error: {e}")
