import joblib

# Load models
svm_model = joblib.load("svm4_model.pkl")
rf_model = joblib.load("rf1_model.pkl")
xgb_model = joblib.load("xgb2_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # <-- NEW

# Sidebar to choose model
st.sidebar.title("Model Selector")
model_choice = st.sidebar.radio("Choose Model", ["SVM", "Random Forest", "XGBoost"])

# Choose correct model
if model_choice == "SVM":
    model = svm_model
elif model_choice == "Random Forest":
    model = rf_model
else:
    model = xgb_model

# Make prediction
X_pred = features_df.drop(columns=["start_time", "end_time", "latitude", "longitude"], errors="ignore")
pred_encoded = model.predict(X_pred)

# Decode if using XGB
if model_choice == "XGBoost":
    pred_labels = label_encoder.inverse_transform(pred_encoded)
else:
    pred_labels = pred_encoded

features_df["prediction"] = pred_labels
