
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide"
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "logistic_regression_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "models", "scaler.pkl"
)

# Load trained model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# Application title
st.title("🎗️ Breast Cancer Classification")
st.write(
    "Machine Learning based Breast Cancer Prediction "
    "using Logistic Regression."
)

st.info(
    "Upload a CSV file containing the 30 breast cancer "
    "measurement features."
)

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write("Dataset Shape:", data.shape)
    st.dataframe(data.head())

    # Separate actual labels if available
    actual_labels = None

    if "target" in data.columns:
        actual_labels = data["target"]
        X_input = data.drop(columns=["target"])
    else:
        X_input = data.copy()

    # Get expected feature names
    expected_features = list(scaler.feature_names_in_)

    # Validate columns
    missing_features = [
        col for col in expected_features
        if col not in X_input.columns
    ]

    extra_features = [
        col for col in X_input.columns
        if col not in expected_features
    ]

    if missing_features or extra_features:

        st.error("Dataset feature columns do not match.")

        if missing_features:
            st.write("Missing columns:", missing_features)

        if extra_features:
            st.write("Unexpected columns:", extra_features)

    elif X_input.isnull().values.any():

        st.error("Dataset contains missing values.")

    else:

        # Arrange features in training order
        X_input = X_input[expected_features]

        # Apply saved scaler
        X_scaled = scaler.transform(X_input)

        # Generate predictions
        predictions = model.predict(X_scaled)

        # Map class labels
        class_names = {
            0: "Malignant",
            1: "Benign"
        }

        results = pd.DataFrame({
            "Actual": (
                actual_labels.values
                if actual_labels is not None
                else ["Not provided"] * len(predictions)
            ),
            "Predicted": [
                class_names[pred] for pred in predictions
            ]
        })

        st.subheader("Prediction Results")
        st.dataframe(results)

        # Summary
        malignant_count = sum(predictions == 0)
        benign_count = sum(predictions == 1)

        col1, col2 = st.columns(2)

        col1.metric("Malignant Predictions", malignant_count)
        col2.metric("Benign Predictions", benign_count)

        # Download predictions
        output = pd.DataFrame({
            "Predicted_Label": predictions,
            "Predicted_Diagnosis": [
                class_names[pred] for pred in predictions
            ]
        })

        csv = output.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Results",
            data=csv,
            file_name="breast_cancer_predictions.csv",
            mime="text/csv"
        )

        st.success("Predictions generated successfully!")
