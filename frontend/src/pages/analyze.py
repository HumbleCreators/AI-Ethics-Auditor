import streamlit as st
import requests

# Constants for the backend API
API_BASE_URL = "http://127.0.0.1:8000"
API_KEY = "secret-token"

def run_analysis():
    st.title("Data & Model Analysis")
    st.write("Use this page to analyze datasets and models for bias and fairness.")

    # --- Dataset Bias Analysis Section ---
    st.header("Dataset Bias Analysis")
    dataset_file = st.file_uploader("Upload a CSV file (must include a 'label' column)", type=["csv"], key="dataset")
    if dataset_file is not None:
        content = dataset_file.read()
        st.info("Uploading dataset and analyzing bias...")
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze/dataset",
                headers={"X-API-Key": API_KEY},
                files={"file": ("dataset.csv", content, "text/csv")}
            )
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get("success"):
                    st.success("Dataset bias analysis complete!")
                    st.json(json_data["data"]["bias_analysis"])
                else:
                    st.error(json_data.get("error", "Unknown error during dataset analysis."))
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # --- Model Bias Analysis Section ---
    st.header("Model Bias Analysis")
    model_file = st.file_uploader("Upload a pickled model (.pkl)", type=["pkl"], key="model")
    if model_file is not None:
        model_content = model_file.read()
        st.info("Uploading model and analyzing bias...")
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze/model",
                headers={"X-API-Key": API_KEY},
                files={"file": ("model.pkl", model_content, "application/octet-stream")}
            )
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get("success"):
                    st.success("Model bias analysis complete!")
                    st.json(json_data["data"]["model_bias"])
                else:
                    st.error(json_data.get("error", "Unknown error during model analysis."))
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    run_analysis()
