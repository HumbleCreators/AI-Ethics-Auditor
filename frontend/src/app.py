import streamlit as st
import requests

# Adjust this to match your backend's URL and port
API_BASE_URL = "http://127.0.0.1:8000"
API_KEY = "secret-token"

def main():
    st.title("AI Ethics Auditor")
    st.write("Welcome to the AI Ethics Auditor frontend demo using Streamlit!")

    # --------------------------------------------
    # Example 1: Dataset Bias Analysis
    # --------------------------------------------
    st.header("Analyze Dataset Bias")
    dataset_file = st.file_uploader("Upload a CSV with a 'label' column", type=["csv"])
    if dataset_file is not None:
        # Convert file to bytes for backend
        content = dataset_file.read()

        # Make POST request to the backend's dataset analysis endpoint
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze/dataset",
                headers={"X-API-Key": API_KEY},
                files={"file": ("dataset.csv", content, "text/csv")}
            )

            if response.status_code == 200:
                # Our backend uses a consistent JSON structure with keys: success, data, error
                json_data = response.json()
                if json_data.get("success"):
                    st.subheader("Bias Analysis Results")
                    st.json(json_data["data"]["bias_analysis"])
                else:
                    st.error(json_data.get("error"))
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(str(e))

    # --------------------------------------------
    # Example 2: Model Bias Analysis
    # --------------------------------------------
    st.header("Analyze Model Bias")
    model_file = st.file_uploader("Upload a pickled model (.pkl)", type=["pkl"])
    if model_file is not None:
        # Convert file to bytes for backend
        model_content = model_file.read()

        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze/model",
                headers={"X-API-Key": API_KEY},
                files={"file": ("model.pkl", model_content, "application/octet-stream")}
            )
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get("success"):
                    st.subheader("Model Bias Results")
                    st.json(json_data["data"]["model_bias"])
                else:
                    st.error(json_data.get("error"))
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(str(e))

    # --------------------------------------------
    # Additional sections can be added similarly:
    # - Fairness Analysis
    # - Privacy Analysis
    # - Explainability (SHAP, LIME)
    # - Mitigation
    # --------------------------------------------

if __name__ == "__main__":
    main()
