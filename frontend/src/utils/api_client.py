import requests

class AIEthicsClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"X-API-Key": "secret-token"}

    def analyze_dataset(self, file_bytes):
        """
        Sends the dataset file to the backend for analysis.
        """
        url = f"{self.base_url}/analyze/dataset"
        files = {"file": file_bytes}
        response = requests.post(url, files=files, headers=self.headers)
        return response.json()

    def analyze_fairness(self, file_bytes):
        """
        Sends the dataset file to compute fairness metrics.
        """
        url = f"{self.base_url}/analyze/fairness"
        files = {"file": file_bytes}
        response = requests.post(url, files=files, headers=self.headers)
        return response.json()

    def mitigate_bias(self, file_bytes):
        """
        Sends the dataset file to mitigate bias.
        """
        url = f"{self.base_url}/mitigate"
        files = {"file": file_bytes}
        response = requests.post(url, files=files, headers=self.headers)
        return response.json()

    def analyze_model(self, model_bytes):
        """
        Sends the model file to analyze its performance (model audit).
        """
        url = f"{self.base_url}/analyze/model"
        files = {"file": model_bytes}
        response = requests.post(url, files=files, headers=self.headers)
        return response.json()

    def explain_shap(self, model_bytes):
        """
        Sends the model file to generate a SHAP explanation.
        """
        url = f"{self.base_url}/explain/shap"
        files = {"file": model_bytes}
        response = requests.post(url, files=files, headers=self.headers)
        return response.json()

    def explain_lime(self, model_bytes):
        """
        Sends the model file to generate a LIME explanation.
        """
        url = f"{self.base_url}/explain/lime"
        files = {"file": model_bytes}
        response = requests.post(url, files=files, headers=self.headers)
        return response.json()

    def analyze_privacy(self, model_bytes, train_bytes):
        """
        Sends both a model file and a training dataset to evaluate privacy.
        """
        url = f"{self.base_url}/analyze/privacy"
        files = {"model": model_bytes, "train": train_bytes}
        response = requests.post(url, files=files, headers=self.headers)
        return response.json()
