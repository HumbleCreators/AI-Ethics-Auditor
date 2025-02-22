document.addEventListener("DOMContentLoaded", function () {
  console.log("Frontend assets loaded.");

  // --- Dataset Analysis Section ---
  const datasetForm = document.getElementById("dataset-form");
  const datasetFileInput = document.getElementById("dataset-file");
  const datasetResultBox = document.getElementById("dataset-result");
  const datasetJson = document.getElementById("dataset-json");
  const datasetChartDiv = document.getElementById("dataset-chart");

  const fairnessResultBox = document.getElementById("fairness-result");
  const fairnessJson = document.getElementById("fairness-json");
  const fairnessChartDiv = document.getElementById("fairness-chart");

  const mitigationResultBox = document.getElementById("mitigation-result");
  const mitigationJson = document.getElementById("mitigation-json");

  // --- Model Audit Section ---
  const modelForm = document.getElementById("model-form");
  const modelFileInput = document.getElementById("model-file");
  const trainingFileInput = document.getElementById("training-file");
  const modelResultBox = document.getElementById("model-result");
  const modelJson = document.getElementById("model-json");
  const confusionMatrixChartDiv = document.getElementById("confusion-matrix-chart");

  const explainResultBox = document.getElementById("explain-result");
  const explainJson = document.getElementById("explain-json");

  const privacyResultBox = document.getElementById("privacy-result");
  const privacyJson = document.getElementById("privacy-json");

  // Helper to display JSON in a <pre> element.
  function displayJSON(element, data) {
    element.textContent = JSON.stringify(data, null, 2);
  }

  // Helper to show result box
  function showResultBox(box) {
    box.style.display = "block";
  }

  // --- Additional Visualization Functions ---

  // Plot a pie chart for class distribution from dataset analysis.
  function plotDatasetPieChart(datasetData) {
    if (datasetData && datasetData.class_counts) {
      const labels = Object.keys(datasetData.class_counts);
      const values = Object.values(datasetData.class_counts);
      const data = [{
        labels: labels,
        values: values,
        type: 'pie',
        textinfo: 'label+percent',
        insidetextorientation: 'radial'
      }];
      const layout = {
        title: 'Class Distribution'
      };
      Plotly.newPlot(datasetChartDiv, data, layout);
    }
  }

  // Plot a bar chart for fairness metrics (group accuracies).
  function plotFairnessChart(fairnessData) {
    if (fairnessData && fairnessData.group_accuracies) {
      const groups = Object.keys(fairnessData.group_accuracies);
      const accuracies = Object.values(fairnessData.group_accuracies);
      const trace = {
        x: groups,
        y: accuracies,
        type: 'bar'
      };
      const layout = {
        title: 'Group Accuracies',
        xaxis: { title: 'Group' },
        yaxis: { title: 'Accuracy' }
      };
      Plotly.newPlot(fairnessChartDiv, [trace], layout);
    }
  }

  // Plot a heatmap for the confusion matrix.
  function plotConfusionMatrix(confusionMatrix) {
    if (confusionMatrix) {
      const data = [{
        z: confusionMatrix,
        type: 'heatmap',
        colorscale: 'Viridis'
      }];
      const layout = {
        title: 'Confusion Matrix',
        xaxis: { title: 'Predicted Label' },
        yaxis: { title: 'True Label' }
      };
      Plotly.newPlot(confusionMatrixChartDiv, data, layout);
    }
  }

  // Plot a gauge chart for the differential privacy epsilon value.
  function plotPrivacyGauge(privacyData) {
    if (privacyData && privacyData.differential_privacy_epsilon !== undefined) {
      const epsilon = privacyData.differential_privacy_epsilon;
      const data = [{
        type: "indicator",
        mode: "gauge+number",
        value: epsilon,
        title: { text: "Differential Privacy Epsilon" },
        gauge: {
          axis: { range: [null, 10] },
          steps: [
            { range: [0, 2], color: "lightgreen" },
            { range: [2, 5], color: "yellow" },
            { range: [5, 10], color: "red" }
          ],
          threshold: {
            line: { color: "black", width: 4 },
            thickness: 0.75,
            value: epsilon
          }
        }
      }];
      const layout = { width: 300, height: 250, margin: { t: 0, b: 0 } };
      Plotly.newPlot(privacyResultBox.querySelector("div"), data, layout);
    }
  }

  // Helper function for API calls.
  async function callEndpoint(endpoint, formData) {
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "X-API-Key": "secret-token"
        },
        body: formData
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Error calling " + endpoint, error);
      return { error: error.message };
    }
  }

  // --- Handle Dataset Analysis ---
  datasetForm.addEventListener("submit", async function (event) {
    event.preventDefault();
    const file = datasetFileInput.files[0];
    if (!file) {
      alert("Please select a dataset file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    // Endpoints for dataset analysis, fairness, and mitigation.
    const endpoints = [
      "http://127.0.0.1:8000/analyze/dataset",
      "http://127.0.0.1:8000/analyze/fairness",
      "http://127.0.0.1:8000/mitigate"
    ];

    try {
      const [datasetRes, fairnessRes, mitigateRes] = await Promise.all(
        endpoints.map(endpoint => callEndpoint(endpoint, formData))
      );
      showResultBox(datasetResultBox);
      displayJSON(datasetJson, datasetRes);
      showResultBox(fairnessResultBox);
      displayJSON(fairnessJson, fairnessRes);
      showResultBox(mitigationResultBox);
      displayJSON(mitigationJson, mitigateRes);

      // Render additional visualizations.
      plotDatasetPieChart(datasetRes);
      if (!fairnessRes.error && fairnessRes.group_accuracies) {
        plotFairnessChart(fairnessRes);
      }
    } catch (error) {
      console.error("Error processing dataset analysis:", error);
    }
  });

  // --- Handle Model Audit ---
  modelForm.addEventListener("submit", async function (event) {
    event.preventDefault();
    const modelFile = modelFileInput.files[0];
    const trainingFile = trainingFileInput.files[0];

    if (!modelFile || !trainingFile) {
      alert("Please select both a model file and a training dataset file.");
      return;
    }
    
    // For endpoints that need only the model file.
    const formDataModel = new FormData();
    formDataModel.append("file", modelFile);

    // For privacy, need both model and training data.
    const formDataPrivacy = new FormData();
    formDataPrivacy.append("model", modelFile);
    formDataPrivacy.append("train", trainingFile);

    const endpoints = [
      "http://127.0.0.1:8000/analyze/model",
      "http://127.0.0.1:8000/explain/shap",
      "http://127.0.0.1:8000/explain/lime"
    ];
    
    try {
      const [modelRes, shapRes, limeRes, privacyRes] = await Promise.all([
        callEndpoint("http://127.0.0.1:8000/analyze/model", formDataModel),
        callEndpoint("http://127.0.0.1:8000/explain/shap", formDataModel),
        callEndpoint("http://127.0.0.1:8000/explain/lime", formDataModel),
        callEndpoint("http://127.0.0.1:8000/analyze/privacy", formDataPrivacy)
      ]);

      showResultBox(modelResultBox);
      displayJSON(modelJson, modelRes);
      if (!modelRes.error && modelRes.confusion_matrix) {
        plotConfusionMatrix(modelRes.confusion_matrix);
      }

      showResultBox(explainResultBox);
      displayJSON(explainJson, { SHAP: shapRes, LIME: limeRes });

      showResultBox(privacyResultBox);
      displayJSON(privacyJson, privacyRes);
      // Render privacy gauge if epsilon is present.
      if (!privacyRes.error && privacyRes.differential_privacy_epsilon !== undefined) {
        // Create a container inside privacyResultBox for the gauge chart if it doesn't exist.
        let gaugeContainer = privacyResultBox.querySelector(".gauge-chart");
        if (!gaugeContainer) {
          gaugeContainer = document.createElement("div");
          gaugeContainer.className = "gauge-chart";
          privacyResultBox.appendChild(gaugeContainer);
        }
        plotPrivacyGauge(privacyRes);
      }
    } catch (error) {
      console.error("Error processing model audit:", error);
    }
  });
});
