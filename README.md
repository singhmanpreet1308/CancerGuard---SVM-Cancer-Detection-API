# CancerGuard-SVM-Cancer Detection-API

CancerGuard is a machine learning REST API for breast cancer prediction built with **FastAPI** and  **scikit-learn** . It uses a trained **Support Vector Machine (SVM)** classification model to predict whether a breast tumour is **malignant or benign** from diagnostic tumour measurements.

The project implements an end-to-end machine learning workflow covering data preprocessing, exploratory data analysis, feature scaling, SVM model training and optimisation, model evaluation and explainability, model serialization, REST API development, API-key authentication, automated testing, and interactive API documentation through Swagger UI.

![1780090438479](image/README/1780090438479.png)

## Key Features

* Breast cancer classification using a trained SVM model
* 30 diagnostic tumour measurements used as model inputs
* Standardised feature preprocessing using a saved `<span>StandardScaler</span>`
* Serialized model and preprocessing pipeline for inference
* FastAPI REST API for real-time predictions
* Prediction confidence/probability returned with each result
* API-key authentication for protected prediction requests
* Pydantic-based request validation
* `<span>/health</span>` endpoint for API health monitoring
* Interactive Swagger/OpenAPI documentation
* Automated API and model tests using `<span>pytest</span>`
* Prediction latency validation

## Dataset

The model is developed using the  **Wisconsin Breast Cancer Diagnostic dataset** , containing **569 observations and 30 numerical diagnostic features** derived from digitised images of breast mass fine-needle aspirates.

The target variable is:

* `<span>M</span>` — Malignant
* `<span>B</span>` — Benign

During model development, the target was encoded as:

```
Malignant (M) = 1
Benign (B)    = 0
```

Non-predictive identifier fields and empty CSV artifacts were removed before model training.

## Technology Stack

| Component            | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Machine Learning     | scikit-learn                 |
| ML Algorithm         | Support Vector Machine (SVM) |
| Data Processing      | pandas, NumPy                |
| Feature Scaling      | StandardScaler               |
| API Framework        | FastAPI                      |
| Data Validation      | Pydantic                     |
| Model Serialization  | joblib                       |
| API Server           | Uvicorn                      |
| API Documentation    | Swagger UI / OpenAPI         |
| Authentication       | API Key                      |
| Automated Testing    | pytest                       |
| Model Development    | Jupyter Notebook             |

## Project Architecture

CancerGuard follows a separation between offline machine-learning development and online API inference.

```
Breast Cancer Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Scaling
        │
        ▼
SVM Training & Optimisation
        │
        ▼
Model Evaluation
        │
        ▼
Serialized Model + Scaler
        │
        ▼
FastAPI Application
        │
        ├── /health
        │
        └── /predict
                │
                ▼
        Prediction + Confidence
```

The trained model and scaler are loaded by the FastAPI application for inference, avoiding model retraining for every prediction request.


# How to Run

### STEP 1 — Create a Conda Environment

After opening the project repository, create a new Conda environment:

```
conda create -n cancerguard python=3.12 -y
```

Activate the environment:

```
conda activate cancerguard
```

### STEP 2 — Install the Requirements

Install all required project dependencies:

```
pip install -r requirements.txt
```


### STEP 3 — Configure the API Key

Create a `<span>.env</span>` file in the project root directory and add your API key:

```
API_KEY=your_api_key_here
```

The API key is required to access the protected `<span>/predict</span>` endpoint.

> **Important:** Do not commit the `<span>.env</span>` file to GitHub. Make sure `<span>.env</span>` is included in `<span>.gitignore</span>`.

---

### STEP 4 — Start the FastAPI Server

Run the FastAPI application using Uvicorn:

```
uvicorn app.main:app --reload
```

Once the server starts successfully, the API will be available locally at:

```
http://127.0.0.1:8000
```

You can verify that the API is running using the health endpoint:

```
http://127.0.0.1:8000/health
```

A successful health check confirms that the CancerGuard API is running and ready to accept requests.

---

### STEP 5 — Open Swagger UI

FastAPI automatically provides interactive Swagger documentation.

After starting the server, open:

```
http://127.0.0.1:8000/docs
```

Swagger UI allows you to:

* View available API endpoints
* Inspect request and response schemas
* Enter the required API key
* Send prediction requests directly from the browser
* Inspect API responses and HTTP status codes

---

### STEP 6 — Test the `<span>/predict</span>` Endpoint

Open the `<span>/predict</span>` endpoint in Swagger UI and click  **Try it out** .

Enter the required API key and provide all 30 breast cancer diagnostic features in the JSON request body.

Example:

```
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "smoothness_mean": 0.1184,
  "compactness_mean": 0.2776,
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.07871,
  "radius_se": 1.095,
  "texture_se": 0.9053,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "smoothness_se": 0.006399,
  "compactness_se": 0.04904,
  "concavity_se": 0.05373,
  "concave_points_se": 0.01587,
  "symmetry_se": 0.03003,
  "fractal_dimension_se": 0.006193,
  "radius_worst": 25.38,
  "texture_worst": 17.33,
  "perimeter_worst": 184.6,
  "area_worst": 2019.0,
  "smoothness_worst": 0.1622,
  "compactness_worst": 0.6656,
  "concavity_worst": 0.7119,
  "concave_points_worst": 0.2654,
  "symmetry_worst": 0.4601,
  "fractal_dimension_worst": 0.1189
}
```

Click **Execute** to send the request.

The API will validate the input, preprocess the features using the saved scaler, perform inference using the trained SVM model, and return the prediction result together with its confidence score.
