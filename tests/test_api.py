from fastapi.testclient import TestClient
from app.main import app
import time


def test_health_endpoint():

    # Context manager ensures FastAPI startup event runs
    with TestClient(app) as client:

        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model"] == "CancerGuard SVM Pipeline"
        assert data["version"] == "1.0.0"


def test_predict_valid_input():

    payload = {
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

    headers = {
        "X-API-Key": "cancerguard-secret-key"
    }

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=payload,
            headers=headers
        )

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "confidence" in data


def test_predict_missing_field():

    payload = {
        # radius_mean intentionally missing
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

    headers = {
        "X-API-Key": "cancerguard-secret-key"
    }

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=payload,
            headers=headers
        )

        assert response.status_code == 422

def test_predict_missing_api_key():

    payload = {
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

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=payload
        )

        assert response.status_code == 401


def test_predict_invalid_api_key():

    payload = {
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

    headers = {
        "X-API-Key": "wrong-api-key"
    }

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json=payload,
            headers=headers
        )

        assert response.status_code == 401



def test_predict_latency():

    payload = {
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

    headers = {
        "X-API-Key": "cancerguard-secret-key"
    }

    with TestClient(app) as client:

        start_time = time.perf_counter()

        response = client.post(
            "/predict",
            json=payload,
            headers=headers
        )

        end_time = time.perf_counter()

    response_time = end_time - start_time

    print(f"\nPrediction response time: {response_time:.4f} seconds")

    assert response.status_code == 200
    assert response_time < 1.0