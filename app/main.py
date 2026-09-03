import pandas as pd
from fastapi import FastAPI, HTTPException

from app.config import MODEL_NAME, MODEL_VERSION
from app.model_handler import SVMModelHandler
from app.schemas import CancerInput, CancerOutput
from fastapi import FastAPI, HTTPException, Depends
from app.auth import verify_api_key
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CancerGuard API",
    description="SVM-based breast cancer prediction API",
    version=MODEL_VERSION
)


model_handler = SVMModelHandler()

@app.on_event("startup")
def load_model():
    try:
        model_handler.load_model()
        logger.info("CancerGuard model loaded successfully.")

    except Exception:
        logger.exception("Failed to load CancerGuard model.")
        raise

@app.get("/")
def root():
    return {
        "message": "CancerGuard API is running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


@app.get("/health")
def health():
    model_loaded = model_handler.model is not None

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


@app.post("/predict", response_model=CancerOutput)
def predict_cancer(input_data: CancerInput,api_key: str = Depends(verify_api_key)):
    """
    Predict whether breast cancer is present or absent.
    """

    try:
        # Convert validated input into a one-row DataFrame
        input_df = pd.DataFrame([input_data.model_dump()])

        # Rename 3 API fields to the exact names used
        # when the production pipeline was trained
        input_df = input_df.rename(columns={
            "concave_points_mean": "concave points_mean",
            "concave_points_se": "concave points_se",
            "concave_points_worst": "concave points_worst"
        })

        # Put columns in the exact order expected by the pipeline
        input_df = input_df[
            model_handler.model.feature_names_in_
        ]

        # Prediction + confidence
        prediction, confidence = (
            model_handler.predict_with_confidence(input_df)
        )

        prediction_label = (
            "present" if prediction == 1 else "absent"
        )

        return CancerOutput(
            prediction=prediction_label,
            confidence=confidence
        )

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Internal prediction error"
        )


