import joblib
import numpy as np
import pandas as pd

from app.config import MODEL_PATH


class SVMModelHandler:

    def __init__(self):
        self.model = None

    def load_model(self):
        """
        Load the trained SVM pipeline from disk.
        """

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

        return self.model

    def predict(self, features):
        """
        Generate class prediction.
        """

        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        prediction = self.model.predict(features)

        return int(prediction[0])

    def predict_proba(self, features):
        """
        Generate prediction probabilities.
        """

        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        probabilities = self.model.predict_proba(features)

        return probabilities[0]

    def predict_with_confidence(self, features):
        """
        Generate prediction and confidence score.
        """

        prediction = self.predict(features)

        probabilities = self.predict_proba(features)

        confidence = float(probabilities[prediction])

        return prediction, confidence