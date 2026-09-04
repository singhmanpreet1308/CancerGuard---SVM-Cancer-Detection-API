import joblib

from app.config import MODEL_PATH


class SVMModelHandler:

    # Experimental operating threshold selected during model evaluation
    OPERATING_THRESHOLD = 0.30

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

    def predict_proba(self, features):
        """
        Generate prediction probabilities.
        """

        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        probabilities = self.model.predict_proba(features)

        return probabilities[0]

    def predict(self, features):
        """
        Generate class prediction using the selected
        experimental operating threshold.

        Class 0: Benign / absent
        Class 1: Malignant / present
        """

        probabilities = self.predict_proba(features)

        malignant_probability = float(probabilities[1])

        prediction = (
            1
            if malignant_probability >= self.OPERATING_THRESHOLD
            else 0
        )

        return prediction

    def predict_with_confidence(self, features):
        """
        Generate threshold-based prediction and confidence score.
        """

        probabilities = self.predict_proba(features)

        malignant_probability = float(probabilities[1])

        prediction = (
            1
            if malignant_probability >= self.OPERATING_THRESHOLD
            else 0
        )

        confidence = float(probabilities[prediction])

        return prediction, confidence