from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_svm_pipeline.pkl"

MODEL_NAME = "CancerGuard SVM Pipeline"

MODEL_VERSION = "1.0.0"

API_KEY = "cancerguard-secret-key"