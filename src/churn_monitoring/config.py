"""Пути к артефактам. Корень можно переопределить переменной ARTIFACTS_DIR."""
import os
from pathlib import Path

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
REFERENCE_PATH = ARTIFACTS_DIR / "reference_data.csv"
METRICS_PATH = ARTIFACTS_DIR / "train_metrics.json"
DB_PATH = Path(os.getenv("PREDICTIONS_DB", str(ARTIFACTS_DIR / "predictions.db")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))
