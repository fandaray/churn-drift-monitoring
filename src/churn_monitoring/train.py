"""Обучение модели оттока и сохранение эталонной выборки для мониторинга.

Запуск: python -m churn_monitoring.train
"""
from __future__ import annotations

import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data import FEATURES, generate_clients

SEED = 42
N_CLIENTS = 1000


def build_model() -> Pipeline:
    # Скейлер внутри пайплайна: API и мониторинг не смогут
    # случайно применить модель к немасштабированным данным.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=SEED)),
    ])


def main() -> dict:
    data = generate_clients(N_CLIENTS, seed=SEED)
    X, y = data[FEATURES], data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate": round(float(y.mean()), 3),
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "f1_churn": round(f1_score(y_test, y_pred), 3),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 3),
    }
    print(classification_report(y_test, y_pred, target_names=["Остался", "Ушёл"], zero_division=0))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    reference = X_train.copy()
    reference["target"] = y_train.values
    reference["prediction"] = model.predict(X_train)
    reference.to_csv(config.REFERENCE_PATH, index=False)

    config.METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    main()
