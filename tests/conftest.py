import importlib

import pytest


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory, monkeypatch_session):
    """Обучает модель один раз во временную папку."""
    root = tmp_path_factory.mktemp("artifacts")
    from churn_monitoring import config

    monkeypatch_session.setattr(config, "ARTIFACTS_DIR", root)
    monkeypatch_session.setattr(config, "MODEL_PATH", root / "model.joblib")
    monkeypatch_session.setattr(config, "REFERENCE_PATH", root / "reference_data.csv")
    monkeypatch_session.setattr(config, "METRICS_PATH", root / "train_metrics.json")
    monkeypatch_session.setattr(config, "DB_PATH", root / "predictions.db")
    monkeypatch_session.setattr(config, "REPORTS_DIR", root / "reports")

    train = importlib.import_module("churn_monitoring.train")
    metrics = train.main()
    return root, metrics


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
