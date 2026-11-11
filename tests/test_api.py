import pytest
from fastapi.testclient import TestClient

CLIENT = {"age": 25, "tenure": 3, "balance": 1000, "num_products": 1, "is_active": 0, "salary": 30000}


@pytest.fixture(scope="module")
def client(artifacts):
    from churn_monitoring.api import app

    with TestClient(app) as c:
        yield c


def test_train_metrics(artifacts):
    _, metrics = artifacts
    assert metrics["accuracy"] > 0.6


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_stats_empty_before_predictions(client):
    assert client.get("/stats").status_code == 404


def test_predict_and_log(client):
    r = client.post("/predict", json=CLIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in ("low", "medium", "high")

    logs = client.get("/logs").json()
    assert logs["total"] == 1
    assert logs["logs"][0]["age"] == 25

    stats = client.get("/stats").json()
    assert stats["total"] == 1


def test_predict_rejects_bad_input(client):
    assert client.post("/predict", json={**CLIENT, "is_active": 5}).status_code == 422


def test_export_csv(client):
    text = client.get("/export").text
    assert text.splitlines()[0] == "age,tenure,balance,num_products,is_active,salary,prediction"
