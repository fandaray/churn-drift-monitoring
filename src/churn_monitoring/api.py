"""REST API, имитирующий продакшен: отдаёт предсказания и пишет каждый
запрос в SQLite, чтобы потом сравнить поток данных с эталоном.

Запуск: uvicorn churn_monitoring.api:app --reload
"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from . import config
from .data import FEATURES

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    age           INTEGER NOT NULL,
    tenure        INTEGER NOT NULL,
    balance       REAL    NOT NULL,
    num_products  INTEGER NOT NULL,
    is_active     INTEGER NOT NULL,
    salary        REAL    NOT NULL,
    prediction    INTEGER NOT NULL,
    probability   REAL    NOT NULL
)
"""


class Client(BaseModel):
    age: int = Field(ge=14, le=120, examples=[35])
    tenure: int = Field(ge=0, examples=[24])
    balance: float = Field(ge=0, examples=[50_000])
    num_products: int = Field(ge=1, le=10, examples=[2])
    is_active: int = Field(ge=0, le=1, examples=[1])
    salary: float = Field(ge=0, examples=[60_000])


def connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.MODEL_PATH.exists():
        raise RuntimeError(f"Нет модели {config.MODEL_PATH}: сначала запустите python -m churn_monitoring.train")
    state["model"] = joblib.load(config.MODEL_PATH)
    with closing(connect()) as conn:
        conn.execute(SCHEMA)
        conn.commit()
    yield
    state.clear()


app = FastAPI(title="Churn model monitoring API", version="2.0.0", lifespan=lifespan)


def risk_level(p: float) -> str:
    return "high" if p > 0.7 else "medium" if p > 0.4 else "low"


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(UTC).isoformat(timespec="seconds")}


@app.post("/predict")
def predict(client: Client):
    row = client.model_dump()
    X = pd.DataFrame([row])[FEATURES]
    proba = float(state["model"].predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    with closing(connect()) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, age, tenure, balance, num_products, is_active, salary,"
            " prediction, probability) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (created_at, *(row[f] for f in FEATURES), pred, proba),
        )
        conn.commit()
    return {
        "prediction": pred,
        "churn_probability": round(proba, 4),
        "risk_level": risk_level(proba),
        "timestamp": created_at,
    }


@app.get("/logs")
def logs(limit: int = 20):
    with closing(connect()) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return {"total": total, "logs": [dict(r) for r in rows]}


@app.get("/stats")
def stats():
    with closing(connect()) as conn:
        r = conn.execute(
            """
            SELECT COUNT(*)                      AS total,
                   COALESCE(SUM(prediction), 0)  AS churned,
                   ROUND(AVG(probability), 4)    AS avg_probability,
                   SUM(probability > 0.7)        AS high_risk
            FROM predictions
            """
        ).fetchone()
    if r["total"] == 0:
        raise HTTPException(status_code=404, detail="Предсказаний пока нет")
    return {**dict(r), "churn_rate": round(r["churned"] / r["total"], 4)}


@app.get("/export", response_class=PlainTextResponse)
def export():
    """Лог запросов в CSV — это и есть «текущие данные» для мониторинга."""
    with closing(connect()) as conn:
        df = pd.read_sql_query(f"SELECT {', '.join(FEATURES)}, prediction FROM predictions", conn)
    return df.to_csv(index=False)
