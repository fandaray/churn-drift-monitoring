"""Синтетические данные клиентов банка для задачи оттока.

Все генераторы собраны здесь, чтобы обучение, API и мониторинг
пользовались одной и той же логикой.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["age", "tenure", "balance", "num_products", "is_active", "salary"]

# Границы признаков в «нормальном» состоянии и при максимальном дрейфе.
# Дрейф имитирует уход молодых активных клиентов и приток пожилых
# неактивных клиентов с низким доходом.
NORMAL_RANGES = {
    "age": (18, 70),
    "tenure": (1, 60),
    "balance": (0, 100_000),
    "num_products": (1, 5),       # верхняя граница не включается
    "salary": (20_000, 120_000),
}
DRIFT_RANGES = {
    "age": (50, 80),
    "tenure": (1, 20),
    "balance": (0, 15_000),
    "num_products": (1, 2),
    "salary": (10_000, 30_000),
}
NORMAL_ACTIVE_SHARE = 0.5
DRIFT_ACTIVE_SHARE = 0.0
DRIFT_CHURN_RATE = 0.85


def churn_probability(df: pd.DataFrame) -> pd.Series:
    """Бизнес-логика оттока: неактивные клиенты с низким балансом
    и малым числом продуктов уходят чаще, молодые — чуть чаще."""
    p = (
        0.6
        - 0.3 * df["is_active"]
        - 0.08 * df["num_products"]
        - 0.000003 * df["balance"]
        + 0.002 * (45 - df["age"]).clip(lower=0)
    )
    return p.clip(0.1, 0.9)


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def generate_clients(n: int, seed: int, drift: float = 0.0) -> pd.DataFrame:
    """Сгенерировать n клиентов с целевой переменной `churn`.

    drift — интенсивность дрейфа от 0 (как в обучающей выборке)
    до 1 (максимальный сдвиг): границы всех признаков линейно
    сдвигаются от NORMAL_RANGES к DRIFT_RANGES.
    """
    if not 0.0 <= drift <= 1.0:
        raise ValueError("drift должен быть в диапазоне [0, 1]")
    rng = np.random.default_rng(seed)
    r = {k: _lerp(NORMAL_RANGES[k], DRIFT_RANGES[k], drift) for k in NORMAL_RANGES}
    active_share = NORMAL_ACTIVE_SHARE + (DRIFT_ACTIVE_SHARE - NORMAL_ACTIVE_SHARE) * drift

    df = pd.DataFrame({
        "age": rng.integers(round(r["age"][0]), round(r["age"][1]), n),
        "tenure": rng.integers(round(r["tenure"][0]), max(round(r["tenure"][1]), 2), n),
        "balance": rng.uniform(*r["balance"], n).round(2),
        "num_products": rng.integers(round(r["num_products"][0]), max(round(r["num_products"][1]), 2), n),
        "is_active": (rng.random(n) < active_share).astype(int),
        "salary": rng.uniform(*r["salary"], n).round(2),
    })
    # Concept drift: при сдвиге меняется и сама зависимость —
    # вероятность ухода подтягивается к DRIFT_CHURN_RATE.
    p = churn_probability(df) * (1 - drift) + DRIFT_CHURN_RATE * drift
    df["churn"] = (rng.random(n) < p).astype(int)
    return df
