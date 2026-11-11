import pytest

from churn_monitoring.data import FEATURES, generate_clients


def test_shape_and_columns():
    df = generate_clients(200, seed=1)
    assert list(df.columns) == FEATURES + ["churn"]
    assert len(df) == 200


def test_reproducible():
    assert generate_clients(50, seed=7).equals(generate_clients(50, seed=7))


def test_normal_ranges():
    df = generate_clients(2000, seed=1)
    assert df["age"].between(18, 69).all()
    assert df["num_products"].between(1, 4).all()
    assert 0.15 < df["churn"].mean() < 0.40


def test_full_drift_shifts_population():
    df = generate_clients(2000, seed=1, drift=1.0)
    assert df["age"].min() >= 50
    assert (df["is_active"] == 0).all()
    assert (df["num_products"] == 1).all()
    assert df["churn"].mean() > 0.8


def test_drift_out_of_range():
    with pytest.raises(ValueError):
        generate_clients(10, seed=1, drift=1.5)
