"""Мониторинг дрейфа данных и качества модели на Evidently AI.

Два режима:

    python -m churn_monitoring.monitor experiment
        Эксперимент из курсовой: нормальные и дрейфовые данные
        + анализ чувствительности к интенсивности дрейфа.

    python -m churn_monitoring.monitor check path/to/current.csv
        Проверка реального потока (например, выгрузки GET /export).
        Код возврата 1 при обнаружении дрейфа — удобно для cron/CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset
from sklearn.metrics import accuracy_score, f1_score, recall_score

from . import config
from .data import FEATURES, generate_clients

warnings.filterwarnings("ignore")

DRIFT_SHARE_THRESHOLD = 0.5  # дрейф датасета, если дрейфуют > 50% колонок
N_CURRENT = 400


def load_reference() -> tuple[pd.DataFrame, object]:
    return pd.read_csv(config.REFERENCE_PATH), joblib.load(config.MODEL_PATH)


def to_dataset(df: pd.DataFrame, with_target: bool) -> Dataset:
    if with_target:
        definition = DataDefinition(
            numerical_columns=FEATURES,
            classification=[BinaryClassification(target="target", prediction_labels="prediction")],
        )
        cols = FEATURES + ["target", "prediction"]
    else:
        definition = DataDefinition(numerical_columns=FEATURES + ["prediction"])
        cols = FEATURES + ["prediction"]
    return Dataset.from_pandas(df[cols], data_definition=definition)


def drift_summary(snapshot) -> dict:
    """Достаёт из отчёта Evidently долю дрейфующих колонок и p-value по каждой."""
    out = {"drifted_columns": 0, "drift_share": 0.0, "columns": {}}
    for m in snapshot.dict()["metrics"]:
        name = m.get("metric_name") or m.get("metric_id", "")
        value = m.get("value")
        if name.startswith("DriftedColumnsCount"):
            out["drifted_columns"] = int(value["count"])
            out["drift_share"] = float(value["share"])
        elif name.startswith("ValueDrift(") and isinstance(value, (int, float)):
            column = name.split("column=")[1].split(",")[0].rstrip(")")
            out["columns"][column] = round(float(value), 4)
    out["dataset_drift"] = out["drift_share"] > DRIFT_SHARE_THRESHOLD
    return out


def quality(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "f1_churn": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
        "recall_stayed": round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 3),
        "real_churn_rate": round(float(np.mean(y_true)), 3),
        "predicted_churn_rate": round(float(np.mean(y_pred)), 3),
    }


def scored(model, drift: float, seed: int) -> pd.DataFrame:
    df = generate_clients(N_CURRENT, seed=seed, drift=drift).rename(columns={"churn": "target"})
    df["prediction"] = model.predict(df[FEATURES])
    return df


def run_report(reference: pd.DataFrame, current: pd.DataFrame, with_target: bool, html_name: str | None):
    metrics = [DataDriftPreset()] + ([ClassificationPreset()] if with_target else [])
    snapshot = Report(metrics=metrics).run(
        reference_data=to_dataset(reference, with_target), current_data=to_dataset(current, with_target))
    if html_name:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(config.REPORTS_DIR / html_name))
    return drift_summary(snapshot)


def experiment() -> dict:
    reference, model = load_reference()
    results = {}

    # 1. Две крайние ситуации: данные как при обучении и сильный дрейф
    for label, drift, seed in [("normal", 0.0, 99), ("drifted", 1.0, 77)]:
        current = scored(model, drift, seed)
        summary = run_report(reference, current, with_target=True, html_name=f"report_{label}.html")
        results[label] = {**summary, **quality(current["target"], current["prediction"])}
        print(f"[{label}] drift_share={summary['drift_share']:.2f} "
              f"dataset_drift={summary['dataset_drift']} accuracy={results[label]['accuracy']}")

    # 2. Чувствительность: при какой интенсивности дрейфа детектор срабатывает
    sweep = []
    for t in np.round(np.linspace(0, 1, 11), 1):
        current = scored(model, float(t), seed=1000 + int(t * 10))
        summary = run_report(reference, current, with_target=True, html_name=None)
        sweep.append({"intensity": float(t), "drift_share": summary["drift_share"],
                      **quality(current["target"], current["prediction"])})
    results["sensitivity"] = sweep
    first = next((s["intensity"] for s in sweep if s["drift_share"] > DRIFT_SHARE_THRESHOLD), None)
    results["detection_threshold_intensity"] = first
    print(f"Дрейф датасета фиксируется начиная с интенсивности t = {first}")

    # 3. Устойчивость вывода: тот же эксперимент для 20 моделей с разными seed
    results["seed_study"] = seed_study()
    s = results["seed_study"]
    print(f"Из {s['runs']} моделей при дрейфе Accuracy выросла у {s['accuracy_up']}, "
          f"macro-F1 упала у {s['f1_macro_down']}")

    plot_sensitivity(sweep)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def seed_study(runs: int = 20) -> dict:
    """Обучает модель с разными seed и сравнивает метрики на нормальных
    и дрейфовых данных. Один запуск может быть случайностью — 20 уже нет."""
    from .train import build_model

    rows = []
    for seed in range(runs):
        train = generate_clients(800, seed=seed)
        model = build_model().fit(train[FEATURES], train["churn"])
        q = {}
        for label, drift, offset in [("normal", 0.0, 100), ("drifted", 1.0, 200)]:
            df = generate_clients(N_CURRENT, seed=seed + offset, drift=drift)
            q[label] = quality(df["churn"], model.predict(df[FEATURES]))
        rows.append({"seed": seed, **{f"{k}_{lbl}": v for lbl in q for k, v in q[lbl].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(config.REPORTS_DIR / "seed_study.csv", index=False)
    plot_seed_study(df)

    def med(col):
        return round(float(df[col].median()), 3)

    return {
        "runs": runs,
        "accuracy_up": int((df["accuracy_drifted"] > df["accuracy_normal"]).sum()),
        "f1_macro_down": int((df["f1_macro_drifted"] < df["f1_macro_normal"]).sum()),
        "median": {c: med(c) for c in [
            "accuracy_normal", "accuracy_drifted", "f1_macro_normal", "f1_macro_drifted",
            "recall_stayed_normal", "recall_stayed_drifted",
            "predicted_churn_rate_normal", "predicted_churn_rate_drifted"]},
    }


def plot_seed_study(df: pd.DataFrame) -> None:
    """Две метрики «до» и «после» дрейфа: линия на каждую модель, медиана жирным."""
    panels = [("accuracy", "Accuracy"), ("f1_macro", "Macro-F1"), ("recall_stayed", "Recall класса «Остался»")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        a, b = df[f"{key}_normal"], df[f"{key}_drifted"]
        for y0, y1 in zip(a, b):
            color = "#d62728" if y1 > y0 else "#7f7f7f"
            ax.plot([0, 1], [y0, y1], color=color, alpha=0.35, lw=1)
        ax.plot([0, 1], [a.median(), b.median()], "o-", color="black", lw=2.5, label="медиана")
        ax.set_xticks([0, 1], ["норма", "дрейф"])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"{len(df)} моделей с разными seed: красным — метрика при дрейфе выросла")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "seed_study.png", dpi=150)
    plt.close(fig)


def plot_sensitivity(sweep: list[dict]) -> None:
    t = [s["intensity"] for s in sweep]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, [s["drift_share"] for s in sweep], "o-", lw=2, label="Доля дрейфующих колонок (Evidently)")
    ax.plot(t, [s["accuracy"] for s in sweep], "s--", label="Accuracy")
    ax.plot(t, [s["f1_macro"] for s in sweep], "^--", label="Macro-F1")
    ax.axhline(DRIFT_SHARE_THRESHOLD, color="grey", ls=":", label="Порог дрейфа датасета")
    ax.set_xlabel("Интенсивность дрейфа t")
    ax.set_ylim(0, 1.05)
    ax.set_title("Чувствительность детектора дрейфа")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.REPORTS_DIR / "sensitivity.png", dpi=150)
    plt.close(fig)


def check(path: str) -> int:
    reference, _ = load_reference()
    current = pd.read_csv(path)
    summary = run_report(reference, current, with_target="target" in current.columns,
                         html_name="report_current.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["dataset_drift"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("experiment")
    p_check = sub.add_parser("check")
    p_check.add_argument("csv")
    args = parser.parse_args(argv)
    if args.cmd == "experiment":
        experiment()
        return 0
    return check(args.csv)


if __name__ == "__main__":
    sys.exit(main())
