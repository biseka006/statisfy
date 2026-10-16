"""Tes inti Statify (tanpa Streamlit). Jalankan: python -m pytest -q"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statify import clustering, config, eda, ml
from statify.data import DataLoadError, clean_data, load_dataset

SAMPLES = Path(__file__).resolve().parents[1] / "sample_data"


def _bytes(name):
    return (SAMPLES / name).read_bytes()


def test_load_csv_and_semicolon():
    res = load_dataset(_bytes("iris.csv"), "iris.csv")
    assert res.df.shape == (150, 5)
    raw = "a;b\n1;2\n3;4\n".encode()
    assert load_dataset(raw, "x.csv").df.shape == (2, 2)


def test_load_excel_roundtrip():
    buf = io.BytesIO()
    pd.read_csv(SAMPLES / "iris.csv").to_excel(buf, index=False)
    res = load_dataset(buf.getvalue(), "iris.xlsx")
    assert res.df.shape == (150, 5)


def test_reject_unknown_format():
    with pytest.raises(DataLoadError):
        load_dataset(b"hello", "data.txt")


def test_sampling_limit():
    big = pd.DataFrame({"x": np.arange(config.MAX_ROWS + 500), "y": 1.0}).to_csv(index=False).encode()
    res = load_dataset(big, "big.csv")
    assert res.sampled and len(res.df) == config.MAX_ROWS and res.original_rows == config.MAX_ROWS + 500


def test_cleaning_fills_and_dedupes():
    df = load_dataset(_bytes("penjualan_toko.csv"), "p.csv").df
    cleaned, rep = clean_data(df)
    assert cleaned.isna().sum().sum() == 0
    assert rep.duplicates_removed == 12
    assert df.isna().sum().sum() > 0  # data asli tidak ikut berubah


def test_classification_iris():
    df = load_dataset(_bytes("iris.csv"), "iris.csv").df
    res = ml.train_models(df, "species", n_trees=50)
    assert res.problem_type == ml.CLASSIFICATION
    assert res.best.metrics["Accuracy"] > 0.85
    assert ml.interpret(res)


def test_regression_and_rmsle_guard():
    df = load_dataset(_bytes("diabetes_progresi.csv"), "d.csv").df
    res = ml.train_models(df, "target", n_trees=50)
    assert res.problem_type == ml.REGRESSION
    assert "R²" in res.best.metrics
    df2 = df.assign(target=df["target"] - 200)  # target negatif → RMSLE tidak ditampilkan
    res2 = ml.train_models(df2, "target", ml.REGRESSION, model_names=["Linear Regression"])
    assert "RMSLE" not in res2.best.metrics


def test_categorical_features_and_smote():
    df, _ = clean_data(load_dataset(_bytes("penjualan_toko.csv"), "p.csv").df)
    res = ml.train_models(df, "promo", use_smote=True, n_trees=30)
    assert res.smote_used
    assert any(c.startswith("kategori_") for c in res.feature_names)


def test_clustering_does_not_touch_source_and_samples_hierarchical():
    big = pd.DataFrame(np.random.default_rng(0).normal(size=(config.HIERARCHICAL_SAMPLE + 1000, 3)),
                       columns=list("abc"))
    res = clustering.run_hierarchical(big, ["a", "b", "c"], 3)
    assert len(res.data) == config.HIERARCHICAL_SAMPLE and res.sampled_from == len(big)
    assert "Cluster" not in big.columns
    km = clustering.run_kmeans(big, ["a", "b"], 3)
    assert km.n_clusters == 3 and km.silhouette is not None


def test_eda_insights():
    df, _ = clean_data(load_dataset(_bytes("penjualan_toko.csv"), "p.csv").df)
    items = eda.auto_insights(df)
    assert 1 <= len(items) <= 4
    for item in items:
        fig = eda.render_insight(df, item)
        assert fig is not None


# ---------- regresi untuk temuan audit deploy ----------
def test_single_column_and_semicolon_decimal_comma():
    assert load_dataset(b"x\n1\n2\n3\n", "a.csv").df.shape == (3, 1)
    df = load_dataset("harga;qty\n1,5;2\n2,5;3\n".encode(), "a.csv").df
    assert pd.api.types.is_float_dtype(df["harga"]) and df["harga"].iloc[0] == pytest.approx(1.5)


def test_empty_csv_and_duplicate_columns():
    with pytest.raises(DataLoadError):
        load_dataset(b"   ", "a.csv")
    assert list(load_dataset(b"a,a,b\n1,2,3\n", "a.csv").df.columns) == ["a", "a.1", "b"]


def test_inf_and_unsafe_feature_names_train():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"<x[1]>": rng.normal(size=80), "y": rng.normal(size=80)})
    df.loc[3, "<x[1]>"] = np.inf
    cleaned, _ = clean_data(load_dataset(df.to_csv(index=False).encode(), "a.csv").df)
    res = ml.train_models(cleaned, "y", n_trees=20)
    assert "XGBoost" in res.results or not ml.HAS_XGB


def test_clustering_tiny_data_is_handled():
    tiny = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    with pytest.raises(clustering.ClusteringError):
        clustering.run_kmeans(tiny, ["a", "b"], 3)
    small = pd.DataFrame(np.random.default_rng(1).normal(size=(12, 2)), columns=["a", "b"])
    assert clustering.run_kmeans(small, ["a", "b"], 10).n_clusters <= 11
