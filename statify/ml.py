"""Persiapan fitur, training, dan evaluasi model. Tidak bergantung pada Streamlit."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, mean_squared_log_error, precision_score,
                             r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:  # pragma: no cover
    HAS_SMOTE = False

CLASSIFICATION = "Klasifikasi"
REGRESSION = "Regresi"
MAX_ONEHOT_LEVELS = 20
MAX_CLASSES = 50


class ModelingError(Exception):
    """Pesan error yang aman ditampilkan ke pengguna."""


def detect_problem_type(y: pd.Series) -> str:
    """Target teks/kategori atau numerik dengan sedikit nilai unik → klasifikasi."""
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return CLASSIFICATION
    n_unique = y.nunique(dropna=True)
    is_whole = np.allclose(y.dropna() % 1, 0)
    if is_whole and n_unique <= 10:
        return CLASSIFICATION
    return REGRESSION


def build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Fitur numerik apa adanya + one-hot untuk kategori ≤ 20 level. Kolom mirip ID dibuang."""
    X = df.drop(columns=[target])
    keep_num = [c for c in X.select_dtypes(include=np.number).columns if X[c].nunique() > 1]
    cat_cols = [c for c in X.columns if c not in X.select_dtypes(include=np.number).columns]
    keep_cat = [c for c in cat_cols if 1 < X[c].nunique() <= MAX_ONEHOT_LEVELS]
    parts = [X[keep_num].astype("float32")]
    if keep_cat:
        parts.append(pd.get_dummies(X[keep_cat].astype(str), drop_first=True, dtype="float32"))
    features = pd.concat(parts, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0)
    # XGBoost menolak nama fitur yang mengandung [ ] <
    features.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "‹") for c in features.columns]
    features = features.loc[:, ~features.columns.duplicated()].copy()
    if features.shape[1] == 0:
        raise ModelingError("Tidak ada kolom fitur yang bisa dipakai. Pastikan ada kolom numerik "
                            "atau kategori dengan ≤ 20 nilai unik selain target.")
    return features


def _models(problem_type: str, n_trees: int, n_classes: int = 2) -> dict:
    nj, rs = config.N_JOBS, config.RANDOM_STATE
    if problem_type == CLASSIFICATION:
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=n_trees, random_state=rs, n_jobs=nj),
            "Logistic Regression": make_pipeline(StandardScaler(),
                                                 LogisticRegression(max_iter=1000, random_state=rs)),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=n_trees, max_depth=6, learning_rate=0.1, n_jobs=nj, random_state=rs,
                tree_method="hist", eval_metric="mlogloss" if n_classes > 2 else "logloss")
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=n_trees, random_state=rs, n_jobs=nj),
            "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=n_trees, max_depth=6, learning_rate=0.1,
                                             n_jobs=nj, random_state=rs, tree_method="hist")
    return models


@dataclass
class ModelResult:
    name: str
    model: object
    metrics: dict
    y_pred: np.ndarray
    confusion: np.ndarray | None = None
    importance: pd.Series | None = None


@dataclass
class TrainResult:
    problem_type: str
    target: str
    feature_names: list
    y_test: np.ndarray
    class_labels: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    smote_used: bool = False
    best_name: str = ""

    @property
    def best(self) -> ModelResult:
        return self.results[self.best_name]


def _importance(model, names) -> pd.Series | None:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    return pd.Series(vals, index=names).sort_values(ascending=False)


def train_models(df: pd.DataFrame, target: str, problem_type: str | None = None,
                 model_names: list[str] | None = None, n_trees: int = 100,
                 test_size: float = 0.2, use_smote: bool = False) -> TrainResult:
    if target not in df.columns:
        raise ModelingError(f"Kolom target '{target}' tidak ditemukan.")
    data = df[df[target].notna()]
    if len(data) < 20:
        raise ModelingError("Data terlalu sedikit (minimal 20 baris dengan target terisi).")

    problem_type = problem_type or detect_problem_type(data[target])
    n_trees = int(min(max(n_trees, 10), config.MAX_TREES))
    X = build_features(data, target)
    y_raw = data[target]

    class_labels: list = []
    stratify = None
    if problem_type == CLASSIFICATION:
        if y_raw.nunique() < 2:
            raise ModelingError("Target hanya punya satu kelas; klasifikasi butuh minimal dua.")
        if y_raw.nunique() > MAX_CLASSES:
            raise ModelingError(f"Target punya {y_raw.nunique()} kelas (maks. {MAX_CLASSES}). "
                                "Mungkin ini masalah regresi?")
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw.astype(str))
        class_labels = list(enc.classes_)
        if np.bincount(y).min() >= 2:
            stratify = y
    else:
        if not pd.api.types.is_numeric_dtype(y_raw):
            raise ModelingError("Regresi butuh kolom target numerik.")
        y = y_raw.to_numpy(dtype="float64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=stratify)

    smote_used = False
    if use_smote and problem_type == CLASSIFICATION and HAS_SMOTE:
        min_count = np.bincount(y_train).min()
        if min_count >= 2:
            sm = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=min(5, min_count - 1))
            X_train, y_train = sm.fit_resample(X_train, y_train)
            smote_used = True

    available = _models(problem_type, n_trees, len(class_labels))
    chosen = model_names or list(available)
    out = TrainResult(problem_type, target, list(X.columns), np.asarray(y_test),
                      class_labels, smote_used=smote_used)

    for name in chosen:
        if name not in available:
            continue
        model = available[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if problem_type == CLASSIFICATION:
            avg = "binary" if len(class_labels) == 2 else "weighted"
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
                "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, average=avg, zero_division=0),
            }
            cm = confusion_matrix(y_test, y_pred, labels=range(len(class_labels)))
        else:
            metrics = {
                "R²": r2_score(y_test, y_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "MAE": mean_absolute_error(y_test, y_pred),
            }
            if (np.asarray(y_test) >= 0).all() and (np.asarray(y_pred) >= 0).all():
                metrics["RMSLE"] = float(np.sqrt(mean_squared_log_error(y_test, y_pred)))
            cm = None
        out.results[name] = ModelResult(name, model, metrics, y_pred, cm, _importance(model, X.columns))

    if not out.results:
        raise ModelingError("Pilih minimal satu model.")
    key = "F1-Score" if problem_type == CLASSIFICATION else "R²"
    out.best_name = max(out.results, key=lambda n: out.results[n].metrics[key])
    return out


def metrics_table(result: TrainResult) -> pd.DataFrame:
    return pd.DataFrame({n: r.metrics for n, r in result.results.items()}).T


def interpret(result: TrainResult) -> list[str]:
    """Penjelasan singkat hasil dalam bahasa sederhana."""
    best = result.best
    notes = []
    if result.problem_type == CLASSIFICATION:
        acc, f1 = best.metrics["Accuracy"], best.metrics["F1-Score"]
        level = "sangat baik" if f1 >= 0.9 else "baik" if f1 >= 0.75 else "cukup" if f1 >= 0.6 else "lemah"
        notes.append(f"Model terbaik **{best.name}** menebak benar {acc:.0%} data uji; "
                     f"F1-Score {f1:.2f} tergolong **{level}**.")
        baseline = np.bincount(result.y_test).max() / len(result.y_test)
        notes.append(f"Sebagai pembanding, selalu menebak kelas terbanyak sudah benar {baseline:.0%}. "
                     + ("Model jelas lebih baik dari tebakan itu." if acc > baseline + 0.05
                        else "Model belum jauh lebih baik dari tebakan itu; coba tambah fitur atau SMOTE."))
    else:
        r2, rmse = best.metrics["R²"], best.metrics["RMSE"]
        level = "sangat baik" if r2 >= 0.9 else "baik" if r2 >= 0.7 else "cukup" if r2 >= 0.5 else "lemah"
        notes.append(f"Model terbaik **{best.name}** menjelaskan {max(r2, 0):.0%} variasi "
                     f"'{result.target}' (R² = {r2:.2f}, **{level}**).")
        notes.append(f"Rata-rata meleset sekitar {rmse:,.2f} satuan (RMSE); "
                     f"bandingkan dengan simpangan baku target {np.std(result.y_test):,.2f}.")
    if best.importance is not None and len(best.importance):
        top = ", ".join(f"'{c}'" for c in best.importance.head(3).index)
        notes.append(f"Fitur paling berpengaruh: {top}.")
    if result.smote_used:
        notes.append("SMOTE dipakai pada data latih untuk menyeimbangkan kelas.")
    return notes
