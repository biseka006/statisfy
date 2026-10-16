"""Clustering dengan batas memori: operasi O(n²) selalu dijalankan pada sampel."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config


class ClusteringError(Exception):
    """Pesan error yang aman ditampilkan ke pengguna."""


@dataclass
class ClusterResult:
    method: str
    data: pd.DataFrame          # baris yang dipakai (bisa sampel), sudah dengan kolom 'Cluster'
    scaled: np.ndarray
    labels: np.ndarray
    sampled_from: int | None    # jumlah baris asli jika disampling
    silhouette: float | None
    davies_bouldin: float | None
    linkage_matrix: np.ndarray | None = None

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels) - {-1})

    @property
    def n_noise(self) -> int:
        return int((self.labels == -1).sum())


def _prepare(df: pd.DataFrame, cols: list[str], max_rows: int | None):
    data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        raise ClusteringError("Clustering butuh minimal 10 baris lengkap pada kolom yang dipilih.")
    sampled_from = None
    if max_rows and len(data) > max_rows:
        sampled_from = len(data)
        data = data.sample(n=max_rows, random_state=config.RANDOM_STATE)
    data = data.reset_index(drop=True)
    return data, StandardScaler().fit_transform(data), sampled_from


def _scores(X: np.ndarray, labels: np.ndarray):
    mask = labels != -1
    n_labels = len(set(labels[mask]))
    if n_labels < 2 or n_labels >= mask.sum():
        return None, None
    Xm, lm = X[mask], labels[mask]
    sample = min(config.SILHOUETTE_SAMPLE, len(Xm))
    sil = silhouette_score(Xm, lm, sample_size=sample, random_state=config.RANDOM_STATE)
    return float(sil), float(davies_bouldin_score(Xm, lm))


def _finish(method, data, X, labels, sampled_from, link=None) -> ClusterResult:
    sil, dbi = _scores(X, labels)
    out = data.copy()  # salinan: label tidak pernah menimpa dataset utama di session
    out["Cluster"] = labels
    return ClusterResult(method, out, X, labels, sampled_from, sil, dbi, link)


def run_kmeans(df, cols, k: int) -> ClusterResult:
    data, X, sampled = _prepare(df, cols, None)
    k = min(k, len(data) - 1)
    labels = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(X)
    return _finish("K-Means", data, X, labels, sampled)


def run_dbscan(df, cols, eps: float, min_samples: int) -> ClusterResult:
    data, X, sampled = _prepare(df, cols, None)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return _finish("DBSCAN", data, X, labels, sampled)


def run_hierarchical(df, cols, k: int, method: str = "ward") -> ClusterResult:
    data, X, sampled = _prepare(df, cols, config.HIERARCHICAL_SAMPLE)
    k = min(k, len(data) - 1)
    link = linkage(X, method=method)
    labels = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(X)
    return _finish("Hierarchical", data, X, labels, sampled, link)


def elbow(df, cols, k_max: int = 10) -> pd.DataFrame:
    data, X, _ = _prepare(df, cols, config.SILHOUETTE_SAMPLE)
    rows = []
    for k in range(2, min(k_max, len(X) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=config.RANDOM_STATE).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(len(X), 2000),
                               random_state=config.RANDOM_STATE)
        rows.append({"K": k, "Inertia": km.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)


def pca_2d(X: np.ndarray) -> np.ndarray:
    if X.shape[1] < 2:
        return np.column_stack([X[:, 0], np.zeros(len(X))])
    return PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X)


def describe(result: ClusterResult) -> list[str]:
    notes = [f"{result.method} menemukan **{result.n_clusters} cluster**"
             + (f" dan {result.n_noise} titik noise." if result.n_noise else ".")]
    if result.sampled_from:
        notes.append(f"Dihitung pada sampel {len(result.data):,} dari {result.sampled_from:,} baris "
                     "agar hemat memori.")
    if result.silhouette is not None:
        s = result.silhouette
        quality = "terpisah jelas" if s >= 0.5 else "cukup terpisah" if s >= 0.25 else "saling tumpang tindih"
        notes.append(f"Silhouette {s:.2f}: cluster **{quality}** (rentang -1 s.d. 1, makin tinggi makin baik).")
        notes.append(f"Davies-Bouldin {result.davies_bouldin:.2f} (makin rendah makin baik).")
    else:
        notes.append("Skor kualitas tidak bisa dihitung karena kurang dari dua cluster.")
    return notes
