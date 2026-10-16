"""Analisis eksploratif: korelasi, insight otomatis, dan grafik. Grafik dikembalikan sebagai figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .data import categorical_columns, numeric_columns


def correlation_matrix(df: pd.DataFrame, max_cols: int = config.MAX_HEATMAP_COLS) -> pd.DataFrame:
    num = df[numeric_columns(df)]
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] > max_cols:  # pilih kolom dengan variansi terbesar (setelah distandarkan)
        cv = (num.std() / num.mean().abs().replace(0, np.nan)).fillna(0)
        num = num[cv.sort_values(ascending=False).index[:max_cols]]
    return num.corr()


def top_correlations(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["Kolom 1", "Kolom 2", "Korelasi", "Kekuatan"])
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().reset_index()
    pairs.columns = ["Kolom 1", "Kolom 2", "Korelasi"]
    pairs = pairs.reindex(pairs["Korelasi"].abs().sort_values(ascending=False).index).head(n)
    pairs["Kekuatan"] = pairs["Korelasi"].abs().map(strength)
    return pairs.reset_index(drop=True)


def strength(r: float) -> str:
    r = abs(r)
    return "Sangat kuat" if r >= 0.8 else "Kuat" if r >= 0.6 else "Sedang" if r >= 0.4 else "Lemah" if r >= 0.2 else "Sangat lemah"


def plot_heatmap(corr: pd.DataFrame):
    size = max(5, 0.55 * len(corr))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(corr, annot=len(corr) <= 12, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Matriks Korelasi")
    fig.tight_layout()
    return fig


def plot_pairplot(df: pd.DataFrame, cols: list[str], hue: str | None = None):
    cols = cols[: config.PAIRPLOT_MAX_COLS]
    use = cols + ([hue] if hue and hue not in cols else [])
    data = df[use].dropna()
    if len(data) > config.PAIRPLOT_SAMPLE:
        data = data.sample(config.PAIRPLOT_SAMPLE, random_state=config.RANDOM_STATE)
    grid = sns.pairplot(data, vars=cols, hue=hue, corner=True, plot_kws={"s": 12, "alpha": 0.6},
                        height=2.2)
    return grid.figure


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(config.PLOT_SAMPLE, random_state=config.RANDOM_STATE) if len(df) > config.PLOT_SAMPLE else df


def plot_histogram(df, col):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(_sample(df)[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribusi {col}")
    fig.tight_layout()
    return fig


def plot_box(df, col, by: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    d = _sample(df)
    if by:
        top = d[by].astype(str).value_counts().head(10).index
        d = d[d[by].astype(str).isin(top)]
        sns.boxplot(x=d[by].astype(str), y=d[col], ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(f"{col} per {by}")
    else:
        sns.boxplot(x=d[col], ax=ax)
        ax.set_title(f"Sebaran & outlier {col}")
    fig.tight_layout()
    return fig


def plot_scatter(df, x, y, hue: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = _sample(df)
    sns.scatterplot(data=d, x=x, y=y, hue=hue, s=18, alpha=0.7, ax=ax)
    if hue is None:
        sns.regplot(data=d, x=x, y=y, scatter=False, ax=ax, color="#F2A541")
    ax.set_title(f"{y} terhadap {x}")
    fig.tight_layout()
    return fig


def plot_bar(df, col, top: int = 10):
    counts = df[col].astype(str).value_counts().head(top)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, orient="h")
    ax.set_title(f"{top} kategori teratas {col}" if df[col].nunique() > top else f"Jumlah per {col}")
    ax.set_xlabel("Jumlah")
    fig.tight_layout()
    return fig


def outlier_share(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return 0.0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())


def auto_insights(df: pd.DataFrame) -> list[dict]:
    """Pilih hingga 4 grafik paling informatif, masing-masing dengan alasan (reasoning)."""
    num, cat = numeric_columns(df), categorical_columns(df)
    num = [c for c in num if df[c].nunique() > 1]
    items: list[dict] = []

    if len(num) >= 2:
        pairs = top_correlations(correlation_matrix(df), 1)
        if not pairs.empty:
            a, b, r = pairs.iloc[0][["Kolom 1", "Kolom 2", "Korelasi"]]
            arah = "naik bersama" if r > 0 else "bergerak berlawanan"
            items.append({"kind": "scatter", "args": (a, b),
                          "title": f"Hubungan terkuat: {a} vs {b}",
                          "why": f"Korelasi {r:.2f} ({strength(r).lower()}): saat {a} naik, {b} cenderung {arah}."})
    if num:
        skews = df[num].skew(numeric_only=True).abs().sort_values(ascending=False)
        col = skews.index[0]
        items.append({"kind": "hist", "args": (col,), "title": f"Distribusi paling miring: {col}",
                      "why": f"Skewness {df[col].skew():.2f}. "
                             + ("Data menumpuk di satu sisi; pertimbangkan transformasi log sebelum modeling."
                                if abs(df[col].skew()) > 1 else "Distribusi relatif simetris.")})
        shares = {c: outlier_share(df[c]) for c in num}
        col = max(shares, key=shares.get)
        if shares[col] > 0:
            by = next((c for c in cat if 1 < df[c].nunique() <= 10), None)
            items.append({"kind": "box", "args": (col, by), "title": f"Outlier terbanyak: {col}",
                          "why": f"{shares[col]:.1%} nilai berada di luar batas IQR (1,5×)."
                                 + (f" Dipecah per '{by}' untuk melihat kelompok mana penyebabnya." if by else "")})
    if cat:
        col = min(cat, key=lambda c: df[c].nunique() if df[c].nunique() > 1 else 10**9)
        if df[col].nunique() > 1:
            top_share = df[col].astype(str).value_counts(normalize=True).iloc[0]
            items.append({"kind": "bar", "args": (col,), "title": f"Komposisi kategori: {col}",
                          "why": f"Kategori terbanyak mencakup {top_share:.0%} data. "
                                 + ("Data tidak seimbang; hati-hati jika kolom ini dijadikan target."
                                    if top_share > 0.7 else "Sebaran kategori cukup merata.")})
    return items[:4]


def render_insight(df: pd.DataFrame, item: dict):
    kind, args = item["kind"], item["args"]
    if kind == "scatter":
        return plot_scatter(df, *args)
    if kind == "hist":
        return plot_histogram(df, *args)
    if kind == "box":
        return plot_box(df, *args)
    return plot_bar(df, *args)
