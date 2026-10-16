"""Batas sumber daya Statify, disetel untuk server 2 vCPU / 4 GB RAM (maks. 5 pengguna bersamaan).

Semua angka bisa diubah lewat environment variable tanpa menyentuh kode, misalnya
STATIFY_MAX_ROWS=10000 jika RAM server sering penuh.
"""
import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


APP_NAME = "Statify"
APP_TAGLINE = "Stats and Simplify"

# Upload & sampling
MAX_UPLOAD_MB = _int_env("STATIFY_MAX_UPLOAD_MB", 20)
MAX_ROWS = _int_env("STATIFY_MAX_ROWS", 20_000)
MAX_EXCEL_ROWS = 200_000  # Excel dibaca maks. sekian baris pertama sebelum disampling

# Visualisasi
MAX_HEATMAP_COLS = 15
PAIRPLOT_MAX_COLS = 4
PAIRPLOT_SAMPLE = 2_000
PLOT_SAMPLE = 5_000

# Clustering (operasi O(n^2) selalu lewat sampel)
HIERARCHICAL_SAMPLE = _int_env("STATIFY_HIERARCHICAL_SAMPLE", 3_000)
SILHOUETTE_SAMPLE = 5_000

# Modeling
MAX_TREES = _int_env("STATIFY_MAX_TREES", 200)
N_JOBS = 1  # tiap sesi memakai 1 core supaya training paralel antar-pengguna tetap jalan
RANDOM_STATE = 42

# Antrean: maks. proses berat (training/clustering) bersamaan di seluruh server
MAX_HEAVY_JOBS = _int_env("STATIFY_MAX_HEAVY_JOBS", 2)
QUEUE_TIMEOUT_S = 240

# Cache
CACHE_TTL = 1800
CACHE_MAX_ENTRIES = 10
