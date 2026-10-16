"""Pemuatan, optimasi, sampling, dan cleaning data. Fungsi di sini tidak bergantung pada Streamlit."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


class DataLoadError(Exception):
    """Pesan error yang aman ditampilkan ke pengguna."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    original_rows: int
    sampled: bool


def _decode_head(raw: bytes) -> tuple[str, str]:
    """Tentukan encoding dari potongan awal file dan kembalikan baris pertamanya."""
    head = raw[:64_000]
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = head.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 selalu berhasil
        enc, text = "latin-1", head.decode("latin-1", errors="replace")
    if enc == "utf-8-sig":
        try:
            raw.decode("utf-8-sig")
        except UnicodeDecodeError:  # bagian lain file bukan UTF-8
            enc = "latin-1"
    first_line = text.splitlines()[0] if text.strip() else ""
    return enc, first_line


def _detect_sep(first_line: str) -> str:
    """Pemisah yang paling sering muncul di header; default koma."""
    counts = {sep: first_line.count(sep) for sep in (",", ";", "\t", "|")}
    sep, n = max(counts.items(), key=lambda kv: kv[1])
    return sep if n > 0 else ","


def read_table(raw: bytes, filename: str) -> pd.DataFrame:
    """Baca CSV/XLSX dari bytes, memilih pembaca berdasarkan ekstensi."""
    name = filename.lower()
    if name.endswith(".csv"):
        if not raw.strip():
            raise DataLoadError("File CSV kosong.")
        enc, first_line = _decode_head(raw)
        sep = _detect_sep(first_line)
        # CSV ekspor Excel berbahasa Indonesia memakai ';' dengan koma desimal (1,5)
        decimal = "," if sep == ";" else "."
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, decimal=decimal, encoding=enc,
                               low_memory=False, on_bad_lines="skip")
        except Exception as err:  # noqa: BLE001 - pesan diteruskan ke pengguna
            raise DataLoadError(f"File CSV tidak bisa dibaca: {err}") from err
    if name.endswith(".xlsx"):
        try:
            # nrows membatasi memori untuk file Excel yang sangat panjang
            return pd.read_excel(io.BytesIO(raw), engine="openpyxl", nrows=config.MAX_EXCEL_ROWS)
        except Exception as err:  # noqa: BLE001
            raise DataLoadError(f"File Excel tidak bisa dibaca: {err}") from err
    raise DataLoadError("Format tidak didukung. Gunakan file .csv atau .xlsx.")


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Perkecil tipe numerik dan ubah teks berulang menjadi category untuk menghemat RAM."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            if len(s) and s.nunique(dropna=True) / len(s) < 0.5:
                df[col] = s.astype("category")
    return df


def sample_rows(df: pd.DataFrame, max_rows: int = config.MAX_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=config.RANDOM_STATE).reset_index(drop=True)


def load_dataset(raw: bytes, filename: str, max_rows: int = config.MAX_ROWS) -> LoadResult:
    df = read_table(raw, filename)
    if df.empty or df.shape[1] == 0:
        raise DataLoadError("File kosong atau tidak memiliki kolom.")
    df.columns = _unique_names([str(c).strip() or "kolom" for c in df.columns])
    num = df.select_dtypes(include=np.number).columns
    if len(num):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)  # inf diperlakukan sebagai data kosong
    original_rows = len(df)
    df = sample_rows(df, max_rows)
    return LoadResult(optimize_dtypes(df), original_rows, original_rows > len(df))


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        seen.setdefault(n, 0)
        out.append(n)
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=np.number).columns.tolist()


@dataclass
class CleaningReport:
    duplicates_removed: int = 0
    rows_before: int = 0
    rows_after: int = 0
    filled: dict = field(default_factory=dict)  # kolom -> (jumlah diisi, metode, nilai)
    dropped_cols: list = field(default_factory=list)


def clean_data(df: pd.DataFrame, drop_empty_threshold: float = 0.9) -> tuple[pd.DataFrame, CleaningReport]:
    """Hapus duplikat, buang kolom yang hampir kosong, isi missing value (median/modus).

    Ditulis dengan assignment eksplisit (bukan inplace) agar benar di pandas Copy-on-Write.
    """
    report = CleaningReport(rows_before=len(df))
    out = df.drop_duplicates().reset_index(drop=True)
    report.duplicates_removed = report.rows_before - len(out)

    for col in list(out.columns):
        missing_ratio = out[col].isna().mean()
        if missing_ratio >= drop_empty_threshold:
            out = out.drop(columns=col)
            report.dropped_cols.append(col)
            continue
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            value = out[col].median()
            out[col] = out[col].fillna(value)
            report.filled[col] = (n_missing, "median", round(float(value), 4))
        else:
            mode = out[col].mode(dropna=True)
            value = mode.iloc[0] if not mode.empty else "Tidak diketahui"
            if isinstance(out[col].dtype, pd.CategoricalDtype) and value not in out[col].cat.categories:
                out[col] = out[col].cat.add_categories([value])
            out[col] = out[col].fillna(value)
            report.filled[col] = (n_missing, "modus", value)

    report.rows_after = len(out)
    return out, report


def summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
