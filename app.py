"""Statify — Stats and Simplify.

Jalankan: streamlit run app.py
"""
import logging
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Statify — Stats and Simplify", page_icon="📊", layout="wide",
                   initial_sidebar_state="auto")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from statify import __version__, config, views  # noqa: E402
from statify.data import DataLoadError, clean_data, load_dataset, summary  # noqa: E402
from statify.limits import BusyError  # noqa: E402
from statify.theme import apply_theme, esc, hero_svg  # noqa: E402

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_INFO = {
    "iris": ("🌸 Iris Dataset", "150 baris · 5 kolom · Klasifikasi spesies bunga"),
    "diabetes_progresi": ("🩺 Diabetes Progresi", "442 baris · 11 kolom · Regresi skor progresi"),
    "penjualan_toko": ("🛒 Penjualan Toko", "612 baris · 8 kolom · Uji cleaning & transaksi"),
}
PAGES = {
    "overview": ":material/table_view: Overview",
    "cleaning": ":material/cleaning_services: Cleaning",
    "eda": ":material/insights: Eksplorasi",
    "ml": ":material/model_training: Modeling",
    "clustering": ":material/bubble_chart: Clustering",
    "automl": ":material/auto_awesome: Auto-ML",
}
DATA_KEYS = ("df", "df_cleaned", "clean_report", "load_info", "file_id",
             "ml_result", "automl_result", "cluster_result", "elbow")


def init_state() -> None:
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("page", "overview")


def clear_data() -> None:
    for key in DATA_KEYS:
        st.session_state.pop(key, None)


def ingest(raw: bytes, name: str, file_id: str) -> bool:
    """Muat + cleaning SEKALI per file. Rerun berikutnya memakai hasil di session."""
    if st.session_state.get("file_id") == file_id:
        return False
    if len(raw) > config.MAX_UPLOAD_MB * 1024**2:
        st.error(f"File lebih besar dari {config.MAX_UPLOAD_MB} MB. Kurangi baris atau kolomnya, lalu unggah lagi.")
        return False
    with st.spinner("Membaca dan membersihkan data..."):
        try:
            loaded = load_dataset(raw, name)
        except DataLoadError as err:
            st.error(str(err))
            return False
        cleaned, report = clean_data(loaded.df)
    clear_data()
    st.session_state.update(
        df=loaded.df, df_cleaned=cleaned, clean_report=report, file_id=file_id, page="overview",
        load_info={"name": name, "original_rows": loaded.original_rows, "sampled": loaded.sampled},
    )
    st.toast(f"{name} siap dianalisis ({len(loaded.df):,} baris)", icon=":material/check_circle:")
    return True


def uploader(label: str, key_suffix: str) -> None:
    f = st.file_uploader(label, type=["csv", "xlsx"], key=f"up_{key_suffix}_{st.session_state['uploader_key']}",
                         help=f"CSV atau Excel, maks. {config.MAX_UPLOAD_MB} MB. Data lebih dari "
                              f"{config.MAX_ROWS:,} baris diambil sampelnya secara acak.")
    if f is not None and ingest(f.getvalue(), f.name, f"upload:{f.name}:{f.size}"):
        st.rerun()


def _health_score(df) -> int:
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing_ratio = df.isna().sum().sum() / total_cells
    dup_ratio = df.duplicated().sum() / max(len(df), 1)
    score = int(round(max(0.0, 100.0 * (1.0 - 0.7 * missing_ratio - 0.3 * dup_ratio))))
    return min(100, max(0, score))


def sidebar() -> None:
    sb = st.sidebar
    sb.markdown(
        f"<div class='sf-brand-box'>"
        f"<div class='sf-wordmark'>Stat<b>ify</b></div>"
        f"<span class='sf-badge-ver'>v{__version__}</span>"
        f"</div>"
        f"<div class='sf-tagline'>{config.APP_TAGLINE} · Interactive Data Studio</div>",
        unsafe_allow_html=True,
    )
    sb.write("")
    if "df" in st.session_state:
        info = st.session_state["load_info"]
        raw = st.session_state["df"]
        cleaned = st.session_state["df_cleaned"]
        rows, cols = cleaned.shape
        raw_health = _health_score(raw)
        clean_health = _health_score(cleaned)

        with sb.container(border=True):
            st.markdown(f"**:material/dataset: {esc(info['name'])}**")
            st.caption(
                f"{rows:,} baris bersih · {cols} kolom"
                + (f" (sampel dari {info['original_rows']:,})" if info["sampled"] else "")
            )
            st.progress(clean_health / 100.0, text=f"Kualitas data: {clean_health}% (mentah {raw_health}%)")

            with st.popover("Ganti dataset", icon=":material/swap_horiz:", use_container_width=True):
                uploader("Unggah file baru", "side")
                st.divider()
                st.caption("Atau pilih dataset contoh:")
                samples = sorted(SAMPLE_DIR.glob("*.csv")) if SAMPLE_DIR.exists() else []
                for path in samples:
                    title, _ = SAMPLE_INFO.get(path.stem, (path.stem.title(), ""))
                    if st.button(title, key=f"side_sample_{path.stem}", use_container_width=True):
                        if ingest(path.read_bytes(), path.name, f"sample:{path.name}"):
                            st.rerun()

            if st.button("Tutup & hapus data", icon=":material/delete:", use_container_width=True):
                clear_data()
                st.session_state["uploader_key"] += 1
                st.rerun()

        # Status progres sesi interaktif
        steps_done = [
            ("Cleaning Otomatis", True),
            ("Model ML", "ml_result" in st.session_state),
            ("Clustering", "cluster_result" in st.session_state),
            ("Auto-ML", "automl_result" in st.session_state),
        ]
        items_html = "".join(
            f"<div class='sf-status-item'><span>{label}</span>"
            f"<span class='{'sf-dot-ok' if ok else 'sf-dot-wait'}' title='{'Selesai' if ok else 'Belum dijalankan'}'></span></div>"
            for label, ok in steps_done
        )
        sb.markdown("**Status Analisis Sesi**")
        sb.markdown(f"<div class='sf-status-list'>{items_html}</div>", unsafe_allow_html=True)
    else:
        with sb.container(border=True):
            st.markdown("**:material/cloud_upload: Belum ada dataset aktif**")
            st.caption("Unggah file CSV/Excel atau klik salah satu dataset contoh di halaman utama untuk mulai bereksplorasi.")

    sb.divider()
    sb.caption("💡 **Tips:** Tema terang/gelap dapat diubah lewat menu **⋮ → Settings**.")
    sb.caption("🔒 Privasi terjaga: Data hanya di memori sesi Anda dan otomatis hilang saat tab ditutup.")


def welcome() -> None:
    st.markdown(
        f"""<section class='sf-hero'>
          <div>
            <div class='sf-kicker'>✨ Tanpa Koding · Instan · Interaktif</div>
            <h1>Dari tabel mentah ke <span>keputusan cerdas.</span></h1>
            <p>Unggah CSV atau Excel Anda. Statify membersihkan data secara otomatis, menyajikan visualisasi interaktif,
            melatih model Machine Learning, hingga menyediakan simulator prediksi langsung.</p>
            <div class='sf-pills-row'>
              <span class='sf-mini-pill'>🧹 Auto-Cleaning</span>
              <span class='sf-mini-pill'>📊 Smart EDA</span>
              <span class='sf-mini-pill'>🤖 ML + Simulator</span>
              <span class='sf-mini-pill'>🔍 3 Metode Clustering</span>
            </div>
          </div>
          <div class='sf-plot'>{hero_svg()}</div>
        </section>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### :material/upload_file: Unggah Data Anda")
            st.caption("Mendukung format `.csv` (koma/titik-koma otomatis) dan `.xlsx` hingga 20 MB.")
            uploader("Pilih atau seret file CSV / Excel ke sini", "main")
    with right:
        with st.container(border=True):
            st.markdown("### :material/science: Coba Dataset Interaktif")
            st.caption("Klik salah satu dataset di bawah untuk langsung mencoba seluruh fitur:")
            samples = sorted(SAMPLE_DIR.glob("*.csv")) if SAMPLE_DIR.exists() else []
            for path in samples:
                title, desc = SAMPLE_INFO.get(path.stem, (path.stem.replace("_", " ").title(), ""))
                if st.button(f"**{title}** — {desc}" if desc else title, key=f"sample_{path.stem}",
                             use_container_width=True):
                    if ingest(path.read_bytes(), path.name, f"sample:{path.name}"):
                        st.rerun()

    st.markdown(
        """<ol class='sf-steps'>
          <li><strong>Unggah atau Pilih Contoh</strong><span>Deteksi pemisah CSV otomatis, hapus duplikat, dan isi data kosong dalam sekejap.</span></li>
          <li><strong>Eksplorasi & Eksperimen</strong><span>Filter baris, inspeksi distribusi tiap kolom, atur parameter cleaning, dan buat grafik kustom.</span></li>
          <li><strong>Latih Model & Simulasikan</strong><span>Bandingkan Random Forest, Regression, & XGBoost, lalu uji tebakan data baru lewat simulator interaktif.</span></li>
        </ol>
        <div class='sf-features'>
          <div><strong>🧹 Cleaning Kustom & Otomatis</strong><span>Atur ambang batas kolom kosong, metode pengisian nilai, lihat perbandingan sebelum/sesudah, dan unduh CSV bersih.</span></div>
          <div><strong>📊 Eksplorasi & Inspektor Kolom</strong><span>Empat grafik insight otomatis + inspektor profil tiap kolom dan pembuat grafik bebas (Histogram, Box, Violin, Scatter, Line).</span></div>
          <div><strong>🎯 Modeling + Simulator Prediksi</strong><span>Latih model klasifikasi/regresi dan gunakan slider interaktif untuk mengetes prediksi pada skenario baru secara langsung.</span></div>
          <div><strong>🔍 Clustering Multi-Dimensi</strong><span>K-Means, Hierarchical, & DBSCAN dengan kurva Elbow/Silhouette dan visualisasi sebaran PCA maupun sumbu kolom asli.</span></div>
          <div><strong>✨ Auto-ML Satu Klik</strong><span>Cukup tentukan kolom target; Statify mendeteksi tipe masalah, memilih fitur terbaik, dan menjelaskan hasilnya.</span></div>
          <div><strong>🔒 Privasi & Performa Terjaga</strong><span>Pemrosesan 100% di memori sesi tanpa menulis ke disk, dilengkapi antrean pintar agar server selalu responsif.</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


def nav() -> str:
    choice = st.pills("Menu", list(PAGES), format_func=PAGES.get, key="page",
                      selection_mode="single", label_visibility="collapsed")
    return choice or "overview"


def main() -> None:
    init_state()
    apply_theme()
    sidebar()

    if "df" not in st.session_state:
        welcome()
        return

    page = nav()
    raw, cleaned = st.session_state["df"], st.session_state["df_cleaned"]
    render = {
        "overview": lambda: views.page_overview(raw),
        "cleaning": lambda: views.page_cleaning(raw, cleaned),
        "eda": lambda: views.page_eda(cleaned),
        "ml": lambda: views.page_ml(cleaned),
        "clustering": lambda: views.page_clustering(cleaned),
        "automl": lambda: views.page_automl(cleaned),
    }[page]
    try:
        render()
    except BusyError as err:
        st.warning(str(err), icon=":material/hourglass_top:")
    except MemoryError:
        st.error("Memori server tidak cukup untuk operasi ini. Kurangi kolom yang dipilih atau gunakan data lebih kecil.")
    except Exception as err:  # noqa: BLE001 - pesan ramah ke pengguna, detail ke log server
        st.error("Data ini tidak bisa diproses dengan pilihan sekarang. Ubah kolom atau parameter lalu coba lagi.")
        print(f"[statify] error di halaman {page}: {err!r}", flush=True)
        import traceback
        traceback.print_exc()


main()
