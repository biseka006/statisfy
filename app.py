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
from statify.data import DataLoadError, clean_data, load_dataset  # noqa: E402
from statify.limits import BusyError  # noqa: E402
from statify.theme import apply_theme, hero_svg  # noqa: E402

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_INFO = {
    "iris": ("Iris", "150 bunga, tebak spesiesnya"),
    "diabetes_progresi": ("Diabetes", "442 pasien, prediksi angka progresi"),
    "penjualan_toko": ("Penjualan toko", "612 transaksi, ada data kosong & duplikat"),
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


def sidebar() -> None:
    sb = st.sidebar
    sb.markdown(f"<div class='sf-wordmark'>Stat<b>ify</b></div><div class='sf-tagline'>{config.APP_TAGLINE}</div>",
                unsafe_allow_html=True)
    sb.write("")
    if "df" in st.session_state:
        info = st.session_state["load_info"]
        rows, cols = st.session_state["df"].shape
        with sb.container(border=True):
            st.markdown(f"**{info['name']}**")
            st.caption(f"{rows:,} baris, {cols} kolom"
                       + (f" (sampel dari {info['original_rows']:,})" if info["sampled"] else ""))
            with st.popover("Ganti dataset", icon=":material/swap_horiz:", use_container_width=True):
                uploader("Unggah file lain", "side")
            if st.button("Hapus data", icon=":material/delete:", use_container_width=True):
                clear_data()
                st.session_state["uploader_key"] += 1
                st.rerun()
    else:
        sb.caption("Belum ada data. Unggah file di halaman utama.")

    sb.divider()
    sb.caption("Tema terang atau gelap bisa dipilih di menu ⋮ → Settings.")
    sb.caption(f"Data hanya tersimpan di sesi ini dan hilang saat dihapus atau tab ditutup. Statify v{__version__}.")


def welcome() -> None:
    st.markdown(
        f"""<section class='sf-hero'>
          <div>
            <h1>Dari tabel mentah ke jawaban yang jelas.</h1>
            <p>Unggah CSV atau Excel. Statify membersihkan data, menampilkan grafik yang paling penting,
            dan melatih model, lalu menjelaskan hasilnya dengan bahasa sehari-hari.</p>
          </div>
          <div class='sf-plot'>{hero_svg()}</div>
        </section>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Unggah data Anda**")
            uploader("Pilih file CSV atau Excel", "main")
    with right:
        with st.container(border=True):
            st.markdown("**Atau coba dataset contoh**")
            samples = sorted(SAMPLE_DIR.glob("*.csv")) if SAMPLE_DIR.exists() else []
            for path in samples:
                title, desc = SAMPLE_INFO.get(path.stem, (path.stem.replace("_", " ").title(), ""))
                if st.button(f"**{title}** · {desc}" if desc else title, key=f"sample_{path.stem}",
                             use_container_width=True):
                    if ingest(path.read_bytes(), path.name, f"sample:{path.name}"):
                        st.rerun()

    st.markdown(
        """<ol class='sf-steps'>
          <li><strong>Unggah data</strong><span>Duplikat dan sel kosong dibereskan otomatis.</span></li>
          <li><strong>Pilih analisis</strong><span>Eksplorasi, modeling, clustering, atau Auto-ML.</span></li>
          <li><strong>Baca penjelasannya</strong><span>Setiap grafik dan skor diberi arti singkat.</span></li>
        </ol>
        <div class='sf-features'>
          <div><strong>Cleaning otomatis</strong><span>Hapus duplikat, isi data kosong, unduh hasilnya sebagai CSV.</span></div>
          <div><strong>Eksplorasi dengan alasan</strong><span>Empat grafik paling informatif dipilih otomatis, lengkap dengan alasannya.</span></div>
          <div><strong>Modeling</strong><span>Klasifikasi dan regresi dengan Random Forest, Logistic/Linear, dan XGBoost.</span></div>
          <div><strong>Clustering</strong><span>K-Means, Hierarchical, dan DBSCAN beserta skor kualitas cluster.</span></div>
          <div><strong>Auto-ML</strong><span>Cukup pilih kolom yang ingin ditebak; model terbaik dipilihkan.</span></div>
          <div><strong>Privasi</strong><span>Data tidak disimpan ke disk server dan hilang saat sesi berakhir.</span></div>
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
