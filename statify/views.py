"""Halaman-halaman Statify. Hasil berat (model, cluster) disimpan di session agar tidak dihitung ulang."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from . import clustering, config, eda, ml
from .limits import heavy_job
from .data import categorical_columns, numeric_columns, summary
from .theme import current_theme, esc, note, page_header, show_figure


# ---------- cache ringan (dibatasi jumlah entri & umur) ----------
@st.cache_data(ttl=config.CACHE_TTL, max_entries=config.CACHE_MAX_ENTRIES, show_spinner=False)
def _corr(df: pd.DataFrame) -> pd.DataFrame:
    return eda.correlation_matrix(df)


@st.cache_data(ttl=config.CACHE_TTL, max_entries=config.CACHE_MAX_ENTRIES, show_spinner=False)
def _insights(df: pd.DataFrame) -> list[dict]:
    return eda.auto_insights(df)


def _bullets(lines: list[str]) -> None:
    st.markdown("\n".join(f"- {line}" for line in lines))


# ---------- 1. Upload & Overview ----------
def page_overview(df: pd.DataFrame) -> None:
    page_header("Overview data", "Gambaran cepat isi file: ukuran, tipe kolom, data kosong, dan statistik dasar.")
    info = st.session_state.get("load_info", {})
    if info.get("sampled"):
        st.info(f"Data disampling acak menjadi {len(df):,} dari {info['original_rows']:,} baris "
                f"(batas server {config.MAX_ROWS:,} baris).")
    s = summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", f"{s['rows']:,}", border=True)
    c2.metric("Kolom", s["cols"], border=True)
    c3.metric("Missing value", f"{s['missing']:,}", border=True)
    mem = s["memory_mb"]
    c4.metric("Memori", f"{mem:.1f} MB" if mem >= 0.1 else f"{mem * 1024:.0f} KB", border=True)

    st.subheader("5 baris pertama")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Tipe & kelengkapan kolom")
    info_df = pd.DataFrame({
        "Tipe": df.dtypes.astype(str),
        "Nilai unik": df.nunique(),
        "Missing": df.isna().sum(),
        "Missing (%)": (df.isna().mean() * 100).round(1),
    })
    st.dataframe(info_df, use_container_width=True)

    num = numeric_columns(df)
    if num:
        st.subheader("Statistik deskriptif (kolom numerik)")
        st.dataframe(df[num].describe().T.round(3), use_container_width=True)


# ---------- 2. Cleaning ----------
def page_cleaning(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    page_header("Data cleaning", "Apa saja yang dibereskan secara otomatis saat file diunggah, dan unduhan versi bersihnya.")
    rep = st.session_state["clean_report"]
    st.caption("Cleaning otomatis dijalankan sekali saat file diunggah: hapus duplikat, buang kolom "
               "≥ 90% kosong, isi missing value numerik dengan median dan kategori dengan modus.")

    before, after = summary(raw), summary(cleaned)
    comp = pd.DataFrame({
        "Sebelum": [before["rows"], before["cols"], before["missing"], before["duplicates"]],
        "Sesudah": [after["rows"], after["cols"], after["missing"], after["duplicates"]],
    }, index=["Baris", "Kolom", "Missing value", "Duplikat"])
    st.dataframe(comp, use_container_width=True)

    if rep.duplicates_removed:
        st.write(f":material/content_copy: {rep.duplicates_removed:,} baris duplikat dihapus.")
    if rep.dropped_cols:
        st.write(f":material/delete_sweep: Kolom dibuang karena hampir kosong: {', '.join(rep.dropped_cols)}")
    if rep.filled:
        st.write(":material/healing: Missing value yang diisi:")
        st.dataframe(pd.DataFrame(
            [{"Kolom": c, "Jumlah diisi": n, "Metode": m, "Nilai pengisi": str(v)}
             for c, (n, m, v) in rep.filled.items()]), use_container_width=True, hide_index=True)
    if not (rep.duplicates_removed or rep.dropped_cols or rep.filled):
        st.success("Data sudah bersih: tidak ada duplikat maupun missing value.")

    st.download_button("Unduh data bersih (CSV)", icon=":material/download:", data=cleaned.to_csv(index=False).encode("utf-8"),
                       file_name="statify_data_bersih.csv", mime="text/csv")


# ---------- 3. EDA ----------
def page_eda(df: pd.DataFrame) -> None:
    page_header("Eksplorasi data", "Pola, hubungan antar-kolom, dan nilai yang tidak biasa, beserta cara membacanya.")
    tab_auto, tab_corr, tab_custom = st.tabs([":material/lightbulb: Insight otomatis", ":material/hub: Korelasi", ":material/tune: Grafik bebas"])

    with tab_auto:
        items = _insights(df)
        if not items:
            st.info("Belum ada insight: data perlu minimal satu kolom numerik atau kategori.")
        for i in range(0, len(items), 2):
            for col, item in zip(st.columns(2, gap="large"), items[i:i + 2]):
                with col:
                    st.markdown(f"**{esc(item['title'])}**")
                    show_figure(eda.render_insight(df, item))
                    note(esc(item["why"]))

    with tab_corr:
        corr = _corr(df)
        if corr.shape[0] < 2:
            st.info("Butuh minimal dua kolom numerik untuk analisis korelasi.")
        else:
            if len(numeric_columns(df)) > config.MAX_HEATMAP_COLS:
                st.caption(f"Menampilkan {config.MAX_HEATMAP_COLS} kolom numerik paling bervariasi.")
            show_figure(eda.plot_heatmap(corr))
            note("<b>Cara membaca:</b> mendekati 1 berarti naik bersama, mendekati -1 berlawanan arah, "
                       "dan sekitar 0 berarti tidak ada hubungan linear.")
            st.subheader("10 pasangan korelasi terkuat")
            top = eda.top_correlations(corr)
            top["Korelasi"] = top["Korelasi"].round(3)
            st.dataframe(top, use_container_width=True, hide_index=True)

            st.subheader("Pairplot")
            num = corr.columns.tolist()
            cols = st.multiselect(f"Pilih maks. {config.PAIRPLOT_MAX_COLS} kolom", num,
                                  default=num[: min(3, len(num))], max_selections=config.PAIRPLOT_MAX_COLS)
            hue_opts = ["(tanpa warna)"] + [c for c in categorical_columns(df) if df[c].nunique() <= 10]
            hue = st.selectbox("Warnai berdasarkan", hue_opts)
            if st.button("Buat pairplot", disabled=len(cols) < 2):
                with heavy_job(f"Menggambar pairplot dari sampel {config.PAIRPLOT_SAMPLE:,} baris..."):
                    show_figure(eda.plot_pairplot(df, cols, None if hue == hue_opts[0] else hue))

    with tab_custom:
        num, cat = numeric_columns(df), categorical_columns(df)
        kind = st.selectbox("Jenis grafik", ["Histogram", "Box plot", "Scatter", "Bar (kategori)"])
        if kind in ("Histogram", "Box plot", "Scatter") and not num:
            st.info("Grafik ini butuh kolom numerik.")
        elif kind == "Histogram":
            show_figure(eda.plot_histogram(df, st.selectbox("Kolom", num)))
        elif kind == "Box plot":
            col = st.selectbox("Kolom", num)
            by = st.selectbox("Kelompokkan per", ["(tidak)"] + cat)
            show_figure(eda.plot_box(df, col, None if by == "(tidak)" else by))
        elif kind == "Scatter":
            if len(num) < 2:
                st.info("Scatter butuh dua kolom numerik.")
            else:
                x = st.selectbox("Sumbu X", num)
                y = st.selectbox("Sumbu Y", [c for c in num if c != x])
                hue = st.selectbox("Warna", ["(tidak)"] + [c for c in cat if df[c].nunique() <= 10])
                show_figure(eda.plot_scatter(df, x, y, None if hue == "(tidak)" else hue))
        else:
            if not cat:
                st.info("Tidak ada kolom kategori.")
            else:
                show_figure(eda.plot_bar(df, st.selectbox("Kolom", cat)))


# ---------- tampilan hasil model (dipakai ML & Auto-ML) ----------
def _show_train_result(res: ml.TrainResult) -> None:
    st.subheader("Perbandingan model")
    table = ml.metrics_table(res)
    key = "F1-Score" if res.problem_type == ml.CLASSIFICATION else "R²"
    fmt = {c: ("{:,.3f}" if table[c].abs().max() < 100 else "{:,.0f}") for c in table.columns}
    styled = (table.style.format(fmt, na_rep="–")
              .highlight_max(subset=[key], props=f"background-color: {current_theme()['accent_soft']}; font-weight: 700"))
    st.dataframe(styled, use_container_width=True)
    st.success(f"Model terbaik: **{res.best_name}** ({key} = {res.best.metrics[key]:.3f})")

    name = st.selectbox("Lihat detail model", list(res.results), index=list(res.results).index(res.best_name))
    r = res.results[name]
    left, right = st.columns(2, gap="medium")
    with left:
        if res.problem_type == ml.CLASSIFICATION:
            labels = [str(c)[:15] for c in res.class_labels]
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(r.confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                        yticklabels=labels, ax=ax, cbar=False)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title(f"Confusion Matrix — {name}")
        else:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(res.y_test, r.y_pred, s=12, alpha=0.6, color=current_theme()["accent"])
            lo, hi = float(np.min(res.y_test)), float(np.max(res.y_test))
            ax.plot([lo, hi], [lo, hi], "--", color=current_theme()["marker"], lw=1)
            ax.set_xlabel("Aktual")
            ax.set_ylabel("Prediksi")
            ax.set_title(f"Aktual vs Prediksi — {name}")
        fig.tight_layout()
        show_figure(fig)
    with right:
        if r.importance is not None:
            top = r.importance.head(10)[::-1]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(top.index.astype(str), top.values, color=current_theme()["accent"])
            ax.set_title("10 fitur paling berpengaruh")
            fig.tight_layout()
            show_figure(fig)
    st.subheader("Artinya")
    _bullets(ml.interpret(res))


# ---------- 4. ML Modeling ----------
def page_ml(df: pd.DataFrame) -> None:
    page_header("Modeling", "Latih beberapa model untuk menebak satu kolom, lalu bandingkan ketepatannya.")
    with st.form("ml_form"):
        c1, c2 = st.columns(2)
        target = c1.selectbox("Kolom target", df.columns.tolist(), index=len(df.columns) - 1)
        problem = c2.selectbox("Tipe masalah", ["Deteksi otomatis", ml.CLASSIFICATION, ml.REGRESSION])
        names_cls = ["Random Forest", "Logistic Regression"] + (["XGBoost"] if ml.HAS_XGB else [])
        names_reg = ["Random Forest", "Linear Regression"] + (["XGBoost"] if ml.HAS_XGB else [])
        c3, c4 = st.columns(2)
        n_trees = c3.slider("Jumlah pohon (RF/XGBoost)", 10, config.MAX_TREES, 100, step=10)
        test_size = c4.slider("Porsi data uji", 0.1, 0.4, 0.2, step=0.05)
        use_smote = st.checkbox("Seimbangkan kelas dengan SMOTE (khusus klasifikasi)",
                                disabled=not ml.HAS_SMOTE)
        st.caption(f"Model klasifikasi: {', '.join(names_cls)} · Model regresi: {', '.join(names_reg)}")
        submitted = st.form_submit_button("Latih model", type="primary", icon=":material/play_arrow:")

    if submitted:
        ptype = None if problem == "Deteksi otomatis" else problem
        with heavy_job("Melatih model..."):
            try:
                st.session_state["ml_result"] = ml.train_models(
                    df, target, ptype, n_trees=n_trees, test_size=test_size, use_smote=use_smote)
            except ml.ModelingError as err:
                st.session_state.pop("ml_result", None)
                st.error(str(err))

    res = st.session_state.get("ml_result")
    if res is None:
        st.info("Pilih target lalu tekan **Latih model**. Training hanya berjalan saat tombol ditekan.")
        return
    st.caption(f"Hasil terakhir: target **{res.target}**, {res.problem_type.lower()}, "
               f"{len(res.feature_names)} fitur.")
    _show_train_result(res)


# ---------- 5. Clustering ----------
def page_clustering(df: pd.DataFrame) -> None:
    page_header("Clustering", "Kelompokkan baris yang mirip tanpa perlu kolom jawaban.")
    num = [c for c in numeric_columns(df) if df[c].nunique() > 1]
    if len(num) < 2:
        st.warning("Clustering butuh minimal dua kolom numerik.")
        return

    with st.form("cluster_form"):
        cols = st.multiselect("Kolom untuk clustering", num, default=num[: min(4, len(num))])
        method = st.radio("Metode", ["K-Means", "Hierarchical", "DBSCAN"], horizontal=True)
        c1, c2 = st.columns(2)
        k = c1.slider("Jumlah cluster (K-Means/Hierarchical)", 2, 10, 3)
        linkage_m = c2.selectbox("Linkage (Hierarchical)", ["ward", "complete", "average", "single"])
        c3, c4 = st.columns(2)
        eps = c3.slider("eps (DBSCAN)", 0.1, 5.0, 0.5, step=0.1)
        min_samples = c4.slider("min_samples (DBSCAN)", 2, 20, 5)
        st.caption(f"Hierarchical dihitung pada sampel maks. {config.HIERARCHICAL_SAMPLE:,} baris "
                   "karena kebutuhan memorinya tumbuh kuadratik.")
        submitted = st.form_submit_button("Jalankan clustering", type="primary", icon=":material/play_arrow:")

    with st.expander("Bantu pilih jumlah cluster (Elbow & Silhouette)", icon=":material/query_stats:"):
        if st.button("Hitung untuk K = 2–10", disabled=len(cols) < 2):
            with heavy_job("Menghitung..."):
                try:
                    st.session_state["elbow"] = clustering.elbow(df, cols)
                except clustering.ClusteringError as err:
                    st.error(str(err))
        if "elbow" in st.session_state and not st.session_state["elbow"].empty:
            e = st.session_state["elbow"]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
            a1.plot(e["K"], e["Inertia"], marker="o")
            a1.set_title("Elbow (Inertia)")
            a2.plot(e["K"], e["Silhouette"], marker="o", color=current_theme()["marker"])
            a2.set_title("Silhouette")
            fig.tight_layout()
            show_figure(fig)
            st.caption(f"Silhouette tertinggi di K = {int(e.loc[e['Silhouette'].idxmax(), 'K'])}.")

    if submitted:
        if len(cols) < 2:
            st.error("Pilih minimal dua kolom.")
        else:
            with heavy_job(f"Menjalankan {method}..."):
                try:
                    if method == "K-Means":
                        st.session_state["cluster_result"] = clustering.run_kmeans(df, cols, k)
                    elif method == "Hierarchical":
                        st.session_state["cluster_result"] = clustering.run_hierarchical(df, cols, k, linkage_m)
                    else:
                        st.session_state["cluster_result"] = clustering.run_dbscan(df, cols, eps, min_samples)
                except clustering.ClusteringError as err:
                    st.session_state.pop("cluster_result", None)
                    st.error(str(err))

    res: clustering.ClusterResult | None = st.session_state.get("cluster_result")
    if res is None:
        st.info("Atur parameter lalu tekan **Jalankan clustering**.")
        return

    st.subheader(f"Hasil {res.method}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah cluster", res.n_clusters, border=True)
    c2.metric("Silhouette", "-" if res.silhouette is None else f"{res.silhouette:.3f}", border=True)
    c3.metric("Davies-Bouldin", "-" if res.davies_bouldin is None else f"{res.davies_bouldin:.3f}", border=True)
    _bullets(clustering.describe(res))

    pts = clustering.pca_2d(res.scaled)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=res.labels, cmap="tab10", s=14, alpha=0.75)
    ax.legend(*sc.legend_elements(), title="Cluster", fontsize=8, loc="best")
    ax.set_title("Sebaran cluster (proyeksi PCA 2D)")
    ax.set_xlabel("Komponen 1")
    ax.set_ylabel("Komponen 2")
    fig.tight_layout()
    show_figure(fig)

    if res.linkage_matrix is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        dendrogram(res.linkage_matrix, truncate_mode="lastp", p=30, ax=ax, no_labels=True,
                   color_threshold=None)
        ax.set_title("Dendrogram (30 cabang terakhir)")
        fig.tight_layout()
        show_figure(fig)

    st.subheader("Profil rata-rata tiap cluster")
    profile = res.data.groupby("Cluster").mean(numeric_only=True).round(3)
    profile.insert(0, "Jumlah", res.data["Cluster"].value_counts().sort_index())
    st.dataframe(profile, use_container_width=True)
    st.download_button("Unduh data + label cluster (CSV)", icon=":material/download:", data=res.data.to_csv(index=False).encode("utf-8"),
                       file_name="statify_cluster.csv", mime="text/csv")


# ---------- 6. Auto-ML ----------
def page_automl(df: pd.DataFrame) -> None:
    page_header("Auto-ML", "Pilih kolom yang ingin ditebak. Statify menentukan sisanya dan menjelaskan hasilnya.")
    st.caption("Pilih target saja. Statify menentukan jenis masalah, menyiapkan fitur, melatih semua "
               "model, lalu merekomendasikan yang terbaik.")
    with st.form("automl_form"):
        target = st.selectbox("Kolom yang ingin diprediksi", df.columns.tolist(), index=len(df.columns) - 1)
        submitted = st.form_submit_button("Jalankan Auto-ML", type="primary", icon=":material/auto_awesome:")

    if submitted:
        ptype = ml.detect_problem_type(df[target].dropna())
        with heavy_job(f"Terdeteksi masalah {ptype.lower()}. Melatih model..."):
            try:
                st.session_state["automl_result"] = ml.train_models(df, target, ptype, n_trees=100)
            except ml.ModelingError as err:
                st.session_state.pop("automl_result", None)
                st.error(str(err))

    res = st.session_state.get("automl_result")
    if res is None:
        return
    reason = ("target berupa kategori/teks atau hanya punya sedikit nilai bulat"
              if res.problem_type == ml.CLASSIFICATION else "target numerik dengan banyak nilai berbeda")
    st.info(f"Jenis masalah: **{res.problem_type}**, karena {reason}. "
            f"{len(res.feature_names)} fitur disiapkan otomatis (kategori ≤ 20 nilai di-one-hot, "
            "kolom mirip ID dibuang).")
    _show_train_result(res)
