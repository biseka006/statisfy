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
from .data import CleaningReport, categorical_columns, clean_data, numeric_columns, summary
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
    page_header("Overview Data", "Jelajahi isi tabel, periksa kelengkapan tiap kolom, dan inspeksi distribusi data secara interaktif.")
    info = st.session_state.get("load_info", {})
    if info.get("sampled"):
        st.info(f"Data disampling acak menjadi {len(df):,} dari {info['original_rows']:,} baris "
                f"(batas server {config.MAX_ROWS:,} baris).")

    s = summary(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baris", f"{s['rows']:,}", border=True)
    c2.metric("Kolom", s["cols"], border=True)
    c3.metric("Missing Value", f"{s['missing']:,}", border=True)
    c4.metric("Baris Duplikat", f"{s['duplicates']:,}", border=True)
    mem = s["memory_mb"]
    c5.metric("Memori", f"{mem:.1f} MB" if mem >= 0.1 else f"{mem * 1024:.0f} KB", border=True)

    tab_table, tab_cols, tab_inspect, tab_stats = st.tabs([
        ":material/table: Penjelajah Tabel",
        ":material/fact_check: Kelengkapan Kolom",
        ":material/manage_search: Inspektor Kolom",
        ":material/analytics: Statistik Deskriptif",
    ])

    with tab_table:
        cleaned = st.session_state.get("df_cleaned", df)
        ctl1, ctl2, ctl3 = st.columns([1.2, 1.8, 1])
        dataset_mode = ctl1.radio(
            "Versi data", ["Data Mentah", "Data Bersih"], horizontal=True, key="ov_mode"
        )
        active_df = df if dataset_mode == "Data Mentah" else cleaned
        search_q = ctl2.text_input("Cari kata / angka di tabel", placeholder="Ketik untuk menyaring baris...", key="ov_search")
        n_preview = ctl3.slider("Jumlah baris tampil", 5, min(100, max(5, len(active_df))), min(15, len(active_df)), key="ov_rows")

        selected_cols = st.multiselect(
            "Kolom yang ditampilkan (kosongkan untuk melihat semua kolom)",
            active_df.columns.tolist(),
            default=[],
            key="ov_cols",
        )
        view_df = active_df[selected_cols] if selected_cols else active_df
        if search_q.strip():
            mask = view_df.astype(str).apply(lambda col: col.str.contains(search_q.strip(), case=False, na=False)).any(axis=1)
            view_df = view_df[mask]
            st.caption(f"Menampilkan {min(n_preview, len(view_df)):,} dari {len(view_df):,} baris yang cocok dengan pencarian **'{esc(search_q)}'**.")
        st.dataframe(view_df.head(n_preview), use_container_width=True)

    with tab_cols:
        missing_pct = (df.isna().mean() * 100).round(1)
        filled_pct = (100 - missing_pct).round(1)
        info_df = pd.DataFrame({
            "Kolom": df.columns,
            "Tipe Data": df.dtypes.astype(str).values,
            "Nilai Unik": df.nunique().values,
            "Missing (Sel)": df.isna().sum().values,
            "Kelengkapan (%)": filled_pct.values,
            "Contoh Nilai": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
        })
        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Kelengkapan (%)": st.column_config.ProgressColumn(
                    "Kelengkapan (%)",
                    help="Persentase sel yang terisi (tidak kosong)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
        )

    with tab_inspect:
        col_pick = st.selectbox("Pilih kolom untuk melihat profil detail & grafiknya", df.columns.tolist(), key="ov_inspect_col")
        s_col = df[col_pick]
        left, right = st.columns([1, 1.4], gap="large")
        with left:
            with st.container(border=True):
                st.markdown(f"#### Profil `{esc(col_pick)}`")
                st.write(f"- **Tipe data:** `{s_col.dtype}`")
                st.write(f"- **Terisi:** {s_col.notna().sum():,} baris ({(s_col.notna().mean() * 100):.1f}%)")
                st.write(f"- **Kosong (Missing):** {s_col.isna().sum():,} baris")
                st.write(f"- **Nilai unik:** {s_col.nunique(dropna=True):,}")
                if pd.api.types.is_numeric_dtype(s_col) and s_col.notna().any():
                    st.divider()
                    m1, m2 = st.columns(2)
                    m1.metric("Rata-rata (Mean)", f"{s_col.mean():,.2f}")
                    m2.metric("Nilai Tengah (Median)", f"{s_col.median():,.2f}")
                    m3, m4 = st.columns(2)
                    m3.metric("Minimum", f"{s_col.min():,.2f}")
                    m4.metric("Maksimum", f"{s_col.max():,.2f}")
                elif s_col.notna().any():
                    st.divider()
                    vc = s_col.astype(str).value_counts().head(5)
                    st.caption("5 nilai yang paling sering muncul:")
                    st.dataframe(
                        pd.DataFrame({"Nilai": vc.index, "Jumlah": vc.values, "Porsi (%)": (vc.values / len(s_col) * 100).round(1)}),
                        use_container_width=True,
                        hide_index=True,
                    )
        with right:
            if pd.api.types.is_numeric_dtype(s_col) and s_col.nunique(dropna=True) > 1:
                show_figure(eda.plot_histogram(df, col_pick))
            elif s_col.notna().any():
                show_figure(eda.plot_bar(df, col_pick))
            else:
                st.info("Kolom ini seluruhnya kosong.")

    with tab_stats:
        num = numeric_columns(df)
        cat = categorical_columns(df)
        if num:
            st.subheader("Kolom Numerik")
            st.dataframe(df[num].describe().T.round(3), use_container_width=True)
        if cat:
            st.subheader("Kolom Kategori / Teks")
            cat_desc = pd.DataFrame({
                "Jumlah Terisi": df[cat].notna().sum(),
                "Nilai Unik": df[cat].nunique(),
                "Modus (Terbanyak)": [df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else "—" for c in cat],
            })
            st.dataframe(cat_desc, use_container_width=True)


# ---------- 2. Cleaning ----------
def _custom_clean(df: pd.DataFrame, drop_dups: bool, drop_thresh: float, num_strategy: str) -> tuple[pd.DataFrame, CleaningReport]:
    if drop_dups and num_strategy == "median":
        return clean_data(df, drop_empty_threshold=drop_thresh)
    report = CleaningReport(rows_before=len(df))
    out = df.drop_duplicates().reset_index(drop=True) if drop_dups else df.copy().reset_index(drop=True)
    report.duplicates_removed = report.rows_before - len(out)

    for col in list(out.columns):
        if out[col].isna().mean() >= drop_thresh:
            out = out.drop(columns=col)
            report.dropped_cols.append(col)
            continue
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            if num_strategy == "mean":
                value = float(out[col].mean())
                label = "mean"
            elif num_strategy == "zero":
                value = 0.0
                label = "nol (0)"
            else:
                value = float(out[col].median())
                label = "median"
            out[col] = out[col].fillna(value)
            report.filled[col] = (n_missing, label, round(float(value), 4))
        else:
            mode = out[col].mode(dropna=True)
            value = mode.iloc[0] if not mode.empty else "Tidak diketahui"
            if isinstance(out[col].dtype, pd.CategoricalDtype) and value not in out[col].cat.categories:
                out[col] = out[col].cat.add_categories([value])
            out[col] = out[col].fillna(value)
            report.filled[col] = (n_missing, "modus", value)

    report.rows_after = len(out)
    return out, report


def page_cleaning(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    page_header("Data Cleaning Interaktif", "Bandingkan kondisi sebelum vs sesudah pembersihan, atau sesuaikan aturan cleaning sesuai kebutuhan Anda.")
    rep = st.session_state["clean_report"]

    before, after = summary(raw), summary(cleaned)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baris Data", f"{after['rows']:,}", delta=after["rows"] - before["rows"], border=True)
    m2.metric("Jumlah Kolom", after["cols"], delta=after["cols"] - before["cols"], border=True)
    m3.metric("Missing Value", f"{after['missing']:,}", delta=after["missing"] - before["missing"], delta_color="inverse", border=True)
    m4.metric("Baris Duplikat", f"{after['duplicates']:,}", delta=after["duplicates"] - before["duplicates"], delta_color="inverse", border=True)

    with st.expander("⚙️ Kustomisasi Aturan Cleaning (Interaktif)", icon=":material/tune:"):
        st.caption("Secara bawaan Statify menghapus duplikat, membuang kolom ≥ 90% kosong, dan mengisi angka kosong dengan median. Anda bisa mengubahnya di sini:")
        c1, c2, c3 = st.columns(3)
        drop_dups = c1.checkbox("Hapus baris duplikat", value=True, key="cl_dups")
        drop_pct = c2.slider("Buang kolom jika kosong melebihi (%)", 50, 100, 90, step=5, key="cl_thresh")
        num_strat = c3.selectbox(
            "Metode isi angka kosong",
            ["median", "mean", "zero"],
            format_func={"median": "Median (Tahan outlier)", "mean": "Rata-rata (Mean)", "zero": "Isi dengan 0"}.get,
            key="cl_strat",
        )
        if st.button("Terapkan Pengaturan Cleaning", type="primary", icon=":material/refresh:"):
            new_cleaned, new_rep = _custom_clean(raw, drop_dups, drop_pct / 100.0, num_strat)
            st.session_state["df_cleaned"] = new_cleaned
            st.session_state["clean_report"] = new_rep
            st.toast("Aturan cleaning berhasil diperbarui!", icon=":material/check_circle:")
            st.rerun()

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Rincian Tindakan Cleaning")
        if rep.duplicates_removed:
            st.info(f":material/content_copy: **{rep.duplicates_removed:,} baris duplikat** ditemukan dan dihapus.")
        if rep.dropped_cols:
            st.warning(f":material/delete_sweep: **Kolom dibuang** karena terlalu kosong: `{', '.join(rep.dropped_cols)}`")
        if rep.filled:
            st.write(":material/healing: **Kolom yang diisi otomatis:**")
            st.dataframe(
                pd.DataFrame([
                    {"Kolom": c, "Sel Diisi": n, "Metode": m, "Nilai Pengisi": str(v)}
                    for c, (n, m, v) in rep.filled.items()
                ]),
                use_container_width=True,
                hide_index=True,
            )
        if not (rep.duplicates_removed or rep.dropped_cols or rep.filled):
            st.success("🎉 Data sudah bersih sejak awal: tidak ada duplikat maupun missing value yang perlu diubah.")

    with right:
        st.subheader("Tabel Perbandingan Ringkas")
        comp = pd.DataFrame({
            "Sebelum Cleaning": [before["rows"], before["cols"], before["missing"], before["duplicates"]],
            "Sesudah Cleaning": [after["rows"], after["cols"], after["missing"], after["duplicates"]],
        }, index=["Baris", "Kolom", "Missing Value", "Baris Duplikat"])
        st.dataframe(comp, use_container_width=True)
        st.download_button(
            "Unduh Data Bersih (CSV)",
            icon=":material/download:",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name="statify_data_bersih.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    with st.expander("🔍 Pratinjau Data Bersih (10 Baris Pertama)", icon=":material/visibility:"):
        st.dataframe(cleaned.head(10), use_container_width=True)


# ---------- 3. EDA ----------
def page_eda(df: pd.DataFrame) -> None:
    page_header("Eksplorasi Data (EDA)", "Temukan pola tersembunyi, korelasi antar-variabel, dan buat visualisasi interaktif.")
    tab_auto, tab_corr, tab_custom = st.tabs([
        ":material/lightbulb: Insight Otomatis",
        ":material/hub: Matriks Korelasi & Pairplot",
        ":material/tune: Studio Grafik Interaktif",
    ])

    with tab_auto:
        items = _insights(df)
        if not items:
            st.info("Belum ada insight: data perlu minimal satu kolom numerik atau kategori.")
        for i in range(0, len(items), 2):
            for col, item in zip(st.columns(2, gap="large"), items[i:i + 2]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"#### {esc(item['title'])}")
                        show_figure(eda.render_insight(df, item))
                        note(esc(item["why"]))

    with tab_corr:
        corr = _corr(df)
        if corr.shape[0] < 2:
            st.info("Butuh minimal dua kolom numerik untuk analisis korelasi.")
        else:
            left_c, right_c = st.columns([1.15, 1], gap="large")
            with left_c:
                if len(numeric_columns(df)) > config.MAX_HEATMAP_COLS:
                    st.caption(f"Menampilkan {config.MAX_HEATMAP_COLS} kolom numerik paling bervariasi.")
                show_figure(eda.plot_heatmap(corr))
                note("<b>Cara membaca:</b> mendekati <b>+1</b> berarti naik bersama, mendekati <b>-1</b> berlawanan arah, "
                     "dan sekitar <b>0</b> berarti tidak ada hubungan linear.")
            with right_c:
                st.subheader("Pasangan Korelasi Terkuat")
                min_r = st.slider("Filter kekuatan korelasi minimal (|r|)", 0.0, 0.9, 0.0, step=0.05, key="eda_min_r")
                top = eda.top_correlations(corr, n=15)
                top = top[top["Korelasi"].abs() >= min_r].reset_index(drop=True)
                top["Korelasi"] = top["Korelasi"].round(3)
                st.dataframe(top, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Pairplot Multi-Kolom")
            num = corr.columns.tolist()
            p1, p2 = st.columns([2, 1])
            cols = p1.multiselect(
                f"Pilih maks. {config.PAIRPLOT_MAX_COLS} kolom numerik",
                num,
                default=num[: min(3, len(num))],
                max_selections=config.PAIRPLOT_MAX_COLS,
            )
            hue_opts = ["(tanpa warna)"] + [c for c in categorical_columns(df) if df[c].nunique() <= 10]
            hue = p2.selectbox("Warnai berdasarkan kategori", hue_opts)
            if st.button("Buat Pairplot", type="primary", icon=":material/grid_view:", disabled=len(cols) < 2):
                with heavy_job(f"Menggambar pairplot dari sampel {config.PAIRPLOT_SAMPLE:,} baris..."):
                    show_figure(eda.plot_pairplot(df, cols, None if hue == hue_opts[0] else hue))

    with tab_custom:
        num, cat = numeric_columns(df), categorical_columns(df)
        kind = st.segmented_control(
            "Pilih jenis visualisasi",
            ["Histogram", "Box plot", "Violin plot", "Scatter", "Bar (kategori)", "Line (tren)"],
            default="Histogram",
        ) or "Histogram"

        if kind in ("Histogram", "Box plot", "Violin plot", "Scatter", "Line (tren)") and not num:
            st.info("Grafik ini membutuhkan minimal satu kolom numerik.")
        elif kind == "Histogram":
            c1, c2, c3 = st.columns([1.5, 1, 1])
            col = c1.selectbox("Kolom numerik", num, key="hist_col")
            bins = c2.slider("Jumlah batang (bins)", 5, 80, 25, key="hist_bins")
            show_kde = c3.checkbox("Tampilkan garis kurva (KDE)", value=True, key="hist_kde")
            fig, ax = plt.subplots(figsize=(7.5, 4))
            sample_s = df[col].dropna()
            if len(sample_s) > config.PLOT_SAMPLE:
                sample_s = sample_s.sample(config.PLOT_SAMPLE, random_state=config.RANDOM_STATE)
            sns.histplot(sample_s, bins=bins, kde=show_kde, ax=ax, color=current_theme()["accent"])
            ax.set_title(f"Distribusi {col} ({bins} bins)")
            fig.tight_layout()
            show_figure(fig)
        elif kind == "Box plot":
            c1, c2 = st.columns(2)
            col = c1.selectbox("Kolom numerik", num, key="box_col")
            by = c2.selectbox("Kelompokkan per kategori", ["(tidak)"] + cat, key="box_by")
            show_figure(eda.plot_box(df, col, None if by == "(tidak)" else by))
        elif kind == "Violin plot":
            c1, c2 = st.columns(2)
            col = c1.selectbox("Kolom numerik", num, key="vio_col")
            by = c2.selectbox("Kelompokkan per kategori", ["(tidak)"] + [c for c in cat if df[c].nunique() <= 10], key="vio_by")
            fig, ax = plt.subplots(figsize=(7.5, 4))
            d = df.sample(config.PLOT_SAMPLE, random_state=config.RANDOM_STATE) if len(df) > config.PLOT_SAMPLE else df
            if by != "(tidak)":
                sns.violinplot(data=d, x=d[by].astype(str), y=col, ax=ax, inner="quartile")
                ax.tick_params(axis="x", rotation=25)
                ax.set_title(f"Kepadatan & Kuartil {col} per {by}")
            else:
                sns.violinplot(x=d[col], ax=ax, inner="quartile", color=current_theme()["accent"])
                ax.set_title(f"Kepadatan & Kuartil {col}")
            fig.tight_layout()
            show_figure(fig)
        elif kind == "Scatter":
            if len(num) < 2:
                st.info("Scatter plot membutuhkan minimal dua kolom numerik.")
            else:
                c1, c2, c3 = st.columns(3)
                x = c1.selectbox("Sumbu X", num, key="sc_x")
                y = c2.selectbox("Sumbu Y", [c for c in num if c != x], key="sc_y")
                hue = c3.selectbox("Warna kategori", ["(tidak)"] + [c for c in cat if df[c].nunique() <= 10], key="sc_hue")
                show_figure(eda.plot_scatter(df, x, y, None if hue == "(tidak)" else hue))
        elif kind == "Line (tren)":
            c1, c2 = st.columns([1.5, 1])
            col = c1.selectbox("Kolom numerik untuk dilihat trennya", num, key="line_col")
            window = c2.slider("Rata-rata bergerak (Moving Average)", 1, 50, 1, help="1 = data asli tanpa penghalusan")
            fig, ax = plt.subplots(figsize=(8, 3.8))
            series = df[col].dropna().reset_index(drop=True).head(config.PLOT_SAMPLE)
            ax.plot(series.index, series.values, alpha=0.45 if window > 1 else 0.85, label="Data Asli", lw=1.2)
            if window > 1:
                ax.plot(series.index, series.rolling(window).mean(), color=current_theme()["marker"], lw=2.2,
                        label=f"Rata-rata bergerak ({window})")
                ax.legend()
            ax.set_title(f"Tren Urutan Baris — {col}")
            ax.set_xlabel("Indeks Baris")
            ax.set_ylabel(col)
            fig.tight_layout()
            show_figure(fig)
        else:
            if not cat:
                st.info("Tidak ada kolom kategori pada dataset ini.")
            else:
                c1, c2 = st.columns([1.5, 1])
                col = c1.selectbox("Kolom kategori", cat, key="bar_col")
                top_n = c2.slider("Tampilkan N kategori teratas", 3, 25, 10, key="bar_topn")
                show_figure(eda.plot_bar(df, col, top=top_n))


# ---------- Simulator Prediksi Interaktif ----------
def _prediction_simulator(res: ml.TrainResult, model_name: str, df_source: pd.DataFrame | None) -> None:
    r = res.results[model_name]
    with st.expander(f"🎯 Simulator Prediksi Interaktif — Uji Coba Model {model_name}", expanded=True, icon=":material/joystick:"):
        st.caption(
            f"Geser nilai fitur di bawah ini untuk melihat bagaimana model **{model_name}** "
            f"menebak nilai **{esc(res.target)}** secara langsung."
        )
        # Tentukan hingga 6 fitur paling berpengaruh agar form ringkas & responsif
        if r.importance is not None and len(r.importance):
            top_feats = [f for f in r.importance.head(6).index if f in res.feature_names]
        else:
            top_feats = res.feature_names[:6]

        X_ref = None
        if df_source is not None and res.target in df_source.columns:
            try:
                X_ref = ml.build_features(df_source[df_source[res.target].notna()], res.target)
            except Exception:  # noqa: BLE001
                X_ref = None

        base_vals: dict[str, float] = {}
        for f in res.feature_names:
            if X_ref is not None and f in X_ref.columns:
                base_vals[f] = float(X_ref[f].median())
            else:
                base_vals[f] = 0.0

        cols_ui = st.columns(min(3, max(1, len(top_feats))))
        user_vals = dict(base_vals)
        for idx, feat in enumerate(top_feats):
            col_box = cols_ui[idx % len(cols_ui)]
            if X_ref is not None and feat in X_ref.columns:
                s = X_ref[feat].dropna()
                lo, hi, med = float(s.min()), float(s.max()), float(s.median())
                if set(s.unique()).issubset({0.0, 1.0}):
                    picked = col_box.selectbox(
                        f"{feat}",
                        options=[0.0, 1.0],
                        index=1 if med >= 0.5 else 0,
                        format_func=lambda v: "Ya (1)" if v == 1.0 else "Tidak (0)",
                        key=f"sim_{res.target}_{model_name}_{feat}",
                    )
                    user_vals[feat] = float(picked)
                elif hi > lo:
                    step = round((hi - lo) / 100.0, 4) or 0.01
                    user_vals[feat] = float(
                        col_box.slider(
                            f"{feat}",
                            min_value=lo,
                            max_value=hi,
                            value=med,
                            step=step,
                            key=f"sim_{res.target}_{model_name}_{feat}",
                        )
                    )
                else:
                    user_vals[feat] = float(
                        col_box.number_input(f"{feat}", value=med, key=f"sim_{res.target}_{model_name}_{feat}")
                    )
            else:
                user_vals[feat] = float(
                    col_box.number_input(f"{feat}", value=0.0, key=f"sim_{res.target}_{model_name}_{feat}")
                )

        row_df = pd.DataFrame([user_vals], columns=res.feature_names).astype("float32")
        pred = r.model.predict(row_df)[0]

        if res.problem_type == ml.CLASSIFICATION:
            pred_idx = int(pred)
            pred_label = res.class_labels[pred_idx] if 0 <= pred_idx < len(res.class_labels) else str(pred)
            st.success(f"🔮 **Hasil Prediksi ({res.target}):** `{pred_label}`")
            if hasattr(r.model, "predict_proba"):
                probs = r.model.predict_proba(row_df)[0]
                prob_df = pd.DataFrame({
                    "Kelas": [str(c) for c in res.class_labels],
                    "Probabilitas (%)": (probs * 100).round(1),
                }).sort_values("Probabilitas (%)", ascending=False)
                st.dataframe(
                    prob_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Probabilitas (%)": st.column_config.ProgressColumn(
                            "Keyakinan Model (%)",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        )
                    },
                )
        else:
            st.success(f"🔮 **Hasil Prediksi ({res.target}):** `{float(pred):,.3f}`")


# ---------- tampilan hasil model (dipakai ML & Auto-ML) ----------
def _show_train_result(res: ml.TrainResult, df_source: pd.DataFrame | None = None) -> None:
    st.subheader("Perbandingan Performa Model")
    table = ml.metrics_table(res)
    key = "F1-Score" if res.problem_type == ml.CLASSIFICATION else "R²"
    fmt = {c: ("{:,.3f}" if table[c].abs().max() < 100 else "{:,.0f}") for c in table.columns}
    styled = (table.style.format(fmt, na_rep="–")
              .highlight_max(subset=[key], props=f"background-color: {current_theme()['accent_soft']}; font-weight: 700"))
    st.dataframe(styled, use_container_width=True)
    st.success(f"🏆 Model terbaik: **{res.best_name}** ({key} = {res.best.metrics[key]:.3f})")

    name = st.selectbox("Pilih model untuk melihat evaluasi mendalam & simulator", list(res.results),
                        index=list(res.results).index(res.best_name), key=f"pick_model_{res.target}_{res.problem_type}")
    r = res.results[name]

    # Kartu metrik model terpilih
    m_cols = st.columns(len(r.metrics))
    for col_ui, (m_name, m_val) in zip(m_cols, r.metrics.items()):
        col_ui.metric(m_name, f"{m_val:.3f}" if abs(m_val) < 100 else f"{m_val:,.1f}", border=True)

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
            ax.scatter(res.y_test, r.y_pred, s=14, alpha=0.65, color=current_theme()["accent"])
            lo, hi = float(np.min(res.y_test)), float(np.max(res.y_test))
            ax.plot([lo, hi], [lo, hi], "--", color=current_theme()["marker"], lw=1.5)
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
            ax.set_title("10 Fitur Paling Berpengaruh")
            fig.tight_layout()
            show_figure(fig)

    st.subheader("Interpretasi Hasil")
    _bullets(ml.interpret(res))

    # Simulator Prediksi Interaktif
    _prediction_simulator(res, name, df_source)

    # Unduh hasil prediksi data uji
    if res.problem_type == ml.CLASSIFICATION and res.class_labels:
        actual_vals = [res.class_labels[int(i)] for i in res.y_test]
        pred_vals = [res.class_labels[int(i)] for i in r.y_pred]
    else:
        actual_vals = res.y_test
        pred_vals = r.y_pred
    pred_export = pd.DataFrame({"Aktual": actual_vals, f"Prediksi_{name}": pred_vals})
    st.download_button(
        f"Unduh Hasil Uji Aktual vs Prediksi ({name}) — CSV",
        icon=":material/download:",
        data=pred_export.to_csv(index=False).encode("utf-8"),
        file_name=f"statify_prediksi_{name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ---------- 4. ML Modeling ----------
def page_ml(df: pd.DataFrame) -> None:
    page_header("Machine Learning Modeling", "Pilih kolom target, atur algoritma & parameter, lalu uji model dengan simulator interaktif.")
    with st.form("ml_form"):
        c1, c2 = st.columns(2)
        target = c1.selectbox("Kolom target (yang ingin diprediksi)", df.columns.tolist(), index=len(df.columns) - 1)
        problem = c2.selectbox("Tipe masalah", ["Deteksi otomatis", ml.CLASSIFICATION, ml.REGRESSION])
        names_cls = ["Random Forest", "Logistic Regression"] + (["XGBoost"] if ml.HAS_XGB else [])
        names_reg = ["Random Forest", "Linear Regression"] + (["XGBoost"] if ml.HAS_XGB else [])
        c3, c4 = st.columns(2)
        n_trees = c3.slider("Jumlah pohon (RF/XGBoost)", 10, config.MAX_TREES, 100, step=10)
        test_size = c4.slider("Porsi data uji (Test Split)", 0.1, 0.4, 0.2, step=0.05)
        use_smote = st.checkbox("Seimbangkan kelas dengan SMOTE (khusus klasifikasi)",
                                disabled=not ml.HAS_SMOTE)
        st.caption(f"Algoritma klasifikasi: {', '.join(names_cls)} · Algoritma regresi: {', '.join(names_reg)}")
        submitted = st.form_submit_button("Latih & Bandingkan Model", type="primary", icon=":material/play_arrow:")

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
        st.info("Pilih kolom target lalu tekan **Latih & Bandingkan Model**. Training hanya berjalan saat tombol ditekan.")
        return
    st.caption(f"Hasil terakhir: target **{res.target}**, {res.problem_type.lower()}, "
               f"{len(res.feature_names)} fitur.")
    _show_train_result(res, df)


# ---------- 5. Clustering ----------
def page_clustering(df: pd.DataFrame) -> None:
    page_header("Clustering & Segmentasi", "Kelompokkan baris data yang memiliki kemiripan pola tanpa memerlukan kolom target.")
    num = [c for c in numeric_columns(df) if df[c].nunique() > 1]
    if len(num) < 2:
        st.warning("Clustering butuh minimal dua kolom numerik.")
        return

    method = st.segmented_control(
        "Pilih Algoritma Clustering",
        ["K-Means", "Hierarchical", "DBSCAN"],
        default="K-Means",
    ) or "K-Means"

    with st.form("cluster_form"):
        cols = st.multiselect("Kolom numerik untuk clustering", num, default=num[: min(4, len(num))])
        k, linkage_m, eps, min_samples = 3, "ward", 0.5, 5
        if method == "K-Means":
            k = st.slider("Jumlah cluster (K)", 2, 10, 3)
            st.caption("K-Means mengelompokkan data ke dalam K pusat terdekat secara cepat.")
        elif method == "Hierarchical":
            c1, c2 = st.columns(2)
            k = c1.slider("Jumlah cluster (K)", 2, 10, 3)
            linkage_m = c2.selectbox("Metode Linkage", ["ward", "complete", "average", "single"])
            st.caption(f"Hierarchical membuat pohon dendrogram (dihitung pada sampel maks. {config.HIERARCHICAL_SAMPLE:,} baris).")
        else:
            c1, c2 = st.columns(2)
            eps = c1.slider("Radius ketetanggaan (eps)", 0.1, 5.0, 0.5, step=0.1)
            min_samples = c2.slider("Minimum titik per cluster (min_samples)", 2, 20, 5)
            st.caption("DBSCAN menentukan jumlah cluster secara otomatis berdasarkan kepadatan dan menandai pencilan sebagai noise (-1).")

        submitted = st.form_submit_button(f"Jalankan {method}", type="primary", icon=":material/play_arrow:")

    with st.expander("📐 Bantu Pilih Jumlah Cluster Optimal (Kurva Elbow & Silhouette)", icon=":material/query_stats:"):
        if st.button("Hitung Skor untuk K = 2 s.d. 10", disabled=len(cols) < 2):
            with heavy_job("Menghitung kurva Elbow & Silhouette..."):
                try:
                    st.session_state["elbow"] = clustering.elbow(df, cols)
                except clustering.ClusteringError as err:
                    st.error(str(err))
        if "elbow" in st.session_state and not st.session_state["elbow"].empty:
            e = st.session_state["elbow"]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
            a1.plot(e["K"], e["Inertia"], marker="o")
            a1.set_title("Elbow (Inertia)")
            a1.set_xlabel("Jumlah Cluster (K)")
            a2.plot(e["K"], e["Silhouette"], marker="o", color=current_theme()["marker"])
            a2.set_title("Skor Silhouette")
            a2.set_xlabel("Jumlah Cluster (K)")
            fig.tight_layout()
            show_figure(fig)
            best_k = int(e.loc[e["Silhouette"].idxmax(), "K"])
            st.success(f"💡 Rekomendasi otomatis: **K = {best_k}** menghasilkan skor Silhouette tertinggi.")

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
        st.info("Pilih kolom & algoritma, lalu tekan tombol **Jalankan**.")
        return

    st.subheader(f"Hasil {res.method}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Cluster", res.n_clusters, border=True)
    c2.metric("Titik Noise", res.n_noise, border=True)
    c3.metric("Silhouette Score", "-" if res.silhouette is None else f"{res.silhouette:.3f}", border=True)
    c4.metric("Davies-Bouldin", "-" if res.davies_bouldin is None else f"{res.davies_bouldin:.3f}", border=True)
    _bullets(clustering.describe(res))

    # Visualisasi interaktif: pilih PCA 2D atau sumbu kolom asli
    cluster_cols = [c for c in res.data.columns if c != "Cluster"]
    viz_mode = st.radio(
        "Mode sumbu grafik sebaran cluster",
        ["Proyeksi PCA 2D (Merangkum Semua Kolom)", "Pilih 2 Kolom Asli (Sumbu X & Y)"],
        horizontal=True,
        key="cl_viz_mode",
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if viz_mode.startswith("Proyeksi") or len(cluster_cols) < 2:
        pts = clustering.pca_2d(res.scaled)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=res.labels, cmap="tab10", s=18, alpha=0.8)
        ax.set_title("Sebaran Cluster (Proyeksi PCA 2D)")
        ax.set_xlabel("Komponen Utama 1 (PCA 1)")
        ax.set_ylabel("Komponen Utama 2 (PCA 2)")
    else:
        ax_c1, ax_c2 = st.columns(2)
        x_col = ax_c1.selectbox("Sumbu X", cluster_cols, index=0, key="cl_x")
        y_col = ax_c2.selectbox("Sumbu Y", cluster_cols, index=min(1, len(cluster_cols) - 1), key="cl_y")
        sc = ax.scatter(res.data[x_col], res.data[y_col], c=res.labels, cmap="tab10", s=20, alpha=0.8)
        ax.set_title(f"Sebaran Cluster: {y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    ax.legend(*sc.legend_elements(), title="Cluster", fontsize=8, loc="best")
    fig.tight_layout()
    show_figure(fig)

    if res.linkage_matrix is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        dendrogram(res.linkage_matrix, truncate_mode="lastp", p=30, ax=ax, no_labels=True,
                   color_threshold=None)
        ax.set_title("Dendrogram (30 Cabang Terakhir)")
        fig.tight_layout()
        show_figure(fig)

    st.subheader("Profil Rata-Rata Tiap Cluster")
    profile = res.data.groupby("Cluster").mean(numeric_only=True).round(3)
    profile.insert(0, "Jumlah Anggota", res.data["Cluster"].value_counts().sort_index())
    profile.insert(1, "Porsi (%)", (res.data["Cluster"].value_counts(normalize=True).sort_index() * 100).round(1))
    st.dataframe(profile, use_container_width=True)

    with st.expander("🔎 Inspeksi Data per Cluster", icon=":material/filter_alt:"):
        unique_clusters = sorted(res.data["Cluster"].unique().tolist())
        chosen_cl = st.selectbox("Pilih nomor Cluster", unique_clusters, key="cl_inspect_pick")
        sub_df = res.data[res.data["Cluster"] == chosen_cl]
        st.caption(f"Menampilkan {len(sub_df):,} baris yang tergabung dalam **Cluster {chosen_cl}**:")
        st.dataframe(sub_df.head(50), use_container_width=True)

    st.download_button(
        "Unduh Data + Label Cluster (CSV)",
        icon=":material/download:",
        data=res.data.to_csv(index=False).encode("utf-8"),
        file_name="statify_cluster.csv",
        mime="text/csv",
        type="primary",
    )


# ---------- 6. Auto-ML ----------
def page_automl(df: pd.DataFrame) -> None:
    page_header("Auto-ML", "Pilih satu kolom yang ingin ditebak. Statify mengurus sisanya secara otomatis dan menyediakan simulator prediksi.")
    st.caption("Cukup pilih kolom target. Statify mendeteksi jenis masalah, menyiapkan fitur terbaik, melatih seluruh "
               "algoritma, lalu memilihkan pemenangnya.")
    with st.form("automl_form"):
        target = st.selectbox("Kolom yang ingin diprediksi", df.columns.tolist(), index=len(df.columns) - 1)
        submitted = st.form_submit_button("Jalankan Auto-ML", type="primary", icon=":material/auto_awesome:")

    if submitted:
        ptype = ml.detect_problem_type(df[target].dropna())
        with heavy_job(f"Terdeteksi masalah {ptype.lower()}. Melatih semua model..."):
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
    st.info(f"✨ Jenis masalah terdeteksi otomatis: **{res.problem_type}**, karena {reason}. "
            f"**{len(res.feature_names)} fitur** disiapkan otomatis (kategori ≤ 20 nilai di-one-hot, "
            "kolom mirip ID dibuang).")
    _show_train_result(res, df)
