"""Identitas visual Statify: studio analitik interaktif bergaya modern.

Warna dasar dan font diatur di .streamlit/config.toml ([theme.light] / [theme.dark]) agar tabel,
input, dan tombol konsisten. Modul ini menambah CSS interaktif, animasi halus, dan menyelaraskan grafik matplotlib.
Pengguna berpindah terang/gelap lewat menu ⋮ → Settings (default mengikuti sistem).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cycler import cycler
import streamlit as st

PALETTES = {
    "light": {
        "bg": "#F8FAFC", "panel": "#FFFFFF", "grid": "#E2E8F0", "ink": "#0F172A",
        "muted": "#64748B", "accent": "#4F46E5", "accent2": "#0D9488", "marker": "#F59E0B",
        "accent_soft": "#EEF2FF", "success_soft": "#ECFDF5",
        "series": ["#4F46E5", "#0D9488", "#F59E0B", "#EC4899", "#8B5CF6", "#64748B"],
    },
    "dark": {
        "bg": "#0B1120", "panel": "#151F32", "grid": "#24324B", "ink": "#F1F5F9",
        "muted": "#94A3B8", "accent": "#818CF8", "accent2": "#2DD4BF", "marker": "#FBBF24",
        "accent_soft": "#1E293B", "success_soft": "#064E3B",
        "series": ["#818CF8", "#2DD4BF", "#FBBF24", "#F472B6", "#A78BFA", "#94A3B8"],
    },
}


def theme_type() -> str:
    try:
        t = st.context.theme.type
    except Exception:  # noqa: BLE001
        t = None
    return t if t in PALETTES else "light"


def current_theme() -> dict:
    return PALETTES[theme_type()]


def _css(t: dict) -> str:
    return f"""
<style>
  /* Latar belakang dot-grid interaktif modern */
  [data-testid="stMain"] {{
    background-image:
      radial-gradient({t['accent']}22 1px, transparent 1px),
      linear-gradient(180deg, {t['accent']}08 0%, transparent 260px);
    background-size: 24px 24px, 100% 100%;
  }}
  [data-testid="stMainBlockContainer"] {{ max-width: 1220px; padding-top: 3.5rem; }}
  [data-testid="stMetricValue"], [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

  /* Kartu metrik interaktif */
  [data-testid="stMetric"] {{
    background: {t['panel']};
    border-top: 3px solid {t['accent']} !important;
    border-radius: 12px;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
  }}
  [data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px -6px {t['accent']}22;
  }}

  /* Kontainer berbingkai lebih halus */
  [data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }}
  [data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: {t['accent']}55 !important;
  }}

  [data-testid="stMain"] h3 {{ font-size: clamp(1.15rem, 1.8vw, 1.4rem); font-weight: 700; }}
  :focus-visible {{ outline: 2px solid {t['marker']} !important; outline-offset: 2px; }}

  /* Wordmark di sidebar */
  .sf-brand-box {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.25rem 0 0.5rem;
  }}
  .sf-wordmark {{
    font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: {t['ink']};
    line-height: 1; margin: 0;
  }}
  .sf-wordmark b {{
    background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800;
  }}
  .sf-badge-ver {{
    font-size: 0.72rem; font-weight: 700; padding: 0.18rem 0.55rem; border-radius: 999px;
    background: {t['accent_soft']}; color: {t['accent']}; border: 1px solid {t['accent']}33;
  }}
  .sf-tagline {{ color: {t['muted']}; font-size: 0.84rem; margin: 0.15rem 0 0; }}

  /* Hero interaktif */
  .sf-hero {{
    display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.95fr); gap: 2.2rem;
    align-items: center; margin: 0.25rem 0 1.75rem;
    padding: 1.75rem 2rem; border-radius: 18px;
    background: linear-gradient(135deg, {t['panel']} 0%, {t['accent_soft']} 100%);
    border: 1px solid {t['grid']};
    box-shadow: 0 10px 30px -10px {t['accent']}18;
  }}
  .sf-kicker {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
    padding: 0.28rem 0.75rem; border-radius: 999px; margin-bottom: 0.85rem;
    background: {t['panel']}; color: {t['accent']}; border: 1px solid {t['accent']}33;
  }}
  .sf-hero h1 {{
    font-size: clamp(1.85rem, 3.6vw, 2.85rem); line-height: 1.12; letter-spacing: -0.025em;
    font-weight: 800; color: {t['ink']}; margin: 0 0 0.85rem; padding: 0;
  }}
  .sf-hero h1 span {{
    background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .sf-hero p {{ font-size: 1.02rem; line-height: 1.6; color: {t['muted']}; max-width: 36em; margin: 0 0 1.1rem; }}
  .sf-pills-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }}
  .sf-mini-pill {{
    font-size: 0.78rem; font-weight: 600; padding: 0.25rem 0.65rem; border-radius: 8px;
    background: {t['panel']}; color: {t['ink']}; border: 1px solid {t['grid']};
  }}
  .sf-plot {{
    background: {t['panel']}; border: 1px solid {t['grid']}; border-radius: 14px; padding: 0.85rem;
    box-shadow: 0 6px 18px -6px rgba(0,0,0,0.08);
  }}
  .sf-plot svg {{ display: block; width: 100%; height: auto; }}
  .sf-trend {{ stroke-dasharray: 520; stroke-dashoffset: 520; animation: sf-draw 1.4s 0.3s ease-out forwards; }}
  .sf-dot {{ animation: sf-settle 1.1s ease-out both; transform-origin: center; transition: r 0.2s ease; }}
  .sf-dot:hover {{ r: 8px; }}
  @keyframes sf-draw {{ to {{ stroke-dashoffset: 0; }} }}
  @keyframes sf-settle {{ from {{ opacity: 0; transform: translateY(var(--dy)); }} to {{ opacity: 1; transform: none; }} }}
  @media (prefers-reduced-motion: reduce) {{
    .sf-trend {{ animation: none; stroke-dashoffset: 0; }}
    .sf-dot {{ animation: none; }}
  }}

  /* Langkah memulai */
  .sf-steps {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 1.25rem 0 1.5rem;
    padding: 0; list-style: none; counter-reset: step;
  }}
  .sf-steps li {{
    counter-increment: step; padding: 1rem 1.1rem 1rem 3.2rem; position: relative;
    border: 1px solid {t['grid']}; border-left: 4px solid {t['accent']};
    background: {t['panel']}; border-radius: 12px;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
  }}
  .sf-steps li:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 20px -8px {t['accent']}22;
  }}
  .sf-steps li::before {{
    content: counter(step); position: absolute; left: 0.9rem; top: 0.95rem;
    width: 1.7rem; height: 1.7rem; border-radius: 50%; background: {t['accent_soft']};
    color: {t['accent']}; font-weight: 800; display: grid; place-items: center; font-size: 0.9rem;
  }}
  .sf-steps strong {{ display: block; color: {t['ink']}; margin-bottom: 0.15rem; }}
  .sf-steps span {{ color: {t['muted']}; font-size: 0.9rem; line-height: 1.45; display: block; }}

  /* Kartu kemampuan interaktif */
  .sf-features {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem;
    margin: 0.5rem 0 1rem;
  }}
  .sf-features div {{
    background: {t['panel']}; border: 1px solid {t['grid']}; border-radius: 12px;
    padding: 1rem 1.1rem; transition: transform 0.18s ease, border-color 0.18s ease;
  }}
  .sf-features div:hover {{
    transform: translateY(-2px); border-color: {t['accent']};
  }}
  .sf-features strong {{ color: {t['ink']}; display: block; margin-bottom: 0.25rem; font-size: 0.96rem; }}
  .sf-features span {{ color: {t['muted']}; font-size: 0.88rem; line-height: 1.5; display: block; }}

  /* Kepala halaman */
  .sf-page {{
    margin: 0.25rem 0 1.25rem; padding: 1.1rem 1.35rem; border-radius: 14px;
    background: {t['panel']}; border: 1px solid {t['grid']};
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.75rem;
  }}
  .sf-page h2 {{
    font-size: clamp(1.35rem, 2.3vw, 1.75rem); font-weight: 800; letter-spacing: -0.015em;
    color: {t['ink']}; margin: 0; padding: 0;
  }}
  .sf-page p {{ color: {t['muted']}; margin: 0.25rem 0 0; max-width: 46em; font-size: 0.94rem; }}
  .sf-chip {{
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.35rem 0.8rem; border-radius: 999px;
    background: {t['accent_soft']}; color: {t['accent']};
    font-size: 0.84rem; font-weight: 700; border: 1px solid {t['accent']}30;
  }}

  /* Kotak "cara membaca" berwarna stabilo */
  .sf-note {{
    border-left: 4px solid {t['marker']}; background: {t['marker']}18; padding: 0.75rem 1rem;
    border-radius: 0 10px 10px 0; color: {t['ink']}; margin: 0.35rem 0 1rem; font-size: 0.93rem;
  }}

  /* Status tahapan di sidebar */
  .sf-status-list {{ display: flex; flex-direction: column; gap: 0.35rem; margin: 0.5rem 0; }}
  .sf-status-item {{
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.82rem; padding: 0.35rem 0.6rem; border-radius: 8px;
    background: {t['bg']}; border: 1px solid {t['grid']}; color: {t['ink']};
  }}
  .sf-dot-ok {{ width: 8px; height: 8px; border-radius: 50%; background: #10B981; display: inline-block; }}
  .sf-dot-wait {{ width: 8px; height: 8px; border-radius: 50%; background: {t['muted']}66; display: inline-block; }}

  /* Ponsel & tablet */
  @media (max-width: 900px) {{
    .sf-hero {{ grid-template-columns: 1fr; gap: 1.25rem; padding: 1.25rem; }}
    .sf-features {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  }}
  @media (max-width: 640px) {{
    [data-testid="stMainBlockContainer"] {{ padding: 3.25rem 1rem 2.5rem; }}
    .sf-steps, .sf-features {{ grid-template-columns: 1fr; }}
    .sf-hero p {{ font-size: 0.96rem; }}
  }}
</style>
"""


def apply_theme() -> dict:
    t = current_theme()
    st.markdown(_css(t), unsafe_allow_html=True)
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": t["panel"], "savefig.facecolor": t["panel"], "axes.facecolor": t["panel"],
        "text.color": t["ink"], "axes.labelcolor": t["muted"], "axes.titlecolor": t["ink"],
        "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "axes.edgecolor": t["grid"], "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.color": t["muted"], "ytick.color": t["muted"], "legend.frameon": False,
        "axes.prop_cycle": cycler(color=t["series"]), "image.cmap": "Blues",
        "figure.dpi": 110,
    })
    return t


def page_header(title: str, description: str) -> None:
    info = st.session_state.get("load_info")
    chip = ""
    if info and "df" in st.session_state:
        rows, cols = st.session_state["df_cleaned"].shape
        chip = f"<span class='sf-chip'>📊 {esc(info['name'])} · {rows:,} baris × {cols} kolom</span>"
    st.markdown(
        f"<div class='sf-page'><div><h2>{title}</h2><p>{description}</p></div>{chip}</div>",
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Kotak penjelasan 'cara membaca' bergaya stabilo."""
    st.markdown(f"<div class='sf-note'>{text}</div>", unsafe_allow_html=True)


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def hero_svg() -> str:
    """Titik-titik acak 'disederhanakan' menjadi satu garis tren: wujud visual Stats and Simplify."""
    t = current_theme()
    pts = [(38, 186, 34), (62, 160, -26), (84, 172, 40), (108, 140, -30), (126, 152, 28), (150, 124, -38),
           (170, 132, 22), (192, 106, -20), (214, 118, 36), (236, 92, -24), (258, 98, 30), (280, 74, -34),
           (300, 84, 20), (322, 60, -22), (346, 66, 26), (368, 44, -18)]
    dots = "".join(
        f"<circle class='sf-dot' cx='{x}' cy='{y}' r='5.8' fill='{t['accent']}' fill-opacity='0.88' "
        f"style='--dy:{dy}px; animation-delay:{i * 35}ms'/>" for i, (x, y, dy) in enumerate(pts))
    grid = "".join(f"<line x1='20' x2='392' y1='{y}' y2='{y}' stroke='{t['grid']}' stroke-dasharray='4 4'/>"
                   for y in (40, 90, 140, 190))
    return f"""
<svg viewBox='0 0 412 230' role='img' aria-label='Titik data acak yang disederhanakan menjadi satu garis tren'>
  <defs>
    <linearGradient id='sfGrad' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0%' stop-color='{t['accent2']}' stop-opacity='0.22'/>
      <stop offset='100%' stop-color='{t['accent2']}' stop-opacity='0.0'/>
    </linearGradient>
  </defs>
  {grid}
  <polygon points='30,190 376,46 376,206 30,206' fill='url(#sfGrad)'/>
  <line x1='20' x2='392' y1='206' y2='206' stroke='{t['muted']}' stroke-opacity='0.45'/>
  <line x1='20' x2='20' y1='24' y2='206' stroke='{t['muted']}' stroke-opacity='0.45'/>
  {dots}
  <path class='sf-trend' d='M30 190 L376 46' stroke='{t['marker']}' stroke-width='4.5' stroke-linecap='round' fill='none'/>
  <rect x='288' y='14' width='96' height='26' rx='13' fill='{t['accent_soft']}'/>
  <text x='336' y='31' text-anchor='middle' font-size='12' font-weight='700' fill='{t['accent']}'>R² = 0,94 ✨</text>
</svg>"""


def _compact(x, _pos=None) -> str:
    """Angka sumbu ringkas ala Indonesia: 1,5 jt / 250 rb, tanpa notasi 1e6."""
    a = abs(x)
    for limit, suffix in ((1e9, " M"), (1e6, " jt"), (1e3, " rb")):
        if a >= limit:
            v = f"{x / limit:.1f}".rstrip("0").rstrip(".")
            return v.replace(".", ",") + suffix
    return f"{x:g}".replace(".", ",")


def show_figure(fig) -> None:
    """Rapikan angka sumbu, tampilkan selebar kolom, lalu tutup agar memori matplotlib dilepas."""
    for ax in fig.get_axes():
        for axis, lim in ((ax.yaxis, ax.get_ylim()), (ax.xaxis, ax.get_xlim())):
            if max(abs(lim[0]), abs(lim[1])) >= 10_000 and axis.get_scale() == "linear" \
                    and isinstance(axis.get_major_formatter(), ticker.ScalarFormatter):
                axis.set_major_formatter(ticker.FuncFormatter(_compact))
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)
