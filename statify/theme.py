"""Identitas visual Statify: "kertas grafik".

Warna dasar dan font diatur di .streamlit/config.toml ([theme.light] / [theme.dark]) agar tabel,
input, dan tombol konsisten. Modul ini menambah CSS tata letak dan menyelaraskan grafik matplotlib.
Pengguna berpindah terang/gelap lewat menu ⋮ → Settings (default mengikuti sistem).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cycler import cycler
import streamlit as st

PALETTES = {
    "light": {
        "bg": "#F7F9FC", "panel": "#FFFFFF", "grid": "#E3E9F2", "ink": "#1B2A41",
        "muted": "#5B6B82", "accent": "#2F5BEA", "marker": "#F2A541", "accent_soft": "#E6ECFD",
        "series": ["#2F5BEA", "#F2A541", "#17A398", "#D1495B", "#7C5CD6", "#5B6B82"],
    },
    "dark": {
        "bg": "#0F1A2B", "panel": "#17253A", "grid": "#26364F", "ink": "#E8EEF7",
        "muted": "#9AA8BD", "accent": "#7B96FF", "marker": "#F2B45A", "accent_soft": "#1E2D4A",
        "series": ["#7B96FF", "#F2B45A", "#3CC7B8", "#EF6F7F", "#A48BEF", "#9AA8BD"],
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
  /* Kertas grafik: petak tipis di latar halaman utama */
  [data-testid="stMain"] {{
    background-image:
      linear-gradient({t['grid']}55 1px, transparent 1px),
      linear-gradient(90deg, {t['grid']}55 1px, transparent 1px);
    background-size: 28px 28px;
  }}
  [data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 4.25rem; }}
  [data-testid="stMetricValue"], [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
  [data-testid="stMetric"] {{ background: {t['panel']}; }}
  [data-testid="stMain"] h3 {{ font-size: clamp(1.15rem, 1.8vw, 1.45rem); font-weight: 700; }}
  :focus-visible {{ outline: 2px solid {t['marker']} !important; outline-offset: 2px; }}

  /* Wordmark di sidebar */
  .sf-wordmark {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; color: {t['ink']};
                  line-height: 1; margin: 0; }}
  .sf-wordmark b {{ color: {t['accent']}; font-weight: 800; }}
  .sf-tagline {{ color: {t['muted']}; font-size: 0.85rem; margin: 0.25rem 0 0; }}

  /* Hero */
  .sf-hero {{ display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 2.5rem;
              align-items: center; margin: 0.5rem 0 2rem; }}
  .sf-hero h1 {{ font-size: clamp(2rem, 4.2vw, 3.3rem); line-height: 1.1; letter-spacing: -0.02em;
                 font-weight: 800; color: {t['ink']}; margin: 0 0 1rem; padding: 0; }}
  .sf-hero p {{ font-size: 1.08rem; line-height: 1.6; color: {t['muted']}; max-width: 34em; margin: 0; }}
  .sf-plot {{ background: {t['panel']}; border: 1px solid {t['grid']}; border-radius: 14px; padding: 0.75rem; }}
  .sf-plot svg {{ display: block; width: 100%; height: auto; }}
  .sf-trend {{ stroke-dasharray: 520; stroke-dashoffset: 520; animation: sf-draw 1.4s 0.4s ease-out forwards; }}
  .sf-dot {{ animation: sf-settle 1.2s ease-out both; }}
  @keyframes sf-draw {{ to {{ stroke-dashoffset: 0; }} }}
  @keyframes sf-settle {{ from {{ opacity: 0; transform: translateY(var(--dy)); }} to {{ opacity: 1; transform: none; }} }}
  @media (prefers-reduced-motion: reduce) {{
    .sf-trend {{ animation: none; stroke-dashoffset: 0; }}
    .sf-dot {{ animation: none; }}
  }}

  /* Langkah memulai (memang berurutan, jadi diberi nomor) */
  .sf-steps {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 0 0 1.5rem;
               padding: 0; list-style: none; counter-reset: step; }}
  .sf-steps li {{ counter-increment: step; padding: 0.9rem 1rem 0.9rem 3.1rem; position: relative;
                  border-left: 3px solid {t['accent']}; background: {t['panel']}; border-radius: 0 10px 10px 0; }}
  .sf-steps li::before {{ content: counter(step); position: absolute; left: 0.9rem; top: 0.85rem;
                          width: 1.6rem; height: 1.6rem; border-radius: 50%; background: {t['accent_soft']};
                          color: {t['accent']}; font-weight: 800; display: grid; place-items: center; font-size: 0.9rem; }}
  .sf-steps strong {{ display: block; color: {t['ink']}; }}
  .sf-steps span {{ color: {t['muted']}; font-size: 0.92rem; }}

  /* Daftar kemampuan: daftar sederhana, bukan kartu */
  .sf-features {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.25rem 2rem;
                  margin: 0.5rem 0 1rem; }}
  .sf-features div {{ border-top: 1px solid {t['grid']}; padding-top: 0.75rem; }}
  .sf-features strong {{ color: {t['ink']}; display: block; margin-bottom: 0.2rem; }}
  .sf-features span {{ color: {t['muted']}; font-size: 0.92rem; line-height: 1.5; }}

  /* Kepala halaman */
  .sf-page {{ margin: 0.25rem 0 1.25rem; }}
  .sf-page h2 {{ font-size: clamp(1.5rem, 2.6vw, 2rem); font-weight: 800; letter-spacing: -0.015em;
                 color: {t['ink']}; margin: 0; padding: 0; }}
  .sf-page p {{ color: {t['muted']}; margin: 0.35rem 0 0; max-width: 46em; }}
  .sf-chip {{ display: inline-block; margin-top: 0.6rem; padding: 0.2rem 0.65rem; border-radius: 999px;
              background: {t['accent_soft']}; color: {t['accent']}; font-size: 0.85rem; font-weight: 600; }}

  /* Kotak "cara membaca" berwarna stabilo */
  .sf-note {{ border-left: 4px solid {t['marker']}; background: {t['marker']}1F; padding: 0.7rem 0.9rem;
              border-radius: 0 8px 8px 0; color: {t['ink']}; margin: 0.25rem 0 1.25rem; font-size: 0.95rem; }}

  /* Ponsel & tablet */
  @media (max-width: 900px) {{
    .sf-hero {{ grid-template-columns: 1fr; gap: 1.25rem; }}
    .sf-features {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  }}
  @media (max-width: 640px) {{
    [data-testid="stMainBlockContainer"] {{ padding: 3.75rem 1rem 3rem; }}
    .sf-steps, .sf-features {{ grid-template-columns: 1fr; }}
    .sf-hero p {{ font-size: 1rem; }}
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
        rows, cols = st.session_state["df"].shape
        chip = f"<span class='sf-chip'>{esc(info['name'])} — {rows:,} baris, {cols} kolom</span>"
    st.markdown(f"<div class='sf-page'><h2>{title}</h2><p>{description}</p>{chip}</div>",
                unsafe_allow_html=True)


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
        f"<circle class='sf-dot' cx='{x}' cy='{y}' r='5.5' fill='{t['accent']}' fill-opacity='0.85' "
        f"style='--dy:{dy}px; animation-delay:{i * 40}ms'/>" for i, (x, y, dy) in enumerate(pts))
    grid = "".join(f"<line x1='20' x2='392' y1='{y}' y2='{y}' stroke='{t['grid']}'/>" for y in (40, 90, 140, 190))
    return f"""
<svg viewBox='0 0 412 230' role='img' aria-label='Titik data acak yang disederhanakan menjadi satu garis tren'>
  {grid}
  <line x1='20' x2='392' y1='206' y2='206' stroke='{t['muted']}' stroke-opacity='0.5'/>
  <line x1='20' x2='20' y1='24' y2='206' stroke='{t['muted']}' stroke-opacity='0.5'/>
  {dots}
  <path class='sf-trend' d='M30 190 L376 46' stroke='{t['marker']}' stroke-width='5' stroke-linecap='round' fill='none'/>
  <text x='372' y='30' text-anchor='end' font-size='13' font-weight='700' fill='{t['ink']}'>r = 0,97</text>
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
