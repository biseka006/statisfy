"""Antrean komputasi berat yang berlaku untuk SEMUA pengguna di server.

Di server 2 core, lima training bersamaan membuat semuanya lambat dan RAM melonjak bersamaan.
Dengan antrean, paling banyak `config.MAX_HEAVY_JOBS` proses berat berjalan sekaligus; sisanya
menunggu giliran dengan pesan yang jelas.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import streamlit as st

from . import config


@st.cache_resource
def _gate() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(config.MAX_HEAVY_JOBS)


class BusyError(Exception):
    """Antrean terlalu lama; minta pengguna mencoba lagi."""


@contextmanager
def heavy_job(label: str):
    gate = _gate()
    if not gate.acquire(blocking=False):
        with st.spinner("Server sedang melayani pengguna lain. Menunggu giliran..."):
            if not gate.acquire(timeout=config.QUEUE_TIMEOUT_S):
                raise BusyError("Server sedang sangat sibuk. Tunggu sebentar lalu tekan tombolnya lagi.")
    try:
        with st.spinner(label):
            yield
    finally:
        gate.release()
