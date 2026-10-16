# 📊 Statify — Stats and Simplify

Aplikasi web untuk analisis data dan machine learning tanpa coding: unggah CSV/Excel, lalu dapatkan
cleaning otomatis, EDA dengan penjelasan, modeling, clustering, dan Auto-ML. Dibangun dengan Streamlit
dan disetel untuk **server 2 vCPU / 4 GB RAM** (± 5 pengguna bersamaan).

## Fitur

| Menu | Isi |
| --- | --- |
| 📁 Upload & Overview | CSV/XLSX ≤ 20 MB, sampling otomatis ke 20.000 baris, ringkasan & statistik |
| 🧹 Data Cleaning | Hapus duplikat, buang kolom ≥ 90% kosong, isi median/modus, unduh CSV |
| 📊 EDA | 4 insight otomatis + reasoning, heatmap & top-10 korelasi, pairplot, grafik bebas |
| 🤖 ML Modeling | Klasifikasi & regresi (Random Forest, Logistic/Linear, XGBoost), SMOTE, confusion matrix, feature importance |
| 🔍 Clustering | K-Means, Hierarchical (dendrogram), DBSCAN, Elbow, Silhouette & Davies-Bouldin |
| ✨ Auto-ML | Deteksi jenis masalah, siapkan fitur, pilih & jelaskan model terbaik |

Tampilan bertema "kertas grafik" dengan font Plus Jakarta Sans (di-host sendiri di `static/fonts`, tanpa
akses ke Google), responsif dari ponsel sampai desktop. Tema terang/gelap dipilih lewat menu
**⋮ → Settings** (default mengikuti sistem).

## Menjalankan di komputer sendiri

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py          # buka http://localhost:8501
python -m pytest -q           # 10 tes inti
```

| Skenario | Hasil |
| --- | --- |
| 20 pengguna mengunggah CSV 19 MB bersamaan, 5 di antaranya melatih model | Tanpa error, puncak RAM 1,4 GB, upload ≤ 50 detik, training ≤ 2 menit (termasuk antre) |
| 5 pengguna melatih model bersamaan | Tanpa error, tiap training ≤ 1 menit (termasuk antre) |
| Satu pengguna, data 20.000 baris, 3 model × 200 pohon | Klasifikasi ± 10 detik, regresi ± 30 detik |

Maksimal 2 proses berat (training, clustering, pairplot) berjalan sekaligus di seluruh server; sisanya
antre dengan pesan "Menunggu giliran". Untuk satu kelas penuh, minta siswa memakai dataset contoh atau
file kecil dan bergiliran menekan tombol latih.

## Batas sumber daya

Diatur di `statify/config.py`, bisa diubah lewat environment variable di `docker-compose.yml`:

| Variabel | Default | Keterangan |
| --- | --- | --- |
| `STATIFY_MAX_UPLOAD_MB` | 20 | Ukuran file maksimum (samakan dengan `maxUploadSize` di `.streamlit/config.toml`) |
| `STATIFY_MAX_ROWS` | 20000 | Data lebih besar disampling acak |
| `STATIFY_MAX_TREES` | 200 | Batas jumlah pohon RF/XGBoost |
| `STATIFY_HIERARCHICAL_SAMPLE` | 3000 | Sampel untuk Hierarchical & dendrogram |
| `STATIFY_MAX_HEAVY_JOBS` | 2 | Proses berat bersamaan di seluruh server (samakan dengan jumlah core) |

Pengaman lain: training hanya saat tombol ditekan, cache dibatasi 10 entri / 30 menit, model memakai
1 core per sesi, figure matplotlib selalu ditutup, container dibatasi 3,5 GB RAM dan restart otomatis.
Jika RAM sering penuh, turunkan `STATIFY_MAX_ROWS` ke 10000 dan `STATIFY_MAX_TREES` ke 100.

Pantau pemakaian dengan `docker stats statify` dan log dengan `docker logs -f statify`. Error aplikasi
hanya tercatat di log; pengunjung melihat pesan singkat tanpa detail teknis.

## Struktur

```
app.py                 entry point + navigasi + upload
statify/config.py      batas sumber daya
statify/data.py        baca file, optimasi tipe, sampling, cleaning
statify/eda.py         korelasi, insight otomatis, grafik
statify/ml.py          fitur, training, evaluasi, interpretasi
statify/clustering.py  K-Means, Hierarchical, DBSCAN, elbow
statify/views.py       halaman Streamlit
statify/limits.py      antrean proses berat untuk semua pengguna
statify/theme.py       identitas visual: CSS, hero, warna & format angka grafik
static/fonts/          Plus Jakarta Sans (lisensi OFL)
sample_data/           iris, diabetes, penjualan toko (contoh)
deploy/                skrip setup server, update, config Nginx
tests/                 tes inti (pytest)
```

## Privasi

Data hanya disimpan di memori sesi pengguna, tidak ditulis ke disk, dan hilang saat tombol Hapus data
ditekan atau tab ditutup.
