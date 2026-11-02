# DataDesa

Aplikasi visualisasi dan clustering **Indeks Desa Membangun (IDM) Indonesia tahun 2024**. Aplikasi menyajikan status desa berdasarkan IDM, IKS, IKE, dan IKL serta membandingkan hasil pengelompokan menggunakan K-Means dan Fuzzy C-Means.

## Tautan

- [Aplikasi Streamlit](https://nr14-018-indeksdesamembangun.streamlit.app)
- [Sumber data melalui data.go.id](https://data.go.id/dataset/dataset/data-indeks-desa-membangun-tahun-2024)

## Fitur

- Visualisasi distribusi status IDM secara nasional dan per provinsi.
- Menampilkan 20 desa dengan nilai IDM tertinggi.
- Mengelompokkan desa dengan metode K-Means.
- Mengelompokkan desa dengan metode Fuzzy C-Means.
- Menampilkan hasil clustering dalam bentuk grafik dua dimensi menggunakan PCA.
- Membandingkan hasil clustering berdasarkan jumlah cluster yang dipilih.

## Teknologi

- Python
- Streamlit
- Pandas dan NumPy
- Scikit-learn
- Scikit-Fuzzy
- Plotly

## Prasyarat

Pastikan Python 3.9 atau versi yang lebih baru telah terpasang di sistem.

## Instalasi

1. Masuk ke direktori proyek:

   ```bash
   cd DataDesa
   ```

2. Buat dan aktifkan lingkungan virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Pasang seluruh dependensi:

   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit, biasanya `http://localhost:8501`.

## Cara Penggunaan

1. Pilih metode clustering pada sidebar: `kmeans`, `fuzzy`, atau `all`.
2. Tentukan jumlah cluster menggunakan slider.
3. Tekan tombol **Run** untuk memproses data.
4. Jelajahi pratinjau data, distribusi status, tabel desa teratas, dan hasil clustering.

## Struktur Proyek

```text
DataDesa/
├── app.py                   # Antarmuka aplikasi Streamlit
├── clustering/
│   ├── clusterer.py         # Implementasi model clustering
│   ├── fuzzycmeans.py       # Implementasi Fuzzy C-Means
│   ├── pipeline.py          # Alur preprocessing dan clustering
│   └── preprocessor.py      # Utilitas pembersihan dan analisis data
├── indeks-desa-membangun-tahun-2024-hasil-pemutakhiran.xlsx
├── requirements.txt         # Dependensi Python
└── README.md
```

## Dataset

Aplikasi membaca dataset Excel yang disimpan di direktori utama proyek dengan nama:

```text
indeks-desa-membangun-tahun-2024-hasil-pemutakhiran.xlsx
```

Data yang digunakan mencakup desa; data kelurahan tidak disertakan dalam aplikasi. Jika nama atau lokasi dataset diubah, sesuaikan nilai `DATA_PATH` pada `app.py`.

## Indikator

| Singkatan | Indikator |
| --- | --- |
| IDM | Indeks Desa Membangun |
| IKS | Indeks Ketahanan Sosial |
| IKE | Indeks Ketahanan Ekonomi |
| IKL | Indeks Ketahanan Lingkungan |

## Status IDM

Status hasil pengkodean yang digunakan aplikasi adalah:

| Nilai | Status |
| --- | --- |
| -1 | Sangat tertinggal |
| 0 | Tertinggal |
| 1 | Berkembang |
| 2 | Maju |
