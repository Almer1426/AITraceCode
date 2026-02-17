# AITraceCode
Sistem deteksi kode berbasis AI untuk verifikasi autentisitas tugas pemrograman.

## 📂 Repository Structure
Proyek ini dibangun dengan pendekatan modular untuk memastikan skalabilitas antara pengembangan model AI dan implementasi web backend. Berikut adalah rincian struktur direktori pada repositori ini:
```
AICodeTrace/
├── data/                   # Penyimpanan dataset utama
│   ├── raw/                # Data mentah asli (CodeNet/AIGCodeSet)
│   └── processed/          # Data siap training (setelah tokenisasi)
├── notebooks/              # Eksperimen awal, prototyping, & EDA (Jupyter)
├── src/                    # Source code utama (Modular Logic)
│   ├── data_loader.py      # Script untuk pembacaan & preprocessing dataset
│   ├── model.py            # Definisi arsitektur CodeBERT + Classification Head
│   ├── trainer.py          # Implementasi training & validation loop
│   └── utils.py            # Fungsi pembantu (logging, metrics calculation)
├── models/                 # Output penyimpanan model (.pt/.bin) pasca-training
├── app/                    # Web Application layer (FastAPI)
│   ├── main.py             # Entry point untuk server backend
│   └── templates/          # Interface pengguna (HTML/CSS)
├── requirements.txt        # Daftar dependensi library (Transformers, Torch, FastAPI)
├── .gitignore              # Konfigurasi pengecualian file untuk Git
└── README.md               # Dokumentasi utama proyek
```