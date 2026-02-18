# AICodeTrace
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

## 🤝 Panduan Kolaborasi Git & GitHub
Untuk menjaga kualitas kode dan menghindari konflik saat pengerjaan proyek AICodeTrace, kita akan mengikuti alur kerja (workflow) standar. Aturan utamanya: Dilarang melakukan push langsung ke branch main.

### 1. Setup Awal: Clone Project
Langkah pertama adalah menyalin repositori dari GitHub ke komputer lokal kamu.
```
# Clone menggunakan HTTPS
git clone https://github.com/Almer1426/AICodeTrace.git

# Masuk ke direktori project
cd AICodeTrace
```

### 2. Sinkronisasi Branch Development
Kita menggunakan branch `development` sebagai tempat coba-coba integrasi semua fitur sebelum dianggap stabil untuk masuk ke `main`.
```
# Pastikan berada di branch development
git checkout development

# Ambil update terbaru dari cloud ke lokal
git pull origin development
```

### 3. Membuat Feature Branch (Isolasi Tugas)
Jangan mengerjakan tugas langsung di branch `development`. Buatlah branch baru untuk setiap tugas spesifik (fitur NLP, atau perbaikan bug).
- Format Nama: `feat/nama-fitur` atau `fix/nama-bug`.
```
# Contoh: Membuat branch untuk modul preprocessing
git checkout -b feat/text-preprocessing
```

### 4. Siklus Kerja: Add, Commit, Push
Setelah melakukan perubahan pada kode, simpan progres kamu dengan pesan commit yang jelas. Gunakan standar Conventional Commits agar riwayat proyek mudah ditelusuri.
- `feat`: Untuk fitur baru (misal: implementasi model LSTM).
- `fix`: Untuk perbaikan error/bug.
- `docs`: Untuk perubahan dokumentasi.
```
# Cek file yang berubah
git status

# Tambahkan perubahan ke staging area
git add .

# Simpan dengan pesan yang deskriptif
git commit -m "feat: implement tokenization using HuggingFace"

# Push branch fitur kamu ke GitHub
git push origin feat/text-preprocessing
```

### 5. Pull Request (PR) ke Development
Setelah kamu selesai dengan fitur tersebut di GitHub, saatnya menggabungkannya ke branch utama tim, yaitu branch `development`:
1. Buka repositori AICodeTrace di browser.
2. Klik tombol `Compare & pull request`.
3. **Penting**: Pastikan `base: development` dan `compare: feat/nama-fitur-kamu`.
4. Berikan deskripsi singkat apa yang kamu kerjakan.
5. Minta anggota tim lain (reviewer) untuk mengecek kode kamu. Jika sudah disetujui (Approve), baru lakukan Merge oleh kamu atau timmu.

## 👥 Team
1. Almer Aji Valentino - 2802544280
2. Frederick Kamsono - 2802549205
3. Kevin Briando Saputra Rinaldy - 2802534172
4. Rizky Mirzaviandy Priambodo - 2802549041
5. Wilson Christian - 2802542804