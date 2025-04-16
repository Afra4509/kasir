# Barcode Cashier System

![Barcode Cashier System](https://github.com/afrafadhma/barcode-cashier/raw/main/app_screenshot.png)

## 🛒 Overview

Barcode Cashier System adalah aplikasi desktop yang menggunakan AI untuk memindai barcode produk melalui gambar dan mengoperasikan sistem kasir dengan interface GUI yang user-friendly. Aplikasi ini dibuat untuk memudahkan proses transaksi penjualan tanpa memerlukan scanner barcode fisik.

**Dibuat oleh: Afra Fadhma Dinata**

## ⚡ Fitur Utama

- 🔍 Pemindaian barcode melalui gambar dengan AI
- 🧠 Sistem pembelajaran mesin untuk mengenali barcode
- 🛍️ Manajemen keranjang belanja
- 💰 Perhitungan total belanja otomatis
- 🧾 Pembuatan dan penyimpanan struk belanja
- 📊 Database produk terintegrasi
- 🌈 Antarmuka grafis yang modern dan intuitif

## 🚀 Cara Instalasi

### Persyaratan

- Python 3.8 atau lebih baru
- Pip (Python package manager)

### Langkah Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/afrafadhma/barcode-cashier.git
cd barcode-cashier
```

2. Buat dan aktifkan virtual environment (opsional tapi disarankan):
```bash
python -m venv venv
# Di Windows
venv\Scripts\activate
# Di Linux/MacOS
source venv/bin/activate
```

3. Install semua dependensi:
```bash
pip install -r requirements.txt
```

4. Jalankan aplikasi:
```bash
python cashier_app.py
```

## 📚 Cara Penggunaan

1. **Persiapan Dataset**:
   - Buat folder "dataset" di direktori utama
   - Di dalam folder dataset, buat subfolder untuk setiap jenis barcode
   - Masukkan gambar barcode ke subfolder yang sesuai

2. **Menjalankan Aplikasi**:
   - Jalankan `python cashier_app.py`
   - Pada tampilan utama, klik "Browse Image" untuk memilih gambar barcode
   - Klik "Scan Barcode" untuk memindai barcode
   - Masukkan jumlah produk dan klik "Add to Cart"
   - Ulangi proses untuk menambahkan produk lain
   - Klik "Checkout" untuk menyelesaikan transaksi

3. **Hasil Checkout**:
   - Struk transaksi akan disimpan dalam format .txt
   - Format struk lengkap dengan daftar produk, harga, jumlah, dan total belanja

## 🛠️ Struktur Kode

```
barcode-cashier/
├── cashier_app.py            # File utama aplikasi
├── README.md                 # Dokumen ini
├── requirements.txt          # Daftar dependensi
├── barcode_model.h5          # Model AI terlatih (auto-generated)
├── product_database.json     # Database produk
├── dataset/                  # Folder dataset gambar barcode
│   ├── 123456789012/         # Contoh subfolder untuk barcode tertentu
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
└── receipts/                 # Folder struk transaksi (auto-generated)
```

## 🧪 Teknologi Yang Digunakan

- **Python**: Bahasa pemrograman utama
- **TensorFlow/Keras**: Framework machine learning untuk model AI
- **Tkinter**: Library untuk membuat GUI
- **Pillow**: Library untuk manipulasi gambar
- **OpenCV**: Library computer vision
- **Matplotlib**: Library visualisasi data

## 🔍 Cara Kerja Model AI

Model AI menggunakan arsitektur MobileNetV2 dengan transfer learning untuk mengenali pola barcode dari gambar. Proses training melibatkan:

1. Pengumpulan dataset gambar barcode
2. Preprocessing dan augmentasi data
3. Fine-tuning model MobileNetV2 pre-trained
4. Validasi dan evaluasi model

## 🚧 Pengembangan Selanjutnya

- [ ] Implementasi pemindaian langsung dari kamera
- [ ] Sistem manajemen produk dan inventory
- [ ] Autentikasi pengguna dan sistem multi-user
- [ ] Laporan penjualan dan analitik
- [ ] Integrasi dengan printer termal untuk struk
- [ ] Versi mobile app untuk platform Android/iOS
- [ ] 

## 📞 Kontak

Afra Fadhma Dinata - [afrafadmadinata@gmail.com](mailto:afrafadmadinata@gmail.com)

Link Proyek: [https://github.com/afrafadhma/barcode-cashier](https://github.com/afrafadhma/barcode-cashier)

---

⭐ Jangan lupa beri stars jika project ini membantu! ⭐
