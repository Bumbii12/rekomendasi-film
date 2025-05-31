# Laporan Sistem Rekomendasi: [MovieLens Latest Datasets]

## 1. Project Overview

Dalam era digital saat ini, pengguna internet semakin dibanjiri dengan berbagai pilihan produk, layanan, maupun konten hiburan seperti film. Di tengah banyaknya pilihan tersebut, pengguna sering mengalami kesulitan dalam menemukan konten yang sesuai dengan preferensi pribadi mereka. Hal ini menjadi latar belakang utama munculnya sistem rekomendasi (recommender system), yaitu sistem cerdas yang mampu mempelajari preferensi pengguna dan memberikan rekomendasi yang relevan.

Salah satu domain yang sangat membutuhkan sistem rekomendasi adalah industri film. Platform seperti Netflix, Disney+, dan IMDb telah berhasil menerapkan sistem rekomendasi untuk meningkatkan kepuasan pengguna, durasi penggunaan platform, hingga jumlah transaksi (subscription). Teknologi ini secara langsung berkontribusi pada pertumbuhan bisnis dan peningkatan loyalitas pelanggan.

Menurut Gomez-Uribe & Hunt (2015), sekitar 80% tayangan yang ditonton di Netflix berasal dari sistem rekomendasi. Hal ini menunjukkan betapa pentingnya teknologi ini dalam menunjang pengalaman pengguna.

Untuk itu, dalam proyek ini dikembangkan sistem rekomendasi film menggunakan dataset MovieLens 100K, dengan memanfaatkan dua pendekatan utama, yaitu:

Collaborative Filtering – menggunakan kesamaan antar pengguna untuk merekomendasikan film yang belum ditonton.

Hybrid Filtering – menggabungkan Collaborative Filtering dengan Content-Based Filtering berbasis genre film, guna meningkatkan relevansi dan akurasi hasil rekomendasi.

## 2. Business Understanding

### Problem Statements

- Bagaimana cara merekomendasikan film yang relevan bagi pengguna berdasarkan data interaksi dan/atau informasi konten film?
- Pendekatan sistem rekomendasi mana yang lebih efektif dalam menghasilkan rekomendasi akurat dan personal: Content-Based Filtering atau Collaborative Filtering?

### Goals

- Mengembangkan sistem rekomendasi film berbasis data pengguna dan konten film.

- Menerapkan dan membandingkan dua pendekatan rekomendasi: Content-Based Filtering dan Collaborative Filtering.

- Mengevaluasi performa sistem menggunakan metrik seperti Mean Absolute Error (MAE).
### Manfaat

- Membantu pengguna menemukan film sesuai preferensi dengan lebih cepat dan efisien.

- Meningkatkan pengalaman pengguna dengan memberikan rekomendasi yang personal.

- Memberikan insight bagi pengembang sistem atau platform film dalam menyusun sistem rekomendasi yang optimal.

- Menjadi studi awal dalam penerapan teknik machine learning untuk sistem rekomendasi di bidang hiburan.

### Solusi yang Diterapkan

1. **Preprocessing Data**  
   - Menggunakan dataset MovieLens 100K, dilakukan pembersihan data seperti menghapus nilai duplikat dan menggabungkan informasi film dan rating.
   - Membuat pivot table rating pengguna dan mengolah genre film menjadi representasi vektor (multi-hot encoding) untuk pendekatan content-based.
   - Melakukan normalisasi dan filtering untuk mengatasi data sparsity.

2. **Content-Based Filtering**  
   - Membuat profil pengguna berdasarkan rata-rata genre film yang disukai.
   - Menghitung kemiripan antar film menggunakan cosine similarity pada fitur genre.
   - Merekomendasikan film berdasarkan kemiripan konten dengan film yang sebelumnya diberi rating tinggi oleh pengguna.

3. **Collaborative Filtering**  
   - Menggunakan pendekatan user-based atau item-based collaborative filtering.
   - Membangun sistem prediksi rating berdasarkan kemiripan antar pengguna atau antar item menggunakan cosine similarity.
   - Memprediksi rating yang belum diberikan dan merekomendasikan film dengan prediksi tertinggi.

4. **Evaluasi Model**
   - Mengukur kinerja kedua pendekatan menggunakan metrik Mean Absolute Error (MAE) pada data uji.
   - Membandingkan nilai MAE dari kedua metode untuk menentukan pendekatan terbaik.


## 3. Data Understanding

### Informasi Dataset

- Sumber: [Grouplens-MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/latest/). 
- Format: CSV  (hanya datset movies.csv dan rating.csv)
- dimensi dataset movies: (9742, 3)
- dimensi dataset ratings: (100836, 4)

#### Kolom datset movies.csv:
- movieId: ID unik untuk setiap film
- NAMA RUMAH : title rumah.
- title: Judul film beserta tahun rilis
- genres: Genre film, dipisahkan oleh tanda | (string).

- Tampilan dataset movies.csv awal dalam bentuk _DataFrame pandas_.  
| movieId | title                              | genres                                     |
|---------|------------------------------------|--------------------------------------------|
| 1       | Toy Story (1995)                   | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 2       | Jumanji (1995)                     | Adventure\|Children\|Fantasy                |
| 3       | Grumpier Old Men (1995)            | Comedy\|Romance                             |
| 4       | Waiting to Exhale (1995)           | Comedy\|Drama\|Romance                      |
| 5       | Father of the Bride Part II (1995) | Comedy                                      |

#### Kolom datset rating.csv:
- userId: ID unik untuk setiap pengguna (integer).
- movieId: ID film yang diberi rating (integer).
- rating: Nilai rating (0.5 sampai 5.0, kelipatan 0.5).
- timestamp: Waktu rating diberikan (integer, format Unix timestamp).

- Tampilan dataset ratings.csv awal dalam bentuk _DataFrame pandas_.  
| userId | movieId | rating | timestamp |
|--------|---------|--------|-----------|
| 1      | 1       | 4.0    | 964982703 |
| 1      | 3       | 4.0    | 964981247 |
| 1      | 6       | 4.0    | 964982224 |
| 1      | 47      | 5.0    | 964983815 |
| 1      | 50      | 5.0    | 964982931 |


3. Kondisi Data
- Pemeriksaan Kualitas Data
  - Missing Values: 0
  - Duplicate Data: 0
  - Distribusi Data

![distribusi data](https://raw.githubusercontent.com/Bumbii12/regresi-harga-rumah/refs/heads/main/images/distribusi_data.png)

  - Tampilan Dataset

Tampilan dataset awal dalam bentuk _DataFrame pandas_.  

| NO | NAMA RUMAH                                                      | HARGA        | LB  | LT  | KT | KM | GRS |
|----|------------------------------------------------------------------|--------------|-----|-----|----|----|-----|
| 1  | Rumah Murah Hook Tebet Timur, Tebet, Jakarta Selatan            | 3.800.000.000 | 220 | 220 | 3  | 3  | 0   |
| 2  | Rumah Modern di Tebet dekat Stasiun, Tebet, Jakarta Selatan     | 4.600.000.000 | 180 | 137 | 4  | 3  | 2   |
| 3  | Rumah Mewah 2 Lantai Hanya 3 Menit Ke Tebet, Tebet, Jakarta     | 3.000.000.000 | 267 | 250 | 4  | 4  | 4   |
| 4  | Rumah Baru Tebet, Tebet, Jakarta Selatan                        |   430.000.000 |  40 |  25 | 2  | 2  | 0   |
| 5  | Rumah Bagus Tebet komp Gudang Peluru lt 350m, Tebet, Jakarta    | 9.000.000.000 | 400 | 355 | 6  | 5  | 3   |


***

## 4. Data Preparation

Langkah-langkah persiapan data:

1. **Pengecekan Duplikat & Missing Values**  
   Data diperiksa untuk mengetahui apakah terdapat duplikasi dan nilai kosong pada setiap kolom.  
   ✅ Tidak ditemukan data duplikat maupun nilai kosong.

2. **Standarisasi Nama Kolom**  
   Nama-nama kolom diubah menjadi huruf kecil semua agar konsisten dan memudahkan proses analisis selanjutnya.

3. **Normalisasi Harga & Pembersihan Data**  
   - Nilai pada kolom `harga` dikonversi dari satuan Rupiah ke juta Rupiah agar lebih mudah dibaca.  
   - Kolom `nomor` juga dihapus karena tidak memiliki pengaruh signifikan terhadap analisis.

4. **Pembuatan Label Kategori Harga**  
   Kategori harga ditentukan berdasarkan nilai kuartil:  
   - `Murah`: harga ≤ Q1  
   - `Menengah`: Q1 < harga ≤ Median  
   - `Mahal`: harga > Median  
   Kolom baru bernama `tingkat_harga` ditambahkan ke dataset.

5. **Visualisasi Distribusi Fitur**  
   Distribusi untuk fitur numerik divisualisasikan:  
   - Fitur `harga`, `luas bangunan (lb)`, dan `luas tanah (lt)` memiliki distribusi miring ke kanan (positively skewed).  
   - Fitur `jumlah kamar tidur (kt)`, `kamar mandi (km)`, dan `garasi (grs)` menunjukkan distribusi diskrit dengan puncak pada nilai 3–5.

6. **Deteksi Outlier dengan Boxplot**  
   Hampir semua fitur mengandung outlier, terutama pada `harga`, `lt`, dan `lb`.  
   Outlier **tidak dihapus** karena:
   - Mewakili variasi nyata dalam pasar properti (misal rumah mewah).
   - Penting untuk model pembelajaran, terutama regresi.
   - Tidak mengganggu hubungan antar variabel utama.

7. **Analisis Korelasi**  
   Korelasi antara variabel numerik dihitung dan divisualisasikan:  
   - Korelasi kuat ditemukan antara `harga` dengan `lt (0.81)` dan `lb (0.75)`.  
   - Fitur lain seperti `kt`, `km`, dan `grs` memiliki korelasi lebih rendah tapi tetap memberikan kontribusi.

8. **Visualisasi Kategori Harga**  
   Diagram batang dibuat untuk menunjukkan distribusi jumlah rumah berdasarkan kategori harga:  
   - `Mahal` adalah kategori dengan jumlah rumah terbanyak.  
   - Diikuti oleh `Menengah`, dan terakhir `Murah`.

9. **Pemilihan Fitur (Feature Selection)**
   Untuk membangun model regresi, dilakukan pemilihan fitur yang digunakan sebagai variabel independen (X).
   Fitur-fitur yang dipilih adalah:
   - lb (luas bangunan)
   - lt (luas tanah)
   - kt (jumlah kamar tidur)
   - km (jumlah kamar mandi)
   - grs (jumlah garasi)
> Pemilihan ini didasarkan pada domain knowledge bahwa variabel-variabel tersebut secara langsung memengaruhi harga rumah, serta hasil analisis korelasi yang menunjukkan bahwa lt dan lb memiliki hubungan yang signifikan terhadap harga.

10. **Data Splitting**
   Data dipecah menjadi 3 bagian:
   - 70% untuk training model
   - 30% untuk pengujian model

## 5. Modeling
   Modeling dilakukan untuk memprediksi harga rumah berdasarkan fitur numerik yang telah dipilih. Dua algoritma digunakan dan dibandingkan performanya, yaitu Linear Regression dan Random Forest Regressor.


   ### 1. Linear Regression
   Linear Regression adalah algoritma regresi yang sederhana dan umum digunakan untuk memodelkan hubungan linear antara fitur (misal: luas bangunan, luas tanah, jumlah kamar) dengan target (harga rumah).

   #### Proses:
   - Model dilatih menggunakan data pelatihan.
   - Tidak memerlukan tuning hyperparameter dalam implementasi dasarnya.
   - Cocok untuk data dengan hubungan linear antar variabel.

   #### Kelebihan:
   - Mudah dipahami dan diinterpretasikan.
   - Proses pelatihan cepat dan efisien.
   - Cocok untuk pola data linear.

   #### Kekurangan:
   - Kurang fleksibel untuk data non-linear.
   - Rentan terhadap outlier dan multikolinearitas.
   - Akurasi menurun jika data tidak linear.

   ---

   ### 2. Random Forest Regressor
   Random Forest Regressor adalah metode ensemble berbasis decision tree yang membangun banyak pohon dan menggabungkan hasilnya untuk prediksi yang lebih baik.

   #### Proses:
   - Model diinisialisasi dengan random_state=42 untuk memastikan reprodusibilitas.
   - Tuning hyperparameter dilakukan menggunakan GridSearchCV dengan kombinasi:
   - n_estimators: [50, 100, 200]
   - max_depth: [None, 10, 20]
   - Model terbaik dipilih berdasarkan nilai R² tertinggi pada validasi silang (cross-validation).
   Parameter Akhir Model:
   - random_state: 42
   - n_estimators: 200
   - max_depth: 20
   > (diperoleh dari grid_search.best_params_)

   #### Kelebihan:
   - Mampu menangani pola non-linear.
   - Lebih tahan terhadap outlier.
   - Memberikan estimasi pentingnya fitur.
   - Performa lebih stabil daripada single decision tree.

   #### Kekurangan:
   - Lebih lambat dalam pelatihan dan prediksi.
   - Kurang interpretatif dibanding linear regression.
   - Risiko overfitting jika tuning tidak optimal.

   ---


## 6. Evaluasi
   ## 📊 Evaluasi Model

Evaluasi dilakukan untuk mengukur performa kedua model (Linear Regression dan Random Forest Regressor) pada data pelatihan dan data pengujian dengan metrik utama:

- **MAE (Mean Absolute Error):** Rata-rata nilai absolut selisih prediksi dan nilai aktual.
- **MSE (Mean Squared Error):** Rata-rata kuadrat selisih prediksi dan nilai aktual, memberikan penalti lebih besar pada kesalahan besar.
- **R2 Score (Koefisien Determinasi):** Proporsi variansi target yang bisa dijelaskan oleh model, dengan nilai maksimum 1 (semakin tinggi semakin baik).

---

### Linear Regression

- Model ini menghasilkan nilai R2 sekitar 0.70 pada data pelatihan dan 0.76 pada data pengujian, yang menunjukkan model dapat menjelaskan sekitar 70-76% variansi harga rumah.
- Nilai MAE dan MSE cukup besar, menunjukkan masih terdapat kesalahan prediksi yang cukup signifikan, terutama pada data pengujian.

---

### Random Forest Regressor

- Setelah dilakukan tuning hyperparameter dengan GridSearchCV, model Random Forest menunjukkan peningkatan performa.
- Nilai R2 pada data pelatihan sangat tinggi (~0.95), dan pada data pengujian meningkat menjadi sekitar 0.79, menunjukkan model dapat menangkap pola data dengan lebih baik.
- MAE dan MSE pada data pengujian lebih kecil dibanding Linear Regression, menandakan prediksi yang lebih akurat dan kesalahan yang lebih rendah.

---

### Perbandingan Kinerja Model

| Model             | R2 Train | R2 Test | MAE Test   | MSE Test      |
|-------------------|----------|---------|------------|---------------|
| Linear Regression | 0.70     | 0.76    | 2064.45    | 11,744,520.00 |
| Random Forest     | 0.95     | 0.79    | 1757.68    | 9,974,966.00  |

- Random Forest menunjukkan performa yang lebih baik secara keseluruhan, terutama pada data pengujian, dengan R2 lebih tinggi dan error lebih kecil.
- Hal ini menunjukkan Random Forest lebih mampu menangani kompleksitas data harga rumah dibanding Linear Regression.
![Visualisasi Regresi Random Forest: Prediksi vs Nilai Aktual](https://raw.githubusercontent.com/Bumbii12/regresi-harga-rumah/refs/heads/main/images/random_f.png)


## 7. Kesimpulan
Proyek ini berhasil mengembangkan dan membandingkan dua model prediksi harga rumah di wilayah Tebet, Jakarta Selatan, yaitu regresi linear dan Random Forest. Berdasarkan evaluasi dengan metrik MAE, MSE, dan R², model Random Forest memberikan performa lebih baik dengan kemampuan menangkap pola data yang kompleks dan non-linear, sehingga menghasilkan prediksi harga yang lebih akurat dibanding regresi linear. Hasil ini menunjukkan bahwa Random Forest lebih sesuai digunakan untuk prediksi harga rumah di pasar properti Tebet yang variatif. Model ini dapat membantu pembeli, penjual, dan pengembang properti dalam pengambilan keputusan harga yang lebih objektif dan realistis.


Referensi:  
  [Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 1–19.](https://doi.org/10.1145/2843948)

  [Harper, F. M., & Konstan, J. A. (2016). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 1–19.](https://doi.org/10.1145/2827872)

  [Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook (2nd ed.). Springer.]( https://doi.org/10.1007/978-1-4899-7637-6)