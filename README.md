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

![distribusi genre ](https://raw.githubusercontent.com/Bumbii12/rekomendasi-film/refs/heads/main/img/dis_genre.png)
> Distribusi genre seperti ini terlihat setelah data genre setiap film dipisah terlebih dahulu, karen terdapat film yang memiliki lebih dari satu genre. Distribusi tertinggi terdapat pada genre Drama.

![distribusi rating ](https://raw.githubusercontent.com/Bumbii12/rekomendasi-film/refs/heads/main/img/dis_ratings.png)
> Distribusi tertinggi terdapat pada rating 4.0, sedangkan distribusi rentang yang terkecil terdapat pada rating 0.5
***

## 4. Data Preparation

Langkah-langkah persiapan data:

1. **Ekstraksi Genre Menjadi Kolom Biner**  
   Data genre pada dataset movies.csv awalnya berada dalam satu kolom dengan format teks, dipisahkan oleh karakter | (contoh: Action|Comedy|Romance).
   Kolom genres kemudian diubah menjadi beberapa kolom biner (0/1) menggunakan teknik One-Hot Encoding, sehingga setiap genre menjadi fitur tersendiri.
   > ✅ Hal ini mempermudah analisis dan memungkinkan penggunaan genre sebagai fitur tambahan di model hybrid atau content-based.

2. **Penggabungan Dataset movies dan ratings**  
   Dataset movies (yang sudah mengandung kolom genre biner) digabungkan dengan dataset ratings berdasarkan kolom movieId.
   Hasilnya adalah dataframe gabungan df yang menyimpan informasi rating, judul film, serta genre-genre film.
   > ✅ Penggabungan ini penting untuk menyatukan informasi pengguna dan film dalam satu struktur data.

3. **Pembagian Data (Train-Test Split)**  
   Dataset df dibagi menjadi dua bagian:
   - df_train: 70% data untuk pelatihan model.
   - df_test: 30% data untuk pengujian model.
   Digunakan random_state=42 untuk memastikan hasil pembagian selalu konsisten saat dijalankan ulang.

4. **Pembuatan User-Item Rating Matrix**  
   Dibuat pivot table dari df_train yang menempatkan userId sebagai baris, movieId sebagai kolom, dan rating sebagai nilai.
   Nilai kosong diisi dengan 0 (artinya belum memberikan rating).
   > ✅ Matriks ini dibutuhkan sebagai input utama untuk algoritma Collaborative Filtering.

5. **Konversi Matriks ke Bentuk Sparse Matrix**  
   Karena mayoritas sel pada user-item matrix bernilai 0, matriks tersebut dikonversi ke format Compressed Sparse Row (CSR) menggunakan csr_matrix.
   > ✅ Format sparse ini menghemat memori dan mempercepat proses komputasi saat melatih model.

6. **Pemetaan userId ke Index dan Sebaliknya**  
   Dibuat dua dictionary:
   - user_id_to_index: mapping dari userId ke indeks numerik.
   -index_to_user_id: mapping dari indeks kembali ke userId.
   > ✅ Mapping ini diperlukan karena beberapa algoritma seperti KNN bekerja dengan indeks numerik, bukan ID asli.


## 5. Modeling
   Sistem rekomendasi dikembangkan untuk membantu pengguna menemukan film yang sesuai dengan preferensi mereka berdasarkan histori rating pengguna lain dan informasi konten film (genre). Dua pendekatan berbeda digunakan, yaitu Collaborative Filtering dan Hybrid Filtering.


   ### 1. Collaborative Filtering (CF)
   Model Collaborative Filtering berbasis kemiripan antar pengguna (user-user similarity). Model ini memanfaatkan rating yang diberikan oleh pengguna-pengguna lain yang memiliki preferensi serupa untuk memprediksi rating yang mungkin diberikan oleh target user terhadap film yang belum ditonton.

   #### Proses:
   - Menggunakan algoritma K-Nearest Neighbors (KNN) dengan cosine similarity untuk menemukan tetangga terdekat dari pengguna target.
   - Rating diprediksi menggunakan rata-rata tertimbang dari rating pengguna-pengguna serupa terhadap film tertentu.
   - Top-N rekomendasi disajikan berdasarkan prediksi rating tertinggi dari film-film yang belum dirating oleh pengguna.

   ![Top-10 Rekomendasi CF untuk userId=2: ]()
   > Insight :
   * Semua film direkomendasikan dengan prediksi rating maksimal (5.00), yang berarti:
      * User-user tetangga yang mirip memberikan rating tinggi pada film tersebut.
      * Model sangat yakin bahwa userId=2 juga akan menyukai film-film ini.
   * Film yang direkomendasikan sebagian besar adalah film klasik populer dan kultus, seperti:
      * Pulp Fiction, The Shining, Memento, Donnie Darko.
      * Ini mengindikasikan userId=2 mungkin memiliki preferensi terhadap film-film drama, thriller, atau misteri dengan rating tinggi.

   #### Kelebihan:
   - Tidak memerlukan informasi konten film.
   - Cocok untuk skenario dengan jumlah pengguna aktif yang besar.
   - Menghasilkan rekomendasi yang bersifat personalized dan seringkali akurat jika data rating cukup padat.

   #### Kekurangan:
   - Tidak dapat memberikan rekomendasi untuk user atau item baru (cold start problem).
   - Rentan terhadap sparsity jika banyak pengguna belum memberikan rating.
   - Bergantung penuh pada rating historis.

   ---

   ### 2. Random Forest Regressor
   Model Hybrid Filtering menggabungkan Collaborative Filtering dengan pendekatan Content-Based Filtering. Pendekatan ini bertujuan untuk mengatasi kekurangan masing-masing metode dengan memadukan keunggulan keduanya.

   #### Proses:
   - Menggunakan algoritma CF seperti sebelumnya untuk menghitung prediksi rating dari user-user yang mirip.
   - Membangun user profile berdasarkan rata-rata preferensi genre yang disukai oleh pengguna.
   - nMenentukan movie profile dari representasi genre setiap film.
   - Menggabungkan skor CF dan skor genre-based menggunakan parameter alpha sebagai bobot penggabungan.
   
   ![Top-10 Rekomendasi Hybrid untuk userId=2:]()
   > Insight :
   * Prediksi rating yang relatif seragam dan tidak terlalu tinggi (sekitar 3) bisa menandakan model hybrid memberikan rekomendasi yang lebih konservatif dibanding model CF murni (yang memberi rating 5).
   * Pendekatan hybrid ini mencoba menyeimbangkan antara kesamaan user (CF) dan kesesuaian genre (content-based), sehingga hasilnya cenderung lebih realistis dan tidak berlebihan.
   * Ini dapat membantu memberikan rekomendasi yang lebih bervariasi dan tidak terlalu optimistik, tapi mungkin perlu peningkatan agar prediksi lebih akurat dan nilai ratingnya lebih dekat ke preferensi nyata user.
   
   #### Kelebihan:
   - Mengatasi masalah cold start pada Collaborative Filtering dengan memanfaatkan konten film.
   - Meningkatkan akurasi dengan mengombinasikan informasi perilaku dan konten.
   - Memberikan rekomendasi yang lebih stabil dan tidak terlalu bias pada kelompok tertentu.

   #### Kekurangan:
   - Lebih kompleks dalam implementasi dan tuning parameter (misal: alpha).
   - Bergantung pada kualitas informasi konten (genre, metadata film).
   - Bisa menghasilkan prediksi yang terlalu konservatif jika bobot genre terlalu dominan.

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