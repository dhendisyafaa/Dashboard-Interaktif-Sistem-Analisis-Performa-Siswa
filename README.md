# Dashboard Interaktif Sistem Analisis Performa Mahasiswa

Dashboard interaktif berbasis web untuk analisis dan prediksi performa akademik mahasiswa menggunakan Machine Learning dan visualisasi data.

## Features

**Prediksi IPK Individual** - Prediksi IPK mahasiswa berdasarkan aktivitas akademik  
**Klasifikasi Prestasi** - Klasifikasi otomatis ke kategori: Sangat Baik, Baik, Cukup, Kurang  
**Clustering Mahasiswa** - Segmentasi mahasiswa berdasarkan pola aktivitas  
**Visualisasi Interaktif** - Charts dan graphs untuk analisis data  
**Dashboard Real-time** - Interface user-friendly tanpa coding

## Tujuan

Dashboard ini dikembangkan untuk membantu institusi pendidikan dalam:
- Memprediksi performa akademik mahasiswa sejak dini
- Mengidentifikasi mahasiswa yang memerlukan intervensi akademik
- Melakukan segmentasi mahasiswa untuk targeted intervention
- Memvisualisasikan pola dan tren performa akademik

## Machine Learning Models

### 1. Random Forest Regression
- **Fungsi:** Prediksi nilai IPK (1.5 - 4.0)
- **Input:** Kehadiran, Partisipasi Diskusi, Nilai Tugas, Aktivitas E-Learning
- **Metrik:** R² Score, MAE, RMSE

### 2. Random Forest Classification
- **Fungsi:** Klasifikasi kategori prestasi
- **Output:** Sangat Baik (3.51-4.0), Baik (3.01-3.5), Cukup (2.51-3.0), Kurang (1.5-2.5)
- **Metrik:** Accuracy, Precision, Recall, F1-Score

### 3. K-Means Clustering
- **Fungsi:** Segmentasi mahasiswa berdasarkan pola aktivitas
- **Metode:** Elbow Method untuk penentuan cluster optimal
- **Output:** Profil karakteristik setiap cluster

## Dataset

Dataset berisi 500 data mahasiswa dengan fitur:

| Fitur | Deskripsi | Range | Satuan |
|-------|-----------|-------|--------|
| Kehadiran | Persentase kehadiran kuliah | 0-100 | % |
| Partisipasi Diskusi | Tingkat partisipasi dalam diskusi | 0-100 | % |
| Nilai Tugas | Rata-rata nilai tugas | 0-100 | Point |
| Aktivitas E-Learning | Tingkat aktivitas dalam LMS | 0-100 | % |
| IPK | Indeks Prestasi Kumulatif | 1.5-4.0 | Scale |

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/dhendisyafaa/Dashboard-Interaktif-Sistem-Analisis-Performa-Siswa.git
cd Dashboard-Interaktif-Sistem-Analisis-Performa-Siswa

# Install dependencies
pip install -r requirements.txt

# Train models (optional - models sudah disediakan)
python analisis_data.py

# Run dashboard
streamlit run dashboard_student.py
```

Dashboard akan berjalan di `http://localhost:8501`

## Live Demo

**[Dashboard Student Performance Analysis](https://dashboard-interaktif-sistem-analisis-performa-siswa.streamlit.app/)**

## Tech Stack

- **Framework:** Streamlit
- **Machine Learning:** Scikit-learn (Random Forest, K-Means)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud

## Project Structure

```
uas_andat/
├── dashboard_student.py      # Main Streamlit app
├── analisis_data.py          # ML model training script
├── mahasiswa.csv             # Dataset mahasiswa
├── requirements.txt          # Python dependencies
├── models/                   # Trained ML models
│   ├── regression_model.pkl
│   ├── classification_model.pkl
│   ├── clustering_model.pkl
│   └── scaler.pkl
└── .streamlit/               # Streamlit config
    └── config.toml
```

## Dashboard Pages

### 1. Prediksi IPK
- Input form untuk 4 fitur akademik
- Prediksi IPK real-time
- Kategori prestasi mahasiswa
- Visualisasi hasil prediksi

### 2. Analisis Clustering
- Interactive clustering dengan pilihan jumlah cluster
- Scatter plot visualisasi 2D (PCA)
- Tabel profil karakteristik setiap cluster
- Statistik per cluster

### 3. Visualisasi & Insights
- Distribusi IPK (histogram, boxplot)
- Correlation heatmap antar fitur
- Feature importance analysis
- Statistical summary

## Use Cases

**Untuk Dosen/Pembimbing Akademik:**
- Monitoring performa mahasiswa bimbingan
- Early warning system untuk mahasiswa berisiko
- Data-driven intervention planning

**Untuk Staff Akademik:**
- Analisis performa program studi
- Identifikasi mahasiswa untuk beasiswa
- Perencanaan program remedial

**Untuk Pimpinan Institusi:**
- Overview performa akademik institusi
- Data untuk policy making
- Evaluasi efektivitas program akademik

## How to Use

1. **Akses Dashboard** melalui link atau run locally
2. **Pilih Page** dari sidebar:
   - "Prediksi IPK" untuk prediksi individual
   - "Analisis Clustering" untuk segmentasi
   - "Visualisasi & Insights" untuk analisis data
3. **Input Data** atau explore visualisasi yang tersedia
4. **Interpretasi Hasil** untuk decision making

## Model Performance

**Random Forest Regression:**
- R² Score: ~0.82
- MAE: ~0.18
- RMSE: ~0.25

**Random Forest Classification:**
- Accuracy: ~84%
- F1-Score: 0.76 - 0.91 (per kategori)

**K-Means Clustering:**
- Optimal Clusters: 3
- Segmentasi: High Performer (20%), Aktif Seimbang (36%), Perlu Perhatian (44%)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Authors

**Dhendi Syafa A.P**
- GitHub: [@dhendisyafaa](https://github.com/dhendisyafaa)
- Email: dhendisyafa@upi.edu

## Acknowledgments

- Dataset: Data Mahasiswa (Simulasi/Institusi)
- Framework: Streamlit Team
- Libraries: Scikit-learn, Pandas, Plotly

## Contact

For questions or collaboration:
- GitHub Issues: [Create an issue](https://github.com/dhendisyafaa/Dashboard-Interaktif-Sistem-Analisis-Performa-Siswa/issues)
- Email: dhendisyafa@upi.edu

---

Jika project ini bermanfaat, jangan lupa kasih star!