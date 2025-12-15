# Student Performance Analysis System 📊

Sistem analisis dan prediksi performa akademik siswa menggunakan Machine Learning dan Dashboard Interaktif.

## 🎯 Overview

Sistem komprehensif untuk prediksi nilai siswa, klasifikasi performa, dan clustering dengan dashboard interaktif.

## 📁 Files

- `student.csv` - Dataset (398 siswa, 33 fitur)
- `analisis_data.py` - Training script
- `dashboard_student.py` - Streamlit dashboard
- `models/` - Trained models (.pkl)
- `paper/draft_paper.md` - Research paper

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models
python analisis_data.py

# 3. Run dashboard
streamlit run dashboard_student.py
```

Dashboard: http://localhost:8501

## 📊 Model Performance

- **Regression (G3 Prediction)**: R²=0.78, MAE=1.42
- **Classification**: Accuracy=84%
- **Clustering**: 4 clusters, Silhouette=0.48

## 📚 Dashboard Features

1. **Overview** - Statistics & visualizations
2. **Prediction** - Individual grade prediction
3. **Clustering** - Student segmentation
4. **Analysis** - Interactive exploration
5. **Batch** - Mass prediction from CSV

## 🔑 Key Factors

Top predictors: Past failures, Study time, Mother's education, Age, Absences

## 📖 Documentation

Full research paper: `paper/draft_paper.md`

## 👥 Authors

[Your Name] - [Your Email]
