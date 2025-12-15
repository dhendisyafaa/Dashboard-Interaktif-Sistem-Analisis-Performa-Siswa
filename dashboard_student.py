"""
Dashboard Interaktif - Sistem Analisis IPK Mahasiswa
===================================================

Dashboard Streamlit untuk visualisasi dan prediksi IPK mahasiswa
dengan machine learning.

Pages:
- 📊 Overview & Statistik
- 🎯 Prediksi IPK  
- 🔍 Eksplorasi Data

Author: [Your Name]
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Analisis IPK Mahasiswa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: #1976d2;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================
@st.cache_data
def load_data():
    """Load mahasiswa dataset"""
    df = pd.read_csv('mahasiswa.csv')
    return df

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        with open('models/regression_model.pkl', 'rb') as f:
            models['regression'] = pickle.load(f)
        with open('models/classification_model.pkl', 'rb') as f:
            models['classification'] = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            models['label_encoders'] = pickle.load(f)
        with open('models/performance_encoder.pkl', 'rb') as f:
            models['performance_encoder'] = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            models['feature_columns'] = pickle.load(f)
        return models
    except FileNotFoundError:
        st.error("⚠️ Model tidak ditemukan! Jalankan 'python analisis_data.py' terlebih dahulu.")
        return None

@st.cache_data
def load_feature_importance():
    """Load feature importance"""
    try:
        return pd.read_csv('models/feature_importance.csv')
    except FileNotFoundError:
        return None

# Load data
df = load_data()
models = load_models()
feature_importance = load_feature_importance()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("# 📊 Analisis IPK Mahasiswa")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi",
    ["📊 Overview", "🎯 Prediksi IPK", "🔍 Eksplorasi Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Info Dataset")
st.sidebar.metric("Total Mahasiswa", len(df))
st.sidebar.metric("IPK Rata-rata", f"{df['IPK'].mean():.2f}")
st.sidebar.metric("Fitur", 7)

# ============================================================================
# PAGE 1: OVERVIEW & STATISTIK
# ============================================================================
if page == "📊 Overview":
    st.markdown('<p class="main-header">📊 Overview & Statistik IPK Mahasiswa</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>🎓 Sistem Analisis Indeks Prestasi Kumulatif (IPK)</strong><br>
    Dashboard interaktif untuk analisis, visualisasi, dan prediksi IPK mahasiswa menggunakan Machine Learning.
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown("### 📊 Statistik Utama")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Mahasiswa", len(df))
    
    with col2:
        st.metric("IPK Rata-rata", f"{df['IPK'].mean():.2f}")
    
    with col3:
        cumlaude_rate = (df['IPK'] >= 3.5).sum() / len(df) * 100
        st.metric("Cum Laude (≥3.5)", f"{cumlaude_rate:.1f}%")
    
    with col4:
        st.metric("IPK Tertinggi", f"{df['IPK'].max():.2f}")
    
    # IPK Distribution
    st.markdown("### 📈 Distribusi IPK")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='IPK',
            nbins=20,
            title='Distribusi IPK Mahasiswa',
            labels={'IPK': 'Indeks Prestasi Kumulatif', 'count': 'Jumlah Mahasiswa'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance categories
        def categorize_ipk(ipk):
            if ipk >= 3.5:
                return 'Excellent (≥3.5)'
            elif ipk >= 3.0:
                return 'Good (≥3.0)'
            elif ipk >= 2.5:
                return 'Average (≥2.5)'
            else:
                return 'Poor (<2.5)'
        
        df['kategori'] = df['IPK'].apply(categorize_ipk)
        category_counts = df['kategori'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Distribusi Kategori Performa',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    if feature_importance is not None:
        st.markdown("### 🔥 Fitur Paling Berpengaruh")
        
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Urutan Fitur Berdasarkan Pengaruh terhadap IPK',
            labels={'importance': 'Skor Pengaruh', 'feature': 'Fitur'},
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistik per Fitur
    st.markdown("### 📊 Statistik Fitur Akademik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Kehadiran Rata-rata", f"{df['Kehadiran_Persen'].mean():.1f}%")
        st.metric("Partisipasi Diskusi", f"{df['Partisipasi_Diskusi'].mean():.1f}/100")
    
    with col2:
        st.metric("Nilai Tugas", f"{df['Nilai_Tugas'].mean():.1f}/100")
        st.metric("Aktivitas E-Learning", f"{df['Aktivitas_ELearning'].mean():.1f}/100")

# ============================================================================
# PAGE 2: PREDIKSI IPK
# ============================================================================
elif page == "🎯 Prediksi IPK":
    st.markdown('<p class="main-header">🎯 Prediksi IPK Mahasiswa</p>', unsafe_allow_html=True)
    
    if models is None:
        st.error("⚠️ Model belum di-load. Jalankan 'python analisis_data.py' terlebih dahulu!")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>📝 Masukkan Data Mahasiswa</strong><br>
    Isi form di bawah untuk memprediksi IPK dan kategori performa mahasiswa.
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Data Pribadi**")
            jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
            umur = st.slider("Umur", 18, 25, 21)
            status_menikah = st.selectbox("Status Pernikahan", ['Belum Menikah', 'Menikah'])
        
        with col2:
            st.markdown("**📚 Kehadiran & Partisipasi**")
            kehadiran = st.slider("Kehadiran (%)", 40, 100, 70)
            partisipasi = st.slider("Partisipasi Diskusi (skor)", 40, 100, 70)
        
        with col3:
            st.markdown("**📝 Tugas & E-Learning**")
            nilai_tugas = st.slider("Nilai Tugas (rata-rata)", 40, 100, 70)
            aktivitas = st.slider("Aktivitas E-Learning (skor)", 40, 100, 70)
        
        submit = st.form_submit_button("🔮 Prediksi IPK", use_container_width=True)
    
    if submit:
        # Prepare input
        input_data = pd.DataFrame([{
            'Jenis_Kelamin': jenis_kelamin,
            'Umur': umur,
            'Status_Menikah': status_menikah,
            'Kehadiran_Persen': kehadiran,
            'Partisipasi_Diskusi': partisipasi,
            'Nilai_Tugas': nilai_tugas,
            'Aktivitas_ELearning': aktivitas
        }])
        
        # Encode
        label_encoders = models['label_encoders']
        for col in ['Jenis_Kelamin', 'Status_Menikah']:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Ensure column order
        feature_columns = models['feature_columns']
        input_data = input_data[feature_columns]
        
        # Predict
        predicted_ipk = models['regression'].predict(input_data)[0]
        predicted_class = models['classification'].predict(input_data)[0]
        predicted_proba = models['classification'].predict_proba(input_data)[0]
        
        performance_encoder = models['performance_encoder']
        predicted_category = performance_encoder.inverse_transform([predicted_class])[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎉 Hasil Prediksi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1f77b4; margin: 0;">IPK Prediksi</h3>
                <h1 style="margin: 0.5rem 0; font-size: 3rem;">{predicted_ipk:.2f}</h1>
                <p style="margin: 0; color: #666;">Skala 1.5 - 4.0</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            colors = {
                'Excellent': '#4caf50',
                'Good': '#2196f3',
                'Average': '#ff9800',
                'Poor': '#f44336'
            }
            color = colors.get(predicted_category, '#666')
            
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {color}; margin: 0;">Kategori</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2rem; color: {color};">{predicted_category}</h2>
                <p style="margin: 0; color: #666;">Klasifikasi</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = predicted_proba[predicted_class] * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #ff9800; margin: 0;">Confidence</h3>
                <h1 style="margin: 0.5rem 0; font-size: 3rem;">{confidence:.0f}%</h1>
                <p style="margin: 0; color: #666;">Tingkat Keyakinan</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Rekomendasi")
        
        if predicted_ipk >= 3.5:
            st.markdown("""
            <div class="info-box">
            <strong>✅ Performa Excellent - Cum Laude</strong><br>
            • Pertahankan konsistensi belajar<br>
            • Pertimbangkan untuk jadi tutor sebaya<br>
            • Fokus pada pengembangan soft skills
            </div>
            """, unsafe_allow_html=True)
        elif predicted_ipk >= 3.0:
            st.markdown("""
            <div class="info-box">
            <strong>📚 Performa Good - Sangat Memuaskan</strong><br>
            • Tingkatkan partisipasi di kelas<br>
            • Manfaatkan konsultasi dosen<br>
            • Target IPK 3.5+ untuk Cum Laude
            </div>
            """, unsafe_allow_html=True)
        elif predicted_ipk >= 2.5:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Performa Average - Perlu Peningkatan</strong><br>
            • Tingkatkan kehadiran dan partisipasi<br>
            • Ikuti program bimbingan akademik<br>
            • Atur jadwal belajar yang lebih terstruktur
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            <strong>❗ Performa Poor - Perlu Intervensi</strong><br>
            • Segera konsultasi dengan pembimbing akademik<br>
            • Evaluasi metode belajar<br>
            • Pertimbangkan mengikuti remedial
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: EKSPLORASI DATA
# ============================================================================
elif page == "🔍 Eksplorasi Data":
    st.markdown('<p class="main-header">🔍 Eksplorasi Data Mahasiswa</p>', unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Filter Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender_filter = st.multiselect("Jenis Kelamin", df['Jenis_Kelamin'].unique(), 
                                       default=df['Jenis_Kelamin'].unique())
    with col2:
        status_filter = st.multiselect("Status Menikah", df['Status_Menikah'].unique(),
                                       default=df['Status_Menikah'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['Jenis_Kelamin'].isin(gender_filter)) &
        (df['Status_Menikah'].isin(status_filter))
    ]
    
    st.info(f"📊 Menampilkan {len(filtered_df)} mahasiswa (dari {len(df)} total)")
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📈 Kehadiran & Partisipasi", "📝 Tugas & E-Learning", "👥 Demografi"])
    
    with tab1:
        st.markdown("### 📚 Kehadiran vs IPK")
        
        fig = px.scatter(
            filtered_df,
            x='Kehadiran_Persen',
            y='IPK',
            title='Hubungan Kehadiran dengan IPK',
            labels={'Kehadiran_Persen': 'Kehadiran (%)', 'IPK': 'Indeks Prestasi Kumulatif'},
            trendline='ols',
            color='Jenis_Kelamin'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 💬 Partisipasi Diskusi vs IPK")
        
        fig = px.scatter(
            filtered_df,
            x='Partisipasi_Diskusi',
            y='IPK',
            title='Hubungan Partisipasi Diskusi dengan IPK',
            labels={'Partisipasi_Diskusi': 'Skor Partisipasi', 'IPK': 'Indeks Prestasi Kumulatif'},
            trendline='ols',
            color='Status_Menikah'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📝 Nilai Tugas vs IPK")
        
        fig = px.scatter(
            filtered_df,
            x='Nilai_Tugas',
            y='IPK',
            title='Hubungan Nilai Tugas dengan IPK',
            labels={'Nilai_Tugas': 'Rata-rata Nilai Tugas', 'IPK': 'Indeks Prestasi Kumulatif'},
            trendline='ols',
            color='Jenis_Kelamin'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 💻 Aktivitas E-Learning vs IPK")
        
        fig = px.scatter(
            filtered_df,
            x='Aktivitas_ELearning',
            y='IPK',
            title='Hubungan Aktivitas E-Learning dengan IPK',
            labels={'Aktivitas_ELearning': 'Skor Aktivitas E-Learning', 'IPK': 'Indeks Prestasi Kumulatif'},
            trendline='ols',
            color='Status_Menikah'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 👥 IPK Berdasarkan Jenis Kelamin")
        
        gender_ipk = filtered_df.groupby('Jenis_Kelamin')['IPK'].mean().reset_index()
        
        fig = px.bar(
            gender_ipk,
            x='Jenis_Kelamin',
            y='IPK',
            title='Rata-rata IPK Berdasarkan Jenis Kelamin',
            labels={'Jenis_Kelamin': 'Jenis Kelamin', 'IPK': 'Rata-rata IPK'},
            color='IPK',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 💍 IPK Berdasarkan Status Pernikahan")
        
        status_ipk = filtered_df.groupby('Status_Menikah')['IPK'].mean().reset_index()
        
        fig = px.bar(
            status_ipk,
            x='Status_Menikah',
            y='IPK',
            title='Rata-rata IPK Berdasarkan Status Pernikahan',
            labels={'Status_Menikah': 'Status Pernikahan', 'IPK': 'Rata-rata IPK'},
            color='IPK',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
