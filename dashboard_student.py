"""
Dashboard Interaktif - Sistem Analisis Performa Siswa
=====================================================

Dashboard Streamlit untuk visualisasi dan prediksi performa akademik siswa
dengan machine learning.

Features:
- Overview & Statistics
- Grade Prediction
- Clustering Analysis
- Exploratory Analysis
- Batch Prediction

Author: [Your Name]
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Student Performance Analysis",
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
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
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
        color: #2196f3;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
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
    """Load student dataset"""
    df = pd.read_csv('student.csv')
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
        with open('models/clustering_model.pkl', 'rb') as f:
            models['clustering'] = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            models['label_encoders'] = pickle.load(f)
        with open('models/performance_encoder.pkl', 'rb') as f:
            models['performance_encoder'] = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            models['feature_columns'] = pickle.load(f)
        return models
    except FileNotFoundError:
        st.error("⚠️ Models not found! Please run 'python analisis_data.py' first.")
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
st.sidebar.markdown("# 📊 Student Performance")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🎯 Grade Prediction", "👥 Clustering Analysis", 
     "🔍 Exploratory Analysis", "📤 Batch Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Dataset Info")
st.sidebar.metric("Total Students", len(df))
st.sidebar.metric("Features", len(df.columns))
st.sidebar.metric("Schools", df['school'].nunique())

# ============================================================================
# PAGE 1: OVERVIEW & STATISTICS
# ============================================================================
if page == "🏠 Overview":
    st.markdown('<p class="main-header">📊 Student Performance Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>🎓 Sistem Analisis Performa Akademik Siswa</strong><br>
    Dashboard interaktif untuk analisis, visualisasi, dan prediksi performa siswa menggunakan Machine Learning.
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown('<p class="sub-header">📊 Key Statistics</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Students",
            len(df),
            delta=None
        )
    
    with col2:
        avg_grade = df['G3'].mean()
        st.metric(
            "Average Final Grade (G3)",
            f"{avg_grade:.2f}",
            delta=None
        )
    
    with col3:
        pass_rate = (df['G3'] >= 10).sum() / len(df) * 100
        st.metric(
            "Pass Rate (≥10)",
            f"{pass_rate:.1f}%",
            delta=None
        )
    
    with col4:
        excellence_rate = (df['G3'] >= 16).sum() / len(df) * 100
        st.metric(
            "Excellence Rate (≥16)",
            f"{excellence_rate:.1f}%",
            delta=None
        )
    
    # Grade Distribution
    st.markdown('<p class="sub-header">📈 Grade Distribution</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='G3',
            nbins=21,
            title='Distribution of Final Grades (G3)',
            labels={'G3': 'Final Grade', 'count': 'Number of Students'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance categories
        def categorize_grade(grade):
            if grade >= 16:
                return 'Excellent (16-20)'
            elif grade >= 12:
                return 'Good (12-15)'
            elif grade >= 8:
                return 'Average (8-11)'
            else:
                return 'Poor (0-7)'
        
        df['category'] = df['G3'].apply(categorize_grade)
        category_counts = df['category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Performance Categories Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Grade Progression
    st.markdown('<p class="sub-header">📉 Grade Progression (G1 → G2 → G3)</p>', unsafe_allow_html=True)
    
    grade_progression = pd.DataFrame({
        'Period': ['G1 (1st Period)', 'G2 (2nd Period)', 'G3 (Final)'],
        'Average Grade': [df['G1'].mean(), df['G2'].mean(), df['G3'].mean()],
        'Std Dev': [df['G1'].std(), df['G2'].std(), df['G3'].std()]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grade_progression['Period'],
        y=grade_progression['Average Grade'],
        mode='lines+markers',
        name='Average Grade',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=12)
    ))
    
    fig.update_layout(
        title='Average Grade Progression Across Periods',
        xaxis_title='Period',
        yaxis_title='Average Grade',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown('<p class="sub-header">🔥 Correlation with Final Grade (G3)</p>', unsafe_allow_html=True)
    
    if feature_importance is not None:
        top_features = feature_importance.head(15)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Most Important Features for Grade Prediction',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Gender Analysis
    st.markdown('<p class="sub-header">👥 Gender Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender_dist = df['sex'].value_counts()
        fig = px.pie(
            values=gender_dist.values,
            names=['Female' if x == 'F' else 'Male' for x in gender_dist.index],
            title='Gender Distribution',
            color_discrete_sequence=['#ff69b4', '#1f77b4']
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        gender_grades = df.groupby('sex')['G3'].mean().reset_index()
        gender_grades['sex'] = gender_grades['sex'].map({'F': 'Female', 'M': 'Male'})
        
        fig = px.bar(
            gender_grades,
            x='sex',
            y='G3',
            title='Average Final Grade by Gender',
            labels={'sex': 'Gender', 'G3': 'Average Final Grade'},
            color='sex',
            color_discrete_sequence=['#ff69b4', '#1f77b4']
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: GRADE PREDICTION
# ============================================================================
elif page == "🎯 Grade Prediction":
    st.markdown('<p class="main-header">🎯 Student Grade Prediction</p>', unsafe_allow_html=True)
    
    if models is None:
        st.error("⚠️ Models not loaded. Please run 'python analisis_data.py' first!")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>📝 Input Student Information</strong><br>
    Fill in the form below to predict the student's final grade and performance category.
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Demographics**")
            school = st.selectbox("School", ['GP', 'MS'])
            sex = st.selectbox("Gender", ['F', 'M'])
            age = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address Type", ['U', 'R'], 
                                 format_func=lambda x: 'Urban' if x == 'U' else 'Rural')
            
            st.markdown("**👨‍👩‍👧‍👦 Family**")
            famsize = st.selectbox("Family Size", ['LE3', 'GT3'],
                                  format_func=lambda x: '≤3' if x == 'LE3' else '>3')
            Pstatus = st.selectbox("Parents Status", ['T', 'A'],
                                  format_func=lambda x: 'Living Together' if x == 'T' else 'Apart')
            Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4],
                               format_func=lambda x: ['None', 'Primary (4th)', 'Middle (9th)', 'Secondary', 'Higher'][x])
            Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4],
                               format_func=lambda x: ['None', 'Primary (4th)', 'Middle (9th)', 'Secondary', 'Higher'][x])
        
        with col2:
            st.markdown("**💼 Parents' Jobs**")
            job_options = ['teacher', 'health', 'services', 'at_home', 'other']
            Mjob = st.selectbox("Mother's Job", job_options)
            Fjob = st.selectbox("Father's Job", job_options)
            
            st.markdown("**🎓 School Related**")
            reason = st.selectbox("Reason to Choose School", 
                                 ['home', 'reputation', 'course', 'other'])
            guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
            traveltime = st.slider("Travel Time (hours)", 1, 4, 1)
            studytime = st.slider("Weekly Study Time", 1, 4, 2)
            failures = st.slider("Past Class Failures", 0, 4, 0)
        
        with col3:
            st.markdown("**🎯 Support & Activities**")
            schoolsup = st.selectbox("Extra Educational Support", ['yes', 'no'])
            famsup = st.selectbox("Family Educational Support", ['yes', 'no'])
            paid = st.selectbox("Extra Paid Classes", ['yes', 'no'])
            activities = st.selectbox("Extra-curricular Activities", ['yes', 'no'])
            nursery = st.selectbox("Attended Nursery School", ['yes', 'no'])
            higher = st.selectbox("Wants Higher Education", ['yes', 'no'])
            internet = st.selectbox("Internet Access at Home", ['yes', 'no'])
            romantic = st.selectbox("In Romantic Relationship", ['yes', 'no'])
            
            st.markdown("**📊 Personal & Health**")
            famrel = st.slider("Family Relationship Quality", 1, 5, 4)
            freetime = st.slider("Free Time After School", 1, 5, 3)
            goout = st.slider("Going Out with Friends", 1, 5, 3)
            Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
            Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)
            health = st.slider("Current Health Status", 1, 5, 3)
            absences = st.number_input("Number of School Absences", 0, 100, 0)
        
        submit = st.form_submit_button("🔮 Predict Grade", use_container_width=True)
    
    if submit:
        # Prepare input data
        input_data = {
            'school': school, 'sex': sex, 'age': age, 'address': address,
            'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
            'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
            'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
            'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
            'activities': activities, 'nursery': nursery, 'higher': higher,
            'internet': internet, 'romantic': romantic, 'famrel': famrel,
            'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc,
            'health': health, 'absences': absences
        }
        
        # Encode categorical features
        input_df = pd.DataFrame([input_data])
        label_encoders = models['label_encoders']
        
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Ensure correct column order
        feature_columns = models['feature_columns']
        input_df = input_df[feature_columns]
        
        # Make predictions
        predicted_grade = models['regression'].predict(input_df)[0]
        predicted_class = models['classification'].predict(input_df)[0]
        predicted_proba = models['classification'].predict_proba(input_df)[0]
        
        performance_encoder = models['performance_encoder']
        predicted_category = performance_encoder.inverse_transform([predicted_class])[0]
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="sub-header">🎉 Prediction Results</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1f77b4; margin: 0;">Predicted Final Grade</h3>
                <h1 style="margin: 0.5rem 0; font-size: 3rem;">{predicted_grade:.2f}</h1>
                <p style="margin: 0; color: #666;">out of 20</p>
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
                <h3 style="color: {color}; margin: 0;">Performance Category</h3>
                <h2 style="margin: 0.5rem 0; font-size: 2rem; color: {color};">{predicted_category}</h2>
                <p style="margin: 0; color: #666;">Classification</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = predicted_proba[predicted_class] * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #ff9800; margin: 0;">Confidence</h3>
                <h1 style="margin: 0.5rem 0; font-size: 3rem;">{confidence:.1f}%</h1>
                <p style="margin: 0; color: #666;">Prediction Certainty</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability distribution
        st.markdown("### 📊 Category Probability Distribution")
        
        proba_df = pd.DataFrame({
            'Category': performance_encoder.classes_,
            'Probability': predicted_proba * 100
        }).sort_values('Probability', ascending=False)
        
        fig = px.bar(
            proba_df,
            x='Category',
            y='Probability',
            title='Probability for Each Performance Category',
            labels={'Probability': 'Probability (%)'},
            color='Probability',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if predicted_grade < 10:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ At Risk - Intervention Needed</strong><br>
            • Encourage more study time and reduce absences<br>
            • Consider extra tutoring or educational support<br>
            • Monitor health and well-being<br>
            • Improve family communication regarding studies
            </div>
            """, unsafe_allow_html=True)
        elif predicted_grade < 14:
            st.markdown("""
            <div class="info-box">
            <strong>📚 Room for Improvement</strong><br>
            • Maintain consistent study habits<br>
            • Participate in school activities<br>
            • Seek help when needed<br>
            • Balance social life and studies
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
            <strong>✅ Excellent Performance Expected</strong><br>
            • Continue current study habits<br>
            • Consider leadership roles or peer tutoring<br>
            • Prepare for higher education<br>
            • Maintain healthy work-life balance
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: CLUSTERING ANALYSIS
# ============================================================================
elif page == "👥 Clustering Analysis":
    st.markdown('<p class="main-header">👥 Student Clustering Analysis</p>', unsafe_allow_html=True)
    
    if models is None:
        st.error("⚠️ Models not loaded. Please run 'python analisis_data.py' first!")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>🔍 Student Segmentation</strong><br>
    K-Means clustering untuk mengelompokkan siswa berdasarkan karakteristik serupa.
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data for clustering
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoders = models['label_encoders']
    for col in df_processed.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels
            df_processed[col] = df_processed[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Get feature columns (exclude G1, G2, G3)
    feature_columns = models['feature_columns']
    X = df_processed[feature_columns]
    
    # Scale and predict clusters
    X_scaled = models['scaler'].transform(X)
    df_processed['cluster'] = models['clustering'].predict(X_scaled)
    
    # Cluster statistics
    n_clusters = df_processed['cluster'].nunique()
    
    st.markdown(f"### 📊 Cluster Overview (K = {n_clusters})")
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_counts = df_processed['cluster'].value_counts().sort_index()
        fig = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title='Student Distribution Across Clusters',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cluster_grades = df_processed.groupby('cluster')['G3'].mean().reset_index()
        cluster_grades['cluster_name'] = cluster_grades['cluster'].apply(lambda x: f'Cluster {x}')
        
        fig = px.bar(
            cluster_grades,
            x='cluster_name',
            y='G3',
            title='Average Final Grade per Cluster',
            labels={'cluster_name': 'Cluster', 'G3': 'Average Grade'},
            color='G3',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed cluster analysis
    st.markdown("### 🔍 Detailed Cluster Characteristics")
    
    for cluster_id in range(n_clusters):
        with st.expander(f"📌 Cluster {cluster_id} Details", expanded=(cluster_id == 0)):
            cluster_df = df[df_processed['cluster'] == cluster_id]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Students", len(cluster_df))
            with col2:
                st.metric("Avg Grade (G3)", f"{cluster_df['G3'].mean():.2f}")
            with col3:
                st.metric("Pass Rate", f"{(cluster_df['G3'] >= 10).mean() * 100:.1f}%")
            with col4:
                st.metric("Avg Absences", f"{cluster_df['absences'].mean():.1f}")
            
            # Characteristics
            st.markdown("**Key Characteristics:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"- **Gender:** {cluster_df['sex'].mode().values[0]} (dominant)")
                st.write(f"- **Average Age:** {cluster_df['age'].mean():.1f} years")
                st.write(f"- **School:** {cluster_df['school'].mode().values[0]}")
                st.write(f"- **Study Time:** {cluster_df['studytime'].mean():.1f}/5")
            
            with col2:
                st.write(f"- **Family Support:** {cluster_df['famsup'].mode().values[0]}")
                st.write(f"- **Want Higher Ed:** {cluster_df['higher'].mode().values[0]}")
                st.write(f"- **Internet:** {cluster_df['internet'].mode().values[0]}")
                st.write(f"- **Avg Failures:** {cluster_df['failures'].mean():.2f}")

# ============================================================================
# PAGE 4: EXPLORATORY ANALYSIS
# ============================================================================
elif page == "🔍 Exploratory Analysis":
    st.markdown('<p class="main-header">🔍 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Data Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        school_filter = st.multiselect("School", df['school'].unique(), default=df['school'].unique())
    with col2:
        sex_filter = st.multiselect("Gender", df['sex'].unique(), default=df['sex'].unique())
    with col3:
        age_range = st.slider("Age Range", int(df['age'].min()), int(df['age'].max()), 
                             (int(df['age'].min()), int(df['age'].max())))
    with col4:
        higher_filter = st.multiselect("Wants Higher Education", df['higher'].unique(), 
                                       default=df['higher'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['school'].isin(school_filter)) &
        (df['sex'].isin(sex_filter)) &
        (df['age'] >= age_range[0]) &
        (df['age'] <= age_range[1]) &
        (df['higher'].isin(higher_filter))
    ]
    
    st.info(f"📊 Showing {len(filtered_df)} students (filtered from {len(df)} total)")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Study Patterns", "👨‍👩‍👧‍👦 Family Impact", 
                                       "📉 Failure Analysis", "🏥 Health & Lifestyle"])
    
    with tab1:
        st.markdown("### 📚 Study Time vs Final Grade")
        
        fig = px.box(
            filtered_df,
            x='studytime',
            y='G3',
            title='Final Grade Distribution by Weekly Study Time',
            labels={'studytime': 'Study Time (1-4)', 'G3': 'Final Grade'},
            color='studytime',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                filtered_df,
                x='absences',
                y='G3',
                title='Absences vs Final Grade',
                labels={'absences': 'Number of Absences', 'G3': 'Final Grade'},
                trendline='ols',
                color='sex',
                color_discrete_map={'F': '#ff69b4', 'M': '#1f77b4'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            support_df = filtered_df.groupby(['schoolsup', 'famsup'])['G3'].mean().reset_index()
            support_df['support_type'] = support_df.apply(
                lambda x: f"School: {x['schoolsup']}, Family: {x['famsup']}", axis=1
            )
            
            fig = px.bar(
                support_df,
                x='support_type',
                y='G3',
                title='Average Grade by Educational Support',
                labels={'support_type': 'Support Type', 'G3': 'Average Final Grade'},
                color='G3',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 👨‍👩‍👧‍👦 Family Education Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            medu_grades = filtered_df.groupby('Medu')['G3'].mean().reset_index()
            medu_grades['Medu_label'] = medu_grades['Medu'].map({
                0: 'None', 1: 'Primary', 2: 'Middle', 3: 'Secondary', 4: 'Higher'
            })
            
            fig = px.line(
                medu_grades,
                x='Medu_label',
                y='G3',
                title="Mother's Education vs Student Grade",
                markers=True,
                labels={'Medu_label': "Mother's Education Level", 'G3': 'Average Final Grade'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fedu_grades = filtered_df.groupby('Fedu')['G3'].mean().reset_index()
            fedu_grades['Fedu_label'] = fedu_grades['Fedu'].map({
                0: 'None', 1: 'Primary', 2: 'Middle', 3: 'Secondary', 4: 'Higher'
            })
            
            fig = px.line(
                fedu_grades,
                x='Fedu_label',
                y='G3',
                title="Father's Education vs Student Grade",
                markers=True,
                labels={'Fedu_label': "Father's Education Level", 'G3': 'Average Final Grade'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Parent status impact
        pstatus_grades = filtered_df.groupby('Pstatus')['G3'].mean().reset_index()
        pstatus_grades['Pstatus_label'] = pstatus_grades['Pstatus'].map({
            'T': 'Living Together', 'A': 'Apart'
        })
        
        fig = px.bar(
            pstatus_grades,
            x='Pstatus_label',
            y='G3',
            title='Average Grade by Parents Living Status',
            labels={'Pstatus_label': 'Parents Status', 'G3': 'Average Final Grade'},
            color='G3',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📉 Past Failures Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            failures_dist = filtered_df['failures'].value_counts().sort_index()
            fig = px.bar(
                x=failures_dist.index,
                y=failures_dist.values,
                title='Distribution of Past Failures',
                labels={'x': 'Number of Failures', 'y': 'Number of Students'},
                color=failures_dist.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            failures_grades = filtered_df.groupby('failures')['G3'].mean().reset_index()
            fig = px.line(
                failures_grades,
                x='failures',
                y='G3',
                title='Average Final Grade by Past Failures',
                markers=True,
                labels={'failures': 'Number of Past Failures', 'G3': 'Average Final Grade'}
            )
            fig.update_traces(line_color='#f44336', marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🏥 Health & Lifestyle Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            health_grades = filtered_df.groupby('health')['G3'].mean().reset_index()
            fig = px.bar(
                health_grades,
                x='health',
                y='G3',
                title='Average Grade by Health Status',
                labels={'health': 'Health Status (1-5)', 'G3': 'Average Final Grade'},
                color='G3',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            goout_grades = filtered_df.groupby('goout')['G3'].mean().reset_index()
            fig = px.line(
                goout_grades,
                x='goout',
                y='G3',
                title='Average Grade by Going Out Frequency',
                markers=True,
                labels={'goout': 'Going Out Frequency (1-5)', 'G3': 'Average Final Grade'}
            )
            fig.update_traces(line_color='#9c27b0', marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
        
        # Alcohol consumption
        st.markdown("### 🍺 Alcohol Consumption Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dalc_grades = filtered_df.groupby('Dalc')['G3'].mean().reset_index()
            fig = px.bar(
                dalc_grades,
                x='Dalc',
                y='G3',
                title='Workday Alcohol vs Grade',
                labels={'Dalc': 'Workday Alcohol (1-5)', 'G3': 'Average Grade'},
                color='G3',
                color_continuous_scale='OrRd'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            walc_grades = filtered_df.groupby('Walc')['G3'].mean().reset_index()
            fig = px.bar(
                walc_grades,
                x='Walc',
                y='G3',
                title='Weekend Alcohol vs Grade',
                labels={'Walc': 'Weekend Alcohol (1-5)', 'G3': 'Average Grade'},
                color='G3',
                color_continuous_scale='OrRd'
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: BATCH PREDICTION
# ============================================================================
elif page == "📤 Batch Prediction":
    st.markdown('<p class="main-header">📤 Batch Prediction</p>', unsafe_allow_html=True)
    
    if models is None:
        st.error("⚠️ Models not loaded. Please run 'python analisis_data.py' first!")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>📁 Upload CSV File</strong><br>
    Upload a CSV file with student data untuk prediksi batch. File harus memiliki format yang sama dengan dataset training.
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read uploaded file
        batch_df = pd.read_csv(uploaded_file)
        
        st.success(f"✓ File uploaded successfully! {len(batch_df)} rows detected.")
        
        # Show preview
        with st.expander("👀 Preview Data", expanded=True):
            st.dataframe(batch_df.head(10))
        
        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner("Processing predictions..."):
                # Prepare data
                batch_processed = batch_df.copy()
                
                # Encode categorical features
                label_encoders = models['label_encoders']
                for col in batch_processed.select_dtypes(include=['object']).columns:
                    if col in label_encoders:
                        le = label_encoders[col]
                        # Handle unseen labels
                        batch_processed[col] = batch_processed[col].map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                
                # Ensure correct columns
                feature_columns = models['feature_columns']
                
                # Add missing columns with default values if needed
                for col in feature_columns:
                    if col not in batch_processed.columns:
                        batch_processed[col] = 0
                
                X_batch = batch_processed[feature_columns]
                
                # Make predictions
                batch_df['Predicted_G3'] = models['regression'].predict(X_batch)
                batch_df['Predicted_Class'] = models['classification'].predict(X_batch)
                
                # Decode class labels
                performance_encoder = models['performance_encoder']
                batch_df['Predicted_Category'] = performance_encoder.inverse_transform(
                    batch_df['Predicted_Class']
                )
                
                # Get probabilities
                probabilities = models['classification'].predict_proba(X_batch)
                batch_df['Confidence'] = probabilities.max(axis=1) * 100
                
                # Show results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Predictions", len(batch_df))
                with col2:
                    st.metric("Average Predicted Grade", f"{batch_df['Predicted_G3'].mean():.2f}")
                with col3:
                    st.metric("Average Confidence", f"{batch_df['Confidence'].mean():.1f}%")
                
                # Category distribution
                category_dist = batch_df['Predicted_Category'].value_counts()
                fig = px.pie(
                    values=category_dist.values,
                    names=category_dist.index,
                    title='Predicted Performance Categories Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show results table
                st.markdown("### 📋 Detailed Results")
                display_df = batch_df[['Predicted_G3', 'Predicted_Category', 'Confidence']].copy()
                display_df['Predicted_G3'] = display_df['Predicted_G3'].round(2)
                display_df['Confidence'] = display_df['Confidence'].round(1).astype(str) + '%'
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download results
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("👆 Please upload a CSV file to start batch prediction")
        
        # Show sample format
        with st.expander("📝 Required CSV Format"):
            st.write("Your CSV should contain the following columns:")
            st.code("""
school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,
traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,
higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences
            """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Student Performance Analysis Dashboard</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p>Dataset: Student Performance in Portugal (UCI ML Repository)</p>
</div>
""", unsafe_allow_html=True)
