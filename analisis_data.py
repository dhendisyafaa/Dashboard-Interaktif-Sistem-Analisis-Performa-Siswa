"""
Analisis Data dan Model Training - Dataset Mahasiswa Indonesia
================================================================

Script ini melakukan:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Training (Regression, Classification, Clustering)
4. Model Evaluation dan Serialization

Dataset: Mahasiswa Indonesia (500 records)
Target: IPK (1.5-4.0)

Author: Kelompok 18 | Dhendi - Bhismart
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("SISTEM ANALISIS DAN PREDIKSI IPK MAHASISWA")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv('mahasiswa.csv')

print(f"\n✓ Dataset berhasil dimuat!")
print(f"  - Total mahasiswa: {len(df)}")
print(f"  - Total kolom: {len(df.columns)}")
print(f"\nKolom yang tersedia:")
print(df.columns.tolist())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n[2] EXPLORATORY DATA ANALYSIS...")

# Info dataset
print("\n--- Dataset Info ---")
print(df.info())

# Statistik deskriptif
print("\n--- Statistik Deskriptif (Numerik) ---")
print(df.describe())

# Cek missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ Tidak ada missing values!")
else:
    print(missing[missing > 0])

# Distribusi target variable (IPK)
print("\n--- Distribusi IPK (Target) ---")
print(df['IPK'].describe())
print(f"  - Min: {df['IPK'].min()}")
print(f"  - Max: {df['IPK'].max()}")
print(f"  - Mean: {df['IPK'].mean():.2f}")
print(f"  - Median: {df['IPK'].median():.2f}")
print(f"  - Std Dev: {df['IPK'].std():.2f}")

# Korelasi dengan IPK
print("\n--- Korelasi Fitur dengan IPK ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['IPK'].sort_values(ascending=False)
print(correlations)

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n[3] DATA PREPROCESSING...")

# Buat copy
df_processed = df.copy()

# Encode categorical variables
print("  - Encoding categorical variables...")
categorical_cols = ['Jenis_Kelamin', 'Status_Menikah', 'Status_Akademik']
label_encoders = {}

for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

print(f"    ✓ Encoded {len(label_encoders)} categorical columns")

# Buat kategori performa berdasarkan IPK
print("  - Creating performance categories...")
def categorize_ipk(ipk):
    if ipk >= 3.5:
        return 'Excellent'  # Cum Laude
    elif ipk >= 3.0:
        return 'Good'       # Sangat Memuaskan
    elif ipk >= 2.5:
        return 'Average'    # Memuaskan  
    else:
        return 'Poor'       # Cukup

df_processed['performance_category'] = df['IPK'].apply(categorize_ipk)
performance_le = LabelEncoder()
df_processed['performance_encoded'] = performance_le.fit_transform(df_processed['performance_category'])

print(f"    ✓ Performance categories distribution:")
print(df_processed['performance_category'].value_counts())

# ============================================================================
# 4. PREPARE DATA FOR MODELS
# ============================================================================
print("\n[4] PREPARING DATA FOR MODELS...")

# Features: exclude Nama, IPK, dan kategori performa
feature_cols = ['Jenis_Kelamin', 'Umur', 'Status_Menikah', 
                'Kehadiran_Persen', 'Partisipasi_Diskusi', 
                'Nilai_Tugas', 'Aktivitas_ELearning']

X = df_processed[feature_cols]
y_regression = df_processed['IPK']
y_classification = df_processed['performance_encoded']

print(f"  - Features: {len(feature_cols)} columns")
print(f"  - Target (Regression): IPK")
print(f"  - Target (Classification): Performance Category")

# Train-test split
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

print(f"  - Training set: {len(X_train)} samples")
print(f"  - Test set: {len(X_test)} samples")

# Scaling untuk clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# 5. MODEL TRAINING - REGRESSION (Prediksi IPK)
# ============================================================================
print("\n[5] TRAINING REGRESSION MODEL (Random Forest)...")

rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_regressor.fit(X_train, y_reg_train)
y_reg_pred = rf_regressor.predict(X_test)

# Evaluation
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("\n--- Regression Model Performance ---") 
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_regressor.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Feature Importance (Regression) ---")
print(feature_importance.to_string(index=False))

# ============================================================================
# 6. MODEL TRAINING - CLASSIFICATION (Kategori Performa)
# ============================================================================
print("\n[6] TRAINING CLASSIFICATION MODEL (Random Forest)...")

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_classifier.fit(X_train, y_clf_train)
y_clf_pred = rf_classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_clf_test, y_clf_pred)

print("\n--- Classification Model Performance ---")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n--- Classification Report ---")
print(classification_report(
    y_clf_test, y_clf_pred,
    target_names=performance_le.classes_
))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_clf_test, y_clf_pred)
print(cm)

# ============================================================================
# 7. MODEL TRAINING - CLUSTERING (K-Means)
# ============================================================================
print("\n[7] TRAINING CLUSTERING MODEL (K-Means)...")

n_clusters = 4

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10,
    max_iter=300
)

cluster_labels = kmeans.fit_predict(X_scaled)

# Evaluation
silhouette = silhouette_score(X_scaled, cluster_labels)

print("\n--- Clustering Model Performance ---")
print(f"  Number of Clusters: {n_clusters}")
print(f"  Silhouette Score: {silhouette:.4f}")

# Analyze clusters
df_clustered = df.copy()
df_clustered['cluster'] = cluster_labels

print("\n--- Cluster Distribution ---")
print(df_clustered['cluster'].value_counts().sort_index())

print("\n--- Average IPK per Cluster ---")
for i in range(n_clusters):
    cluster_df = df_clustered[df_clustered['cluster'] == i]
    avg_ipk = cluster_df['IPK'].mean()
    count = len(cluster_df)
    print(f"  Cluster {i}: Avg IPK = {avg_ipk:.2f} (n={count})")

# ============================================================================
# 8. SAVE MODELS
# ============================================================================
print("\n[8] SAVING MODELS...")

models_to_save = {
    'regression_model.pkl': rf_regressor,
    'classification_model.pkl': rf_classifier,
    'clustering_model.pkl': kmeans,
    'scaler.pkl': scaler,
    'label_encoders.pkl': label_encoders,
    'performance_encoder.pkl': performance_le,
    'feature_columns.pkl': feature_cols
}

for filename, model in models_to_save.items():
    filepath = f'models/{filename}'
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved: {filepath}")

# Save feature importance
feature_importance.to_csv('models/feature_importance.csv', index=False)
print("  ✓ Saved: models/feature_importance.csv")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - MODEL TRAINING COMPLETED")
print("="*80)
print(f"\n✓ Dataset: {len(df)} mahasiswa, {len(df.columns)} features")
print(f"\n✓ Regression Model (IPK Prediction):")
print(f"  - R² Score: {r2:.4f}")
print(f"  - RMSE: {rmse:.4f}")
print(f"  - MAE: {mae:.4f}")

print(f"\n✓ Classification Model (Performance Category):")
print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  - Categories: {', '.join(performance_le.classes_)}")

print(f"\n✓ Clustering Model (Student Segmentation):")
print(f"  - K Clusters: {n_clusters}")
print(f"  - Silhouette Score: {silhouette:.4f}")

print(f"\n✓ All models saved to 'models/' folder")
print("\n" + "="*80)
print("NEXT STEP: Run 'streamlit run dashboard_student.py' to launch dashboard")
print("="*80)
