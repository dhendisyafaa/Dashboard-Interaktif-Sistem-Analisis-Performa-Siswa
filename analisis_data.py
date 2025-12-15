"""
Analisis Data dan Model Training - Student Performance Dataset
================================================================

Script ini melakukan:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Training (Regression, Classification, Clustering)
4. Model Evaluation dan Serialization

Author: [Your Name]
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

# Set style untuk visualisasi
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("SISTEM ANALISIS DAN PREDIKSI PERFORMA SISWA")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv('student.csv')

print(f"\n✓ Dataset berhasil dimuat!")
print(f"  - Total baris: {len(df)}")
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

# Distribusi target variable (G3)
print("\n--- Distribusi Nilai G3 (Target) ---")
print(df['G3'].describe())
print(f"  - Min: {df['G3'].min()}")
print(f"  - Max: {df['G3'].max()}")
print(f"  - Mean: {df['G3'].mean():.2f}")
print(f"  - Median: {df['G3'].median():.2f}")
print(f"  - Std Dev: {df['G3'].std():.2f}")

# Korelasi dengan G3
print("\n--- Top 10 Fitur Berkorelasi dengan G3 ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['G3'].sort_values(ascending=False)
print(correlations.head(11))  # 11 karena include G3 itself

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n[3] DATA PREPROCESSING...")

# Buat copy untuk preprocessing
df_processed = df.copy()

# Encode categorical variables
print("  - Encoding categorical variables...")
categorical_cols = df_processed.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

print(f"    ✓ Encoded {len(categorical_cols)} categorical columns")

# Buat kategori performa untuk klasifikasi
print("  - Creating performance categories...")
def categorize_grade(grade):
    if grade >= 16:
        return 'Excellent'  # 16-20
    elif grade >= 12:
        return 'Good'       # 12-15
    elif grade >= 8:
        return 'Average'    # 8-11
    else:
        return 'Poor'       # 0-7

df_processed['performance_category'] = df['G3'].apply(categorize_grade)
performance_le = LabelEncoder()
df_processed['performance_encoded'] = performance_le.fit_transform(df_processed['performance_category'])

print(f"    ✓ Performance categories distribution:")
print(df_processed['performance_category'].value_counts())

# ============================================================================
# 4. PREPARE DATA FOR MODELS
# ============================================================================
print("\n[4] PREPARING DATA FOR MODELS...")

# Features dan target untuk regression (prediksi G3)
# Include G1, G2 for better prediction accuracy
feature_cols = [col for col in df_processed.columns 
                if col not in ['G3', 'performance_category', 'performance_encoded']]

X = df_processed[feature_cols]
y_regression = df_processed['G3']
y_classification = df_processed['performance_encoded']

print(f"  - Features: {len(feature_cols)} columns")
print(f"  - Target (Regression): G3")
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
# 5. MODEL TRAINING - REGRESSION (Prediksi G3)
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

print("\n--- Top 10 Most Important Features (Regression) ---")
print(feature_importance.head(10).to_string(index=False))

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

# Determine optimal K using elbow method (simplified - use K=4)
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

print("\n--- Average G3 per Cluster ---")
for i in range(n_clusters):
    cluster_df = df_clustered[df_clustered['cluster'] == i]
    avg_g3 = cluster_df['G3'].mean()
    count = len(cluster_df)
    print(f"  Cluster {i}: Avg G3 = {avg_g3:.2f} (n={count})")

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
print(f"\n✓ Dataset: {len(df)} students, {len(df.columns)} features")
print(f"\n✓ Regression Model (G3 Prediction):")
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
