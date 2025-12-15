"""
Generate Dataset Mahasiswa dengan Korelasi yang Benar
======================================================

Script untuk regenerate dataset mahasiswa dengan korelasi positif
yang masuk akal antara aktivitas akademik dan IPK.

Author: Fix untuk Dashboard IPK
Date: December 2025
"""

import pandas as pd
import numpy as np

np.random.seed(42)

print("="*70)
print("GENERATING NEW DATASET")
print("="*70)

# Jumlah mahasiswa
n_students = 500

# Generate data
print("\n[1] Generating student data...")

# Gender dan status
jenis_kelamin = np.random.choice(['Laki-laki', 'Perempuan'], n_students)
umur = np.random.randint(18, 26, n_students)
status_menikah = np.random.choice(['Belum Menikah', 'Menikah'], n_students, p=[0.85, 0.15])

# Generate aktivitas akademik dengan distribusi normal
# Nilai rata-rata sekitar 70, std 15
kehadiran = np.random.normal(70, 15, n_students)
partisipasi = np.random.normal(70, 15, n_students)
tugas = np.random.normal(70, 15, n_students)
elearning = np.random.normal(70, 15, n_students)

# Clip ke range 40-100
kehadiran = np.clip(kehadiran, 40, 100).astype(int)
partisipasi = np.clip(partisipasi, 40, 100).astype(int)
tugas = np.clip(tugas, 40, 100).astype(int)
elearning = np.clip(elearning, 40, 100).astype(int)

# Generate IPK dengan KORELASI POSITIF terhadap aktivitas
# Formula: IPK = base + weighted_average(aktivitas) + noise
# Weight untuk setiap fitur (total = 1.0)
w_kehadiran = 0.25
w_partisipasi = 0.15
w_tugas = 0.40  # Tugas paling berpengaruh
w_elearning = 0.20

# Normalize aktivitas ke scale 0-1
kehadiran_norm = (kehadiran - 40) / 60  # 40-100 -> 0-1
partisipasi_norm = (partisipasi - 40) / 60
tugas_norm = (tugas - 40) / 60
elearning_norm = (elearning - 40) / 60

# Calculate weighted average
weighted_avg = (
    w_kehadiran * kehadiran_norm +
    w_partisipasi * partisipasi_norm +
    w_tugas * tugas_norm +
    w_elearning * elearning_norm
)

# Convert to IPK scale (1.5 - 4.0)
# weighted_avg adalah 0-1, kita mapping ke 1.5-4.0
base_ipk = 1.5
range_ipk = 2.5  # 4.0 - 1.5 = 2.5

ipk = base_ipk + (weighted_avg * range_ipk)

# Add small random noise
noise = np.random.normal(0, 0.15, n_students)
ipk = ipk + noise

# Clip to valid range
ipk = np.clip(ipk, 1.5, 4.0)
ipk = np.round(ipk, 2)

# Status akademik
status_akademik = ['Lulus' if i >= 2.0 else 'Tidak Lulus' for i in ipk]

# Generate nama
first_names = ['Andi', 'Budi', 'Cindy', 'Dian', 'Eka', 'Fani', 'Gita', 'Hadi',
               'Indah', 'Joko', 'Kiki', 'Lina', 'Mira', 'Nuri', 'Omar', 'Putri',
               'Qori', 'Rio', 'Sari', 'Tina', 'Umar', 'Vina', 'Wati', 'Yuni', 'Zahra']

last_names = ['Wijaya', 'Kusuma', 'Pratama', 'Santoso', 'Hidayat', 'Rahman',
              'Sari', 'Putra', 'Dewi', 'Ananda', 'Fauzi', 'Hakim', 'Septiani',
              'Permana', 'Cahaya', 'Nugroho', 'Wardana', 'Utama']

names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" 
         for _ in range(n_students)]

# Create DataFrame
df = pd.DataFrame({
    'Nama': names,
    'Jenis_Kelamin': jenis_kelamin,
    'Umur': umur,
    'Status_Menikah': status_menikah,
    'Kehadiran_Persen': kehadiran,
    'Partisipasi_Diskusi': partisipasi,
    'Nilai_Tugas': tugas,
    'Aktivitas_ELearning': elearning,
    'IPK': ipk,
    'Status_Akademik': status_akademik
})

print(f"  ✓ Generated {len(df)} students")

# Verify correlations
print("\n[2] Verifying correlations...")
print("\nCorrelation with IPK:")
correlations = df[['Kehadiran_Persen', 'Partisipasi_Diskusi', 'Nilai_Tugas', 
                    'Aktivitas_ELearning', 'IPK']].corr()['IPK']
print(correlations)

# Statistics
print("\n[3] Dataset statistics...")
print("\nIPK Distribution:")
print(df['IPK'].describe())

print("\nFeature ranges:")
print(f"  Kehadiran: {df['Kehadiran_Persen'].min()}-{df['Kehadiran_Persen'].max()}")
print(f"  Partisipasi: {df['Partisipasi_Diskusi'].min()}-{df['Partisipasi_Diskusi'].max()}")
print(f"  Tugas: {df['Nilai_Tugas'].min()}-{df['Nilai_Tugas'].max()}")
print(f"  E-Learning: {df['Aktivitas_ELearning'].min()}-{df['Aktivitas_ELearning'].max()}")

# Category distribution
print("\nPerformance distribution:")
def categorize(ipk):
    if ipk >= 3.5:
        return 'Excellent (≥3.5)'
    elif ipk >= 3.0:
        return 'Good (≥3.0)'
    elif ipk >= 2.5:
        return 'Average (≥2.5)'
    else:
        return 'Poor (<2.5)'

df['category'] = df['IPK'].apply(categorize)
print(df['category'].value_counts())

# Save
print("\n[4] Saving dataset...")
df_save = df.drop('category', axis=1)
df_save.to_csv('mahasiswa.csv', index=False)
print("  ✓ Saved to mahasiswa.csv")

# Sample data
print("\n[5] Sample data:")
print(df_save.head(10))

print("\n" + "="*70)
print("DATASET GENERATION COMPLETE")
print("="*70)
print("\nNEXT STEP: Run 'python analisis_data.py' to retrain models")
print("="*70)
