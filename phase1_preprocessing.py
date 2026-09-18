"""
================================================================================
Spotify Audio Analytics — Data Acquisition & Preprocessing Pipeline
Course Outcome Alignment: CO2 (Data Ingestion, Cleaning & Feature Engineering)
================================================================================
This module implements the end-to-end data preprocessing pipeline for the Spotify
audio tracks dataset. It performs data sanitization, metadata imputation,
two-tier deterministic deduplication, acoustic outlier filtering, standard
feature scaling, multicollinearity-safe categorical encoding, and domain feature
engineering.

Deliverables:
1. cleaned_tracks.csv      — Sanitized dataset (76,103 rows × 43 columns)
2. scaler.pkl              — Serialized StandardScaler fitted on audio metrics
3. cleaning_decision_log.md — Technical decision log and methodological justifications
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
INPUT_CSV = "dataset.csv"
OUTPUT_CSV = "cleaned_tracks.csv"
SCALER_FILE = "scaler.pkl"

# Cold-start exposure filter:
# Set to True: filters unpromoted tracks with popularity == 0 to eliminate exposure noise
# Set to False: retains unrated/zero-popularity tracks in the empirical distribution
DROP_ZERO_POPULARITY = True

print("=" * 80)
print("SPOTIFY AUDIO ANALYTICS — DATA ACQUISITION & PREPROCESSING PIPELINE")
print("=" * 80)

# ==============================================================================
# STEP 1: LOAD THE DATASET
# ==============================================================================
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv(INPUT_CSV)
original_len = len(df)
print(f"-> Loaded '{INPUT_CSV}' with initial shape: {df.shape}")

# Drop redundant index column from CSV serialization
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("-> Dropped redundant index column 'Unnamed: 0'.")

print(f"-> Feature columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"-> Initial row count: {len(df):,}")

# ==============================================================================
# STEP 2: INITIAL INSPECTION & PROFILING
# ==============================================================================
print("\n[STEP 2] Inspecting missing values and descriptive statistics...")
null_counts = df.isnull().sum()
missing_summary = null_counts[null_counts > 0]
if not missing_summary.empty:
    print("-> Missing values detected:")
    for col, count in missing_summary.items():
        print(f"   • {col}: {count} missing value(s)")
else:
    print("-> No missing values detected in raw data.")

print(f"-> Unique musical genres: {df['track_genre'].nunique()}")
print(f"-> Current row count: {len(df):,}")

# ==============================================================================
# STEP 3: HANDLE MISSING VALUES
# ==============================================================================
print("\n[STEP 3] Handling missing values...")
step3_start_len = len(df)

# Impute non-critical descriptive metadata with 'Unknown' to preserve acoustic observations
missing_artists = df["artists"].isnull().sum()
missing_albums = df["album_name"].isnull().sum()
missing_tracks = df["track_name"].isnull().sum()
df["artists"] = df["artists"].fillna("Unknown")
df["album_name"] = df["album_name"].fillna("Unknown")
df["track_name"] = df["track_name"].fillna("Unknown")
print(f"-> Imputed missing metadata with 'Unknown' (artists: {missing_artists}, album: {missing_albums}, track: {missing_tracks}).")

# Core audio features and target (popularity) cannot be synthetically imputed without bias
core_features = ["danceability", "energy", "tempo", "valence", "loudness", "acousticness", "popularity"]
df = df.dropna(subset=core_features)
step3_dropped = step3_start_len - len(df)
print(f"-> Audited core audio features {core_features}. Dropped {step3_dropped} rows.")
print(f"-> Row count after Step 3: {len(df):,}")

# ==============================================================================
# STEP 4: TWO-TIER DETERMINISTIC DEDUPLICATION
# ==============================================================================
print("\n[STEP 4] Executing two-tier deduplication...")

# 4a. Exact Spotify track URI deduplication
step4a_start = len(df)
df = df.drop_duplicates(subset=["track_id"])
dropped_exact_id = step4a_start - len(df)
print(f"-> Tier 1: Dropped {dropped_exact_id:,} duplicate track_id rows across playlists.")
print(f"-> Rows after exact deduplication: {len(df):,}")

# 4b. Composition and release normalization (Remasters, Live cuts, Radio Edits)
step4b_start = len(df)
df["track_name_clean"] = df["track_name"].str.replace(
    r"\s*-\s*(Remaster(ed)?|Live|Radio Edit).*", "", regex=True, case=False
).str.strip()

# Sort by popularity descending with deterministic tiebreaker on track_id
# Resolves ties among 997 duplicate groups to ensure exact reproducibility across machines
df = df.sort_values(["popularity", "track_id"], ascending=[False, True])
df = df.drop_duplicates(subset=["track_name_clean", "artists"], keep="first")
dropped_version_dups = step4b_start - len(df)
print(f"-> Tier 2: Dropped {dropped_version_dups:,} duplicate release versions (retaining highest popularity).")
print(f"-> Row count after Step 4: {len(df):,}")

# ==============================================================================
# STEP 5: DOMAIN OUTLIER FILTERING
# ==============================================================================
print("\n[STEP 5] Filtering domain outliers...")
step5_start = len(df)

# 5a. Non-musical / truncated audio fragments (duration <= 30 seconds)
step5a_start = len(df)
df = df[df["duration_ms"] > 30_000]
dropped_duration = step5a_start - len(df)
print(f"-> Dropped {dropped_duration:,} tracks with duration <= 30,000 ms (sound effects / intro artifacts).")

# 5b. Algorithmic tempo extraction errors (outside plausible range 30–250 BPM)
step5b_start = len(df)
df = df[df["tempo"].between(30, 250)]
dropped_tempo = step5b_start - len(df)
print(f"-> Dropped {dropped_tempo:,} tracks with tempo outside [30, 250] BPM.")

# 5c. Unpromoted tracks with popularity == 0 (Cold-start / exposure bias)
step5c_start = len(df)
if DROP_ZERO_POPULARITY:
    df = df[df["popularity"] > 0]
    dropped_zero_pop = step5c_start - len(df)
    print(f"-> [Active Filter] Dropped {dropped_zero_pop:,} unpromoted tracks with popularity == 0.")
else:
    dropped_zero_pop = 0
    print("-> [Inactive Filter] Retained tracks with popularity == 0.")

total_step5_dropped = step5_start - len(df)
print(f"-> Row count after Step 5: {len(df):,} (Total outliers filtered: {total_step5_dropped:,})")

# ==============================================================================
# STEP 6: FEATURE STANDARDIZATION (StandardScaler)
# ==============================================================================
print("\n[STEP 6] Standardizing continuous audio features...")
scale_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
print(f"-> Scaling continuous features to zero mean and unit variance: {scale_cols}")

scaler = StandardScaler()
scaled_feature_names = [f"{col}_scaled" for col in scale_cols]
df[scaled_feature_names] = scaler.fit_transform(df[scale_cols])

# Serialize fitted scaler to prevent data leakage in downstream clustering and inference
joblib.dump(scaler, SCALER_FILE)
print(f"-> Serialized fitted StandardScaler to '{SCALER_FILE}'.")
print(f"-> Verified scaler parameters: n_features = {scaler.n_features_in_}, mean shape = {scaler.mean_.shape}")
print(f"-> Row count after Step 6: {len(df):,}")

# ==============================================================================
# STEP 7: CATEGORICAL ENCODING & MULTICOLLINEARITY PREVENTION
# ==============================================================================
print("\n[STEP 7] Encoding categorical attributes...")
# 7a. Binary explicit flag conversion
df["explicit"] = df["explicit"].astype(int)
print("-> Converted 'explicit' boolean indicator to integer binary (0 / 1).")

# 7b. One-hot encode key and mode with drop_first=True
# Note: drop_first=True prevents the dummy variable trap (perfect multicollinearity),
# ensuring the design matrix X is full rank for Ordinary Least Squares (OLS) regression.
cols_before = len(df.columns)
df = pd.get_dummies(df, columns=["key", "mode"], prefix=["key", "mode"], drop_first=True, dtype=int)
dummy_cols = [c for c in df.columns if c.startswith("key_") or c.startswith("mode_")]
print(f"-> Generated {len(dummy_cols)} dummy variables with drop_first=True: {dummy_cols}")
print(f"-> Column count expanded from {cols_before} to {len(df.columns)}.")
print(f"-> Row count after Step 7: {len(df):,}")

# ==============================================================================
# STEP 8: FEATURE ENGINEERING
# ==============================================================================
print("\n[STEP 8] Constructing engineered features...")
# 8a. Valence-energy interaction term
df["energy_valence"] = df["energy"] * df["valence"]

# 8b. Discretized tempo categorization
df["tempo_bucket"] = pd.cut(df["tempo"], bins=[0, 90, 130, 300], labels=["slow", "mid", "fast"])

# 8c. Composite valence-energy-danceability mood score
df["mood_score"] = 0.5 * df["valence"] + 0.3 * df["energy"] + 0.2 * df["danceability"]

print("-> Created 'energy_valence' interaction feature.")
print(f"-> Created 'tempo_bucket' distribution: {df['tempo_bucket'].value_counts().to_dict()}")
print("-> Created 'mood_score' composite index.")
print(f"-> Row count after Step 8: {len(df):,}")

# ==============================================================================
# STEP 9: INTEGRITY VERIFICATION, EXPORT & AUDIT
# ==============================================================================
print("\n[STEP 9] Verifying dataset integrity and exporting deliverables...")

# Formal Integrity Checks
total_nulls = df.isnull().sum().sum()
tempo_bucket_nulls = df["tempo_bucket"].isnull().sum()
tempo_min, tempo_max = df["tempo"].min(), df["tempo"].max()

print("-> Executing Dataset Integrity Verification:")
print(f"   • Total missing values across all cells: {total_nulls}")
print(f"   • 'tempo_bucket' missing values (pd.cut boundary safety): {tempo_bucket_nulls}")
print(f"   • Validated empirical tempo range: [{tempo_min:.3f}, {tempo_max:.3f}] BPM (within boundary [0, 300])")
assert total_nulls == 0, f"Integrity Failure: Expected 0 null values, found {total_nulls}."
print("   • Status: PASSED (0 missing values across all 43 columns).")

df.to_csv(OUTPUT_CSV, index=False)
print(f"-> Successfully exported sanitized dataset to '{OUTPUT_CSV}'.")

print("\n" + "=" * 80)
print("PREPROCESSING PIPELINE SUMMARY & AUDIT")
print("=" * 80)
print(f"Original Row Count:        {original_len:,}")
print(f"Final Cleaned Row Count:   {len(df):,}")
print(f"Total Observations Filtered: {(original_len - len(df)):,} ({((original_len - len(df)) / original_len) * 100:.2f}%)")
print(f"Final Column Count:        {len(df.columns)} (20 raw + clean_name + 9 scaled - 2 key/mode + 12 dummies + 3 engineered)")
print(f"Total Null Values:         {total_nulls}")
print(f"Cleaned CSV File Size:     {os.path.getsize(OUTPUT_CSV) / (1024 * 1024):.2f} MB")
print(f"Scaler Artifact Size:      {os.path.getsize(SCALER_FILE) / 1024:.2f} KB")
print("=" * 80)
print("Phase 1 preprocessing pipeline completed successfully.")
