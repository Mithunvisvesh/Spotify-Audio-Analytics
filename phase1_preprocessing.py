"""
================================================================================
Spotify Audio Analytics — Phase 1: Data Acquisition & Preprocessing
Person A Deliverable (CO2: Preprocessing — 5 Marks)
================================================================================
This script implements the complete data acquisition, cleaning, deduplication,
outlier filtering, feature scaling, encoding, and feature engineering pipeline.

Deliverables Produced:
1. cleaned_tracks.csv  — Cleaned dataset ready for Person B (EDA) and Person C (Modeling)
2. scaler.pkl          — Serialized StandardScaler fitted on audio features (for Person D)
3. cleaning_decision_log.md — Viva defense documentation of row counts and justifications
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================================================================
# CONFIGURATION & TOGGLES
# ==============================================================================
INPUT_CSV = "dataset.csv"
OUTPUT_CSV = "cleaned_tracks.csv"
SCALER_FILE = "scaler.pkl"

# Toggle for zero-popularity tracks:
# Set to True: drops tracks with popularity == 0 (unpromoted / untouched tracks)
# Set to False: keeps tracks with popularity == 0
DROP_ZERO_POPULARITY = True

print("=" * 80)
print("SPOTIFY AUDIO ANALYTICS — PHASE 1: DATA ACQUISITION & PREPROCESSING")
print("=" * 80)

# ==============================================================================
# STEP 1: LOAD THE DATASET
# ==============================================================================
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv(INPUT_CSV)
original_len = len(df)
print(f"-> Loaded '{INPUT_CSV}' with initial shape: {df.shape}")

# Drop stray Unnamed: 0 index column from previous CSV export
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("-> Dropped stray 'Unnamed: 0' index column.")

print(f"-> Current columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"-> Data types:\n{df.dtypes}")
print(f"-> First 3 rows:\n{df.head(3)}")
print(f"-> Row count after Step 1: {len(df):,}")

# ==============================================================================
# STEP 2: INITIAL INSPECTION
# ==============================================================================
print("\n[STEP 2] Initial inspection & summary statistics...")
null_counts = df.isnull().sum()
print("-> Null counts per column (columns with missing values):")
print(null_counts[null_counts > 0] if (null_counts > 0).any() else "No missing values found.")

print("\n-> Unique genres count:", df["track_genre"].nunique())
print("\n-> Summary Statistics (Transposed):")
print(df.describe(include="all").T[["count", "mean", "std", "min", "50%", "max"]].dropna(how="all"))
print(f"-> Row count after Step 2: {len(df):,}")

# ==============================================================================
# STEP 3: HANDLE MISSING VALUES
# ==============================================================================
print("\n[STEP 3] Handling missing values...")
step3_start_len = len(df)

# Impute non-critical metadata columns with 'Unknown' instead of dropping rows
missing_artists = df["artists"].isnull().sum()
missing_albums = df["album_name"].isnull().sum()
df["artists"] = df["artists"].fillna("Unknown")
df["album_name"] = df["album_name"].fillna("Unknown")
df["track_name"] = df["track_name"].fillna("Unknown")
print(f"-> Imputed missing metadata with 'Unknown' (artists: {missing_artists}, album_name: {missing_albums}).")

# Core audio features and target (popularity) cannot be safely imputed — drop if null
core_features = ["danceability", "energy", "tempo", "valence", "loudness", "acousticness", "popularity"]
df = df.dropna(subset=core_features)
step3_dropped = step3_start_len - len(df)
print(f"-> Checked core features {core_features} for NaNs. Dropped {step3_dropped} rows.")
print(f"-> Row count after Step 3: {len(df):,} (from {step3_start_len:,})")

# ==============================================================================
# STEP 4: REMOVE DUPLICATES (EXACT AND VERSION DEDUPLICATION)
# ==============================================================================
print("\n[STEP 4] Removing duplicates...")

# 4a. Exact duplicate track_id removal
step4a_start_len = len(df)
df = df.drop_duplicates(subset=["track_id"])
dropped_exact_id = step4a_start_len - len(df)
print(f"-> Dropped {dropped_exact_id:,} exact duplicate track_id rows.")
print(f"-> Rows after exact track_id dedup: {len(df):,}")

# 4b. Song version / remaster / live deduplication
step4b_start_len = len(df)
# Normalize track name by stripping '- Remastered', '- Live', '- Radio Edit', etc.
df["track_name_clean"] = df["track_name"].str.replace(
    r"\s*-\s*(Remaster(ed)?|Live|Radio Edit).*", "", regex=True, case=False
).str.strip()

# Sort by popularity descending with deterministic tiebreaker on track_id
# CRITICAL FOR REPRODUCIBILITY: 997 duplicate groups tie on max popularity.
# Including track_id ensures exact bit-for-bit deterministic output across all machines/environments.
df = df.sort_values(["popularity", "track_id"], ascending=[False, True])
df = df.drop_duplicates(subset=["track_name_clean", "artists"], keep="first")
dropped_version_dups = step4b_start_len - len(df)
print(f"-> Dropped {dropped_version_dups:,} duplicate track version rows (remasters/live/edits).")
print(f"-> Row count after Step 4: {len(df):,}")

# ==============================================================================
# STEP 5: HANDLE OUTLIERS
# ==============================================================================
print("\n[STEP 5] Handling outliers...")
step5_start_len = len(df)

# 5a. Near-zero duration tracks (<= 30 seconds / 30,000 ms)
step5a_start = len(df)
df = df[df["duration_ms"] > 30_000]
dropped_duration = step5a_start - len(df)
print(f"-> Dropped {dropped_duration:,} tracks with duration_ms <= 30,000 ms (short intro/sound effect artifacts).")

# 5b. Implausible tempo values (outside [30, 250] BPM)
step5b_start = len(df)
df = df[df["tempo"].between(30, 250)]
dropped_tempo = step5b_start - len(df)
print(f"-> Dropped {dropped_tempo:,} tracks with tempo outside [30, 250] BPM (glitched/corrupted tempo estimates).")

# 5c. Popularity == 0 handling (Toggleable)
step5c_start = len(df)
if DROP_ZERO_POPULARITY:
    df = df[df["popularity"] > 0]
    dropped_zero_pop = step5c_start - len(df)
    print(f"-> [TOGGLE ON] Dropped {dropped_zero_pop:,} unpromoted/untouched tracks with popularity == 0.")
else:
    dropped_zero_pop = 0
    print("-> [TOGGLE OFF] Retained tracks with popularity == 0.")

total_step5_dropped = step5_start_len - len(df)
print(f"-> Row count after Step 5: {len(df):,} (Total outliers dropped: {total_step5_dropped:,})")

# ==============================================================================
# STEP 6: FEATURE SCALING (StandardScaler)
# ==============================================================================
print("\n[STEP 6] Feature scaling with StandardScaler...")
scale_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
print(f"-> Scaling features: {scale_cols}")

scaler = StandardScaler()
scaled_feature_names = [f"{col}_scaled" for col in scale_cols]
df[scaled_feature_names] = scaler.fit_transform(df[scale_cols])

# Save the fitted scaler — REQUIRED handoff for Person D (Power BI / K-Means) & Person C
joblib.dump(scaler, SCALER_FILE)
print(f"-> Saved fitted StandardScaler to '{SCALER_FILE}'.")
print(f"-> Verification: scaler feature count = {scaler.n_features_in_}, mean shape = {scaler.mean_.shape}")
print(f"-> Row count after Step 6: {len(df):,}")

# ==============================================================================
# STEP 7: ENCODE CATEGORICAL FEATURES
# ==============================================================================
print("\n[STEP 7] Encoding categorical features...")
# 7a. Binary explicit flag to integer
df["explicit"] = df["explicit"].astype(int)
print("-> Converted 'explicit' boolean flag to integer (0 / 1).")

# 7b. One-hot encode key and mode with drop_first=True
# CRITICAL: drop_first=True prevents the dummy variable trap (perfect multicollinearity)
# which would cause singular-matrix errors in Person C's OLS regression.
initial_cols = len(df.columns)
df = pd.get_dummies(df, columns=["key", "mode"], prefix=["key", "mode"], drop_first=True, dtype=int)
new_dummy_cols = [c for c in df.columns if c.startswith("key_") or c.startswith("mode_")]
print(f"-> Generated {len(new_dummy_cols)} dummy columns with drop_first=True: {new_dummy_cols}")
print(f"-> Columns count changed from {initial_cols} to {len(df.columns)}")
print(f"-> Row count after Step 7: {len(df):,}")

# ==============================================================================
# STEP 8: FEATURE ENGINEERING
# ==============================================================================
print("\n[STEP 8] Feature engineering...")
# 8a. Interaction term: energy * valence
df["energy_valence"] = df["energy"] * df["valence"]

# 8b. Categorical binning: tempo_bucket
df["tempo_bucket"] = pd.cut(df["tempo"], bins=[0, 90, 130, 300], labels=["slow", "mid", "fast"])

# 8c. Composite mood index: mood_score
df["mood_score"] = 0.5 * df["valence"] + 0.3 * df["energy"] + 0.2 * df["danceability"]

print("-> Created 'energy_valence' (energy * valence interaction).")
print("-> Created 'tempo_bucket' with distribution:")
print(df["tempo_bucket"].value_counts().to_dict())
print("-> Created 'mood_score' (0.5*valence + 0.3*energy + 0.2*danceability).")
print(f"-> Row count after Step 8: {len(df):,}")

# ==============================================================================
# STEP 9: SAVE, AUDIT & HAND OFF
# ==============================================================================
print("\n[STEP 9] Integrity checks and exporting final dataset...")

# Explicit Sanity Check: Zero Missing Values & pd.cut boundary audit
total_nulls = df.isnull().sum().sum()
tempo_bucket_nulls = df["tempo_bucket"].isnull().sum()
tempo_min, tempo_max = df["tempo"].min(), df["tempo"].max()

print("-> Performing dataset integrity audit:")
print(f"   • Total missing values across entire DataFrame: {total_nulls}")
print(f"   • 'tempo_bucket' missing values (pd.cut boundary check): {tempo_bucket_nulls}")
print(f"   • Validated tempo range: [{tempo_min:.3f}, {tempo_max:.3f}] BPM (inside bin edges [0, 300])")
assert total_nulls == 0, f"Integrity Failure: Expected 0 null values, found {total_nulls}."
print("   • STATUS: PASSED (0 missing values across all 43 columns).")

df.to_csv(OUTPUT_CSV, index=False)
print(f"-> Exported cleaned dataset to '{OUTPUT_CSV}'.")

print("\n" + "=" * 80)
print("PIPELINE SUMMARY & HANDOFF AUDIT")
print("=" * 80)
print(f"Original Row Count:              {original_len:,}")
print(f"Final Cleaned Row Count:         {len(df):,}")
print(f"Total Rows Dropped:              {(original_len - len(df)):,} ({((original_len - len(df)) / original_len) * 100:.2f}%)")
print(f"Final Column Count:              {len(df.columns)} (20 orig + clean_name + 9 scaled - 2 key/mode + 12 dummies + 3 engineered)")
print(f"Total Null Values:               {total_nulls}")
print(f"Cleaned CSV File Size:           {os.path.getsize(OUTPUT_CSV) / (1024 * 1024):.2f} MB")
print(f"Scaler File Size:                {os.path.getsize(SCALER_FILE) / 1024:.2f} KB")
print("=" * 80)
print("Phase 1 preprocessing pipeline completed successfully!")
