# Spotify Audio Analytics — Cleaning Decision Log
**Phase 1 Deliverable: Data Acquisition & Preprocessing (Person A)**  
**Target Course Outcome:** CO2 (Preprocessing) — 5 Marks  
**Downstream Dependents:** Person B (EDA), Person C (Regression Modeling), Person D (Power BI & K-Means Clustering)

---

## 1. Executive Summary & Row Progression

| Stage / Step | Starting Rows | Rows Dropped / Affected | Resulting Rows | % Dataset Retained | Key Action / Transformation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0. Raw Dataset** (`dataset.csv`) | 114,000 | 0 | 114,000 | 100.00% | Kaggle 114k Spotify tracks dataset. Dropped stray `Unnamed: 0` CSV index. |
| **Step 3: Missing Value Handling** | 114,000 | 0 dropped<br>(2 imputed) | 114,000 | 100.00% | Imputed 1 missing `artists` & 1 missing `album_name` with `"Unknown"`. Verified 0 NaNs in core audio features. |
| **Step 4a: Exact `track_id` Deduplication** | 114,000 | **24,259** | 89,741 | 78.72% | Dropped identical Spotify track URI duplicates appearing across multiple genre playlists. |
| **Step 4b: Song Version / Remaster Deduplication** | 89,741 | **8,708** | 81,033 | 71.08% | Stripped `- Remastered`, `- Live`, `- Radio Edit` suffixes; sorted by popularity desc; kept most popular version per artist. |
| **Step 5a: Duration Outliers** | 81,033 | **15** | 81,018 | 71.07% | Dropped tracks with `duration_ms <= 30,000` (short intros, sound effects, corrupted audio fragments). |
| **Step 5b: Tempo Outliers** | 81,018 | **143** | 80,875 | 70.94% | Dropped tracks with `tempo < 30` or `tempo > 250` BPM (tempo extraction failure / implausible cadence). |
| **Step 5c: Zero-Popularity Filter** *(Toggleable)* | 80,875 | **4,772** | **76,103** | **66.76%** | Dropped untouched/unpromoted tracks with `popularity == 0` (sampling noise vs listener preference). |

**Net Result:** Cleaned dataset contains **76,103 rows and 43 columns** (37,897 rows dropped in total, 33.24% of original dataset).

---

## 2. Step-by-Step Cleaning Decisions & Technical Rationale

### Step 1: Ingestion and Index Sanitization
- **What was done:** Loaded `dataset.csv` (114,000 rows × 21 columns) and immediately dropped the `Unnamed: 0` column.
- **Rationale:** `Unnamed: 0` is an artifact of saving a pandas DataFrame to CSV with `index=True`. Leaving it in adds redundant noise, consumes unnecessary memory, and creates potential indexing bugs downstream.

### Step 2 & 3: Missing Value Strategy (Imputation vs. Dropping)
- **What was done:**
  - Imputed missing values in `artists` (1 row), `album_name` (1 row), and `track_name` (1 row) with the literal string `"Unknown"`.
  - Audited core audio features (`danceability`, `energy`, `tempo`, `valence`, `loudness`, `acousticness`) and the target variable (`popularity`). Confirmed **0 missing values** existed in these columns.
- **Rationale:** Metadata fields (artist/album name) are non-critical descriptors; dropping an entire song row for an absent album name throws away legitimate acoustic measurement data. Conversely, core audio features and popularity cannot be synthetically imputed without introducing severe bias into Person C's regression models.

### Step 4: Two-Tier Deduplication Strategy
- **Tier 1 (Exact `track_id`):**
  - **Count dropped:** 24,259 rows (21.28% of the raw data).
  - **Rationale:** Spotify's catalog categorizes identical songs under multiple `track_genre` labels (e.g., the same track listed under both `pop` and `dance`). Keeping duplicate track IDs artificially inflates sample size, violates the i.i.d. (independent and identically distributed) assumption of regression models, and leads to data leakage during train/test splits.
- **Tier 2 (Fuzzy Title / Remaster Normalization with Deterministic Tie-Breaker):**
  - **Count dropped:** 8,708 rows.
  - **Rationale & Reproducibility Fix:** Artists frequently release standard, deluxe, remastered (e.g., "- Remastered 2011"), live, and radio edit versions of the exact same song. These versions share near-identical acoustic properties but carry fragmented popularity scores. By stripping regex suffixes (`r"\s*-\s*(Remaster(ed)?|Live|Radio Edit).*"`), sorting by `["popularity", "track_id"]` (descending popularity, ascending `track_id`), and retaining the first occurrence per `[track_name_clean, artists]`, we preserve the primary, most representative version of each distinct musical composition.
  - **Why the `track_id` tiebreaker is critical:** Exactly **997 duplicate groups** share an identical maximum popularity score. Without sorting on `track_id` as a secondary key, pandas' default quicksort does not guarantee sort stability, causing slight variations in which tied version survives across different OS/Python/pandas versions. Adding `track_id` ensures 100% bit-for-bit deterministic reproducibility everywhere.

### Step 5: Domain-Specific Outlier Filtering
- **Duration (`duration_ms > 30,000`):**
  - **Count dropped:** 15 rows.
  - **Rationale:** Any recording under 30 seconds is not a standard commercial song (e.g., silence calibration tracks, sound effect samples, or truncated album interludes).
- **Tempo (`tempo.between(30, 250)`):**
  - **Count dropped:** 143 rows.
  - **Rationale:** Human musical tempo typically falls between 40 and 220 BPM. Tempos below 30 BPM or above 250 BPM indicate Spotify acoustic algorithm failures (e.g., spoken word, octave-doubling errors, ambient noise).
- **Popularity Zero Handling (`popularity > 0`):**
  - **Count dropped:** 4,772 rows.
  - **Rationale:** A popularity of 0 in the Spotify Web API represents an unindexed or newly uploaded track that was never surfaced by the recommendation algorithm to register user impressions (cold-start / exposure bias). Including these tracks confuses Person C's regression model, teaching it to correlate acoustic features with zero visibility rather than consumer preference.
  - **Configurability:** Controlled via `DROP_ZERO_POPULARITY = True` at the top of `phase1_preprocessing.py` so Person C can toggle it to `False` if experimenting with unrated distributions.

### Step 6: Feature Scaling & Model Artifact Serialization
- **What was done:** Standardized 9 continuous numeric features (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`) using `StandardScaler` (zero mean, unit variance). Serialized the fitted scaler to `scaler.pkl` with `joblib`.
- **Rationale:**
  - Loudness (dB, typically -60 to 0) and duration/tempo operate on radically larger scales than bounded audio features (0.0 to 1.0). Unscaled features distort Euclidean distance calculations in Person D's K-Means clustering.
  - **Critical Handoff:** Saving `scaler.pkl` ensures Person D transforms new cluster data using the *exact training parameters* ($\mu, \sigma$) rather than committing data leakage or schema mismatch by refitting a new scaler.

### Step 7: Categorical Encoding & Multicollinearity Prevention
- **What was done:**
  - Cast `explicit` boolean to integer `(0 / 1)`.
  - One-hot encoded `key` (12 musical pitches, 0–11) and `mode` (Major=1, Minor=0) using `pd.get_dummies(..., drop_first=True)`.
- **Rationale:**
  - **The Dummy Variable Trap:** Musical key has 12 discrete states. If all 12 binary dummy columns were included alongside an intercept term, $\sum_{i=0}^{11} \text{key}_i = 1$, introducing perfect linear dependency (multicollinearity).
  - Person C's downstream Ordinary Least Squares (`statsmodels.api.OLS`) requires an invertible covariance matrix $(X^T X)^{-1}$. Without `drop_first=True`, the matrix is singular, immediately crashing the regression with a `LinAlgError / Singular Matrix` failure. `key_0` and `mode_0` serve as the statistical reference baselines.

### Step 8: Domain-Specific Feature Engineering
- **`energy_valence` ($= \text{energy} \times \text{valence}$):** An interaction term capturing intense positive emotionality (high energy + happy mood) vs calm or melancholic profiles.
- **`tempo_bucket` (`slow`: 0–90, `mid`: 90–130, `fast`: 130–300 BPM):** Categorical discretization enabling intuitive dashboard slicers for Person D's Power BI report.
- **`mood_score` ($= 0.5 \times \text{valence} + 0.3 \times \text{energy} + 0.2 \times \text{danceability}$):** A composite positive valence index weighting valence primary, rhythm secondary, and danceability tertiary.

---

## 3. Dataset Integrity & Schema Audit

### A. Mathematical Column Count Verification (43 Columns)
Every column in `cleaned_tracks.csv` is accounted for:
- **20** Original raw columns (`track_id`, `artists`, `album_name`, `track_name`, `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `track_genre`, plus `key` & `mode` before encoding, minus dropped `Unnamed: 0`)
- **+ 1** `track_name_clean` (regex cleaned title for deduplication)
- **+ 9** Scaled audio features (`danceability_scaled`, `energy_scaled`, `loudness_scaled`, `speechiness_scaled`, `acousticness_scaled`, `instrumentalness_scaled`, `liveness_scaled`, `valence_scaled`, `tempo_scaled`)
- **- 2** Raw `key` and `mode` columns (dropped by `pd.get_dummies`)
- **+ 12** Dummy indicator columns (11 for `key_1` through `key_11`; 1 for `mode_1` — `key_0` and `mode_0` omitted via `drop_first=True`)
- **+ 3** Domain-engineered features (`energy_valence`, `tempo_bucket`, `mood_score`)
- **Formula:** $20 + 1 + 9 - 2 + 12 + 3 = \mathbf{43\text{ columns}}$.

### B. Zero Missing Values & `pd.cut` Boundary Verification
- **Total Missing Values Across All Cells:** **`0`** (`df.isnull().sum().sum() == 0`).
- **`pd.cut` Boundary Safety Audit:** `pd.cut` silently assigns `NaN` if any numeric value falls outside the predefined bin edges (`[0, 90, 130, 300]`).
  - Empirical tempo range in cleaned dataset: **Min = 30.322 BPM, Max = 243.372 BPM**.
  - Because Step 5b strictly filters tempos outside `[30, 250]`, 100% of rows fall well within the bin edges $(0, 300]$.
  - Result: `df['tempo_bucket'].isnull().sum() == 0` (0 nulls produced; **11,050 slow, 36,444 mid, 28,609 fast**).

---

## 4. Viva Defense Q&A (Preparation for Examiners)

### Q1: "Why did you drop nearly 38,000 rows (33%) from the original dataset?"
> **Answer:** *"The raw Kaggle dataset contained 114,000 records, but over 24,000 were exact duplicate track IDs resulting from tracks appearing across multiple genre playlists. Another 8,700 were duplicate releases of the exact same composition (such as remaster and live re-issues). Retaining duplicate tracks would artificially skew statistical tests and cause train-test data leakage. Furthermore, removing 4,772 zero-popularity tracks and 158 extreme duration/tempo anomalies ensured that our downstream regression models learn true acoustic determinants of popularity rather than audio artifacts or cold-start exposure noise."*

### Q2: "Why did you use `drop_first=True` when one-hot encoding key and mode?"
> **Answer:** *"In linear regression, including all $k$ levels of a categorical variable alongside a constant intercept term creates exact linear dependency, known as the dummy variable trap. For key (12 pitches) and mode (2 states), the sum of each group's dummies equals 1. By setting `drop_first=True`, we omit the first level as the reference baseline, guaranteeing that the feature matrix $X^TX$ remains full rank and strictly invertible for Person C's OLS regression."*

### Q3: "Why did Person A save `scaler.pkl` instead of letting Person D fit their own scaler in Phase 4?"
> **Answer:** *"Fitting a separate scaler in Phase 4 would violate sound data science protocol. Preprocessing parameters (the exact mean and standard deviation of each audio feature) must be computed once during the data preparation phase and applied consistently across all downstream consumers. Persisting `scaler.pkl` guarantees that Person D's K-Means clustering and any future production inference evaluate features on the exact same scale."*

### Q4: "Why did you impute missing album and artist names with 'Unknown' instead of dropping those rows?"
> **Answer:** *"Only 1 track lacked artist metadata and 1 lacked album metadata, while their core acoustic measurements (danceability, tempo, energy) were 100% complete and valid. Dropping an entire audio observation for missing descriptive text discards valuable empirical signal. Imputing 'Unknown' preserves the sample size without distorting numerical feature distributions."*

### Q5: "How did you guarantee 100% reproducibility in your deduplication step across different operating systems?"
> **Answer:** *"In Step 4b, 997 duplicate song groups tie on their maximum popularity value. Sorting solely on popularity leaves ties unresolved, leading to quicksort instability where different pandas installations or architectures pick different surviving rows. By sorting on `['popularity', 'track_id']` (descending on popularity, ascending on the unique Spotify `track_id`), every tie is resolved deterministically, guaranteeing exact bit-for-bit reproducible data and tempo-bucket counts everywhere."*

---

## 5. Handoff Checklist for Teammates

- [x] **`cleaned_tracks.csv`** (76,103 rows, 43 columns, 0 nulls) -> Handed off to **Person B (EDA)** & **Person C (Modeling)**
- [x] **`scaler.pkl`** (Fitted `StandardScaler`, 9 audio features) -> Handed off to **Person C** & **Person D (Power BI / K-Means)**
- [x] **`cleaning_decision_log.md`** -> Handed off to **All Team Members** for viva preparation
