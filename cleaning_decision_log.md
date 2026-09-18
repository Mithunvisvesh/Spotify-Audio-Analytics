# Spotify Audio Analytics — Data Preprocessing & Decision Log
**Project Module:** Phase 1: Data Acquisition, Sanitization & Preprocessing  
**Course Outcome Alignment:** CO2 (Data Ingestion, Cleaning & Feature Transformation)  
**Dataset Reference:** Spotify Tracks Dataset (114,000 observations)  

---

## 1. Executive Summary & Observation Progression

| Pipeline Stage | Starting Observations | Observations Filtered / Imputed | Resulting Observations | % Dataset Retained | Methodological Action |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0. Raw Ingestion** (`dataset.csv`) | 114,000 | 0 | 114,000 | 100.00% | Ingested Kaggle 114k tracks dataset. Removed redundant `Unnamed: 0` CSV index. |
| **Step 3: Missing Value Handling** | 114,000 | 0 filtered<br>(2 imputed) | 114,000 | 100.00% | Imputed 1 missing `artists` & 1 missing `album_name` with `"Unknown"`. Audited 0 NaNs in core audio features. |
| **Step 4a: Exact URI Deduplication** | 114,000 | **24,259** | 89,741 | 78.72% | Filtered duplicate Spotify track URIs repeated across genre playlist classifications. |
| **Step 4b: Composition Deduplication** | 89,741 | **8,708** | 81,033 | 71.08% | Normalized titles (stripped `- Remastered`, `- Live`, `- Radio Edit`); resolved ties via `track_id`; preserved highest popularity cut. |
| **Step 5a: Duration Filtering** | 81,033 | **15** | 81,018 | 71.07% | Filtered non-musical audio recordings with `duration_ms <= 30,000` ms (sound effects and calibration noise). |
| **Step 5b: Tempo Plausibility** | 81,018 | **143** | 80,875 | 70.94% | Filtered tempo estimation anomalies outside plausible musical cadence `[30, 250]` BPM. |
| **Step 5c: Cold-Start Exposure Filter** | 80,875 | **4,772** | **76,103** | **66.76%** | Filtered unpromoted tracks with `popularity == 0` to remove algorithmic exposure bias. |

**Final Sanitized Dataset:** **76,103 observations and 43 attributes** (37,897 total observations filtered, 33.24% of raw corpus; 0 missing values).

---

## 2. Technical Decisions & Methodological Rationale

### Step 1: Ingestion and Index Sanitization
- **Action:** Loaded `dataset.csv` (114,000 rows × 21 columns) and immediately dropped `Unnamed: 0`.
- **Rationale:** `Unnamed: 0` is an unintended artifact from prior CSV exports with default indexing (`index=True`). Removing it eliminates redundant memory overhead and prevents potential alignment errors in subsequent merges.

### Steps 2 & 3: Missing Value Strategy (Imputation vs. Deletion)
- **Action:**
  - Imputed missing values in `artists` (1 row), `album_name` (1 row), and `track_name` (1 row) with `"Unknown"`.
  - Audited core continuous audio attributes (`danceability`, `energy`, `tempo`, `valence`, `loudness`, `acousticness`) and target variable (`popularity`), verifying **0 missing values**.
- **Rationale:** Text metadata fields are descriptive identifiers; deleting an observation because of an unindexed album title discards valid acoustic measurements. Conversely, missing acoustic features or target popularity cannot be synthetically imputed without distorting empirical distributions.

### Step 4: Two-Tier Deterministic Deduplication
- **Tier 1 (Exact Spotify `track_id`):**
  - **Count filtered:** 24,259 observations (21.28% of corpus).
  - **Rationale:** Spotify’s catalog indexes the same song across multiple playlist genres (e.g., a track categorized under both `pop` and `dance`). Keeping exact duplicate IDs artificially inflates sample size, distorts standard errors, and introduces train-test data leakage.
- **Tier 2 (Composition & Remaster Normalization with Deterministic Tiebreaker):**
  - **Count filtered:** 8,708 observations.
  - **Rationale & Reproducibility:** Artists release remastered editions, live recordings, and radio edits of the same underlying composition. By normalizing titles (regex stripping `r"\s*-\s*(Remaster(ed)?|Live|Radio Edit).*"`), sorting by `["popularity", "track_id"]` (descending popularity, ascending `track_id`), and retaining the first occurrence per `[track_name_clean, artists]`, the primary version is preserved.
  - **Why the `track_id` tiebreaker is essential:** Exactly **997 duplicate groups** share an identical maximum popularity score. Without a secondary unique key, quicksort instability leads to non-deterministic row selection across different operating systems. Incorporating `track_id` guarantees 100% bit-for-bit reproducibility.

### Step 5: Domain-Specific Outlier Filtering
- **Duration (`duration_ms > 30,000`):**
  - **Count filtered:** 15 observations.
  - **Rationale:** Recordings under 30 seconds represent non-song audio fragments (spoken intros, silence calibration, or sound effects).
- **Tempo (`tempo.between(30, 250)`):**
  - **Count filtered:** 143 observations.
  - **Rationale:** Standard musical tempo spans 40 to 220 BPM. Tempos below 30 BPM or above 250 BPM indicate audio feature extraction failures (e.g., ambient silence or octave-doubling errors).
- **Cold-Start Exposure Filtering (`popularity > 0`):**
  - **Count filtered:** 4,772 observations.
  - **Rationale:** In the Spotify Web API, a popularity score of 0 represents an unindexed or newly uploaded track that received zero algorithmic promotion. Including zero-popularity tracks introduces platform exposure bias into predictive regression models, causing them to model lack of distribution rather than acoustic appeal.

### Step 6: Feature Standardization (StandardScaler)
- **Action:** Standardized 9 continuous features (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`) to zero mean and unit variance ($\mu = 0, \sigma = 1$). Persisted the fitted transformer to `scaler.pkl`.
- **Rationale:**
  - Continuous metrics operate across disparate scales (e.g., loudness in negative decibels [-60 to 0 dB], tempo in BPM [30 to 250], and valence bounded in [0, 1]). Normalization prevents features with large magnitudes from dominating Euclidean distance metrics during clustering.
  - Persisting `scaler.pkl` ensures downstream clustering and production models use the exact baseline parameters without refitting, eliminating data leakage.

### Step 7: Categorical Encoding & Multicollinearity Prevention
- **Action:** Converted `explicit` to binary integer ($0 / 1$). One-hot encoded `key` (12 pitch classes) and `mode` (Major/Minor) using `pd.get_dummies(..., drop_first=True)`.
- **Rationale:**
  - **The Dummy Variable Trap:** Including all $k$ binary indicators alongside a constant intercept generates exact linear dependency ($\sum \text{key}_i = 1$).
  - Ordinary Least Squares (OLS) regression requires the design matrix $(X^TX)$ to be full rank and strictly invertible. Applying `drop_first=True` establishes a reference category (`key_0` and `mode_0`), preventing singular matrix (`LinAlgError`) failures.

### Step 8: Domain Feature Engineering
- **`energy_valence`** ($= \text{energy} \times \text{valence}$): Interaction metric modeling high-arousal positive affect.
- **`tempo_bucket`**: Discretization into categorical cadence classes (`slow`: $\le 90$, `mid`: $90–130$, `fast`: $> 130$ BPM).
- **`mood_score`** ($= 0.5 \times \text{valence} + 0.3 \times \text{energy} + 0.2 \times \text{danceability}$): Composite mood index emphasizing emotional valence, rhythmical energy, and rhythmic regularity.

---

## 3. Dataset Integrity & Schema Audit

### A. Mathematical Column Count Verification (43 Attributes)
$$\begin{aligned}
\mathbf{20} & \quad \text{Initial attributes (excluding dropped } \texttt{Unnamed: 0}\text{)} \\
+\; \mathbf{1} & \quad \texttt{track\_name\_clean}\text{ (normalized title for composition deduplication)} \\
+\; \mathbf{9} & \quad \text{Standardized feature metrics (}\texttt{<feature>\_scaled}\text{)} \\
-\; \mathbf{2} & \quad \text{Original categorical attributes (}\texttt{key}\text{, }\texttt{mode}\text{ removed by dummy encoding)} \\
+\; \mathbf{12} & \quad \text{Binary dummy columns (11 for key: }\texttt{key\_1}\dots\texttt{key\_11}\text{, 1 for mode: }\texttt{mode\_1}\text{; reference dropped)} \\
+\; \mathbf{3} & \quad \text{Engineered attributes (}\texttt{energy\_valence}\text{, }\texttt{tempo\_bucket}\text{, }\texttt{mood\_score}\text{)} \\
\hline
=\; \mathbf{43} & \quad \textbf{Final column count in } \texttt{cleaned\_tracks.csv}
\end{aligned}$$

### B. Zero Missing Values & Boundary Verification
- **Total Missing Values Across Dataset:** **`0`** (`df.isnull().sum().sum() == 0`).
- **`pd.cut` Boundary Verification:** `pd.cut` assigns `NaN` to any values outside the defined bin edges (`[0, 90, 130, 300]`).
  - Observed tempo range: **Min = 30.322 BPM, Max = 243.372 BPM**.
  - Because Step 5b filtered tempos outside `[30, 250]` BPM, all observations fall within $(0, 300]$, producing exactly 0 nulls (`slow`: **11,050**, `mid`: **36,444**, `fast`: **28,609**).

---

## 4. Technical Defense & Methodology Justifications

### Q1: "Why were approximately 38,000 observations (33%) filtered from the corpus?"
> **Methodological Justification:** *"The raw Kaggle dataset contained 24,259 exact duplicate track IDs resulting from cross-playlist classification, and 8,708 redundant remaster/live versions. Retaining duplicates introduces train-test data leakage and artificially compresses standard errors. Furthermore, filtering non-musical recordings (<30s, extreme tempo errors) and 4,772 unpromoted 0-popularity tracks ensures downstream predictive models capture true acoustic drivers of popularity rather than platform exposure artifacts."*

### Q2: "Why was `drop_first=True` mandatory during categorical dummy encoding?"
> **Methodological Justification:** *"In linear regression, including all $k$ levels of a categorical attribute alongside an intercept creates the dummy variable trap (perfect collinearity), making the covariance matrix $(X^TX)$ non-invertible. Setting `drop_first=True` leaves one baseline category as the reference, ensuring the design matrix is full rank and avoiding singular-matrix errors in OLS regression."*

### Q3: "Why is `scaler.pkl` serialized and reused rather than refit downstream?"
> **Methodological Justification:** *"Feature normalization parameters ($\mu, \sigma$) must be computed strictly on the baseline preprocessing distribution. Re-fitting a scaler in subsequent clustering or inference introduces data leakage and creates scale misalignment across project modules."*

### Q4: "Why impute missing metadata with 'Unknown' rather than discarding the rows?"
> **Methodological Justification:** *"Only 1 observation lacked artist metadata and 1 lacked album metadata, while their acoustic measurements were 100% complete and valid. Discarding empirical acoustic observations for non-critical descriptive text causes unnecessary information loss. Imputing 'Unknown' preserves statistical sample size without biasing numerical distributions."*

### Q5: "How was deterministic reproducibility achieved in composition deduplication?"
> **Methodological Justification:** *"Across the dataset, 997 duplicate groups share an identical maximum popularity score. Sorting solely on popularity leaves ties unresolved, leading to quicksort instability where different pandas installations select different surviving rows. By sorting on `['popularity', 'track_id']` (popularity descending, unique Spotify track URI ascending), every tie is resolved deterministically, ensuring exact bit-for-bit reproducibility everywhere."*

---

## 5. Artifact Sign-Off & Deliverables

- [x] **`cleaned_tracks.csv`** (76,103 rows, 43 columns, 0 nulls) — Primary dataset for EDA and Predictive Modeling
- [x] **`scaler.pkl`** (StandardScaler artifact, 9 features) — Normalized transformer for K-Means clustering
- [x] **`cleaning_decision_log.md`** — Comprehensive methodology audit and defense reference
- [x] **`phase1_preprocessing.py`** — Headless execution script with row tracking
- [x] **`phase1_preprocessing.ipynb`** — Executed Jupyter notebook with pre-rendered outputs
