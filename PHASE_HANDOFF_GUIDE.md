# Spotify Audio Analytics — Team Phase Handoff Guide
**Project Title:** Spotify Audio Analytics: Predicting Track Popularity from Audio Features  
**Course Outcomes & Grading:** CO1 (EDA: 5 Marks), CO2 (Preprocessing: 5 Marks), CO3 (Dashboard: 5 Marks), CO4 (Modeling: 5 Marks) + Viva (10 Marks)  
**Status:** **PHASE 1 IS 100% COMPLETE & VERIFIED**

---

## 1. Project Status & Completed Work (Phase 1 — Person A)

Person A has completed all data acquisition, index sanitization, missing value imputation, two-tier deduplication, outlier filtering, feature scaling, categorical dummy encoding, and feature engineering.

### Handoff Files Available Now:
1. [`cleaned_tracks.csv`](cleaned_tracks.csv) — 76,103 rows × 43 columns, 0 missing values, fully prepared dataset.
2. [`scaler.pkl`](scaler.pkl) — Fitted `StandardScaler` on 9 continuous audio features.
3. [`cleaning_decision_log.md`](cleaning_decision_log.md) — Comprehensive row audit and viva defense questions.
4. [`phase1_preprocessing.py`](phase1_preprocessing.py) — Standalone pipeline script with before/after row logging.
5. [`phase1_preprocessing.ipynb`](phase1_preprocessing.ipynb) — Executed Jupyter notebook with pre-rendered outputs.

### Summary of Cleaning & Pipeline Transformations:
- **Raw Kaggle Dataset:** 114,000 rows × 21 columns (dropped stray `Unnamed: 0` export column).
- **Missing Values:** Imputed 1 missing `artists`, 1 missing `album_name`, and 1 missing `track_name` with `"Unknown"`. Verified 0 missing values across all numeric audio features.
- **Exact Track Deduplication:** Dropped 24,259 duplicate `track_id` rows (tracks appearing across multiple playlists).
- **Song Version Deduplication (Deterministic):** Stripped `- Remastered`, `- Live`, `- Radio Edit` suffixes; sorted by `["popularity", "track_id"]` (descending popularity with unique `track_id` tie-breaker) and dropped duplicates on `[track_name_clean, artists]`, dropping 8,708 rows.
- **Outlier Filtering:**
  - Dropped 15 tracks with `duration_ms <= 30,000` ms (short non-musical fragments).
  - Dropped 143 tracks with tempo outside `[30, 250]` BPM (algorithmic cadence extraction anomalies).
  - Dropped 4,772 unpromoted tracks with `popularity == 0` via `DROP_ZERO_POPULARITY = True` (removes cold-start noise).
- **Feature Scaling:** Standardized 9 audio features with `StandardScaler` and saved to `scaler.pkl`.
- **Categorical Encoding:** `explicit` boolean converted to integer (0/1); `key` and `mode` one-hot encoded with `drop_first=True` (**CRITICAL:** eliminates perfect multicollinearity so Person C's OLS regression will not crash with a singular matrix error).
- **Feature Engineering:**
  - `energy_valence`: Interaction term ($\text{energy} \times \text{valence}$).
  - `tempo_bucket`: Categorical bins (`slow`: 11,050, `mid`: 36,444, `fast`: 28,609).
  - `mood_score`: Composite valence-energy-danceability index.
- **Final Shape:** **76,103 rows × 43 columns**, exactly **0 missing values**.

---

## 2. Dataset Schema Reference (43 Columns)

Every teammate should understand the structure of [`cleaned_tracks.csv`](cleaned_tracks.csv):

| Column Name | Data Type | Source / Description |
| :--- | :--- | :--- |
| `track_id` | string | Unique Spotify track identifier |
| `artists` | string | Performing artist(s) (missing filled with 'Unknown') |
| `album_name` | string | Release album (missing filled with 'Unknown') |
| `track_name` | string | Official track name |
| `popularity` | int (0–100) | **Target variable ($y$)**: Spotify track popularity score |
| `duration_ms` | int | Track duration in milliseconds (filtered > 30,000 ms) |
| `explicit` | int (0/1) | Converted boolean: 1 if track contains explicit lyrics, 0 otherwise |
| `danceability` | float (0–1) | Spotify danceability metric |
| `energy` | float (0–1) | Spotify perceptual energy metric |
| `loudness` | float (dB) | Overall loudness in decibels (typically -60 to 0 dB) |
| `speechiness` | float (0–1) | Presence of spoken words |
| `acousticness` | float (0–1) | Confidence measure of acoustic instrumentation |
| `instrumentalness` | float (0–1) | Likelihood track contains no vocals |
| `liveness` | float (0–1) | Presence of an audience in the recording |
| `valence` | float (0–1) | Musical positiveness/happiness score |
| `tempo` | float (BPM) | Overall estimated tempo in beats per minute |
| `time_signature` | int | Estimated overall time signature (e.g., 3, 4, 5) |
| `track_genre` | string | Primary genre category (114 unique genres) |
| `track_name_clean` | string | Normalized track name with remaster/live suffixes stripped |
| `danceability_scaled` ... `tempo_scaled` | float | 9 scaled continuous features standardized via `StandardScaler` |
| `key_1` through `key_11` | int (0/1) | 11 binary dummy columns for musical pitch (`key_0` dropped as reference) |
| `mode_1` | int (0/1) | 1 binary dummy column for Major mode (`mode_0` [Minor] dropped as reference) |
| `energy_valence` | float | Engineered interaction feature ($\text{energy} \times \text{valence}$) |
| `tempo_bucket` | category | Engineered tempo class: `'slow'` ($\le 90$), `'mid'` ($90–130$), `'fast'` ($> 130$) |
| `mood_score` | float | Engineered composite mood: $0.5 \times \text{valence} + 0.3 \times \text{energy} + 0.2 \times \text{danceability}$ |

---

## 3. Detailed Instructions for Person B: Phase 2 — Exploratory Data Analysis (EDA)

**Target Outcome:** CO1 (Data Visualization & EDA) — **5 Marks**  
**Input File:** [`cleaned_tracks.csv`](cleaned_tracks.csv)  
**Primary Tool:** Python notebook (`phase2_eda.ipynb`) using `matplotlib`, `seaborn`, `scipy`

### Step-by-Step Execution Plan:
1. **Load Data:**
   ```python
   import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, scipy.stats as stats
   df = pd.read_csv("cleaned_tracks.csv")
   ```
2. **Visual 1: Audio Feature Correlation Heatmap:**
   - Compute Pearson correlation matrix between continuous features (`danceability`, `energy`, `loudness`, `valence`, `acousticness`, `tempo`, `mood_score`, `energy_valence`) and `popularity`.
   - Plot using `sns.heatmap(annot=True, cmap="coolwarm", fmt=".2f")`.
   - Save to `figures/correlation_heatmap.png`.
3. **Visual 2: Mood Quadrant Scatter Plot (Russell's Circumplex Model):**
   - Scatter plot of `valence` (x-axis) vs `energy` (y-axis).
   - Overlay reference lines at $x=0.5$ and $y=0.5$ to divide into 4 quadrants:
     - Top-Right: Happy / Energetic
     - Top-Left: Angry / Turbulent
     - Bottom-Right: Peaceful / Chill
     - Bottom-Left: Sad / Depressing
   - Color points by `popularity` or cluster label.
   - Save to `figures/mood_quadrant.png`.
4. **Visual 3: Top 10 Genres by Average Popularity:**
   - Aggregate mean popularity across `track_genre` and select the top 10 genres by song count or popularity.
   - Horizontal bar plot: `sns.barplot(x="popularity", y="track_genre", data=top_genres)`.
   - Save to `figures/top_genres_popularity.png`.
5. **Visual 4: Feature Distribution Plots:**
   - KDE/Histogram plots for `popularity`, `loudness`, and `danceability` to visualize skewness and normality.
   - Save to `figures/feature_distributions.png`.
6. **Statistical Hypothesis Testing (t-test):**
   - Run a two-sample independent t-test comparing popularity between explicit vs clean songs:
     ```python
     explicit = df[df["explicit"] == 1]["popularity"]
     clean = df[df["explicit"] == 0]["popularity"]
     t_stat, p_val = stats.ttest_ind(explicit, clean)
     print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
     ```
   - Report whether the difference in popularity is statistically significant ($\alpha = 0.05$).
7. **Deliverables for Person B:**
   - `phase2_eda.ipynb` (clean, executed notebook).
   - Saved high-resolution figures in `figures/`.
   - Written summary of EDA insights and t-test result for the team report.

---

## 4. Detailed Instructions for Person C: Phase 3 — Predictive Modeling & Regression

**Target Outcome:** CO4 (Supervised Learning & Regression Modeling) — **5 Marks**  
**Input File:** [`cleaned_tracks.csv`](cleaned_tracks.csv)  
**Primary Tool:** Python notebook (`phase3_modeling.ipynb`) using `scikit-learn`, `statsmodels`, `joblib`

### Step-by-Step Execution Plan:
1. **Prepare Feature Matrix ($X$) & Target ($y$):**
   ```python
   import pandas as pd, numpy as np, statsmodels.api as sm, joblib
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression, Ridge, Lasso
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.metrics import r2_score, mean_squared_error

   df = pd.read_csv("cleaned_tracks.csv")

   # Select feature columns (unscaled or scaled features, dummies, engineered features)
   feature_cols = [
       "danceability", "energy", "loudness", "speechiness", "acousticness",
       "instrumentalness", "liveness", "valence", "tempo", "explicit",
       "energy_valence", "mood_score",
       "key_1", "key_2", "key_3", "key_4", "key_5", "key_6",
       "key_7", "key_8", "key_9", "key_10", "key_11", "mode_1"
   ]
   X = df[feature_cols]
   y = df["popularity"]

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
2. **Baseline Model (Linear Regression):**
   - Fit `LinearRegression()`, predict on test set, calculate $R^2$ and $\text{RMSE} = \sqrt{\text{MSE}}$.
3. **Model Comparison:**
   - Fit Ridge ($\alpha=1.0$), Lasso ($\alpha=0.1$), and Random Forest Regressor (`n_estimators=200`, `random_state=42`).
   - Assemble comparison table sorted by test $R^2$ descending:
     | Model | Train $R^2$ | Test $R^2$ | Test RMSE |
     | :--- | :---: | :---: | :---: |
4. **Regression Assumption Diagnostics:**
   - **Residual Plot:** Plot predicted popularity vs residuals ($y_{\text{test}} - \hat{y}$) with a horizontal zero line to test homoscedasticity. Save as `figures/residuals.png`.
   - **Q-Q Plot:** `stats.probplot(residuals, dist="norm", plot=plt)`. Save as `figures/qq_plot.png`.
5. **Coefficient Significance (OLS Analysis):**
   ```python
   X_sm = sm.add_constant(X_train)
   ols_model = sm.OLS(y_train, X_sm).fit()
   print(ols_model.summary())
   ```
   > **Note:** Because Phase 1 applied `drop_first=True` to `key` and `mode`, this step will run smoothly without any `Singular Matrix` or multicollinearity errors.
6. **Feature Importance Ranking:**
   - Extract `feature_importances_` from Random Forest or use `sklearn.inspection.permutation_importance`.
   - Plot top 10 most predictive features of track popularity.
7. **Save Winning Model:**
   ```python
   joblib.dump(best_rf_model, "rf_model.pkl")
   ```
8. **Deliverables for Person C:**
   - `phase3_modeling.ipynb`.
   - `rf_model.pkl` (serialized model artifact).
   - Model metrics comparison table and assumption diagnostic plots.

---

## 5. Detailed Instructions for Person D: Phase 4 — Dashboard (Power BI) & Integration

**Target Outcome:** CO3 (Dashboard Development & Data Integration) — **5 Marks**  
**Input Files:** [`cleaned_tracks.csv`](cleaned_tracks.csv), [`scaler.pkl`](scaler.pkl)  
**Primary Tool:** Power BI Desktop (`spotify_dashboard.pbix`), Python (`sklearn.cluster.KMeans`)

### Step-by-Step Execution Plan:
1. **K-Means Mood Clustering (Unsupervised Visualization Layer):**
   - **CRITICAL:** Re-use the existing [`scaler.pkl`](scaler.pkl) created in Phase 1 — **do not fit a new scaler**!
   - Features: `["energy_scaled", "valence_scaled", "danceability_scaled", "acousticness_scaled"]`.
   - Compute inertia for $k \in [2, 7]$ to generate the Elbow Method curve (`figures/elbow_curve.png`).
   - Fit $k=4$ clusters:
     ```python
     from sklearn.cluster import KMeans
     import joblib

     cluster_features = ["energy_scaled", "valence_scaled", "danceability_scaled", "acousticness_scaled"]
     kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
     df["mood_cluster"] = kmeans.fit_predict(df[cluster_features])
     joblib.dump(kmeans, "kmeans_model.pkl")

     # Re-export CSV including mood_cluster for Power BI ingestion
     df.to_csv("cleaned_tracks.csv", index=False)
     ```
2. **Build Power BI Dashboard (`spotify_dashboard.pbix`):**
   - Open Power BI Desktop $\to$ Get Data $\to$ Text/CSV $\to$ Load `cleaned_tracks.csv`.
   - Visual 1 (Scatter Chart): Valence ($x$) vs Energy ($y$), colored by `mood_cluster` (quadrant view).
   - Visual 2 (Bar Chart): Average Popularity by `track_genre` (Top 10 genres).
   - Visual 3 (Slicers): Interactive dropdown/slicers for Genre, `explicit`, and `tempo_bucket` (`slow`, `mid`, `fast`).
   - Visual 4 (KPI Cards): Total Tracks (`76,103`), Average Popularity, Average Energy.
   - Visual 5 (What-If Parameter): Add a dynamic parameter (e.g., simulating changes in danceability).
   - Export 2–3 clear screenshots to `figures/dashboard_screenshot1.png`, etc., as viva slide backups.
3. **Compile Final Report and Slides:**
   - Assemble findings from all teammates into the standardized rubric structure:
     1. Introduction & Problem Statement
     2. Data Acquisition & Preprocessing (Person A — CO2)
     3. Exploratory Data Analysis & Visualization (Person B — CO1)
     4. Predictive Modeling & Regression Diagnostics (Person C — CO4)
     5. Power BI Interactive Dashboard (Person D — CO3)
     6. Conclusion & Practical Implications
4. **Deliverables for Person D:**
   - `spotify_dashboard.pbix` Power BI file.
   - High-resolution dashboard screenshots.
   - Master presentation slides and final report draft.

---

## 6. Cross-Phase Dependency & Handoff Matrix

| From | To | Artifact Handed Off | Purpose |
| :--- | :--- | :--- | :--- |
| **Person A** | **Person B, C** | `cleaned_tracks.csv` + `cleaning_decision_log.md` | Cleaned baseline data and preprocessing rationale |
| **Person A** | **Person C, D** | `scaler.pkl` | Exact feature scaling parameters (no refitting allowed) |
| **Person B** | **Person D** | EDA plots & insights | Visualizations to embed into dashboard and report |
| **Person C** | **Person D** | `rf_model.pkl` + metrics table | Final regression model and coefficient table |
| **Person D** | **All Team** | `spotify_dashboard.pbix` + final report draft | Completed integration and presentation materials |

---

## 7. Viva Defense Cheatsheet (For Oral Exam)

| # | Expected Viva Question | Approved Standard Team Answer |
| :--- | :--- | :--- |
| **1** | **Why did you drop ~38,000 rows (33%) from the dataset?** | 24,259 were exact duplicates from multi-genre playlist overlap. 8,708 were remaster/live re-releases of the same songs. Removing these plus extreme outliers (<30s, invalid tempos) and 4,772 unpromoted 0-popularity tracks prevents exposure noise and data leakage. |
| **2** | **How did you ensure deterministic deduplication across different computers?** | 997 duplicate song groups tie on maximum popularity. Sorting on `['popularity', 'track_id']` (descending popularity, ascending unique track ID) eliminates quicksort order instability, guaranteeing exact bit-for-bit reproducibility. |
| **3** | **Why did you use `drop_first=True` when one-hot encoding key and mode?** | Including all dummy levels alongside an intercept creates the dummy variable trap (perfect multicollinearity), causing $(X^TX)$ to be non-invertible. `drop_first=True` avoids singular-matrix errors in Person C's OLS regression. |
| **4** | **Why did Person A save `scaler.pkl` instead of Person D refitting in Phase 4?** | Preprocessing parameters ($\mu, \sigma$) must be computed once to prevent data leakage and guarantee consistent feature representation across EDA, regression, and clustering. |
| **5** | **Why is K-Means clustering not your graded machine learning model?** | K-Means is an unsupervised visual grouping layer used for dashboard mood segmentation (CO3). The graded ML component is Person C's supervised regression predicting continuous popularity (CO4). |
