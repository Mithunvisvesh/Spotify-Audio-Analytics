# Spotify Audio Analytics: Modular Implementation & Phase Handoff Specification
**Project Title:** Spotify Audio Analytics: Predicting Track Popularity from Audio Features  
**Curriculum & Evaluation Alignment:** CO1 (EDA), CO2 (Data Preparation), CO3 (BI & Dashboards), CO4 (Predictive Modeling)  
**Current Milestone:** **PHASE 1 IS COMPLETE AND VALIDATED**

---

## 1. Baseline Deliverables & Phase 1 Execution Summary

Phase 1 provides the sanitized, normalized, and feature-engineered dataset along with model artifacts required by all downstream analytical modules.

### Available Handoff Artifacts:
1. [`cleaned_tracks.csv`](cleaned_tracks.csv) — Sanitized dataset (76,103 observations × 43 attributes, 0 missing values).
2. [`scaler.pkl`](scaler.pkl) — Fitted `StandardScaler` artifact encapsulating baseline feature scaling parameters ($\mu, \sigma$).
3. [`cleaning_decision_log.md`](cleaning_decision_log.md) — Comprehensive technical decision log and audit breakdown.
4. [`phase1_preprocessing.py`](phase1_preprocessing.py) — Standalone automated pipeline script with observation tracking.
5. [`phase1_preprocessing.ipynb`](phase1_preprocessing.ipynb) — Executed Jupyter notebook with pre-rendered outputs.

### Summary of Transformations:
- **Corpus Ingestion:** 114,000 observations × 21 columns from the Kaggle Spotify Tracks dataset. Removed redundant `Unnamed: 0` index column.
- **Missing Value Handling:** Imputed non-critical text metadata (`artists`, `album_name`, `track_name`) with `"Unknown"`. Verified 0 missing values across all continuous audio metrics.
- **Two-Tier Deduplication:**
  - *Tier 1 (Exact Spotify URI):* Filtered 24,259 duplicate `track_id` rows across playlist classifications.
  - *Tier 2 (Composition Normalization):* Stripped version suffixes (`- Remastered`, `- Live`, `- Radio Edit`); sorted deterministically on `["popularity", "track_id"]` to resolve 997 ties and retain the primary version per artist (filtered 8,708 rows).
- **Domain Outlier Filtering:**
  - Filtered 15 recordings with `duration_ms <= 30,000` ms (short non-musical fragments).
  - Filtered 143 recordings with tempo outside `[30, 250]` BPM (cadence extraction errors).
  - Filtered 4,772 unpromoted recordings with `popularity == 0` (eliminates platform cold-start noise).
- **Feature Standardization:** Standardized 9 continuous audio features with `StandardScaler` ($\mu = 0, \sigma = 1$) and serialized to `scaler.pkl`.
- **Multicollinearity-Safe Encoding:** Converted `explicit` to binary integer (0/1); one-hot encoded `key` and `mode` with `drop_first=True` to prevent the dummy variable trap in OLS regression.
- **Feature Engineering:** Constructed `energy_valence` (interaction), `tempo_bucket` (`slow`: 11,050, `mid`: 36,444, `fast`: 28,609), and `mood_score` (composite valence-energy index).
- **Integrity Verification:** Verified 100% completeness (**0 missing values across all 43 columns**).

---

## 2. Dataset Schema & Attribute Reference (43 Columns)

Every analytical module should utilize the standardized schema in [`cleaned_tracks.csv`](cleaned_tracks.csv):

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `track_id` | string | Unique Spotify track URI identifier |
| `artists` | string | Performing artist(s) (missing values filled with 'Unknown') |
| `album_name` | string | Album title (missing values filled with 'Unknown') |
| `track_name` | string | Track title |
| `popularity` | int (0–100) | **Target variable ($y$)**: Spotify track popularity metric |
| `duration_ms` | int | Track duration in milliseconds (filtered > 30,000 ms) |
| `explicit` | int (0/1) | Binary indicator: 1 if explicit content, 0 otherwise |
| `danceability` | float (0–1) | Perceptual danceability score |
| `energy` | float (0–1) | Perceptual intensity and musical activity score |
| `loudness` | float (dB) | Overall loudness in decibels (typically -60 to 0 dB) |
| `speechiness` | float (0–1) | Relative presence of spoken words |
| `acousticness` | float (0–1) | Confidence score of acoustic instrumentation |
| `instrumentalness` | float (0–1) | Likelihood of non-vocal audio content |
| `liveness` | float (0–1) | Presence of an audience in the recording |
| `valence` | float (0–1) | Musical positiveness and emotional cheerfulness score |
| `tempo` | float (BPM) | Estimated musical cadence in beats per minute |
| `time_signature` | int | Estimated musical time signature (e.g., 3, 4, 5) |
| `track_genre` | string | Primary genre category (114 unique categories) |
| `track_name_clean` | string | Normalized title with remaster/live/edit suffixes removed |
| `danceability_scaled` ... `tempo_scaled` | float | 9 continuous attributes standardized via `StandardScaler` |
| `key_1` through `key_11` | int (0/1) | 11 binary dummy variables for pitch class (`key_0` reference dropped) |
| `mode_1` | int (0/1) | 1 binary dummy variable for Major modality (`mode_0` [Minor] reference dropped) |
| `energy_valence` | float | Interaction term ($\text{energy} \times \text{valence}$) |
| `tempo_bucket` | category | Discretized cadence: `'slow'` ($\le 90$), `'mid'` ($90–130$), `'fast'` ($> 130$) |
| `mood_score` | float | Composite mood index: $0.5 \times \text{valence} + 0.3 \times \text{energy} + 0.2 \times \text{danceability}$ |

---

## 3. Phase 2 Specification: Exploratory Data Analysis & Hypothesis Testing

**Curriculum Alignment:** Course Outcome 1 (CO1 — Data Visualization & Exploration)  
**Input File:** [`cleaned_tracks.csv`](cleaned_tracks.csv)  
**Target Environment:** Jupyter Notebook (`phase2_eda.ipynb`) using `seaborn`, `matplotlib`, and `scipy.stats`

### Required Methodologies & Visualizations:
1. **Audio Feature Correlation Matrix:**
   - Compute Pearson correlation coefficients between continuous features (`danceability`, `energy`, `loudness`, `valence`, `acousticness`, `tempo`, `mood_score`, `energy_valence`) and `popularity`.
   - Render heatmap with `sns.heatmap(annot=True, cmap="coolwarm", fmt=".2f")`.
   - Save high-resolution visual to `figures/correlation_heatmap.png`.
2. **Russell's Circumplex Mood Quadrant Scatter Plot:**
   - Plot `valence` (x-axis) vs `energy` (y-axis).
   - Overlay quadrant dividers at $x = 0.5$ and $y = 0.5$:
     - Quadrant I (Top-Right): High Valence, High Energy (Happy / Energetic)
     - Quadrant II (Top-Left): Low Valence, High Energy (Angry / Turbulent)
     - Quadrant III (Bottom-Left): Low Valence, Low Energy (Melancholic / Sad)
     - Quadrant IV (Bottom-Right): High Valence, Low Energy (Peaceful / Chill)
   - Color points by `popularity` or genre clusters. Save to `figures/mood_quadrant.png`.
3. **Genre Popularity Distribution:**
   - Aggregate mean popularity across `track_genre` and display the top 10 genres.
   - Horizontal bar plot: `sns.barplot(x="popularity", y="track_genre")`. Save to `figures/genre_popularity.png`.
4. **Feature Distribution & Skewness Analysis:**
   - KDE and histogram distributions for `popularity`, `loudness`, and `danceability`. Save to `figures/feature_distributions.png`.
5. **Statistical Hypothesis Testing (Two-Sample t-Test):**
   - Perform an independent two-sample t-test comparing popularity distributions between explicit vs clean tracks:
     ```python
     from scipy import stats
     explicit_pop = df[df["explicit"] == 1]["popularity"]
     clean_pop = df[df["explicit"] == 0]["popularity"]
     t_stat, p_val = stats.ttest_ind(explicit_pop, clean_pop)
     print(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")
     ```
   - Formulate null hypothesis $H_0: \mu_{\text{explicit}} = \mu_{\text{clean}}$ vs $H_1: \mu_{\text{explicit}} \neq \mu_{\text{clean}}$ and evaluate at $\alpha = 0.05$.

---

## 4. Phase 3 Specification: Supervised Predictive Modeling & OLS Diagnostics

**Curriculum Alignment:** Course Outcome 4 (CO4 — Supervised Machine Learning & Regression Analysis)  
**Input Files:** [`cleaned_tracks.csv`](cleaned_tracks.csv), [`scaler.pkl`](scaler.pkl)  
**Target Environment:** Jupyter Notebook (`phase3_modeling.ipynb`) using `scikit-learn`, `statsmodels`, and `joblib`

### Required Methodologies & Model Comparison:
1. **Train/Test Partitioning:**
   - Split dataset into 80% training and 20% test subsets using `train_test_split(..., test_size=0.2, random_state=42)`.
   - Feature matrix $X$ incorporates continuous audio metrics, engineered interaction terms, and one-hot dummy columns.
2. **Model Formulation & Evaluation:**
   - Evaluate 4 candidate regressors:
     1. Baseline Linear Regression (`LinearRegression()`)
     2. Ridge Regularization (`Ridge(alpha=1.0)`)
     3. Lasso Regularization (`Lasso(alpha=0.1)`)
     4. Random Forest Regressor (`RandomForestRegressor(n_estimators=200, random_state=42)`)
   - Compile comparative evaluation metrics table:
     | Model | Train $R^2$ | Test $R^2$ | Test RMSE |
     | :--- | :---: | :---: | :---: |
3. **Regression Assumption Diagnostics:**
   - **Homoscedasticity Assessment:** Scatter plot of predicted values ($\hat{y}$) vs residuals ($y - \hat{y}$) with a horizontal reference line at zero. Save to `figures/residuals.png`.
   - **Normality of Residuals:** Quantile-Quantile (Q-Q) probability plot using `scipy.stats.probplot(residuals, dist="norm")`. Save to `figures/qq_plot.png`.
4. **OLS Coefficient Significance:**
   - Fit `statsmodels.api.OLS` on standardized features to extract coefficient weights, standard errors, $t$-statistics, and $p$-values.
   - *Note:* Because Phase 1 applied `drop_first=True`, the design matrix $X$ is strictly full rank, avoiding singular-matrix errors.
5. **Permutation Feature Importance:**
   - Evaluate feature importance via `sklearn.inspection.permutation_importance` on the top-performing model to identify key acoustic drivers of popularity.
6. **Artifact Serialization:**
   - Serialize the optimal model to `rf_model.pkl` using `joblib.dump(best_model, "rf_model.pkl")`.

---

## 5. Phase 4 Specification: Power BI Dashboard & Unsupervised Clustering

**Curriculum Alignment:** Course Outcome 3 (CO3 — Business Intelligence & Data Integration)  
**Input Files:** [`cleaned_tracks.csv`](cleaned_tracks.csv), [`scaler.pkl`](scaler.pkl)  
**Primary Tool:** Power BI Desktop (`spotify_dashboard.pbix`), Python (`sklearn.cluster.KMeans`)

### Required Methodologies & Visual Layer:
1. **Unsupervised K-Means Clustering (Mood Grouping):**
   - **Crucial Rule:** Re-use the existing [`scaler.pkl`](scaler.pkl) generated in Phase 1 — **never fit a new scaler**.
   - Input cluster features: `["energy_scaled", "valence_scaled", "danceability_scaled", "acousticness_scaled"]`.
   - Compute inertia across $k \in [2, 7]$ to generate the Elbow Method plot (`figures/elbow_curve.png`).
   - Fit $k=4$ clusters:
     ```python
     from sklearn.cluster import KMeans
     import joblib

     cluster_cols = ["energy_scaled", "valence_scaled", "danceability_scaled", "acousticness_scaled"]
     kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
     df["mood_cluster"] = kmeans.fit_predict(df[cluster_cols])
     joblib.dump(kmeans, "kmeans_model.pkl")
     df.to_csv("cleaned_tracks.csv", index=False)
     ```
2. **Interactive Power BI Dashboard (`spotify_dashboard.pbix`):**
   - Visual 1 (Scatter Plot): Valence ($x$) vs Energy ($y$) colored by `mood_cluster` (interactive quadrant view).
   - Visual 2 (Bar Chart): Average popularity across top 10 genres.
   - Visual 3 (Slicers & Filters): Interactive filtering by genre, `explicit` indicator, and `tempo_bucket` (`slow`, `mid`, `fast`).
   - Visual 4 (KPI Cards): Total tracks analyzed (`76,103`), mean popularity, mean energy.
   - Visual 5 (What-If Parameter): Simulation parameter for audio features (e.g., danceability adjustment).
   - Export 2–3 high-resolution screenshots to `figures/dashboard_screenshot.png` for slide backups.
3. **Comprehensive Report Integration:**
   - Synthesize project outcomes into the formal submission report:
     1. Introduction & Research Problem
     2. Data Acquisition & Preprocessing (Phase 1 — CO2)
     3. Exploratory Data Analysis & Hypothesis Testing (Phase 2 — CO1)
     4. Supervised Predictive Modeling & OLS Diagnostics (Phase 3 — CO4)
     5. Power BI Dashboard & Unsupervised Clustering (Phase 4 — CO3)
     6. Conclusion & Industry Implications

---

## 6. Inter-Module Dependency & Artifact Matrix

```
[Phase 1: Preprocessing]
   │
   ├──> cleaned_tracks.csv ───────┬───> [Phase 2: EDA & Hypothesis Testing]
   │                              │
   │                              ├───> [Phase 3: Predictive Modeling & OLS]
   │                              │
   └──> scaler.pkl (reused) ──────┴───> [Phase 4: Power BI & K-Means Clustering]
                                               │
                                               ▼
                                      [Final Academic Report & Presentation]
```

---

## 7. Technical Defense & Evaluation Reference (Viva Preparation)

| Evaluation Topic | Methodological Defense |
| :--- | :--- |
| **Observation Reduction Strategy (~33%)** | 24,259 entries were playlist cross-listing duplicates; 8,708 were release-version duplicates. Removing duplicates, extreme audio errors (<30s, invalid tempos), and 4,772 unpromoted 0-popularity tracks prevents exposure bias, sample variance distortion, and train-test data leakage. |
| **Deterministic Reproducibility** | 997 duplicate groups share identical top popularity values. Sorting on `['popularity', 'track_id']` (descending popularity, ascending unique track URI) ensures deterministic tie-breaking across different operating systems. |
| **Categorical Multicollinearity Avoidance** | Setting `drop_first=True` during one-hot encoding avoids the dummy variable trap ($\sum \text{key}_i = 1$). This guarantees the covariance matrix $(X^TX)$ is full rank and invertible for OLS regression. |
| **StandardScaler Persistence Protocol** | Scaling parameters ($\mu, \sigma$) must be computed strictly on the baseline distribution. Reusing `scaler.pkl` in Phase 4 clustering ensures consistent scale representation without data leakage. |
| **K-Means vs Supervised Modeling Differentiation** | K-Means is an unsupervised visual layer used for mood segmentation in the dashboard (CO3). The primary predictive component is the supervised regression predicting popularity from audio features (CO4). |
