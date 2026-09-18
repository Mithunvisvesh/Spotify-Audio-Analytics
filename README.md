# Spotify Audio Analytics: Predicting Track Popularity from Audio Features

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Phase%201-Completed%20%26%20Verified-brightgreen.svg)]()
[![Course Outcomes](https://img.shields.io/badge/Curriculum%20Alignment-CO1%20%7C%20CO2%20%7C%20CO3%20%7C%20CO4-orange.svg)]()

An end-to-end data science project analyzing acoustic audio descriptors from Spotify to predict track popularity, model non-linear emotional valence patterns, and deliver interactive business intelligence dashboards.

---

## 📌 Modular Project Architecture & Roadmap

| Module | Focus Area | Course Outcome | Core Deliverables | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Phase 1** | **Data Acquisition, Sanitization & Preprocessing** | **CO2** (Data Preparation) | `cleaned_tracks.csv`, `scaler.pkl`, `cleaning_decision_log.md`, `phase1_preprocessing.py`, `phase1_preprocessing.ipynb` | **COMPLETED** |
| **Phase 2** | **Exploratory Data Analysis & Hypothesis Testing** | **CO1** (Data Visualization) | `phase2_eda.ipynb`, Correlation Heatmap, Russell's Mood Quadrant, Two-Sample t-test | *Ready to Start* |
| **Phase 3** | **Supervised Predictive Modeling & OLS Diagnostics** | **CO4** (Predictive Modeling) | `phase3_modeling.ipynb`, Linear/Ridge/Lasso/RF comparison, Residual/Q-Q diagnostics, `rf_model.pkl` | *Ready to Start* |
| **Phase 4** | **Power BI Interactive Dashboard & Mood Clustering** | **CO3** (Data Integration & BI) | `spotify_dashboard.pbix`, K-Means clustering ($k=4$), Final report & slide integration | *Ready to Start* |

> 📘 **Modular Implementation Guide:** For comprehensive specifications, code architectures, and rubrics for all phases, refer to **[PHASE_HANDOFF_GUIDE.md](PHASE_HANDOFF_GUIDE.md)**.

---

## 📊 Phase 1: Data Acquisition & Preprocessing Summary

Phase 1 provides the mathematical and data foundation for all downstream predictive models and dashboards.

### Data Ingestion & Transformation Highlights:
- **Raw Input Corpus:** Kaggle Spotify tracks dataset (114,000 observations × 21 attributes).
- **Index Sanitization:** Dropped redundant `Unnamed: 0` CSV serialization index column.
- **Missing Value Handling:** Imputed non-critical metadata (`artists`, `album_name`, `track_name`) with `"Unknown"`. Confirmed 0 missing values across core audio features.
- **Tier 1 Deduplication:** Removed 24,259 exact duplicate `track_id` records repeated across multi-genre playlist classifications.
- **Tier 2 Deduplication (Deterministic):** Stripped remaster, live, and radio edit suffixes; sorted deterministically on `["popularity", "track_id"]` to preserve the primary composition per artist (removed 8,708 records).
- **Domain Outlier Filtering:**
  - Removed 15 recordings with `duration_ms <= 30,000` ms (short sound effects and calibration artifacts).
  - Removed 143 recordings with tempo outside `[30, 250]` BPM (cadence extraction errors).
  - Filtered 4,772 unpromoted recordings with `popularity == 0` (removes cold-start exposure noise).
- **Continuous Feature Standardization:** Standardized 9 continuous audio features with `StandardScaler` ($\mu = 0, \sigma = 1$) and serialized to `scaler.pkl`.
- **Categorical Dummy Encoding:** Converted `explicit` to binary integer (0/1); one-hot encoded `key` and `mode` with `drop_first=True` to prevent multicollinearity in Ordinary Least Squares (OLS) regression.
- **Feature Engineering:** Constructed `energy_valence` (interaction), `tempo_bucket` (`slow`: 11,050, `mid`: 36,444, `fast`: 28,609), and `mood_score` (composite valence-energy index).
- **Final Cleaned Dimensions:** **76,103 observations × 43 attributes** (exactly **0 missing values**).

---

## 📁 Repository Structure

```text
Spotify-Audio-Analytics/
├── dataset.csv                       # Raw Spotify tracks dataset (114,000 tracks)
├── cleaned_tracks.csv                # [DELIVERABLE] Cleaned dataset for modeling (76,103 rows, 43 cols)
├── scaler.pkl                        # [DELIVERABLE] Serialized StandardScaler fitted on audio metrics
├── cleaning_decision_log.md          # [DELIVERABLE] Technical methodology and row audit log
├── phase1_preprocessing.py           # [DELIVERABLE] Standalone headless execution script
├── phase1_preprocessing.ipynb        # [DELIVERABLE] Executed Jupyter notebook with pre-rendered outputs
├── PHASE_HANDOFF_GUIDE.md            # Modular engineering specifications for Phases 2, 3, and 4
├── requirements.txt                  # Pinned environment dependencies
├── .gitignore                        # Clean repository configuration
└── README.md                         # Main project overview and documentation
```

---

## 🚀 Environment Setup & Reproduction

### 1. Set Up Environment
```bash
# Clone the repository
git clone https://github.com/Oscar-man-shrestha/Spotify-Audio-Analytics.git
cd Spotify-Audio-Analytics

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Preprocessing Script
```bash
python phase1_preprocessing.py
```

### 3. Launch the Interactive Notebook
```bash
jupyter notebook phase1_preprocessing.ipynb
```

---

## 🎓 Technical & Methodological Defense (Viva Preparation)

1. **Why filter approximately 38,000 observations (33%)?**  
   *24,259 records were duplicate track IDs resulting from tracks being tagged in multiple genre playlists, and 8,708 were redundant remaster/live releases. Removing duplicates plus extreme audio anomalies (<30s, invalid tempos) and 4,772 unpromoted 0-popularity tracks prevents exposure bias and train-test data leakage.*
2. **How was deduplication reproducibility guaranteed across systems?**  
   *997 duplicate groups share an identical maximum popularity score. Sorting on `['popularity', 'track_id']` (descending popularity, ascending unique track ID) resolves ties deterministically, eliminating sorting algorithm instability.*
3. **Why use `drop_first=True` during one-hot encoding?**  
   *Omitting one reference category avoids the dummy variable trap (perfect multicollinearity), ensuring the feature covariance matrix $(X^TX)$ is full rank and strictly invertible for Ordinary Least Squares regression.*
4. **Why serialize `scaler.pkl` rather than refitting downstream?**  
   *Feature scaling parameters ($\mu, \sigma$) must be computed strictly on the baseline distribution. Serializing `scaler.pkl` prevents data leakage and ensures uniform scale alignment across clustering and inference.*

---

## 🔗 Project Documentation Links
- 📘 **Technical Implementation Guide:** [PHASE_HANDOFF_GUIDE.md](PHASE_HANDOFF_GUIDE.md)
- 📝 **Preprocessing Decision Log:** [cleaning_decision_log.md](cleaning_decision_log.md)
- 📓 **Interactive Notebook:** [phase1_preprocessing.ipynb](phase1_preprocessing.ipynb)
- 🐍 **Standalone Script:** [phase1_preprocessing.py](phase1_preprocessing.py)
