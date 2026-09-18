# Spotify Audio Analytics: Predicting Track Popularity from Audio Features

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Phase%201-Completed%20%26%20Verified-brightgreen.svg)]()
[![Course Outcomes](https://img.shields.io/badge/Rubric-CO1%20%7C%20CO2%20%7C%20CO3%20%7C%20CO4-orange.svg)]()

A collaborative data science project that analyzes acoustic properties of Spotify tracks to predict popularity and uncover audio-driven mood patterns.

---

## 👥 Team Breakdown & Project Roadmap

| Phase | Owner | Course Outcome | Focus Area | Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Phase 1** | **Person A** | **CO2 (5 Marks)** | **Data Acquisition & Preprocessing** | `cleaned_tracks.csv`, `scaler.pkl`, `cleaning_decision_log.md`, `phase1_preprocessing.py`, `phase1_preprocessing.ipynb` | **COMPLETED** |
| **Phase 2** | **Person B** | **CO1 (5 Marks)** | **Exploratory Data Analysis (EDA)** | `phase2_eda.ipynb`, Correlation Heatmap, Mood Quadrant, t-test hypothesis | *Ready to Start* |
| **Phase 3** | **Person C** | **CO4 (5 Marks)** | **Predictive Modeling & Regression** | `phase3_modeling.ipynb`, Linear/Ridge/Lasso/RF comparison, OLS diagnostics, `rf_model.pkl` | *Ready to Start* |
| **Phase 4** | **Person D** | **CO3 (5 Marks)** | **Power BI Dashboard & Integration** | `spotify_dashboard.pbix`, K-Means clustering, Slides & Final Report assembly | *Ready to Start* |

> 📖 **Teammates:** For detailed instructions, code templates, and rubrics for your specific phase, see the **[Phase Handoff Guide](PHASE_HANDOFF_GUIDE.md)**.

---

## 📊 Phase 1 Summary (Person A)

Phase 1 establishes the verified data foundation and serialized parameters required by downstream phases.

### Pipeline Progression:
- **Raw Input:** `dataset.csv` (114,000 tracks × 21 columns).
- **Index Sanitization:** Dropped redundant `Unnamed: 0` CSV export column.
- **Missing Values:** Imputed 1 missing `artists` and 1 missing `album_name` with `"Unknown"`. 0 missing values in core audio features.
- **Tier 1 Deduplication:** Removed 24,259 exact duplicate `track_id` records.
- **Tier 2 Deduplication (Deterministic):** Stripped remaster/live suffixes; sorted by `["popularity", "track_id"]` (descending popularity with unique `track_id` tie-breaker) and dropped duplicates on `[track_name_clean, artists]`, removing 8,708 records.
- **Outlier Filtering:**
  - Removed 15 tracks with `duration_ms <= 30,000` ms.
  - Removed 143 tracks with tempo outside `[30, 250]` BPM.
  - Removed 4,772 unpromoted tracks with `popularity == 0` (toggleable via `DROP_ZERO_POPULARITY = True`).
- **Feature Scaling:** Standardized 9 continuous audio features with `StandardScaler` and saved to `scaler.pkl`.
- **Categorical Encoding:** Converted `explicit` to 0/1; one-hot encoded `key` and `mode` with `drop_first=True` to prevent multicollinearity in OLS regression.
- **Feature Engineering:** Added `energy_valence` (interaction), `tempo_bucket` (`slow`: 11,050, `mid`: 36,444, `fast`: 28,609), and `mood_score`.
- **Final Cleaned Shape:** **76,103 rows × 43 columns** (0 missing values).

---

## 📁 Repository Structure

```text
Spotify-Audio-Analytics/
├── dataset.csv                       # Raw Kaggle 114k Spotify tracks dataset
├── cleaned_tracks.csv                # [DELIVERABLE] Final cleaned, feature-engineered dataset (76,103 rows, 43 cols)
├── scaler.pkl                        # [DELIVERABLE] Serialized StandardScaler fitted on 9 audio features
├── cleaning_decision_log.md          # [DELIVERABLE] Viva defense documentation and row count audits
├── phase1_preprocessing.py           # [DELIVERABLE] Standalone headless preprocessing script
├── phase1_preprocessing.ipynb        # [DELIVERABLE] Executed Jupyter notebook with pre-rendered outputs
├── PHASE_HANDOFF_GUIDE.md            # Detailed instructions for Person B, Person C, and Person D
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Standard Python/Jupyter/macOS ignore rules
└── README.md                         # Main project overview and documentation
```

---

## 🚀 Getting Started & Reproducing Phase 1

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/Oscar-man-shrestha/Spotify-Audio-Analytics.git
cd Spotify-Audio-Analytics

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all project dependencies
pip install -r requirements.txt
```

### 2. Run the Preprocessing Script
```bash
python phase1_preprocessing.py
```

### 3. Open the Interactive Notebook
```bash
jupyter notebook phase1_preprocessing.ipynb
```

---

## 🎓 Viva Defense Summary

For oral examination preparation, all team members should know these core answers:

1. **Why drop 38,000 rows (33%)?**  
   *24,259 were exact duplicates across multiple playlists and 8,708 were remaster/live duplicates. Removing duplicates plus extreme duration/tempo anomalies and 4,772 unpromoted 0-popularity tracks prevents exposure bias and data leakage.*
2. **How was deduplication made 100% reproducible?**  
   *997 duplicate groups tied on maximum popularity. Sorting on `['popularity', 'track_id']` ensures deterministic tie-breaking regardless of operating system or pandas version.*
3. **Why use `drop_first=True` for key and mode?**  
   *Eliminates the dummy variable trap (perfect multicollinearity) so the covariance matrix $(X^TX)$ remains strictly invertible for Person C's OLS regression.*
4. **Why save `scaler.pkl`?**  
   *Preprocessing scaling parameters ($\mu, \sigma$) must be computed once to prevent data leakage and ensure consistent scaling for Person D's K-Means clustering.*

---

## 📜 Deliverable Handoff Links
- 📘 **Team Implementation Guide:** [PHASE_HANDOFF_GUIDE.md](PHASE_HANDOFF_GUIDE.md)
- 📝 **Cleaning Decision Log:** [cleaning_decision_log.md](cleaning_decision_log.md)
- 📓 **Interactive Notebook:** [phase1_preprocessing.ipynb](phase1_preprocessing.ipynb)
- 🐍 **Standalone Script:** [phase1_preprocessing.py](phase1_preprocessing.py)
