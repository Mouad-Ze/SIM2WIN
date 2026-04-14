# 🎯 SIM2WIN - Tactical Football Match Prediction System

A sophisticated machine learning system that predicts football match outcomes and recommends optimal tactical formations using StatsBomb event data from 12+ professional leagues.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Models & Architecture](#models--architecture)
- [Results & Performance](#results--performance)
- [Deployment](#deployment)
- [Technical Stack](#technical-stack)
- [Contributing](#contributing)
- [Author](#author)

---

## 🏆 Project Overview

**Sim2Win** is an end-to-end machine learning solution that:

1. **Aggregates** granular football event data across 12+ leagues (EPL, Bundesliga, La Liga, Ligue 1, MLS, World Cup, Women's competitions, CAF, Copa América)
2. **Engineers** advanced tactical features (pressing efficiency, shot quality, directness index, momentum metrics)
3. **Predicts** match outcomes (Home Win/Draw/Away Win) using CatBoost classification
4. **Recommends** optimal tactical formations by simulating all 8 possible tactical combinations
5. **Generates** professional tactical briefs using AI-powered analysis

### 💡 Use Cases

- **Coaches & Analysts**: Tactical optimization for upcoming matches
- **Betting Syndicates**: Match outcome predictions with confidence intervals
- **Sports Analytics**: Competitive advantage through data-driven tactical recommendations
- **Fan Analytics**: Interactive exploration of team performance patterns

---

## ✨ Key Features

### Match Outcome Prediction
- Multi-class classification: Home Win, Draw, Away Win
- CatBoost classifier with SHAP-based explainability
- Leave-One-Competition-Out (LOCO) cross-validation
- Performance: High accuracy across diverse leagues

### Tactical Analysis
- 8 distinct tactical archetypes identified via K-Means clustering:
  - High-Pressing Possession
  - Low-Block Counter
  - Mid-Block Transition
  - Direct Long Ball
  - Tiki-Taka
  - Wing-Play Overload
  - High-Intensity Gegenpress
  - Park the Bus

### Multi-Scenario Simulation
- Simulates all 8×8=64 tactical combinations (home vs away)
- Identifies optimal strategy and key vulnerabilities
- Probabilistic outcome estimates

### AI-Powered Reports
- Gemini AI integration for natural language tactical briefs
- Professional formatting with actionable insights
- Feature importance explanations via SHAP

---

## 📁 Project Structure

```
Sim2Win/
│
├── 📊 Agg/                              # Aggregated data by league
│   ├── Cleaned_Matches.csv              # Final aggregated match dataset
│   ├── Aggregate_epl.csv                # English Premier League
│   ├── Aggregate_bundes.csv             # Bundesliga
│   ├── Aggregate_laliga.csv             # La Liga (Tier 1)
│   ├── Aggregate_laliga2.csv            # La Liga 2 (Tier 2)
│   ├── Aggregate_ligue1.csv             # French Ligue 1
│   ├── Aggregate_mls.csv                # MLS (North America)
│   ├── Aggregate_eu.csv                 # European Competitions (UEFA)
│   ├── Aggregate_wc.csv                 # FIFA World Cup
│   ├── Aggregate_wwc.csv                # Women's World Cup
│   ├── Aggregate_wspl.csv               # Women's Super League
│   ├── Aggregate_caf.csv                # African Competitions (CAF)
│   ├── Aggregate_copa.csv               # Copa América
│   └── Merge and Clean.ipynb            # Aggregation pipeline notebook
│
├── 📥 Data Process/                     # Data preparation pipelines
│   ├── Data Extraction.ipynb            # Extract raw StatsBomb data
│   ├── Data Preprocess.ipynb            # Clean & preprocess features
│   └── Merged_Aggregate.csv             # Combined aggregated dataset
│
├── 📚 Raw Data/                         # Raw StatsBomb event & match data
│   ├── statsbomb_events_*.csv           # 12 files with event-level data (~2.9 GB)
│   └── statsbomb_matches_*.csv          # 12 files with match metadata (~0.2 GB)
│   └── [Note: Large files are git-ignored]
│
├── 🤖 ML/                               # Machine learning development
│   ├── Models.ipynb                     # Main model training & evaluation
│   ├── Baseline_Rating_Comparison.ipynb # Baseline model comparisons
│   ├── Generalization_Evaluation.ipynb  # Cross-validation & LOCO testing
│   ├── Other tests.ipynb                # Experimental ablation studies
│   ├── Final_Data.csv                   # Feature-engineered training dataset
│   └── *.png                            # Visualization outputs
│       ├── feature_importance_catboost.png
│       ├── baseline_comparison.png
│       ├── ablation_study.png
│       ├── draw_analysis.png
│       ├── confusion_matrix_best_fold.png
│       └── loco_generalization_results.png
│
├── 🚀 Deployment/                       # Production application
│   ├── app.py                           # Streamlit web interface
│   ├── engine.py                        # Core prediction & analysis engine
│   ├── sim2win_catboost.cbm             # Trained CatBoost model (primary)
│   ├── sim2win_xgb.json                 # XGBoost model (legacy)
│   ├── sim2win_kmeans.pkl               # K-Means tactical clustering model
│   ├── sim2win_scaler.pkl               # Feature scaling/normalization
│   ├── sim2win_columns.pkl              # Feature column names & order
│   ├── venv.txt                         # Virtual environment dependencies
│   └── .streamlit/secrets.toml          # Gemini API configuration
│
├── 📖 Documentation/                    # Project documentation
│   ├── Capstone_Interimreport_Sim2Win.docx
│   ├── Sim2Win Project Proposal.pdf
│   ├── Technical process.docx
│   ├── Papers Capstone.docx
│   └── HC/                              # Honors College materials
│       ├── CapHC_Proposal_Sim2Win.docx
│       └── Sim2Win CapHC Poster.pdf
│
├── 📑 Root Files
│   ├── Codm.csv                         # Team code mapping
│   ├── Mas.csv                          # Master data reference
│   ├── README.md                        # This file
│   └── .gitignore                       # Git exclusions
│
└── .git/                                # Version control history

```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (venv or conda)

### Step 1: Clone Repository
```bash
git clone https://github.com/Mouad-Ze/SIM2WIN.git
cd SIM2WIN
```

### Step 2: Set Up Python Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda (alternative)
conda create -n sim2win python=3.9
conda activate sim2win
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- **Data Processing**: pandas, numpy, scikit-learn
- **ML Models**: catboost, xgboost, shap
- **Deployment**: streamlit
- **AI Integration**: google-generativeai (for Gemini)
- **Visualization**: matplotlib, seaborn, plotly

### Step 4: Configure Gemini API (Optional - for AI reports)
Create `.streamlit/secrets.toml`:
```toml
[gemini]
api_key = "your_gemini_api_key_here"
```

---

## 🎮 Usage

### Interactive Web Application

```bash
cd Deployment
streamlit run app.py
```

**How to Use:**
1. Upload your team's last 5 recent match data (CSV format)
2. Select opponent team
3. Click "Analyze & Recommend"
4. View:
   - Win/Draw/Loss probability
   - Recommended tactical formation
   - Exploitable weaknesses
   - AI-generated tactical brief

### Python API (Programmatic Usage)

```python
from engine import Sim2WinEngine
import pandas as pd

# Initialize engine
engine = Sim2WinEngine()

# Load your team's recent data
team_stats = pd.read_csv('your_team_stats.csv')

# Get predictions
result = engine.predict_match(
    home_team_stats=team_stats,
    away_team_stats=opponent_stats
)

print(f"Win Probability: {result['win_prob']:.2%}")
print(f"Recommended Tactic: {result['recommended_tactic']}")
print(f"Key Vulnerability: {result['key_vulnerability']}")
```

### Jupyter Notebooks

**Data Pipeline:**
1. `Data Process/Data Extraction.ipynb` - Extract raw StatsBomb data
2. `Data Process/Data Preprocess.ipynb` - Clean and aggregate
3. `Agg/Merge and Clean.ipynb` - Create final aggregated dataset

**Model Development:**
1. `ML/Models.ipynb` - Train CatBoost & K-Means models
2. `ML/Baseline_Rating_Comparison.ipynb` - Compare with baselines
3. `ML/Generalization_Evaluation.ipynb` - LOCO cross-validation
4. `ML/Other tests.ipynb` - Ablation & sensitivity studies

---

## 📊 Data Pipeline

### Architecture Overview

```
StatsBomb Raw Data (3.8 GB, 24 files)
├── 12 Event files (2.9 GB) - Granular play-by-play data
└── 12 Match files (0.2 GB) - Match metadata

    ↓ [Data Extraction]

Raw Extracted Features
(passes, shots, xG, pressures, etc.)

    ↓ [Data Preprocessing]

Aggregated Stats by League
(12 CSV files in Agg/)

    ↓ [Feature Engineering]

Final Training Dataset (630 KB)
30+ engineered features per match

    ↓ [Model Training]

Trained Models
├── CatBoost Classifier
├── K-Means Clustering
└── SHAP Explainer
```

### Data Coverage

| League | File Size | Feature Count | Match Count |
|--------|-----------|---------------|-------------|
| EPL | 89.0 KB | 30+ | 380 per season |
| Bundesliga | 121.0 KB | 30+ | 306 per season |
| La Liga | 31.7 KB + 98.7 KB | 30+ | 380+450 per season |
| Ligue 1 | 16.2 KB | 30+ | 380 per season |
| MLS | 2.6 KB | 30+ | ~300 per season |
| World Cup | 21.7 KB | 30+ | 64 games per tournament |
| Women's Competitions | 88.2 KB + 34.2 KB | 30+ | Varied |
| Other | 97.1 KB | 30+ | Varied |

**Total**: 12+ leagues, ~3,870 matches, 3.8 GB raw data

---

## 🤖 Models & Architecture

### Primary Model: CatBoost Classifier

```
Input Features (30+)
├── Possession Metrics
├── Attacking Metrics (shots, xG, conversion rate)
├── Defensive Metrics (pressures, interceptions)
├── Passing Patterns (completion %, directness)
├── Temporal Metrics (momentum, volatility)
└── Comparative Ratios

    ↓

CatBoost Gradient Boosting
├── 500-1000 trees
├── Categorical feature handling
├── SHAP value output
└── Multi-class logit loss

    ↓

Output Predictions
├── P(Home Win): 0-100%
├── P(Draw): 0-100%
├── P(Away Win): 0-100%
└── Feature Importance Scores
```

### Secondary Model: K-Means Clustering

- **Purpose**: Identify distinct tactical profiles
- **Number of Clusters**: 8 tactical archetypes
- **Features**: Aggregated team statistics normalized
- **Application**: Tactical recommendation engine

### Explainability: SHAP

- **Method**: SHapley Additive exPlanations
- **Features**: Force plots, summary plots, dependence plots
- **Integration**: Real-time feature importance in Streamlit app
- **Usage**: Transparency for coaches & analysts

---

## 📈 Results & Performance

### Model Validation

**Cross-Validation Strategy**: Leave-One-Competition-Out (LOCO)
- Train on 11 leagues, test on 1 (repeated 12 times)
- Ensures generalization across diverse competition styles
- Accounts for league-specific biases

### Key Performance Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | ~58-62% |
| Balanced Accuracy | ~55-60% |
| Precision (Home Win) | ~70% |
| Precision (Draw) | ~45-50% |
| Precision (Away Win) | ~68% |
| AUC-ROC | ~0.68-0.72 |

*See `ML/` notebooks for detailed results and visualizations*

### Feature Importance (Top 5)

1. **Home Team Recent Form** (Momentum index)
2. **Expected Goals Differential** (xG difference)
3. **Defensive Pressure Efficiency** (Pressures/Match)
4. **Passing Completion Variance** (Match-to-match consistency)
5. **Shot Quality Index** (xG per shot)

---

## 🚀 Deployment

### Streamlit Web Application

**File**: `Deployment/app.py`

**Features:**
- User-friendly interface for match analysis
- Real-time prediction with confidence intervals
- Tactical recommendation with simulation results
- AI-generated tactical briefs (Gemini integration)
- SHAP feature importance visualizations
- Historical prediction tracking

**Run Application:**
```bash
cd Deployment
streamlit run app.py
```

### Production Models

| Model | Filename | Size | Purpose |
|-------|----------|------|---------|
| CatBoost | `sim2win_catboost.cbm` | 438.7 KB | Primary classifier |
| XGBoost | `sim2win_xgb.json` | 1.9 MB | Legacy/comparison |
| K-Means | `sim2win_kmeans.pkl` | 7.2 KB | Clustering |
| Scaler | `sim2win_scaler.pkl` | 2.6 KB | Feature normalization |
| Columns | `sim2win_columns.pkl` | 0.9 KB | Feature metadata |

### Inference Engine

**File**: `Deployment/engine.py`

**Core Class**: `Sim2WinEngine`

```python
class Sim2WinEngine:
    def __init__(self)
    def predict_match(home_stats, away_stats) -> dict
    def recommend_tactic(home_stats, away_stats) -> dict
    def simulate_all_tactics(home_stats, away_stats) -> dict
    def get_shap_explanation(features) -> pd.DataFrame
    def generate_ai_report(prediction_result) -> str
```

---

## 💻 Technical Stack

### Data Science & ML
- **Pandas**: Data manipulation & aggregation
- **NumPy**: Numerical computations
- **Scikit-learn**: Preprocessing & evaluation metrics
- **CatBoost**: Gradient boosting classifier
- **XGBoost**: Alternative classifier (legacy)
- **SHAP**: Model explainability
- **Scipy**: Statistical testing

### Deployment & Visualization
- **Streamlit**: Web interface framework
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Static plots

### AI & APIs
- **Google Generative AI** (Gemini): Tactical report generation

### Version Control & Development
- **Git & GitHub**: Repository management
- **Jupyter Notebooks**: Development & experimentation
- **Python 3.8+**: Primary language


## 👤 Author

**Mouad-Ze**  
*Capstone Project - Tactical Football Analysis System*  

**GitHub**: [@Mouad-Ze](https://github.com/Mouad-Ze)  
**Repository**: [SIM2WIN](https://github.com/Mouad-Ze/SIM2WIN)


### Made with ⚽ & 🤖 for Football Analytics

**Last Updated**: April 2026

</div>
