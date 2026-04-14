# Sim2Win Deployment Configuration
# ==============================================
# Centralized constants and configuration settings

# Model Files Configuration
MODEL_CONFIG = {
    'catboost_model': 'sim2win_catboost.cbm',
    'kmeans_model': 'sim2win_kmeans.pkl',
    'scaler_model': 'sim2win_scaler.pkl',
    'feature_columns': 'sim2win_columns.pkl',
    'xgb_model': 'sim2win_xgb.json',  # Legacy
}

# Tactical Archetypes (K-Means Clusters)
TACTICAL_ARCHETYPES = {
    0: "High-Pressing Possession",
    1: "Low-Block Counter",
    2: "Mid-Block Transition",
    3: "Direct Long Ball",
    4: "Tiki-Taka",
    5: "Wing-Play Overload",
    6: "High-Intensity Gegenpress",
    7: "Park the Bus"
}

# K-Means Feature Set
KMEANS_FEATURES = [
    'passes', 'shots', 'xg', 'pressures', 'ball_recoveries',
    'interceptions', 'possession_events', 'pressing_efficiency',
    'shot_quality', 'directness_index', 'chaos_index',
    'xg_volatility', 'pressures_volatility'
]

# Non-rolling features (don't apply rolling suffix)
NON_ROLLING_FEATURES = ['days_rest', 'xg_momentum', 'xg_volatility', 'pressures_volatility']

# Feature Engineering Defaults
FEATURE_ENGINEERING_DEFAULTS = {
    'days_rest_default': 7,
    'epsilon': 1e-5,  # To avoid division by zero
}

# Gemini AI Configuration
GEMINI_CONFIG = {
    'model_name': 'gemini-2.5-flash',
    'temperature': 0.7,
    'max_tokens': 1000,
}

# Default Formation (if not provided in data)
DEFAULT_FORMATION = "Standard 4-3-3"

# CSV Column Requirements (for input validation)
CSV_REQUIRED_COLUMNS = {
    'passes', 'shots', 'xg', 'pressures', 'ball_recoveries',
    'interceptions', 'possession_events'
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'Sim2Win Tactical Dashboard',
    'page_icon': '⚽',
    'layout': 'wide',
    'initial_sidebar_state': 'collapsed',
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'sim2win.log',
}

# API Configuration
API_CONFIG = {
    'max_retries': 3,
    'timeout': 30,  # seconds
}
