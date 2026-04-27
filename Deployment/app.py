"""
Sim2Win Tactical Dashboard
Streamlit web interface for football match analysis and tactical recommendations
"""

import streamlit as st
import pandas as pd
import joblib
import os
from typing import Optional, Tuple
from catboost import CatBoostClassifier
from engine import Sim2WinEngine
from validators import validate_and_load_csv, validate_csv
from config import MODEL_CONFIG, STREAMLIT_CONFIG, DEFAULT_FORMATION
from logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=STREAMLIT_CONFIG['initial_sidebar_state']
)

st.title(f"{STREAMLIT_CONFIG['page_icon']} {STREAMLIT_CONFIG['page_title']}")


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
def get_file_timestamps() -> dict:
    """Get modification timestamps for model files for cache invalidation."""
    timestamps = {}
    for filename in MODEL_CONFIG.values():
        if os.path.exists(filename):
            timestamps[filename] = os.path.getmtime(filename)
        else:
            timestamps[filename] = None
    return timestamps


@st.cache_resource
def load_engine(_timestamp: dict) -> Optional[Sim2WinEngine]:
    """
    Load and initialize the Sim2Win engine with all models.
    
    Cache invalidates when model file timestamps change.
    """
    try:
        logger.info("Loading ML models...")
        
        # Load all required models
        kmeans = joblib.load(MODEL_CONFIG['kmeans_model'])
        scaler = joblib.load(MODEL_CONFIG['scaler_model'])
        feature_cols = joblib.load(MODEL_CONFIG['feature_columns'])
        
        # Load CatBoost
        cat_model = CatBoostClassifier()
        cat_model.load_model(MODEL_CONFIG['catboost_model'], format="cbm")
        
        # Get API key
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            logger.error("GEMINI_API_KEY not found in Streamlit secrets")
            st.error("Gemini API key not configured. Some features will be unavailable.")
            api_key = "dummy_key"  # Fallback
        
        engine = Sim2WinEngine(kmeans, cat_model, scaler, feature_cols, api_key)
        logger.info("Engine loaded successfully")
        return engine
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        st.error(f"Required model file not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Engine initialization failed: {e}")
        st.error(f"Failed to load engine: {e}")
        return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main() -> None:
    """Main application logic."""
    
    # Load engine
    timestamps = get_file_timestamps()
    engine = load_engine(timestamps)
    
    if engine is None:
        st.error("Cannot proceed without engine initialization")
        return
    
    st.success("System ready. Upload your team data to begin.")
    st.divider()
    
    # ========================================================================
    # DATA INPUT SECTION
    # ========================================================================
    st.header("Match Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Team")
        user_file = st.file_uploader(
            "Upload Last 5 Matches CSV",
            type=['csv'],
            key="home_team"
        )
    
    with col2:
        st.subheader("Opponent Team")
        opp_file = st.file_uploader(
            "Upload Last 5 Matches CSV",
            type=['csv'],
            key="away_team"
        )
    
    if not user_file or not opp_file:
        st.info("Upload CSV files from both teams to proceed")
        return
    
    # ========================================================================
    # DATA VALIDATION & LOADING
    # ========================================================================
    try:
        df_user, error_msg = validate_and_load_csv(user_file)
        if df_user is None:
            st.error(f"Your Team CSV Error: {error_msg}")
            return
        
        df_opp, error_msg = validate_and_load_csv(opp_file)
        if df_opp is None:
            st.error(f"Opponent CSV Error: {error_msg}")
            return
        
        logger.info(" Both CSV files validated successfully")
        
    except Exception as e:
        logger.error(f"CSV loading error: {e}")
        st.error(f"Error processing files: {e}")
        return
    
    # ========================================================================
    # TACTICAL PROFILING
    # ========================================================================
    st.divider()
    st.header("Tactical Analysis")
    
    try:
        with st.spinner("Analyzing tactical profiles..."):
            user_cluster, user_tactic = engine.get_tactical_profile(df_user)
            opp_cluster, opp_tactic = engine.get_tactical_profile(df_opp)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your Tactic", user_tactic)
        with col2:
            st.metric("Opponent Tactic", opp_tactic)
        
        logger.info(f"Tactical profiles: {user_tactic} vs {opp_tactic}")
        
    except Exception as e:
        logger.error(f"Tactical profiling error: {e}")
        st.error(f"Failed to analyze tactics: {e}")
        return
    
    # ========================================================================
    # MATCHUP SIMULATION & RECOMMENDATION
    # ========================================================================
    st.divider()
    
    if st.button("Run Simulation Engine", type="primary", use_container_width=True):
        try:
            with st.spinner("Simulating all 64 tactical combinations..."):
                results_df, coach_report = engine.simulate_matchup(
                    df_user, df_opp, user_cluster
                )
            
            st.success("Simulation Complete!")
            
            # Display coaching report
            st.header("Tactical Recommendation")
            st.info(coach_report)
            
            # Display results table
            st.divider()
            display_cols = ['Tactic Name', 'Win Prob', 'Draw Prob', 'Loss Prob']
            st.dataframe(
                results_df[display_cols].head(8),
                use_container_width=True,
                height=315
            )
            
            logger.info("Simulation completed and displayed successfully")
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            st.error(f"Simulation failed: {e}\n\nPlease check your data and try again.")


if __name__ == "__main__":
    main()