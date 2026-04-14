from shap import kmeans
import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from engine import Sim2WinEngine 
import os


def get_file_timestamps():
    timestamps = {}
    for filename in ['sim2win_kmeans.pkl', 'sim2win_scaler.pkl', 'sim2win_columns.pkl', 'sim2win_catboost.cbm']:
        if os.path.exists(filename):
            timestamps[filename] = os.path.getmtime(filename)
        else:
            timestamps[filename] = None
    return timestamps


@st.cache_resource
def load_engine(_timestamp):
    kmeans = joblib.load('sim2win_kmeans.pkl')
    scaler = joblib.load('sim2win_scaler.pkl')
    feature_cols = joblib.load('sim2win_columns.pkl')
    
    # Load CatBoost
    cat_model = CatBoostClassifier()
    cat_model.load_model('sim2win_catboost.cbm')
    
    api_key = st.secrets["GEMINI_API_KEY"]
    
    # Pass cat_model to the engine!
    return Sim2WinEngine(kmeans, cat_model, scaler, feature_cols, api_key)

current_timestamps = get_file_timestamps()
engine = load_engine(current_timestamps)

st.title("Sim2Win Tactical Dashboard")

# --- 2. Data Input ---
col1, col2 = st.columns(2)
with col1:
    user_file = st.file_uploader("Upload Your Team (Last 5 Matches)", type=['csv'])
with col2:
    opp_file = st.file_uploader("Upload Opponent Team (Last 5 Matches)", type=['csv'])

if user_file and opp_file:
    df_user = pd.read_csv(user_file)
    df_opp = pd.read_csv(opp_file)
    
    # --- 3. Tactical Profiling ---
    user_cluster, user_tactic = engine.get_tactical_profile(df_user)
    opp_cluster, opp_tactic = engine.get_tactical_profile(df_opp)
    
    st.subheader("Tactical Matchup")
    st.write(f"**Your Team:** {user_tactic}  |  **Opponent:** {opp_tactic}")
    
    # --- 4. Simulation & Recommendation ---
    if st.button("Run Simulation Engine", type="primary"):
        with st.spinner('Simulating All tactical configurations...'):
            
            # Unpack the two returned items
            results_df, coach_report = engine.simulate_matchup(df_user, df_opp, user_cluster)
            
            st.success("Simulation Complete!")
            
            # Display the Humanized Coaching Report in a nice highlighted box
            st.info(coach_report)
            
            st.divider()
            st.subheader("Underlying Simulation Data")
            
            # Display the dataframe for analysts who want the raw numbers
            st.dataframe(
                results_df[['Tactic Name', 'Win Prob', 'Draw Prob', 'Loss Prob']], 
                use_container_width=True
            )