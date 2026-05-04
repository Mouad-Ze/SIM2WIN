"""
Sim2Win ML Inference Engine
Core prediction and tactical analysis engine for football match outcomes and recommendations
"""

from typing import Tuple, Any
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import shap
import warnings
import google.generativeai as genai
from config import (
    TACTICAL_ARCHETYPES, KMEANS_FEATURES, NON_ROLLING_FEATURES,
    FEATURE_ENGINEERING_DEFAULTS, GEMINI_CONFIG, DEFAULT_FORMATION
)
from logger import setup_logger

warnings.filterwarnings('ignore', category=UserWarning)

logger = setup_logger(__name__)


class Sim2WinEngine:
    """
    Core ML inference engine for Sim2Win tactical analysis system.
    
    Handles:
    - Team feature engineering from raw match statistics
    - Tactical profile identification via K-Means clustering
    - Match outcome predictions using CatBoost
    - Matchup simulation across all 8 tactical archetypes
    - AI-powered tactical report generation using Gemini
    """
    
    def __init__(self, kmeans_model: Any, cat_model: CatBoostClassifier, scaler: Any, feature_columns: list, api_key: str) -> None:
        """
        Initialize Sim2Win Engine.
        
        Args:
            kmeans_model: Fitted K-Means clustering model
            cat_model: Fitted CatBoost classifier
            scaler: StandardScaler for feature normalization
            feature_columns: List of feature column names in training order
            api_key: Google Generative AI API key for Gemini
            
        Raises:
            ValueError: If api_key is invalid or empty
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Invalid API key. Must provide valid Gemini API key.")
        
        self.kmeans = kmeans_model
        self.cat_model = cat_model
        self.scaler = scaler
        self.feature_columns = feature_columns
        
        try:
            self.explainer = shap.TreeExplainer(self.cat_model)
            logger.info("SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning(f"SHAP initialization warning: {e}")
            self.explainer = None
        
        try:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel(GEMINI_CONFIG['model_name'])
            logger.info("Gemini API configured")
        except Exception as e:
            logger.error(f"Gemini API initialization failed: {e}")
            self.llm = None
        
        self.tactic_names = TACTICAL_ARCHETYPES

    def _engineer_features(self, raw_team_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced tactical features from raw team statistics.
        
        Computes derived metrics:
        - Pressing Efficiency: Interceptions / Pressures
        - Shot Quality: xG / Shots
        - Directness Index: Passes / Possession Events
        - Chaos Index: Ball Recoveries + Interceptions
        - xG Volatility: Standard deviation of xG across matches
        - xG Momentum: Recent performance vs historical average
        
        Args:
            raw_team_data: DataFrame with raw match statistics
            
        Returns:
            Processed feature DataFrame (1 row)
            
        Raises:
            Exception: If feature engineering fails
        """
        try:
            raw_team_data.columns = raw_team_data.columns.str.lower()
            numeric_data = raw_team_data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                logger.warning("No numeric data found in raw_team_data")
                raise ValueError("No numeric columns in team data")
            
            # Base statistics (mean across all matches)
            rolling_stats = numeric_data.mean().to_dict()
            
            # Derived metrics with safe division
            epsilon = FEATURE_ENGINEERING_DEFAULTS['epsilon']
            rolling_stats['pressing_efficiency'] = rolling_stats.get('interceptions', 0) / (rolling_stats.get('pressures', 0) + epsilon)
            rolling_stats['shot_quality'] = rolling_stats.get('xg', 0) / (rolling_stats.get('shots', 0) + epsilon)
            rolling_stats['directness_index'] = rolling_stats.get('passes', 0) / (rolling_stats.get('possession_events', 0) + epsilon)
            rolling_stats['chaos_index'] = rolling_stats.get('ball_recoveries', 0) + rolling_stats.get('interceptions', 0)
            
            # Volatility and momentum metrics
            if len(numeric_data) > 1:
                rolling_stats['xg_volatility'] = numeric_data['xg'].std() if 'xg' in numeric_data else 0
                rolling_stats['pressures_volatility'] = numeric_data['pressures'].std() if 'pressures' in numeric_data else 0
                rolling_stats['xg_momentum'] = numeric_data['xg'].iloc[-1] - numeric_data['xg'].mean() if 'xg' in numeric_data else 0
            else:
                rolling_stats['xg_volatility'] = 0
                rolling_stats['pressures_volatility'] = 0
                rolling_stats['xg_momentum'] = 0
            
            # Default days_rest if not provided
            if 'days_rest' not in rolling_stats:
                rolling_stats['days_rest'] = FEATURE_ENGINEERING_DEFAULTS['days_rest_default']
            
            logger.debug(f"Features engineered: {len(rolling_stats)} features computed")
            return pd.DataFrame([rolling_stats])
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

    def get_tactical_profile(self, team_data: pd.DataFrame) -> Tuple[int, str]:
        """
        Identify team's tactical profile via K-Means clustering.
        
        Args:
            team_data: DataFrame with team match statistics
            
        Returns:
            Tuple of (cluster_id, tactic_name)
            
        Raises:
            ValueError: If team_data is invalid
        """
        try:
            processed = self._engineer_features(team_data)
            
            if processed.empty:
                raise ValueError("Could not process team data")
            
            cluster_features = processed[KMEANS_FEATURES]
            cluster_id = int(self.kmeans.predict(cluster_features.values)[0])
            tactic_name = self.tactic_names.get(cluster_id, "Unknown Tactic")
            
            logger.info(f"Tactical profile identified: {tactic_name} (Cluster {cluster_id})")
            return cluster_id, tactic_name
            
        except Exception as e:
            logger.error(f"Tactical profiling failed: {e}")
            raise

    def _generate_coach_report(
        self,
        tactic_name: str,
        top_driver: str,
        win_prob: float,
        usual_prob: float,
        draw_prob: float,
        loss_prob: float,
        home_form: str,
        away_form: str
    ) -> str:
        """
        Generate AI-powered tactical report using Gemini.
        
        Args:
            tactic_name: Recommended tactical archetype
            top_driver: Top feature influencing the prediction
            win_prob: Predicted win probability (%)
            usual_prob: Win probability with usual tactic (%)
            draw_prob: Predicted draw probability (%)
            loss_prob: Predicted loss probability (%)
            home_form: Home team's usual formation
            away_form: Away team's usual formation
            
        Returns:
            Professional tactical brief (markdown formatted)
        """
        try:
            if self.llm is None:
                logger.warning("Gemini API not available, returning fallback report")
                return self._fallback_report(tactic_name, top_driver, win_prob, usual_prob)
            
            edge = win_prob - usual_prob
            prompt = f"""
        You are an elite, professional football tactical analyst writing a briefing for the Head Coach and the coaching staff.
        
        Here is the tactical assessment from the SIM2WIN engine for the upcoming match:
        - Opponent's Usual Formation: {away_form}
        - Our Usual Formation: {home_form}
        - Win Probability if we use our usual tactic: {usual_prob}%
        
        SIM2WIN highly recommends we pivot to a "{tactic_name}" tactical setup. 
        If we make this structural adjustment, the probabilities shift to:
        - Win: {win_prob}% (A tactical edge of {edge:.1f}%)
        - Draw: {draw_prob}%
        - Loss: {loss_prob}%
        
        The simulation data indicates that the primary reason this tactic works is because it specifically exploits this opponent vulnerability: "{top_driver}".
        
        Write a professional, 3-paragraph tactical dossier explaining this recommendation to the coaching staff. 
        
        CRITICAL INSTRUCTIONS:
        - Tone: Analytical, objective, confident, and highly tactical. Speak the language of football.
        - Do NOT sound like a data scientist or mathematician. Do not mention algorithms, AI, or math. 
        - Explain tactically why exploiting '{top_driver}' against a '{away_form}' using a '{tactic_name}' makes sense on the pitch. 
        - Attribute the recommendation explicitly to SIM2WIN.
        - Do NOT include a title, heading, or subject line at the top of your response. Start immediately with the first paragraph.
        - Use markdown formatting (bolding, bullet points) to make it highly readable.
        - Keep it concise but detailed enough for assistant coaches to understand the tactical nuances.
            """
            
            response = self.llm.generate_content(prompt)
            logger.info("Tactical report generated via Gemini")
            return response.text
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return self._fallback_report(tactic_name, top_driver, win_prob, usual_prob)

    @staticmethod
    def _fallback_report(tactic_name: str, top_driver: str, win_prob: float, usual_prob: float) -> str:
        """Fallback report when Gemini API is unavailable."""
        edge = win_prob - usual_prob
        return f"""
        **SIM2WIN Tactical Recommendation**
        
        **Recommended Tactic:** {tactic_name}
        
        **Win Probability Gain:** {edge:.1f}% improvement (from {usual_prob}% → {win_prob}%)
        
        **Key Exploit:** The recommended formation specifically targets the opponent's weakness in {top_driver}, providing a significant tactical advantage.
        
        *Note: Full tactical analysis unavailable. Gemini API not configured.*
        """


    def simulate_matchup(
        self,
        home_team_data: pd.DataFrame,
        away_team_data: pd.DataFrame,
        current_home_tactic: int
    ) -> Tuple[pd.DataFrame, str]:
        """
        Simulate all 8×8 tactical combinations and provide recommendation.
        
        Args:
            home_team_data: Home team statistics DataFrame
            away_team_data: Away team statistics DataFrame
            current_home_tactic: Home team's current tactical cluster ID
            
        Returns:
            Tuple of (results_DataFrame, coach_report)
            
        Raises:
            ValueError: If team data is invalid
        """
        try:
            # Get team formations
            home_form = (
                home_team_data['starting_formation'].mode()[0]
                if 'starting_formation' in home_team_data.columns
                else DEFAULT_FORMATION
            )
            away_form = (
                away_team_data['starting_formation'].mode()[0]
                if 'starting_formation' in away_team_data.columns
                else DEFAULT_FORMATION
            )
            
            # Process team data
            away_processed = self._engineer_features(away_team_data)
            away_cluster_id, _ = self.get_tactical_profile(away_team_data)
            home_base_processed = self._engineer_features(home_team_data)
            
            results = []
            
            # Simulate each home tactical variant
            for tactic_id in range(8):
                match_simulation = pd.DataFrame(index=[0])
                
                # Add away team features
                for col in away_processed.columns:
                    suffix = '' if col in NON_ROLLING_FEATURES else '_rolling'
                    match_simulation[f'away_{col}{suffix}'] = away_processed.loc[0, col]
                match_simulation['away_Tactical_Cluster'] = away_cluster_id
                
                # Add home team features (with tested tactic)
                for col in home_base_processed.columns:
                    suffix = '' if col in NON_ROLLING_FEATURES else '_rolling'
                    match_simulation[f'home_{col}{suffix}'] = home_base_processed.loc[0, col]
                match_simulation['home_Tactical_Cluster'] = tactic_id
                
                # Prepare for model
                match_simulation = pd.get_dummies(match_simulation)
                match_simulation = match_simulation.reindex(
                    columns=self.feature_columns, fill_value=0
                )
                X_sim_scaled = self.scaler.transform(match_simulation)
                
                # Predict outcome
                probs = self.cat_model.predict_proba(X_sim_scaled)[0]
                win_prob = round(float(probs[2] * 100), 2)
                draw_prob = round(float(probs[1] * 100), 2)
                loss_prob = round(float(probs[0] * 100), 2)
                
                # Extract feature importance (top driver)
                top_driver = "Match Dynamics"  # Default
                try:
                    if self.explainer is not None:
                        shap_values = self.explainer.shap_values(X_sim_scaled)
                        if isinstance(shap_values, list):
                            win_shap_array = shap_values[2][0]
                        else:
                            if len(shap_values.shape) == 3:
                                win_shap_array = shap_values[0, :, 2]
                            else:
                                win_shap_array = shap_values[0]
                        
                        feature_impacts = pd.Series(
                            win_shap_array, index=self.feature_columns
                        )
                        top_driver = str(feature_impacts.sort_values(ascending=False).index[0])
                except Exception as e:
                    logger.warning(f"Could not extract SHAP values: {e}")
                
                results.append({
                    "Tactic ID": tactic_id,
                    "Tactic Name": self.tactic_names[tactic_id],
                    "Win Prob": win_prob,
                    "Draw Prob": draw_prob,
                    "Loss Prob": loss_prob,
                    "Top Driver": top_driver,
                })
            
            # Generate results and recommendation
            results_df = pd.DataFrame(results).sort_values(
                by="Win Prob", ascending=False
            ).reset_index(drop=True)
            
            best_tactic = results_df.iloc[0]
            try:
                usual_tactic_prob = results_df[
                    results_df["Tactic ID"] == current_home_tactic
                ]["Win Prob"].values[0]
            except IndexError:
                usual_tactic_prob = results_df.iloc[0]["Win Prob"]
            
            # Generate tactical report
            coach_report = self._generate_coach_report(
                tactic_name=best_tactic["Tactic Name"],
                top_driver=best_tactic["Top Driver"],
                win_prob=best_tactic["Win Prob"],
                usual_prob=usual_tactic_prob,
                draw_prob=best_tactic["Draw Prob"],
                loss_prob=best_tactic["Loss Prob"],
                home_form=home_form,
                away_form=away_form
            )
            
            logger.info("Matchup simulation completed successfully")
            return results_df, coach_report
            
        except Exception as e:
            logger.error(f"Matchup simulation failed: {e}")
            raise