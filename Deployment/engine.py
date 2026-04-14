import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import shap
import warnings
import google.generativeai as genai

warnings.filterwarnings('ignore', category=UserWarning)

class Sim2WinEngine:
    def __init__(self, kmeans_model, cat_model, scaler, feature_columns, api_key):
        self.kmeans = kmeans_model
        self.cat_model = cat_model # Swapped to CatBoost
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.explainer = shap.TreeExplainer(self.cat_model) # Swapped to CatBoost
        
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        
        self.tactic_names = {
            0: "High-Pressing Possession", 1: "Low-Block Counter",
            2: "Mid-Block Transition", 3: "Direct Long Ball",
            4: "Tiki-Taka", 5: "Wing-Play Overload",
            6: "High-Intensity Gegenpress", 7: "Park the Bus"
        }

    def _engineer_features(self, raw_team_data):
        raw_team_data.columns = raw_team_data.columns.str.lower()
        numeric_data = raw_team_data.select_dtypes(include=[np.number])
        
        # 1. Base Rolling Averages
        rolling_stats = numeric_data.mean().to_dict()
        
        # 2. Advanced Feature Recreation (THE BUG FIX)
        # We must calculate these here so K-Means doesn't crash on raw CSVs
        rolling_stats['pressing_efficiency'] = rolling_stats.get('interceptions', 0) / (rolling_stats.get('pressures', 0) + 1e-5)
        rolling_stats['shot_quality'] = rolling_stats.get('xg', 0) / (rolling_stats.get('shots', 0) + 1e-5)
        
        # Recreating the custom indexes 
        rolling_stats['directness_index'] = rolling_stats.get('passes', 0) / (rolling_stats.get('possession_events', 0) + 1e-5)
        rolling_stats['chaos_index'] = rolling_stats.get('ball_recoveries', 0) + rolling_stats.get('interceptions', 0)
        
        # Volatility requires calculating the Standard Deviation across the 5 rows
        if len(numeric_data) > 1:
            rolling_stats['xg_volatility'] = numeric_data['xg'].std() if 'xg' in numeric_data else 0
            rolling_stats['pressures_volatility'] = numeric_data['pressures'].std() if 'pressures' in numeric_data else 0
            rolling_stats['xg_momentum'] = numeric_data['xg'].iloc[-1] - numeric_data['xg'].mean() if 'xg' in numeric_data else 0
        else:
            rolling_stats['xg_volatility'] = 0
            rolling_stats['pressures_volatility'] = 0
            rolling_stats['xg_momentum'] = 0
            
        # Hardcoding days_rest if it isn't in the CSV
        if 'days_rest' not in rolling_stats:
            rolling_stats['days_rest'] = 7 
            
        return pd.DataFrame([rolling_stats])

    def get_tactical_profile(self, team_data):
        processed = self._engineer_features(team_data)
        
        kmeans_features = [
            'passes', 'shots', 'xg', 'pressures', 'ball_recoveries', 
            'interceptions', 'possession_events', 'pressing_efficiency', 
            'shot_quality', 'directness_index', 'chaos_index', 
            'xg_volatility', 'pressures_volatility'
        ]
        
        cluster_features = processed[kmeans_features]
        cluster_id = self.kmeans.predict(cluster_features.values)[0]
        return cluster_id, self.tactic_names[cluster_id]

    # ... [Keep your _generate_coach_report exactly the same] ...
    def _generate_coach_report(self, tactic_name, top_driver, win_prob, usual_prob, draw_prob, loss_prob, home_form, away_form):
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
        - The language should be simple and direct enough for the Head Coach to quickly grasp the key insights, but also detailed enough for the assistant coaches to understand the tactical nuances.
        """
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"**System Error:** SIM2WIN failed to generate report. Raw data: Switch to **{tactic_name}** for a **{win_prob}%** win probability driven by exploiting **{top_driver}**. (Error: {str(e)})"


    def simulate_matchup(self, home_team_data, away_team_data, current_home_tactic):
        home_form = home_team_data['starting_formation'].mode()[0] if 'starting_formation' in home_team_data.columns else "Standard"
        away_form = away_team_data['starting_formation'].mode()[0] if 'starting_formation' in away_team_data.columns else "Standard"

        away_processed = self._engineer_features(away_team_data)
        away_cluster_id, _ = self.get_tactical_profile(away_team_data)
        home_base_processed = self._engineer_features(home_team_data)
        
        results = []
        non_rolling = ['days_rest', 'xg_momentum', 'xg_volatility', 'pressures_volatility']
        
        for tactic_id in range(8):
            match_simulation = pd.DataFrame(index=[0])
            
            for col in away_processed.columns:
                suffix = '' if col in non_rolling else '_rolling'
                match_simulation[f'away_{col}{suffix}'] = away_processed.loc[0, col]
            match_simulation['away_Tactical_Cluster'] = away_cluster_id
            
            for col in home_base_processed.columns:
                suffix = '' if col in non_rolling else '_rolling'
                match_simulation[f'home_{col}{suffix}'] = home_base_processed.loc[0, col]
            match_simulation['home_Tactical_Cluster'] = tactic_id
            
            match_simulation = pd.get_dummies(match_simulation)
            match_simulation = match_simulation.reindex(columns=self.feature_columns, fill_value=0)
            X_sim_scaled = self.scaler.transform(match_simulation)
            
            # Use CatBoost predict_proba
            probs = self.cat_model.predict_proba(X_sim_scaled)[0]
            win_prob = round(probs[2] * 100, 2)
            draw_prob = round(probs[1] * 100, 2)
            loss_prob = round(probs[0] * 100, 2)
            
            shap_values = self.explainer.shap_values(X_sim_scaled)
            if isinstance(shap_values, list):
                win_shap_array = shap_values[2][0]
            else:
                if len(shap_values.shape) == 3:
                    win_shap_array = shap_values[0, :, 2]
                else:
                    win_shap_array = shap_values[0]
                    
            feature_impacts = pd.Series(win_shap_array, index=self.feature_columns)
            top_driver = feature_impacts.sort_values(ascending=False).index[0]
            
            results.append({
                "Tactic ID": tactic_id,
                "Tactic Name": self.tactic_names[tactic_id],
                "Win Prob": win_prob,
                "Draw Prob": draw_prob,
                "Loss Prob": loss_prob,
                "Top Driver": top_driver,
            })
            
        results_df = pd.DataFrame(results).sort_values(by="Win Prob", ascending=False).reset_index(drop=True)
        
        best_tactic = results_df.iloc[0]
        usual_tactic_prob = results_df[results_df["Tactic ID"] == current_home_tactic]["Win Prob"].values[0]
        
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
        
        return results_df, coach_report