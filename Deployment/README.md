# Deployment Configuration

This folder contains the Streamlit web application for Sim2Win.

## Setup Instructions

### 1. Configure Secrets

The application requires a Gemini API key to generate tactical reports:

```bash
cp Deployment/.streamlit/secrets.example.toml Deployment/.streamlit/secrets.toml
```

Edit `secrets.toml` and add your Gemini API key:

```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

**Important:** Never commit `secrets.toml` to version control. It contains sensitive credentials.

### 2. Run the Application

```bash
streamlit run Deployment/app.py
```

The app will open in your browser at `http://localhost:8501`

## Application Structure

- `app.py` - Main Streamlit web interface
- `engine.py` - Core ML inference engine with tactical analysis
- `config.py` - Centralized configuration and constants
- `logger.py` - Logging setup and utilities
- `validators.py` - Input validation for CSV files
- `sim2win_*.cbm/.json/.pkl` - Trained ML models (CatBoost, XGBoost, K-Means)
- `.streamlit/config.toml` - Streamlit configuration
- `.streamlit/secrets.toml` - API keys (NOT tracked in git, use secrets.example.toml as template)

## How to Use

1. **Upload Team Data**: Provide 2 CSV files with your team's last 5 matches
2. **Input Format**: Required columns: passes, shots, xG, pressures, ball_recoveries, interceptions, possession_events
3. **Run Simulation**: Click "Run Simulation Engine" to analyze all 8 tactical formations
4. **Results**: View the AI-generated coaching report and probability table for each tactic

## Troubleshooting

- **"API key not configured"**: Make sure `secrets.toml` exists and contains your Gemini API key
- **"Model file not found"**: Ensure all `.cbm`, `.pkl`, and `.json` files are in the Deployment folder
- **"CSV validation error"**: Check that your input CSV has the required columns with numeric values
