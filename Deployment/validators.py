"""
Input Validation Module for Sim2Win
Validates CSV files and ensures data integrity
"""

import pandas as pd
from typing import Tuple
from config import CSV_REQUIRED_COLUMNS
from logger import setup_logger

logger = setup_logger(__name__)


def validate_csv(df: pd.DataFrame, filename: str = "input") -> Tuple[bool, str]:
    """
    Validate CSV file structure and content.
    
    Args:
        df: Pandas DataFrame to validate
        filename: Name of file for error reporting
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            return False, f"{filename} is empty"
        
        # Convert columns to lowercase for consistency
        df.columns = df.columns.str.lower()
        
        # Check for required columns (at least some must be present)
        missing_cols = CSV_REQUIRED_COLUMNS - set(df.columns)
        has_data = len(CSV_REQUIRED_COLUMNS - missing_cols) >= 5  # At least 5 of 7 required
        
        if not has_data:
            return False, f"{filename} missing essential columns. Found: {set(df.columns)}"
        
        # Check for numeric values in required columns
        for col in CSV_REQUIRED_COLUMNS:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return False, f"Column '{col}' in {filename} contains non-numeric values"
        
        # Check for NaN values in required columns
        numeric_cols = df.select_dtypes(include='number').columns
        if numeric_cols.empty:
            return False, f"{filename} has no numeric columns"
        
        logger.info(f"✓ Validation passed for {filename}: {len(df)} rows, {len(df.columns)} columns")
        return True, ""
        
    except Exception as e:
        error_msg = f"Validation error in {filename}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def validate_and_load_csv(uploaded_file) -> Tuple[pd.DataFrame | None, str]:
    """
    Load and validate uploaded CSV file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (DataFrame or None, error_message)
    """
    try:
        if uploaded_file is None:
            return None, "No file provided"
        
        df = pd.read_csv(uploaded_file)
        is_valid, error_msg = validate_csv(df, uploaded_file.name)
        
        if not is_valid:
            return None, error_msg
        
        return df, ""
        
    except pd.errors.ParserError as e:
        error_msg = f"CSV parsing error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error loading CSV: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def validate_feature_columns(features_available: set, features_required: set) -> Tuple[bool, list]:
    """
    Check if all required feature columns are available.
    
    Args:
        features_available: Set of available features
        features_required: Set of required features
        
    Returns:
        Tuple of (all_present, missing_features)
    """
    missing = features_required - features_available
    return len(missing) == 0, list(missing)
