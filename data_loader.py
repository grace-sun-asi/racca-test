"""
Data Loader — Feature Engineering from feature_columns.json
------------------------------------------------------------
Mirrors the server-side data loading pattern:
1. Load from SQL Server (pyodbc) or CSV
2. Build derived columns (Month_Due, Routed_Late, etc.)
3. Construct binary feature matrix from dummy_feature_columns
4. Add additional_feature_columns as numeric
5. Scale for neural network input

Reads feature column definitions from feature_columns.json.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import pyodbc
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# DATABASE CONNECTION CONFIG (from environment variables)
# =============================================================================

APSYS_PROD_SERVER = os.getenv("APSYS_PROD_SERVER")
APSYS_DEV_SERVER = os.getenv("APSYS_DEV_SERVER")
APSYS_USERNAME = os.getenv("APSYS_USERNAME")
APSYS_PASSWORD = os.getenv("APSYS_PASSWORD")
AVIONICS_USERNAME = os.getenv("AVIONICS_USERNAME")
AVIONICS_PASSWORD = os.getenv("AVIONICS_PASSWORD")

# Database names per model target
DATABASE_MAP = {
    "dccs": os.getenv("APSYS_DCCS_DATABASE"),
    "etac": os.getenv("APSYS_ETAC_DATABASE"),
    "csdt": os.getenv("APSYS_CSDT_DATABASE"),
}

# Max categories to keep per column (top N by frequency, rest -> "Other")
MAX_CATEGORIES_PER_COLUMN = {
    "Assigned_To_Group": 20,
    "Assigned_To_SubGroup": 40,
    "Major_Model": 10,
    "M/M_Change_Type": 12,
    "M/M_Prime_Engineer_Name": 30,
    "Month_Due": 12,
    "COMMODITY_NAME": 30,
    "SR_MANAGER_NAME": 30,
    "IPT_NAME": 20,
    "THREE_LETTER_CODE": 30,
    "SUPPLIER_NAME": 40,
    "CoordGrp": 20,
    "Customer": 15,
}

MAX_SAMPLE_ROWS = 50000


# =============================================================================
# SQL CONNECTION HELPERS
# =============================================================================

def get_connection_string(model_name: str, use_dev: bool = False) -> str:
    """Build pyodbc connection string for the given model target."""
    server = APSYS_DEV_SERVER if use_dev else APSYS_PROD_SERVER
    database = DATABASE_MAP.get(model_name)

    if database is None:
        raise ValueError(
            f"No database configured for model '{model_name}'. "
            f"Set APSYS_{model_name.upper()}_DATABASE in .env"
        )

    conn_str = (
        f"Driver={{ODBC Driver 17 for SQL Server}}; "
        f"Server={server}; Database={database}; "
        f"UID={AVIONICS_USERNAME}; PWD={AVIONICS_PASSWORD}; "
        f"Trusted_Connection=yes"
    )
    return conn_str


def construct_df(query: str, conn) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    Mirrors the server-side construct_df function.
    """
    df = pd.read_sql(query, conn)
    return df


def load_from_sql(
    query: str,
    model_name: str,
    conn_str: str = None,
    use_dev: bool = False,
) -> pd.DataFrame:
    """
    Load data from SQL Server.

    Parameters
    ----------
    query : str
        SQL query to execute.
    model_name : str
        Model target (etac, dccs, csdt) — used to select database.
    conn_str : str or None
        Override connection string. If None, builds from env vars.
    use_dev : bool
        Use dev server instead of prod.

    Returns
    -------
    pd.DataFrame
    """
    if conn_str is None:
        conn_str = get_connection_string(model_name, use_dev=use_dev)

    with pyodbc.connect(conn_str) as conn:
        df = construct_df(query, conn)

    print(f"Loaded {len(df)} rows from SQL Server ({model_name})")
    print(df.head())
    return df


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


# =============================================================================
# DERIVED COLUMN BUILDERS (mirrors server pattern)
# =============================================================================

def build_derived_columns(df: pd.DataFrame, data_source: str) -> pd.DataFrame:
    """
    Add derived columns to the DataFrame based on the data source.
    Mirrors the server-side build_derived_columns function.
    """
    # Normalize column names (replace spaces with underscores)
    df.columns = df.columns.str.replace(' ', '_')

    if data_source == "dccs":
        df = _build_dccs_derived_columns(df)
    elif data_source == "etac":
        df = _build_etac_derived_columns(df)
    elif data_source == "csdt":
        df = _build_csdt_derived_columns(df)

    return df


def _build_dccs_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive DCCS-specific columns."""
    # Month_Due from date column
    if "Monthly_ByDueOrClose" in df.columns:
        df["Monthly_ByDueOrClose"] = pd.to_datetime(df["Monthly_ByDueOrClose"], errors="coerce")
        df["Month_Due"] = df["Monthly_ByDueOrClose"].dt.month_name()

    # Routed_Late binary flag
    if "Complete_Status" in df.columns:
        df["Routed_Late"] = df["Complete_Status"].apply(
            lambda x: 1 if str(x).startswith("Routed Late") else 0
        )

    return df


def _build_etac_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ETAC-specific columns."""
    # Month/Week from due date
    if "ECD" in df.columns:
        df["ECD"] = pd.to_datetime(df["ECD"], errors="coerce")
        df["Month_Due"] = df["ECD"].dt.month_name()
        df["Week_Due"] = df["ECD"].dt.isocalendar().week.astype(str)

    # ECD change count thresholds
    if "ECDChangeCount" in df.columns:
        for threshold in [0, 1, 2, 5, 10]:
            df[f"ECDChangeCount_GT{threshold}"] = (
                df["ECDChangeCount"] > threshold
            ).astype(int)

    # OBDue change count thresholds
    if "OBDueChangeCount" in df.columns:
        for threshold in [0, 1, 2, 5, 10]:
            df[f"OBDueChangeCount_GT{threshold}"] = (
                df["OBDueChangeCount"] > threshold
            ).astype(int)

    # Replace "Delinquent" with "Late" in target
    if "MM_Determination_Completion_Status" in df.columns:
        df["MM_Determination_Completion_Status"] = (
            df["MM_Determination_Completion_Status"].replace("Delinquent", "Late")
        )

    return df


def _build_csdt_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive CSDT-specific columns."""
    # IsSCN binary
    if "SCN_NUMBER" in df.columns and "IsSCN" not in df.columns:
        df["IsSCN"] = df["SCN_NUMBER"].notna().astype(int)

    return df


# =============================================================================
# BINARY FEATURE MATRIX (mirrors server pattern)
# =============================================================================

def build_binary_feature_matrix(
    df: pd.DataFrame,
    dummy_columns: List[str],
    max_categories: Dict[str, int] = None,
) -> pd.DataFrame:
    """
    Build a one-hot encoded binary feature matrix from categorical columns.
    Mirrors the server-side build_binary_feature_matrix.

    High-cardinality columns are capped at top-N categories (by frequency),
    with remaining values grouped into "Other".

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    dummy_columns : list
        Columns to one-hot encode.
    max_categories : dict or None
        Max categories per column. Uses MAX_CATEGORIES_PER_COLUMN if None.

    Returns
    -------
    pd.DataFrame
        Binary feature matrix.
    """
    if max_categories is None:
        max_categories = MAX_CATEGORIES_PER_COLUMN

    frames = []

    for col in dummy_columns:
        if col not in df.columns:
            print(f"WARNING: Column '{col}' not in DataFrame, skipping.")
            continue

        series = df[col].astype(str).fillna("Unknown")

        # Cap high-cardinality columns
        if col in max_categories:
            top_n = max_categories[col]
            top_categories = series.value_counts().head(top_n).index.tolist()
            series = series.where(series.isin(top_categories), other="Other")

        dummies = pd.get_dummies(series, prefix=col)
        frames.append(dummies)

    if frames:
        return pd.concat(frames, axis=1)
    else:
        return pd.DataFrame(index=df.index)


def add_feature_matrix_columns(
    X: pd.DataFrame,
    df: pd.DataFrame,
    additional_columns: List[str],
) -> pd.DataFrame:
    """
    Add numeric/binary columns to the feature matrix.
    Mirrors the server-side add_feature_matrix_columns.
    """
    for col in additional_columns:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        else:
            print(f"WARNING: Additional column '{col}' not found, filling with 0.")
            X[col] = 0

    return X


# =============================================================================
# FEATURE CONFIG LOADING
# =============================================================================

def load_feature_config(config_path: str = "feature_columns.json") -> Dict[str, Any]:
    """Load feature column definitions from JSON."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def get_model_names(config_path: str = "feature_columns.json") -> list:
    """Return available model names (etac, dccs, csdt, etc.)."""
    config = load_feature_config(config_path)
    return list(config.keys())


# =============================================================================
# MAIN FEATURE PREPARATION (combines all steps)
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    model_name: str,
    config_path: str = "feature_columns.json",
    target_column: str = "PREDICTION",
    fit_encoders: bool = True,
    encoders: Optional[Dict] = None,
    scaler: Optional[StandardScaler] = None,
    build_derived: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict, StandardScaler, List[str]]:
    """
    Full feature preparation pipeline:
    1. Build derived columns (Month_Due, Routed_Late, etc.)
    2. Build binary feature matrix from dummy_feature_columns
    3. Add additional_feature_columns
    4. Scale for neural network input

    Parameters
    ----------
    df : pd.DataFrame
        Raw data (from SQL or CSV).
    model_name : str
        Which model config to use: 'etac', 'dccs', or 'csdt'.
    config_path : str
        Path to feature_columns.json.
    target_column : str
        Name of the target/label column.
    fit_encoders : bool
        If True, fit new encoders/scaler. If False, reuse provided ones.
    encoders : dict or None
        Pre-fitted encoders (for inference).
    scaler : StandardScaler or None
        Pre-fitted scaler (for inference).
    build_derived : bool
        Whether to run derived column builders. Set False if already applied.

    Returns
    -------
    X : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    y : np.ndarray
        Integer class labels.
    encoders : dict
        Fitted encoders (category lists + label encoder).
    scaler : StandardScaler
        Fitted scaler.
    feature_names : list
        Ordered feature column names after encoding.
    """
    config = load_feature_config(config_path)

    if model_name not in config:
        raise ValueError(
            f"Model '{model_name}' not found in config. "
            f"Available: {list(config.keys())}"
        )

    model_cfg = config[model_name]
    dummy_columns = model_cfg["dummy_feature_columns"]
    additional_columns = model_cfg["additional_feature_columns"]

    # --- Step 1: Build derived columns ---
    if build_derived:
        df = build_derived_columns(df.copy(), data_source=model_name)

    # --- Step 2: Build binary feature matrix ---
    if fit_encoders:
        encoders = {}
        # Store category lists for each dummy column (for inference alignment)
        X_binary = build_binary_feature_matrix(df, dummy_columns)
        encoders["__feature_columns__"] = X_binary.columns.tolist()
    else:
        # Build matrix, then align to training columns
        X_binary = build_binary_feature_matrix(df, dummy_columns)
        training_columns = encoders["__feature_columns__"]
        X_binary = X_binary.reindex(columns=training_columns, fill_value=0)

    # --- Step 3: Add additional numeric columns ---
    X_df = add_feature_matrix_columns(X_binary, df, additional_columns)

    feature_names = X_df.columns.tolist()

    # Store final feature names for inference
    if fit_encoders:
        encoders["__all_feature_columns__"] = feature_names

    X = X_df.values.astype(np.float32)

    # --- Step 4: Scale ---
    if fit_encoders:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        # Align to all training feature columns (in case additional cols differ)
        all_training_cols = encoders.get("__all_feature_columns__", feature_names)
        X_df = X_df.reindex(columns=all_training_cols, fill_value=0)
        X = X_df.values.astype(np.float32)
        X = scaler.transform(X)
        feature_names = all_training_cols

    # --- Step 5: Target labels ---
    if target_column in df.columns:
        y_raw = df[target_column].values
        if y_raw.dtype == object or (len(y_raw) > 0 and isinstance(y_raw[0], str)):
            if fit_encoders:
                label_enc = LabelEncoder()
                y = label_enc.fit_transform(y_raw)
                encoders["__label_encoder__"] = label_enc
            else:
                label_enc = encoders["__label_encoder__"]
                y = label_enc.transform(y_raw)
        else:
            y = y_raw.astype(np.int64)
    else:
        y = np.zeros(len(df), dtype=np.int64)

    return X, y, encoders, scaler, feature_names


# =============================================================================
# CONVENIENCE: LOAD + PREPARE IN ONE CALL
# =============================================================================

def load_and_prepare_from_sql(
    query: str,
    model_name: str,
    target_column: str = "PREDICTION",
    config_path: str = "feature_columns.json",
    conn_str: str = None,
    use_dev: bool = False,
    max_rows: int = MAX_SAMPLE_ROWS,
) -> Tuple[np.ndarray, np.ndarray, Dict, StandardScaler, List[str], pd.DataFrame]:
    """
    One-shot: load from SQL, build features, return everything needed for training.

    Returns
    -------
    X, y, encoders, scaler, feature_names, df (original DataFrame for reference)
    """
    df = load_from_sql(query, model_name, conn_str=conn_str, use_dev=use_dev)

    if len(df) > max_rows:
        print(f"Sampling {max_rows} rows from {len(df)} total.")
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    X, y, encoders, scaler, feature_names = prepare_features(
        df=df,
        model_name=model_name,
        config_path=config_path,
        target_column=target_column,
        fit_encoders=True,
    )

    return X, y, encoders, scaler, feature_names, df


def load_and_prepare_from_csv(
    csv_path: str,
    model_name: str,
    target_column: str = "PREDICTION",
    config_path: str = "feature_columns.json",
    max_rows: int = MAX_SAMPLE_ROWS,
) -> Tuple[np.ndarray, np.ndarray, Dict, StandardScaler, List[str], pd.DataFrame]:
    """
    One-shot: load from CSV, build features, return everything needed for training.

    Returns
    -------
    X, y, encoders, scaler, feature_names, df (original DataFrame for reference)
    """
    df = load_from_csv(csv_path)

    if len(df) > max_rows:
        print(f"Sampling {max_rows} rows from {len(df)} total.")
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    X, y, encoders, scaler, feature_names = prepare_features(
        df=df,
        model_name=model_name,
        config_path=config_path,
        target_column=target_column,
        fit_encoders=True,
    )

    return X, y, encoders, scaler, feature_names, df


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_feature_summary(
    model_name: str,
    config_path: str = "feature_columns.json",
):
    """Print a summary of features for the given model."""
    config = load_feature_config(config_path)
    model_cfg = config[model_name]

    dummy_cols = model_cfg["dummy_feature_columns"]
    additional_cols = model_cfg["additional_feature_columns"]
    published_cols = model_cfg["published_columns"]

    print(f"\n{'='*60}")
    print(f"Feature Summary for model: '{model_name}'")
    print(f"{'='*60}")
    print(f"\nCategorical (one-hot encoded) columns ({len(dummy_cols)}):")
    for col in dummy_cols:
        max_cat = MAX_CATEGORIES_PER_COLUMN.get(col, "unlimited")
        print(f"  - {col} (max categories: {max_cat})")
    print(f"\nNumeric/binary columns ({len(additional_cols)}):")
    for col in additional_cols:
        print(f"  - {col}")
    print(f"\nPublished output columns ({len(published_cols)}):")
    for col in published_cols:
        print(f"  - {col}")
    print(f"\nTotal raw columns needed: {len(dummy_cols) + len(additional_cols)}")
    print(f"(Final input_dim depends on unique categories after capping)")
    print(f"{'='*60}\n")
