"""
Data Loader — Load data and prepare dual-format features for CatBoost + MLP.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import pyodbc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# --- Database config from .env ---
APSYS_PROD_SERVER = os.getenv("APSYS_PROD_SERVER")
APSYS_DEV_SERVER = os.getenv("APSYS_DEV_SERVER")
AVIONICS_USERNAME = os.getenv("AVIONICS_USERNAME")
AVIONICS_PASSWORD = os.getenv("AVIONICS_PASSWORD")

DATABASE_MAP = {
    "dccs": os.getenv("APSYS_DCCS_DATABASE"),
    "etac": os.getenv("APSYS_ETAC_DATABASE"),
    "csdt": os.getenv("APSYS_CSDT_DATABASE"),
}

MAX_CATEGORIES = {
    "Assigned_To_Group": 20, "Assigned_To_SubGroup": 40,
    "Major_Model": 10, "M/M_Change_Type": 12,
    "M/M_Prime_Engineer_Name": 30, "Month_Due": 12,
    "COMMODITY_NAME": 30, "SR_MANAGER_NAME": 30,
    "IPT_NAME": 20, "THREE_LETTER_CODE": 30,
    "SUPPLIER_NAME": 40, "CoordGrp": 20, "Customer": 15,
}

MAX_SAMPLE_ROWS = 810_000


# =============================================================================
# LOADING
# =============================================================================

def load_feature_config(config_path: str = "feature_columns.json") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def load_from_csv(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """Load data from CSV. If max_rows is set, uses chunked reading to avoid full memory load."""
    if max_rows is None:
        max_rows = MAX_SAMPLE_ROWS

    # Use chunked reading for large files
    chunks = []
    rows_loaded = 0
    for chunk in pd.read_csv(csv_path, chunksize=50_000, low_memory=False):
        chunks.append(chunk)
        rows_loaded += len(chunk)
        if rows_loaded >= max_rows * 2:
            # Load enough to get a good sample, then stop
            break

    df = pd.concat(chunks, ignore_index=True)

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"Loaded & sampled to {max_rows} rows from {csv_path}")
    else:
        print(f"Loaded {len(df)} rows from {csv_path}")

    return df


def load_from_sql(query: str, model_name: str, conn_str: str = None) -> pd.DataFrame:
    if conn_str is None:
        server = APSYS_PROD_SERVER
        database = DATABASE_MAP.get(model_name)
        conn_str = (
            f"Driver={{ODBC Driver 17 for SQL Server}}; "
            f"Server={server}; Database={database}; "
            f"UID={AVIONICS_USERNAME}; PWD={AVIONICS_PASSWORD}; "
            f"Trusted_Connection=yes"
        )
    with pyodbc.connect(conn_str) as conn:
        df = pd.read_sql(query, conn)
    print(f"Loaded {len(df)} rows from SQL ({model_name})")
    return df


# =============================================================================
# DERIVED COLUMNS
# =============================================================================

def build_derived_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Add model-specific derived columns."""
    df.columns = df.columns.str.replace(' ', '_')

    if model_name == "dccs":
        if "Monthly_ByDueOrClose" in df.columns:
            df["Monthly_ByDueOrClose"] = pd.to_datetime(df["Monthly_ByDueOrClose"], errors="coerce")
            df["Month_Due"] = df["Monthly_ByDueOrClose"].dt.month_name()
        if "M/M_Form_Review_Due" in df.columns:
            df["M/M_Form_Review_Due"] = pd.to_datetime(df["M/M_Form_Review_Due"], errors="coerce")
            ref = df.get("Monthly_ByDueOrClose", pd.Timestamp.now())
            if isinstance(ref, pd.Timestamp):
                df["Days_Until_Due"] = (df["M/M_Form_Review_Due"] - ref).dt.days.fillna(0).astype(int)
            else:
                df["Days_Until_Due"] = (df["M/M_Form_Review_Due"] - ref).dt.days.fillna(0).astype(int)
        if "Complete_Status" in df.columns:
            df["Routed_Late"] = df["Complete_Status"].apply(
                lambda x: 1 if str(x).startswith("Routed Late") else 0
            )

    elif model_name == "etac":
        if "ECD" in df.columns:
            df["ECD"] = pd.to_datetime(df["ECD"], errors="coerce")
            df["Month_Due"] = df["ECD"].dt.month_name()
            df["Week_Due"] = df["ECD"].dt.isocalendar().week.astype(str)
        if "ECDChangeCount" in df.columns:
            for t in [0, 1, 2, 5, 10]:
                df[f"ECDChangeCount_GT{t}"] = (df["ECDChangeCount"] > t).astype(int)
        if "OBDueChangeCount" in df.columns:
            for t in [0, 1, 2, 5, 10]:
                df[f"OBDueChangeCount_GT{t}"] = (df["OBDueChangeCount"] > t).astype(int)

    elif model_name == "csdt":
        if "SCN_NUMBER" in df.columns and "IsSCN" not in df.columns:
            df["IsSCN"] = df["SCN_NUMBER"].notna().astype(int)

    return df


# =============================================================================
# FEATURE PREPARATION (dual format)
# =============================================================================

def prepare_data(
    df: pd.DataFrame,
    model_name: str,
    target_column: str,
    config_path: str = "feature_columns.json",
    fit: bool = True,
    encoders: Optional[Dict] = None,
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """
    Prepare data in dual format for CatBoost and MLP.

    Returns dict with keys:
        X_cat, X_mlp, y, cat_indices, encoders, scaler,
        cat_feature_names, mlp_feature_names
    """
    config = load_feature_config(config_path)
    model_cfg = config[model_name]
    dummy_cols = model_cfg["dummy_feature_columns"]
    additional_cols = model_cfg["additional_feature_columns"]

    # Build derived columns
    df = build_derived_columns(df.copy(), model_name)

    # --- CatBoost features (raw categoricals + numerics) ---
    cat_cols = [c for c in dummy_cols if c in df.columns]
    num_cols = [c for c in additional_cols if c in df.columns]
    cat_feature_names = cat_cols + num_cols

    X_cat_df = df[cat_feature_names].copy()
    for col in cat_cols:
        X_cat_df[col] = X_cat_df[col].astype(str).fillna("Unknown")
    for col in num_cols:
        X_cat_df[col] = pd.to_numeric(X_cat_df[col], errors="coerce").fillna(0)

    cat_indices = list(range(len(cat_cols)))
    X_cat = X_cat_df.values

    # --- MLP features (one-hot encoded + scaled) ---
    if fit:
        encoders = {}

    # Build one-hot binary matrix
    binary_frames = []
    for col in dummy_cols:
        if col not in df.columns:
            continue
        series = df[col].astype(str).fillna("Unknown")
        if col in MAX_CATEGORIES:
            top = series.value_counts().head(MAX_CATEGORIES[col]).index.tolist()
            series = series.where(series.isin(top), other="Other")
        dummies = pd.get_dummies(series, prefix=col)
        binary_frames.append(dummies)

    X_binary = pd.concat(binary_frames, axis=1) if binary_frames else pd.DataFrame(index=df.index)

    if fit:
        encoders["__feature_columns__"] = X_binary.columns.tolist()
    else:
        X_binary = X_binary.reindex(columns=encoders["__feature_columns__"], fill_value=0)

    # Add numeric columns
    for col in additional_cols:
        if col in df.columns:
            X_binary[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        else:
            X_binary[col] = 0

    mlp_feature_names = X_binary.columns.tolist()
    if fit:
        encoders["__mlp_feature_columns__"] = mlp_feature_names

    X_mlp = X_binary.values.astype(np.float32)
    if fit:
        scaler = StandardScaler()
        X_mlp = scaler.fit_transform(X_mlp)
    else:
        cols = encoders.get("__mlp_feature_columns__", mlp_feature_names)
        X_binary = X_binary.reindex(columns=cols, fill_value=0)
        X_mlp = scaler.transform(X_binary.values.astype(np.float32))
        mlp_feature_names = cols

    # --- Target ---
    if target_column in df.columns:
        y_raw = df[target_column].values
        if y_raw.dtype == object or (len(y_raw) > 0 and isinstance(y_raw[0], str)):
            if fit:
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
                encoders["__label_encoder__"] = le
            else:
                y = encoders["__label_encoder__"].transform(y_raw)
        else:
            y = y_raw.astype(np.int64)
    else:
        y = np.zeros(len(df), dtype=np.int64)

    return {
        "X_cat": X_cat,
        "X_mlp": X_mlp,
        "y": y,
        "cat_indices": cat_indices,
        "encoders": encoders,
        "scaler": scaler,
        "cat_feature_names": cat_feature_names,
        "mlp_feature_names": mlp_feature_names,
    }
