"""
Ensemble Training — CatBoost (Primary) + MLP (Secondary)
---------------------------------------------------------
Trains both models, runs cross-validation, and provides inference.
CatBoost gets raw categorical data. MLP gets one-hot encoded + scaled data.

Usage:
    python ensemble_train.py --model dccs --data training_data.csv --target MM_Determination_Completion_Status
    python ensemble_train.py --model etac --sql "SELECT * FROM vw_etac_training"
    python ensemble_train.py --model dccs  (synthetic demo)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
from pathlib import Path

from catboost import CatBoostClassifier, Pool

from model import FlexibleMLP
from ensemble_model import CatBoostMLP_Ensemble
from data_loader import (
    load_feature_config,
    load_from_sql,
    load_from_csv,
    build_derived_columns,
    build_binary_feature_matrix,
    add_feature_matrix_columns,
    print_feature_summary,
    MAX_CATEGORIES_PER_COLUMN,
    MAX_SAMPLE_ROWS,
)
