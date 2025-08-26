# src/feature_engineering.py
import pandas as pd

def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight features; skip any that don't exist in df."""
    out = df.copy()

    # Example ratio if both present
    if "TransactionAmount" in out.columns and "AccountBalance" in out.columns:
        out["amt_to_bal"] = out["TransactionAmount"] / (out["AccountBalance"].abs() + 1e-6)

    # Time since previous transaction per account (seconds)
    if "PreviousTransactionDate" in out.columns and "TransactionDate" in out.columns:
        delta = (out["TransactionDate"] - out["PreviousTransactionDate"]).dt.total_seconds()
        out["secs_since_prev"] = delta.fillna(delta.median())

    # Number of logins to attempts ratio
    if "LoginAttempts" in out.columns:
        out["logins_log1p"] = (out["LoginAttempts"]).fillna(0)

    # Keep as-is otherwise; ULB V1..V28 will pass through to preprocessing
    return out
