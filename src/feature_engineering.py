import pandas as pd

def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight features; skip any that don't exist in df."""
    out = df.copy()
    if "TransactionAmount" in out.columns and "AccountBalance" in out.columns:
        out["amt_to_bal"] = out["TransactionAmount"] / (out["AccountBalance"].abs() + 1e-6)
    if "PreviousTransactionDate" in out.columns and "TransactionDate" in out.columns:
        delta = (out["TransactionDate"] - out["PreviousTransactionDate"]).dt.total_seconds()
        out["secs_since_prev"] = delta.fillna(delta.median())
    if "LoginAttempts" in out.columns:
        out["logins_log1p"] = (out["LoginAttempts"]).fillna(0)
    return out
