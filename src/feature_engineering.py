import pandas as pd
import numpy as np
from config.config import TIME_COL

def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ratio feature (guard division by zero)
    out["amt_to_bal"] = np.where(out["AccountBalance"].abs() > 1e-9,
                                 out["TransactionAmount"] / out["AccountBalance"], 0.0)

    # Time-based features (hour, dayofweek)
    dt = pd.to_datetime(out[TIME_COL], errors="coerce", utc=True)
    out["txn_hour"] = dt.dt.hour
    out["txn_dow"] = dt.dt.dayofweek
    out["is_weekend"] = out["txn_dow"].isin([5, 6]).astype(int)

    # Basic sanitization for NA
    numeric_cols = ["TransactionAmount","AccountBalance","CustomerAge","TransactionDuration","LoginAttempts","amt_to_bal","txn_hour","txn_dow","is_weekend"]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    return out
