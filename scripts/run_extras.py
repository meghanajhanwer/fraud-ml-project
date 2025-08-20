import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF if present
from src.analysis_extras import run_all_and_save

if __name__ == "__main__":
    run_all_and_save()
    print("✅ Extras complete: test metrics, tuned confusion, calibration, ablation, interpretability saved to GCS.")
