"""
evaluate.py  —  Evaluation Suite for Complaint Auto-Routing System
────────────────────────────────────────────────────────────────

Computes and reports:
    T1 Officer Routing   : Accuracy, F1-macro, F1 per class, Confusion Matrix
    T2 Priority          : Accuracy, F1-macro, F1 per class
    T3 ETA Regressor     : MAE, RMSE, R²
    T4 Similarity Search : Recall@1, Recall@5, Recall@10 (same-department)

Run:
    python evaluation/evaluate.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference.embedding_engine import get_embedding_engine
from inference.vector_store import NumpyVectorStore

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_complaints.csv")
SAVE_DIR  = os.path.join(BASE_DIR, "models", "saved")


def load_artifacts():
    emb     = joblib.load(os.path.join(SAVE_DIR, "embedding_engine.pkl"))
    les     = joblib.load(os.path.join(SAVE_DIR, "label_encoders.pkl"))
    off_clf = joblib.load(os.path.join(SAVE_DIR, "officer_classifier.pkl"))
    pri_clf = joblib.load(os.path.join(SAVE_DIR, "priority_classifier.pkl"))
    eta_reg = joblib.load(os.path.join(SAVE_DIR, "eta_regressor.pkl"))
    vs      = NumpyVectorStore().load(os.path.join(SAVE_DIR, "vector_store.pkl"))
    return emb, les, off_clf, pri_clf, eta_reg, vs


def recall_at_k(store: NumpyVectorStore, X: np.ndarray,
                df: pd.DataFrame, k: int) -> float:
    """
    Recall@K for similarity search.
    A retrieval is considered a hit if ≥1 of the top-K results
    belongs to the same department as the query.
    Self-match is excluded.
    """
    n    = len(df)
    hits = 0
    for i in range(n):
        results = store.search(X[i], top_k=k + 1)
        results = [r for r in results
                   if r["complaint_id"] != df.iloc[i]["complaint_id"]][:k]
        if any(r["department"] == df.iloc[i]["department"] for r in results):
            hits += 1
    return hits / n


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       COMPLAINT AUTO-ROUTING — EVALUATION SUITE         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Load data & artifacts
    df       = pd.read_csv(DATA_PATH)
    emb, les, off_clf, pri_clf, eta_reg, vs = load_artifacts()
    print(f"\nDataset: {len(df)} complaints")
    X = emb.encode(df["text"].tolist())
    print(f"Embeddings: {X.shape}")

    le_off = les["officer"]
    le_pri = les["priority"]
    y_off  = le_off.transform(df["officer_id"])
    y_pri  = le_pri.transform(df["priority"])
    y_eta  = df["eta_days"].values

    # ═══════════════════════════════════════════════════
    # T1: Officer Routing
    # ═══════════════════════════════════════════════════
    print_section("T1 — OFFICER ROUTING (SVM)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_off = cross_validate(
        off_clf, X, y_off, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
    )
    print(f"  CV Accuracy (5-fold) : {cv_off['test_accuracy'].mean():.4f} ± {cv_off['test_accuracy'].std():.4f}")
    print(f"  CV F1-macro (5-fold) : {cv_off['test_f1_macro'].mean():.4f} ± {cv_off['test_f1_macro'].std():.4f}")

    y_pred_off = off_clf.predict(X)
    print(f"\n  Full-data Accuracy   : {accuracy_score(y_off, y_pred_off):.4f}")
    print(f"  Full-data F1-macro   : {f1_score(y_off, y_pred_off, average='macro'):.4f}")
    print("\n  Per-class Report:")
    print(classification_report(
        y_off, y_pred_off,
        target_names=le_off.classes_,
        digits=3,
    ))

    # ═══════════════════════════════════════════════════
    # T2: Priority Prediction
    # ═══════════════════════════════════════════════════
    print_section("T2 — PRIORITY PREDICTION (Random Forest)")
    cv_pri = cross_validate(
        pri_clf, X, y_pri, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
    )
    print(f"  CV Accuracy (5-fold) : {cv_pri['test_accuracy'].mean():.4f} ± {cv_pri['test_accuracy'].std():.4f}")
    print(f"  CV F1-macro (5-fold) : {cv_pri['test_f1_macro'].mean():.4f} ± {cv_pri['test_f1_macro'].std():.4f}")

    y_pred_pri = pri_clf.predict(X)
    print(f"\n  Full-data Accuracy   : {accuracy_score(y_pri, y_pred_pri):.4f}")
    print(f"  Full-data F1-macro   : {f1_score(y_pri, y_pred_pri, average='macro'):.4f}")
    print("\n  Per-class Report:")
    print(classification_report(
        y_pri, y_pred_pri,
        target_names=le_pri.classes_,
        digits=3,
    ))

    # ═══════════════════════════════════════════════════
    # T3: ETA Prediction
    # ═══════════════════════════════════════════════════
    print_section("T3 — ETA PREDICTION (Gradient Boosting Regressor)")
    cv_eta = cross_validate(
        eta_reg, X, y_eta, cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        return_train_score=False,
    )
    cv_mae  = -cv_eta["test_neg_mean_absolute_error"].mean()
    cv_rmse = -cv_eta["test_neg_root_mean_squared_error"].mean()
    cv_r2   =  cv_eta["test_r2"].mean()
    print(f"  CV MAE  (5-fold) : {cv_mae:.3f} days")
    print(f"  CV RMSE (5-fold) : {cv_rmse:.3f} days")
    print(f"  CV R²   (5-fold) : {cv_r2:.4f}")

    y_pred_eta = eta_reg.predict(X)
    print(f"\n  Full-data MAE    : {mean_absolute_error(y_eta, y_pred_eta):.3f} days")
    print(f"  Full-data RMSE   : {np.sqrt(mean_squared_error(y_eta, y_pred_eta)):.3f} days")
    print(f"  Full-data R²     : {r2_score(y_eta, y_pred_eta):.4f}")

    # ═══════════════════════════════════════════════════
    # T4: Similarity Search
    # ═══════════════════════════════════════════════════
    print_section("T4 — SIMILARITY SEARCH (Cosine / NumpyVectorStore)")
    print("  Computing Recall@K … (this may take ~30 seconds)")
    r1  = recall_at_k(vs, X, df, k=1)
    r5  = recall_at_k(vs, X, df, k=5)
    r10 = recall_at_k(vs, X, df, k=10)
    print(f"  Recall@1  : {r1:.4f}")
    print(f"  Recall@5  : {r5:.4f}")
    print(f"  Recall@10 : {r10:.4f}")
    print("  (criterion: ≥1 retrieved complaint from same department)")

    # ═══════════════════════════════════════════════════
    # Summary JSON
    # ═══════════════════════════════════════════════════
    summary = {
        "T1_officer_routing": {
            "cv_accuracy_mean": float(cv_off["test_accuracy"].mean()),
            "cv_accuracy_std":  float(cv_off["test_accuracy"].std()),
            "cv_f1_macro_mean": float(cv_off["test_f1_macro"].mean()),
            "cv_f1_macro_std":  float(cv_off["test_f1_macro"].std()),
        },
        "T2_priority_prediction": {
            "cv_accuracy_mean": float(cv_pri["test_accuracy"].mean()),
            "cv_accuracy_std":  float(cv_pri["test_accuracy"].std()),
            "cv_f1_macro_mean": float(cv_pri["test_f1_macro"].mean()),
            "cv_f1_macro_std":  float(cv_pri["test_f1_macro"].std()),
        },
        "T3_eta_prediction": {
            "cv_mae_days":  float(cv_mae),
            "cv_rmse_days": float(cv_rmse),
            "cv_r2":        float(cv_r2),
        },
        "T4_similarity_search": {
            "recall@1":  float(r1),
            "recall@5":  float(r5),
            "recall@10": float(r10),
        },
    }

    out_path = os.path.join(SAVE_DIR, "evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    print(f"\n[OK] Evaluation report saved -> {out_path}")


if __name__ == "__main__":
    main()
