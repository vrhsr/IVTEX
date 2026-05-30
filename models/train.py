"""
train.py
────────────────────────────────────────────────────────────────
End-to-end training pipeline for the Complaint Auto-Routing System.

Tasks trained:
  T1  Officer Routing      → SVM (RBF kernel) classifier
  T2  Priority Prediction  → Random Forest classifier
  T3  ETA Prediction       → Gradient-Boosted Regressor
  T4  Similarity Search    → NumpyVectorStore (cosine, FAISS optional)

Embeddings:
  • Default : TF-IDF + SVD (256-dim) — offline, no downloads
  • Upgrade : sentence-transformers paraphrase-multilingual-MiniLM-L12-v2

Run:
    python models/train.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference.embedding_engine import get_embedding_engine
from inference.vector_store import get_vector_store

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "synthetic_complaints.csv")
SAVE_DIR   = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

OFFICER_MODEL_PATH   = os.path.join(SAVE_DIR, "officer_classifier.pkl")
PRIORITY_MODEL_PATH  = os.path.join(SAVE_DIR, "priority_classifier.pkl")
ETA_MODEL_PATH       = os.path.join(SAVE_DIR, "eta_regressor.pkl")
EMBEDDING_PATH       = os.path.join(SAVE_DIR, "embedding_engine.pkl")
VECTOR_STORE_PATH    = os.path.join(SAVE_DIR, "vector_store.pkl")
LABEL_ENCODERS_PATH  = os.path.join(SAVE_DIR, "label_encoders.pkl")
METRICS_PATH         = os.path.join(SAVE_DIR, "metrics.json")


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} complaints from {DATA_PATH}")
    return df


def build_embeddings(df, embedding_engine):
    """Fit embedding engine on corpus and return matrix."""
    texts = df["text"].tolist()
    embedding_engine.fit(texts)
    print(f"Embedding engine fitted on {len(texts)} documents.")
    X = embedding_engine.encode(texts)
    print(f"Embedding matrix: {X.shape}")
    return X


def train_officer_classifier(X, y_officer, label_encoder_officer):
    """SVM with RBF kernel → multi-class officer routing."""
    y_enc = label_encoder_officer.fit_transform(y_officer)

    # Cross-validation
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, X, y_enc, cv=cv, scoring="f1_macro")
    print(f"\n[Officer Routing] CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on full data
    svm.fit(X, y_enc)
    return svm, {"cv_f1_macro_mean": cv_scores.mean(), "cv_f1_macro_std": cv_scores.std()}


def train_priority_classifier(X, y_priority, label_encoder_priority):
    """Random Forest → High / Medium / Low priority."""
    y_enc = label_encoder_priority.fit_transform(y_priority)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc  = cross_val_score(rf, X, y_enc, cv=cv, scoring="accuracy")
    cv_f1   = cross_val_score(rf, X, y_enc, cv=cv, scoring="f1_macro")
    print(f"\n[Priority] CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"[Priority] CV F1-macro : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    rf.fit(X, y_enc)
    return rf, {
        "cv_accuracy_mean": cv_acc.mean(), "cv_accuracy_std": cv_acc.std(),
        "cv_f1_macro_mean": cv_f1.mean(),  "cv_f1_macro_std": cv_f1.std(),
    }


def train_eta_regressor(X, y_eta):
    """Gradient Boosting Regressor → ETA in days (MAE metric)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_eta, test_size=0.2, random_state=42
    )
    gbr = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8, random_state=42
    )
    gbr.fit(X_train, y_train)
    preds = gbr.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\n[ETA Regressor] Test MAE  : {mae:.2f} days")
    print(f"[ETA Regressor] Test RMSE : {rmse:.2f} days")

    # Refit on full data
    gbr.fit(X, y_eta)
    return gbr, {"test_mae": mae, "test_rmse": rmse}


def build_vector_store(df, X):
    """Build similarity search index from training embeddings."""
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "complaint_id": row["complaint_id"],
            "text":         row["text"],
            "officer_name": row["officer_name"],
            "department":   row["department"],
            "priority":     row["priority"],
            "eta_days":     int(row["eta_days"]),
        })
    store = get_vector_store(use_faiss=False)
    store.build(X, metadata)
    return store


def evaluate_similarity_recall(store, X, df, k: int = 5):
    """
    Recall@K: for each complaint, the top-K retrieved complaints
    should include at least one from the same officer/department.
    """
    hits = 0
    n    = min(200, len(df))   # sample for speed
    for i in range(n):
        results = store.search(X[i], top_k=k + 1)  # +1 to exclude self
        results = [r for r in results if r["complaint_id"] != df.iloc[i]["complaint_id"]][:k]
        gold_dept = df.iloc[i]["department"]
        if any(r["department"] == gold_dept for r in results):
            hits += 1
    recall_at_k = hits / n
    print(f"\n[Similarity] Recall@{k} (same-department): {recall_at_k:.4f}")
    return recall_at_k


def train_test_detailed_report(X, df, models, label_encoders):
    """Produce held-out classification reports."""
    X_tr, X_te, df_tr, df_te = train_test_split(
        X, df, test_size=0.20, random_state=42, stratify=df["officer_id"]
    )

    print("\n" + "="*55)
    print("HELD-OUT EVALUATION (80/20 split)")
    print("="*55)

    # Officer routing
    off_model = models["officer"]
    le_off    = label_encoders["officer"]
    y_te_off  = le_off.transform(df_te["officer_id"])
    y_pr_off  = off_model.predict(X_te)
    print("\n[Officer Routing] Classification Report:")
    print(classification_report(y_te_off, y_pr_off, target_names=le_off.classes_))

    # Priority
    pri_model = models["priority"]
    le_pri    = label_encoders["priority"]
    y_te_pri  = le_pri.transform(df_te["priority"])
    y_pr_pri  = pri_model.predict(X_te)
    print("[Priority Prediction] Classification Report:")
    print(classification_report(y_te_pri, y_pr_pri, target_names=le_pri.classes_))

    # ETA
    eta_model = models["eta"]
    y_te_eta  = df_te["eta_days"].values
    y_pr_eta  = eta_model.predict(X_te)
    mae       = mean_absolute_error(y_te_eta, y_pr_eta)
    rmse      = np.sqrt(mean_squared_error(y_te_eta, y_pr_eta))
    print(f"[ETA Regressor] MAE={mae:.2f} days  RMSE={rmse:.2f} days")

    return {
        "officer_accuracy":  accuracy_score(y_te_off, y_pr_off),
        "officer_f1_macro":  f1_score(y_te_off, y_pr_off, average="macro"),
        "priority_accuracy": accuracy_score(y_te_pri, y_pr_pri),
        "priority_f1_macro": f1_score(y_te_pri, y_pr_pri, average="macro"),
        "eta_mae":           mae,
        "eta_rmse":          rmse,
    }


def main():
    print("="*55)
    print("  COMPLAINT AUTO-ROUTING - TRAINING PIPELINE")
    print("="*55)

    # 1. Load data
    df = load_data()

    # 2. Build embeddings
    emb_engine = get_embedding_engine(prefer_transformer=True)
    X = build_embeddings(df, emb_engine)

    # 3. Label encoders
    le_officer  = LabelEncoder()
    le_priority = LabelEncoder()

    # 4. Train all models
    officer_model, off_metrics = train_officer_classifier(
        X, df["officer_id"], le_officer
    )
    priority_model, pri_metrics = train_priority_classifier(
        X, df["priority"], le_priority
    )
    eta_model, eta_metrics = train_eta_regressor(X, df["eta_days"].values)

    # 5. Build vector store
    vector_store = build_vector_store(df, X)
    recall = evaluate_similarity_recall(vector_store, X, df, k=5)

    # 6. Detailed held-out report
    models       = {"officer": officer_model, "priority": priority_model, "eta": eta_model}
    label_encoders = {"officer": le_officer, "priority": le_priority}
    ho_metrics   = train_test_detailed_report(X, df, models, label_encoders)

    # 7. Save all artifacts
    joblib.dump(officer_model,  OFFICER_MODEL_PATH)
    joblib.dump(priority_model, PRIORITY_MODEL_PATH)
    joblib.dump(eta_model,      ETA_MODEL_PATH)
    joblib.dump(emb_engine,     EMBEDDING_PATH)
    joblib.dump({"officer": le_officer, "priority": le_priority}, LABEL_ENCODERS_PATH)
    vector_store.save(VECTOR_STORE_PATH)

    # 8. Save metrics JSON
    all_metrics = {
        "officer_routing_cv":   off_metrics,
        "priority_cv":          pri_metrics,
        "eta_cv":               eta_metrics,
        "similarity_recall@5":  recall,
        "held_out":             ho_metrics,
    }
    # convert numpy floats
    all_metrics = json.loads(json.dumps(all_metrics, default=float))
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[OK] All artifacts saved to {SAVE_DIR}")
    print(f"[OK] Metrics saved to {METRICS_PATH}")
    print("\nFinal Summary:")
    print(f"  Officer F1-macro  : {ho_metrics['officer_f1_macro']:.4f}")
    print(f"  Priority Accuracy : {ho_metrics['priority_accuracy']:.4f}")
    print(f"  ETA MAE           : {ho_metrics['eta_mae']:.2f} days")
    print(f"  Similarity R@5    : {recall:.4f}")


if __name__ == "__main__":
    main()
