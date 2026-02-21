import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# Load feature configurations
selected_idx = joblib.load("global_selected_features.pkl")
pca = joblib.load("global_pca_model.pkl")

# Load test set and split validation set
test_mapping = {i: f'./Processed_Datasets/Test_{i}.csv' for i in range(8)}
test_df = pd.concat([pd.read_csv(test_mapping[i]) for i in range(8)], ignore_index=True)
X_test_full = test_df.drop("Label", axis=1).iloc[:, selected_idx]
X_test_full_pca = pca.transform(X_test_full)
y_test_full = (test_df["Label"] != 0).astype(int)

# Extract 20% from test set as validation set
X_test_pca, X_val_pca, y_test, y_val = train_test_split(
    X_test_full_pca, y_test_full, test_size=0.2, stratify=y_test_full, random_state=42
)

# Load client models
client_ids = range(1, 6)
model_paths = [f"./exclude_web/LightGBM_CL_{cid}_10.pkl" for cid in client_ids]
models = [joblib.load(p) for p in model_paths]

# Evaluation function
def evaluate_predictions(y_true, y_pred, y_proba, method_name):
    print(f"---------- {method_name} Evaluation ----------")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("=" * 19)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.6f}")
    print(f"Precision: {precision_score(y_true, y_pred):.6f}")
    print(f"Recall: {recall_score(y_true, y_pred):.6f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.6f}")
    print(f"AUC: {roc_auc_score(y_true, y_proba):.6f}")
    print("=" * 19)

# ==================== Method 1: Hard Voting ====================
print("\nRunning Hard Voting...")
# Get predictions from each client model
test_preds = np.array([model.predict(X_test_pca) for model in models])
val_preds = np.array([model.predict(X_val_pca) for model in models])

# Optimize vote threshold on validation set
best_f1, best_vote_threshold = 0, 1
for vote_th in range(1, len(models) + 1):
    y_val_pred = (np.sum(val_preds, axis=0) >= vote_th).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1, best_vote_threshold = f1, vote_th
print(f"Best Vote Threshold for Hard Voting: At least {best_vote_threshold} votes")

# Apply best vote threshold
y_hard_pred = (np.sum(test_preds, axis=0) >= best_vote_threshold).astype(int)
# Compute AUC
probs_all = np.array([model.predict_proba(X_test_pca)[:, 1] for model in models])
probs_mean = np.mean(probs_all, axis=0)
evaluate_predictions(y_test, y_hard_pred, probs_mean, "Hard Voting")

# ==================== Method 2: Weighted Voting ====================
print("\nRunning Weighted Voting...")
# Compute F1 from val set as weight
val_probs = np.array([model.predict_proba(X_val_pca)[:, 1] for model in models])
val_preds_weighted = (val_probs >= 0.5).astype(int)
weights = np.array([f1_score(y_val, pred) for pred in val_preds_weighted])
weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(models)) / len(models)
print("Model Weights:", {cid: w for cid, w in zip(client_ids, weights)})

# Weighted probabilities average
probs_weighted = np.average(probs_all, axis=0, weights=weights)
val_probs_weighted = np.average(val_probs, axis=0, weights=weights)
best_f1, best_threshold = 0, 0.5
for th in np.arange(0.3, 0.8, 0.05):
    y_val_pred = (val_probs_weighted >= th).astype(int)
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1, best_threshold = f1, th
print(f"Best Threshold for Weighted Voting: {best_threshold:.2f}")

y_weighted_pred = (probs_weighted >= best_threshold).astype(int)
evaluate_predictions(y_test, y_weighted_pred, probs_weighted, "Weighted Voting")

# ==================== Method 3: Stacking ====================
print("\nRunning Stacking...")
# Use val set to train the stacking meta model
val_probs_stacking = val_probs.T
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(val_probs_stacking, y_val)

# Test stacking model
test_probs_stacking = probs_all.T
y_stacking_pred = meta_model.predict(test_probs_stacking)
probs_stacking = meta_model.predict_proba(test_probs_stacking)[:, 1]
evaluate_predictions(y_test, y_stacking_pred, probs_stacking, "Stacking")
