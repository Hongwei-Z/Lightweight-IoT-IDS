import time
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# Evaluation function
def evaluate_model(probs, y_test):
    preds = (probs >= 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))


# LightGBM Optuna
def objective_lgb(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'seed': 42,
        'n_estimators': 500
    }
    model = LGBMClassifier(**param)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


# XGBoost Optuna
def objective_xgb(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'auc',
        'n_estimators': 500,
        'random_state': 42
    }
    model = XGBClassifier(**param)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


# CatBoost Optuna
def objective_cat(trial):
    param = {
        'iterations': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 0,
        'od_type': 'Iter',
        'od_wait': 50
    }
    model = CatBoostClassifier(**param)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


# Run Optuna
def run_optuna(name, objective_func, n_trials=50):
    print(f"\nRunning Optuna for {name} with early stopping and 50 trials...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_func, n_trials=n_trials)
    print(f"Best params for {name}: {study.best_params}")
    return study.best_params


# Dataset loading
file_mapping = {
    1: "./Processed_Datasets/Dataset_BruteForce.csv",
    2: "./Processed_Datasets/Dataset_DDoS.csv",
    3: "./Processed_Datasets/Dataset_DoS.csv",
    4: "./Processed_Datasets/Dataset_Mirai.csv",
    5: "./Processed_Datasets/Dataset_Recon.csv",
    6: "./Processed_Datasets/Dataset_Spoofing.csv",
    7: "./Processed_Datasets/Dataset_Web-based.csv"
}

num_attack_sets = int(input("Enter the number of attack datasets to use: "))
if num_attack_sets <= 0:
    raise ValueError("Please enter a positive integer.")

attack_set_numbers = input(f"Enter {num_attack_sets} attack set numbers (1~7) separated by spaces: ")
attack_set_numbers = list(map(int, attack_set_numbers.strip().split()))
assert len(attack_set_numbers) == num_attack_sets
assert all(1 <= n <= 7 for n in attack_set_numbers)

selected_files = [file_mapping[n] for n in attack_set_numbers]
print(f"Selected files: {selected_files}.")

normal_df = pd.read_csv("./Processed_Datasets/Dataset_Benign.csv")
attack_dfs = [pd.read_csv(f) for f in selected_files]
attack_df = pd.concat(attack_dfs, ignore_index=True)
df = pd.concat([normal_df, attack_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print("Label distribution: ", df["Label"].value_counts())

X = df.drop("Label", axis=1)
y = df["Label"]
y = (y != 0).astype(int)

# Train/test split before feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# The following commented code is used to find the optimal parameters

# 1. LightGBM training and parameter tuning
# best_lgb_params = run_optuna("LightGBM", objective_lgb, n_trials=50)
# lgb_model = LGBMClassifier(**best_lgb_params, n_estimators=1000, random_state=42)
# start_time = time.time()
# lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
# lgb_train_time = time.time() - start_time
# lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
# print(f"LightGBM training time: {lgb_train_time:.2f} sec")
# evaluate_model(lgb_probs, y_test)

# 2. XGBoost training and parameter tuning
# best_xgb_params = run_optuna("XGBoost", objective_xgb, n_trials=50)
# xgb_model = XGBClassifier(**best_xgb_params)
# start_time = time.time()
# xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
# xgb_train_time = time.time() - start_time
# xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
# print(f"XGBoost training time: {xgb_train_time:.2f} sec")
# evaluate_model(xgb_probs, y_test)

# 3. CatBoost training and parameter tuning
# best_cat_params = run_optuna("CatBoost", objective_cat, n_trials=50)
# cat_model = CatBoostClassifier(**best_cat_params)
# start_time = time.time()
# cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
# cat_train_time = time.time() - start_time
# cat_probs = cat_model.predict_proba(X_test)[:, 1]
# print(f"CatBoost training time: {cat_train_time:.2f} sec")
# evaluate_model(cat_probs, y_test)


print("\nTraining LightGBM with best params...")

lgb_model = LGBMClassifier(
    learning_rate=0.19398654921103392,
    num_leaves=128,
    feature_fraction=0.9456935785817141,
    bagging_fraction=0.7876053476487416,
    bagging_freq=9,
    n_estimators=1000,
    random_state=42,
    verbosity=-1
)
start_time = time.time()
lgb_model.fit(X_train, y_train)
lgb_train_time = time.time() - start_time
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
print(f"LightGBM training time: {lgb_train_time:.2f} sec")
evaluate_model(lgb_probs, y_test)


print("\nTraining XGBoost with best params...")
xgb_model = XGBClassifier(
    learning_rate=0.1997145286610102,
    max_depth=10,
    subsample=0.8781392275527573,
    colsample_bytree=0.8767439111501281,
    n_estimators=1000,
    eval_metric='auc',
    random_state=42
)
start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_train_time = time.time() - start_time
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
print(f"XGBoost training time: {xgb_train_time:.2f} sec")
evaluate_model(xgb_probs, y_test)


print("\nTraining CatBoost with best params...")
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.16214117224124838,
    depth=10,
    l2_leaf_reg=1.0071287775690863,
    eval_metric='AUC',
    random_seed=42,
    verbose=0
)

start_time = time.time()
cat_model.fit(X_train, y_train)
cat_train_time = time.time() - start_time
cat_probs = cat_model.predict_proba(X_test)[:, 1]
print(f"CatBoost training time: {cat_train_time:.2f} sec")
evaluate_model(cat_probs, y_test)
