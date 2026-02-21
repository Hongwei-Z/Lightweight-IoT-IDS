from ResourceMonitor import ResourceMonitor
import joblib, os
from time import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")


def train_model(X_train, y_train, classifier='LightGBM', name="Default", output_dir="./"):
    print("Model Training".center(40, "-"))
    print("X_train:", X_train.shape, "| y_train:", y_train.shape)

    resource_monitor = ResourceMonitor()
    resource_monitor.start()

    if classifier == 'LightGBM':
        model = LGBMClassifier(
            n_estimators=430,
            learning_rate=0.3,
            num_leaves=180,
            max_depth=60,
            subsample=0.3,
            colsample_bytree=0.5,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            verbose=-1
        )
    elif classifier == 'XGBoost':
        model = XGBClassifier(
            learning_rate=0.1997145286610102,
            max_depth=10,
            subsample=0.8781392275527573,
            colsample_bytree=0.8767439111501281,
            n_estimators=1000,
            eval_metric='auc',
            random_state=42
        )
    elif classifier == 'CatBoost':
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.16214117224124838,
            depth=10,
            l2_leaf_reg=1.0071287775690863,
            eval_metric='AUC',
            random_seed=42,
            verbose=0
        )
    else:
        raise ValueError("Invalid classifier specified.")

    model.fit(X_train, y_train)

    resource_monitor.stop()
    print("Training Completed".center(40, "-"))

    filename = os.path.join(output_dir, f"{classifier}_{name}.pkl")
    joblib.dump(model, filename)
    print(f"Model saved: {filename} | size {os.path.getsize(filename)/1024:.2f} KB")
    return model


def test_model(model, X_test, y_test):
    print("\n" + "Model Testing".center(40, "-"))
    start = time()
    probs = model.predict_proba(X_test)[:, 1]
    end = time()
    preds = (probs >= 0.5).astype(int)

    acc = balanced_accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    auc = roc_auc_score(y_test, probs)
    elapsed = end - start
    print("=" * 20)
    print(f"Accuracy:  {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1 Score:  {f1:.6f}")
    print(f"AUC:       {auc:.6f}")
    print("=" * 20)
    print(f"Predicted {X_test.shape[0]} samples in {elapsed:.4f} sec, ~{int(X_test.shape[0]/elapsed)} samples/s")
    print("Testing Completed".center(40, "-"))


if __name__ == '__main__':
    classifier = "LightGBM"  # LightGBM, CatBoost, XGBoost

    # Train and test data
    train_mapping = {i: f'./Processed_Datasets/Train_{i}.csv' for i in range(8)}
    test_mapping = {i: f'./Processed_Datasets/Test_{i}.csv' for i in range(8)}

    # Assign attacks to each client
    client_attack_config = {
        1: [0, 2, 3],  # DDoS, DoS
        2: [0, 2, 3, 4],  # DDoS, DoS, Mirai
        3: [0, 2, 4, 5, 6],  # DDoS, Mirai, Recon, Spoofing
        4: [0, 2, 3, 7],  # DDoS, DoS, Web-based
        5: [0, 1, 2, 5],  # DDoS, BruteForce, Recon
    }

    client_id = int(input("Enter client ID (1~5): "))
    if client_id not in client_attack_config:
        raise ValueError("Invalid ID")

    labels_to_load = client_attack_config[client_id]
    print(f"\nClient {client_id} load labels: {labels_to_load}")

    # Load client train data
    train_parts = [pd.read_csv(train_mapping[l]) for l in labels_to_load]
    df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=42)
    X_train = df_train.drop("Label", axis=1)
    y_train = (df_train["Label"] != 0).astype(int)
    print("Train label distribution:", df_train["Label"].value_counts().to_dict())

    # Load test data
    test_parts = [pd.read_csv(test_mapping[l]) for l in labels_to_load]
    df_test = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=42)
    X_test = df_test.drop("Label", axis=1)
    y_test = (df_test["Label"] != 0).astype(int)
    print("Test label distribution:", df_test["Label"].value_counts().to_dict())

    output_dir = f"Client_{client_id}_Output"
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. Direct Training ====================
    print("\n" + "Direct Training".center(50, "="))
    model_direct = train_model(X_train, y_train, classifier, name=f"D_{client_id}", output_dir=output_dir)
    test_model(model_direct, X_test, y_test)

    # Load global configuration models
    print("\n" + "Load Global Configuration models:")
    selected_features = joblib.load("global_selected_features.pkl")
    pca = joblib.load("global_pca_model.pkl")
    optimal_k = joblib.load("global_optimal_k.pkl")
    print(f"Loaded Global: top-{len(selected_features)} features, PCA-> {pca.n_components_} dims, optimal_k={optimal_k}.")

    # ==================== 2. Feature Selection ====================
    print("\n" + " Feature Selection ".center(50, "="))
    X_train_mi = X_train.iloc[:, selected_features]
    X_test_mi = X_test.iloc[:, selected_features]
    print(f"After MI top-{len(selected_features)} features: train {X_train_mi.shape}")
    model_fs = train_model(X_train_mi, y_train, classifier, name=f"FS_{client_id}", output_dir=output_dir)
    test_model(model_fs, X_test_mi, y_test)

    # ==================== 3. Dimension Reduction ==================
    print("\n" + " Dimension Reduction ".center(50, "="))
    X_train_pca = pca.transform(X_train_mi)
    X_test_pca = pca.transform(X_test_mi)
    print(f"PCA reduced to {X_train_pca.shape[1]} dims.")
    model_pca = train_model(X_train_pca, y_train, classifier, name=f"DR_{client_id}", output_dir=output_dir)
    test_model(model_pca, X_test_pca, y_test)

    # ==================== 4. Clustering Compression ===============
    print("\n" + " Clustering Compression ".center(50, "="))
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans_final.fit_predict(X_train_pca)
    cluster_centers = kmeans_final.cluster_centers_

    compression_levels = [0.9, 0.8, 0.6, 0.5, 0.3, 0.1]
    for ratio in compression_levels:
        selected_idx = []
        for cid in range(optimal_k):
            mask = np.where(cluster_labels == cid)[0]
            cluster_points = X_train_pca[mask]
            dists = np.linalg.norm(cluster_points - cluster_centers[cid], axis=1)
            sorted_idx = mask[np.argsort(dists)]
            n_keep = max(1, int(len(sorted_idx)*ratio))
            selected_idx.extend(sorted_idx[:n_keep])

        print(f"Ratio {int(ratio*100)}%: selected {len(selected_idx)} / {len(X_train_pca)} samples.")
        X_sel = X_train_pca[selected_idx]
        y_sel = y_train.iloc[selected_idx]

        model_cl = train_model(X_sel, y_sel, classifier,
                               name=f"CL_{client_id}_{int(ratio*100)}", output_dir=output_dir)
        test_model(model_cl, X_test_pca, y_test)

    print("\n" + "All Done".center(60, "="))
