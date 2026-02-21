import joblib, os
from time import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


def train_model(X_train, y_train, classifier='LightGBM', name="Default", output_dir="./"):
    print("Model Training".center(40, "-"))
    print("X_train:", X_train.shape, "| y_train:", y_train.shape)

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
    else:
        raise ValueError("Invalid classifier specified.")

    model.fit(X_train, y_train)

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

    # Train and test data paths
    train_mapping = {i: f'./Processed_Datasets/Train_{i}.csv' for i in range(8)}
    test_mapping = {i: f'./Processed_Datasets/Test_{i}.csv' for i in range(8)}

    # Client-specific labels for training
    # Remove a specific label to completely exclude a certain type of attack during training
    client_attack_config = {
        1: [0, 2, 3],  # DDoS, DoS
        2: [0, 2, 3, 4],  # DDoS, DoS, Mirai
        3: [0, 2, 4, 5, 6],  # DDoS, Mirai, Recon, Spoofing
        4: [0, 2, 3, 7],  # DDoS, DoS, Web-based
        5: [0, 1, 2, 5],  # DDoS, BruteForce, Recon
    }

    # Client-specific labels for testing (slightly different)
    test_config = {
        1: [0, 2, 3],  # DDoS, DoS
        2: [0, 2, 3, 4],  # DDoS, DoS, Mirai
        3: [0, 2, 4, 5, 6],  # DDoS, Mirai, Recon, Spoofing
        4: [0, 2, 3, 7],  # DDoS, DoS, Web-based
        5: [0, 1, 2, 5],  # DDoS, BruteForce, Recon
    }

    output_dir_base = "exclude_A" # Exclude an attack type
    os.makedirs(output_dir_base, exist_ok=True)

    # Load global configuration models once (outside loop for efficiency)
    selected_features = joblib.load("global_selected_features.pkl")
    pca = joblib.load("global_pca_model.pkl")
    optimal_k = joblib.load("global_optimal_k.pkl")
    print(f"Loaded Global: top-{len(selected_features)} features, PCA-> {pca.n_components_} dims, optimal_k={optimal_k}.")

    # Loop over all clients
    for client_id in range(1, 6):
        print(f"\n{'='*60}")
        print(f"Processing Client {client_id}")
        print(f"{'='*60}")

        train_labels = client_attack_config[client_id]
        test_labels = test_config[client_id]

        print(f"Training labels: {train_labels}")
        print(f"Testing labels:  {test_labels}")

        # Load training data
        train_parts = [pd.read_csv(train_mapping[l]) for l in train_labels]
        df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=42)
        X_train = df_train.drop("Label", axis=1)
        y_train = (df_train["Label"] != 0).astype(int)
        print("Train label distribution:", df_train["Label"].value_counts().to_dict())

        # Load testing data
        test_parts = [pd.read_csv(test_mapping[l]) for l in test_labels]
        df_test = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=42)
        X_test = df_test.drop("Label", axis=1)
        y_test = (df_test["Label"] != 0).astype(int)
        print("Test label distribution:", df_test["Label"].value_counts().to_dict())

        # Direct Training
        print("\n" + "Direct Training".center(50, "="))
        model_direct = train_model(X_train, y_train, classifier, name=f"CL_{client_id}_dir", output_dir=output_dir_base)
        test_model(model_direct, X_test, y_test)

        # Feature Selection
        X_train_mi = X_train.iloc[:, selected_features]
        X_test_mi = X_test.iloc[:, selected_features]

        # Dimension Reduction
        X_train_pca = pca.transform(X_train_mi)
        X_test_pca = pca.transform(X_test_mi)

        # Clustering Compression
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans_final.fit_predict(X_train_pca)
        cluster_centers = kmeans_final.cluster_centers_

        compression_levels = [0.1]
        for ratio in compression_levels:
            selected_idx = []
            for cid in range(optimal_k):
                mask = np.where(cluster_labels == cid)[0]
                if len(mask) == 0:
                    continue
                cluster_points = X_train_pca[mask]
                dists = np.linalg.norm(cluster_points - cluster_centers[cid], axis=1)
                sorted_idx = mask[np.argsort(dists)]
                n_keep = max(1, int(len(sorted_idx) * ratio))
                selected_idx.extend(sorted_idx[:n_keep])

            print(f"Client {client_id}, Ratio {int(ratio*100)}%: selected {len(selected_idx)} / {len(X_train_pca)} samples.")
            X_sel = X_train_pca[selected_idx]
            y_sel = y_train.iloc[selected_idx]

            model_name = f"CL_{client_id}_{int(ratio*100)}"
            model_cl = train_model(X_sel, y_sel, classifier, name=model_name, output_dir=output_dir_base)
            test_model(model_cl, X_test_pca, y_test)

    print("\n" + "All Done".center(60, "="))
