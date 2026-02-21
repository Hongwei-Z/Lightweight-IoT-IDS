from ResourceMonitor import ResourceMonitor
import joblib, os
from time import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
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

    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
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

    # Select an attack
    attack_label = int(input("Select attack set number (1~7): "))
    if attack_label <= 0 or attack_label > 7:
        raise ValueError("Invalid number")
    print(f"Selected file: {train_mapping[attack_label]}")

    # Load client train data
    normal_df = pd.read_csv(train_mapping[0])
    attack_df = pd.read_csv(train_mapping[attack_label])
    df = pd.concat([normal_df, attack_df], ignore_index=True).sample(frac=1, random_state=42)
    X = df.drop("Label", axis=1)
    y = (df["Label"] != 0).astype(int)
    print("Train label distribution:", df["Label"].value_counts().to_dict())

    # Load test data
    test0 = pd.read_csv(test_mapping[0])
    test1 = pd.read_csv(test_mapping[attack_label])
    df_test = pd.concat([test0, test1], ignore_index=True).sample(frac=1, random_state=42)
    X_test = df_test.drop("Label", axis=1)
    y_test = (df_test["Label"] != 0).astype(int)
    print("Test label distribution:", df_test["Label"].value_counts().to_dict())

    output_dir = f"Output_{attack_label}"
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. Direct Training ====================
    print("\n" + "Direct Training".center(50, "="))
    model_direct = train_model(X, y, classifier, name=f"D_{attack_label}", output_dir=output_dir)
    test_model(model_direct, X_test, y_test)

    # Load global configuration models
    print("\n" + "Load Global Configuration models:")
    selected_features = joblib.load("global_selected_features.pkl")
    pca = joblib.load("global_pca_model.pkl")
    optimal_k = joblib.load("global_optimal_k.pkl")
    print(f"Loaded Global: top-{len(selected_features)} features, PCA-> {pca.n_components_} dims, optimal_k={optimal_k}.")

    # ==================== 2. Feature Selection ====================
    print("\n" + " Feature Selection ".center(50, "="))
    X_train_mi = X.iloc[:, selected_features]
    X_test_mi = X_test.iloc[:, selected_features]
    print(f"After MI top-{len(selected_features)} features: train {X_train_mi.shape}")
    model_fs = train_model(X_train_mi, y, classifier, name=f"FS_{attack_label}", output_dir=output_dir)
    test_model(model_fs, X_test_mi, y_test)

    # ==================== 3. Dimension Reduction ==================
    print("\n" + " Dimension Reduction ".center(50, "="))
    X_train_pca = pca.transform(X_train_mi)
    X_test_pca = pca.transform(X_test_mi)
    print(f"PCA reduced to {X_train_pca.shape[1]} dims.")
    model_pca = train_model(X_train_pca, y, classifier, name=f"DR_{attack_label}", output_dir=output_dir)
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
        y_sel = y.iloc[selected_idx]

        model_cl = train_model(X_sel, y_sel, classifier,
                               name=f"CL_{attack_label}_{int(ratio*100)}", output_dir=output_dir)
        test_model(model_cl, X_test_pca, y_test)

    print("\n" + "All Done".center(60, "="))
