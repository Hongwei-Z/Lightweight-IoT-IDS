import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load train sets
train_paths = [f'./Processed_Datasets/Train_{i}.csv' for i in range(0, 8)]
dfs = [pd.read_csv(p) for p in train_paths]
df_all = pd.concat(dfs, ignore_index=True)
X_all = df_all.drop("Label", axis=1)
y_all = (df_all["Label"] != 0).astype(int)
print(f"All training data loaded: {X_all.shape}.")

# Get MI Scores
mi_scores = mutual_info_classif(X_all, y_all, random_state=42)
feature_names = X_all.columns
sorted_idx = np.argsort(mi_scores)[::-1]
sorted_features = feature_names[sorted_idx]
sorted_scores = mi_scores[sorted_idx]

print("\nGlobal Mutual Information Feature Scores:")
for feature, score in zip(sorted_features, sorted_scores):
    print(f"{feature}: {score}")

plt.figure(figsize=(12, 8))
plt.bar(range(len(sorted_scores)), sorted_scores)
plt.xticks(range(len(sorted_scores)), sorted_features, rotation=90)
plt.title("Global Mutual Information Feature Scores")
plt.tight_layout()
plt.grid()
plt.savefig("Global_MI_Scores.png")
plt.show()

# Select Top-K features
top_k = int(input("\nPlease enter MI Top-K number: "))
selected_idx = sorted_idx[:top_k]
selected_feature_names = feature_names[selected_idx]
print(f"Selected top-{top_k} features: {selected_feature_names.tolist()}")

X_all_mi = X_all.iloc[:, selected_idx]

# PCA
pca_full = PCA().fit(X_all_mi)
explained_var_ratio = np.cumsum(pca_full.explained_variance_ratio_)

print("\nPCA:")
for i, variance in enumerate(explained_var_ratio, 1):
    print(f"{i}\t{variance}")

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(explained_var_ratio)+1), explained_var_ratio, marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("PCA Explained Variance Curve (Global)")
plt.grid()
plt.savefig("Global_PCA_Variance.png")
plt.show()

var_threshold = 0.9

# Filter
n_components = np.argmax(explained_var_ratio >= var_threshold) + 1
pca = PCA(n_components=n_components, random_state=42)
X_all_pca = pca.fit_transform(X_all_mi)
print(f"\nPCA selected {n_components} principal components to retain {var_threshold*100:.1f}% variance.")

# Get Elbow curve
print("\nComputing Elbow curve...")
inertia_list = []
K_range = range(2, 20)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_all_pca)
    inertia_list.append(km.inertia_)

print("\nNumber of Clusters\tInertia")
for k, inertia in zip(K_range, inertia_list):
    print(f"{k}\t{inertia}")

plt.figure(figsize=(12, 8))
plt.plot(K_range, inertia_list, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Global Elbow Method for Optimal k')
plt.grid()
plt.savefig("Global_Elbow.png")
plt.show()

optimal_k = int(input("\nEnter the optimal k: "))

# Save the configurations
joblib.dump(selected_idx, "global_selected_features.pkl")
joblib.dump(pca, "global_pca_model.pkl")
joblib.dump(optimal_k, "global_optimal_k.pkl")
print("Global configurations saved.")

print(f"\nTop-{top_k} features, PCA {n_components} dimensions, optimal k={optimal_k}.")
