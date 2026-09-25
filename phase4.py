import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv("cleaned_tracks.csv")
cluster_features = ["energy_scaled","valence_scaled","danceability_scaled","acousticness_scaled"]
print("Dataset shape:", df.shape); print("Clustering features:", cluster_features)

# 0. Verify Phase 1 scaler — fail loudly, don't refit
scaler = joblib.load("scaler.pkl")
expected = ["danceability","energy","loudness","speechiness","acousticness",
            "instrumentalness","liveness","valence","tempo"]
assert list(scaler.feature_names_in_) == expected, "Scaler mismatch — investigate, do not refit."
print("Scaler verified: feature order matches Phase 1.")

# 1. Elbow method
inertia = []
for k in range(2, 8):
    inertia.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(df[cluster_features]).inertia_)
plt.figure(figsize=(7,5)); plt.plot(range(2,8), inertia, marker="o")
plt.xlabel("Number of Clusters (k)"); plt.ylabel("Inertia"); plt.title("Elbow Method for K-Means")
plt.tight_layout(); plt.savefig("elbow.png"); plt.close()
for k, v in zip(range(2,8), inertia): print(f"k={k}: inertia={v:.2f}")

# 2. Fit final model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["mood_cluster"] = kmeans.fit_predict(df[cluster_features])

# 3. Cluster summary — the only source of truth for interpretation; no labels hard-coded here
cluster_summary = df.groupby("mood_cluster")[["energy","valence","danceability","acousticness"]].mean()
print("\nCluster Summary:\n", cluster_summary.round(3))
cluster_summary.to_csv("cluster_summary.csv")

# 4. Sizes + percentages
counts = df["mood_cluster"].value_counts().sort_index()
pcts = (counts / len(df) * 100).round(2)
print("\nCluster Distribution:")
for c in counts.index: print(f"Cluster {c}: {counts[c]} tracks ({pcts[c]}%)")

# 5. Visualization — sample only for the plot; fit already used all 76,103 tracks
plot_sample = df.sample(5000, random_state=42)
plt.figure(figsize=(9,7))
sns.scatterplot(data=plot_sample, x="valence", y="energy", hue="mood_cluster", palette="Set2", alpha=0.6)
plt.title("Mood Clusters: Valence vs Energy (5,000-track sample)")
plt.xlabel("Valence"); plt.ylabel("Energy")
plt.tight_layout(); plt.savefig("mood_clusters.png"); plt.close()

# 6. Save model + dataset
joblib.dump(kmeans, "kmeans_model.pkl")
df.to_csv("cleaned_tracks_with_clusters.csv", index=False)
print("\nSaved: elbow.png, mood_clusters.png, cluster_summary.csv, kmeans_model.pkl, cleaned_tracks_with_clusters.csv")