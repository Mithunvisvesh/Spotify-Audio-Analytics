import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("cleaned_tracks.csv")

# 1. Distributions 
fig, axes = plt.subplots(2, 2, figsize=(10,8))
for ax, col in zip(axes.flat, ["energy","valence","tempo","danceability"]):
    sns.histplot(df[col], kde=True, ax=ax); ax.set_title(f"Distribution of {col}")
plt.tight_layout(); plt.savefig("distributions.png"); plt.clf()

# 2. Correlation heatmap 
num_cols = ["danceability","energy","loudness","speechiness","acousticness",
            "instrumentalness","liveness","valence","tempo","popularity"]
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.tight_layout(); plt.savefig("correlation_heatmap.png"); plt.clf()

# 3. Categorical comparisons
genre_counts = df["track_genre"].value_counts()
valid_genres = genre_counts[genre_counts >= 100].index   # kept: prevents a 1-2 track genre from topping the chart
top_genres = (df[df["track_genre"].isin(valid_genres)]
              .groupby("track_genre")["popularity"].mean()
              .sort_values(ascending=False).head(10))

genre_summary = (df[df["track_genre"].isin(valid_genres)]
                 .groupby("track_genre")["popularity"]
                 .agg(["mean", "count"])
                 .sort_values("mean", ascending=False).head(10))
print("\nTop 10 genres by mean popularity (minimum 100 tracks):")
print(genre_summary.round(2))

fig, axes = plt.subplots(1, 3, figsize=(18,5))

sns.boxplot(data=df, x="explicit", y="popularity", ax=axes[0])
axes[0].set_title("Popularity by Explicit Content")
axes[0].set_xlabel("Explicit"); axes[0].set_ylabel("Popularity")

sns.boxplot(data=df, x="mode_1", y="popularity", ax=axes[1])
axes[1].set_title("Popularity by Mode")
axes[1].set_xlabel("Mode (1 = Major, 0 = Minor)"); axes[1].set_ylabel("Popularity")

sns.barplot(x=top_genres.values, y=top_genres.index,
            hue=top_genres.index, palette="viridis", legend=False, ax=axes[2])
axes[2].set_title("Top 10 Genres by Mean Popularity (min. 100 tracks)")
axes[2].set_xlabel("Mean Popularity"); axes[2].set_ylabel("Genre")

plt.tight_layout()
plt.savefig("categorical_comparisons.png")   
plt.clf()

# 4. Mood quadrant: mean popularity by valence/energy region
plt.figure(figsize=(10,7))
hb = plt.hexbin(df["valence"], df["energy"], C=df["popularity"],
                 reduce_C_function=np.mean, gridsize=40, cmap="viridis", mincnt=20)
plt.colorbar(hb, label="Mean Popularity")
plt.axvline(0.5, color="white", ls="--", lw=1)
plt.axhline(0.5, color="white", ls="--", lw=1)
plt.xlabel("Valence"); plt.ylabel("Energy")
plt.title("Mood Quadrant: Mean Popularity by Valence/Energy Region")
plt.tight_layout(); plt.savefig("mood_quadrant.png"); plt.clf()

# 5. Hit vs not-hit overlays 
df["is_hit"] = (df["popularity"] >= 70).astype(int)
for col in ["energy","valence","danceability"]:
    sns.kdeplot(df[df.is_hit==1][col], label="Hit", fill=True)
    sns.kdeplot(df[df.is_hit==0][col], label="Not Hit", fill=True)
    plt.legend(); plt.title(f"{col.capitalize()}: Hit vs Not Hit")
    plt.xlabel(col.capitalize()); plt.ylabel("Density")
    plt.tight_layout(); plt.savefig(f"{col}_hit_vs_not.png"); plt.clf()

# 6. t-test 
hits, non_hits = df[df.is_hit==1]["energy"], df[df.is_hit==0]["energy"]
t, p = stats.ttest_ind(hits, non_hits, equal_var=False)
d = (hits.mean()-non_hits.mean()) / (((hits.std()**2+non_hits.std()**2)/2)**0.5)
print("Hit count:", len(hits), "| Non-hit count:", len(non_hits))
print("Hit mean energy:", round(hits.mean(),4), "| Non-hit mean energy:", round(non_hits.mean(),4))
print(f"t={t:.2f}, p={p:.4g}, Cohen's d={d:.3f}")