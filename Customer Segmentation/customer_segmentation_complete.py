

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Global plot style ────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor":  "#1a1d27",
    "axes.edgecolor":   "#3a3d4d", "axes.labelcolor": "#e0e0e0",
    "xtick.color":      "#a0a0b0", "ytick.color":     "#a0a0b0",
    "text.color":       "#e0e0e0", "grid.color":      "#2a2d3d",
    "grid.linestyle":   "--",      "grid.alpha":      0.5,
    "font.family":      "DejaVu Sans",
})
PALETTE = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A29BFE", "#FD79A8"]

SEGMENT_NAMES = {
    0: "Premium Shoppers",
    1: "Budget Conscious",
    2: "Balanced Buyers",
    3: "Conservative Wealthy",
    4: "Impulse Buyers",
}

# ════════════════════════════════════════════════════════════
# DATASET  (200 customers, 5 natural clusters)
# ════════════════════════════════════════════════════════════
MALL_CUSTOMERS_DATA = {
    "CustomerID": list(range(1, 201)),
    "Gender": [
        'Male','Female','Female','Male','Male','Female','Male','Female','Male','Male',
        'Female','Female','Female','Female','Female','Female','Female','Male','Female','Male',
        'Female','Male','Male','Female','Female','Female','Male','Female','Female','Male',
        'Male','Male','Male','Female','Female','Male','Female','Female','Male','Female',
        'Male','Female','Female','Male','Male','Female','Male','Female','Female','Female',
        'Female','Male','Male','Male','Male','Male','Female','Male','Female','Female',
        'Female','Female','Male','Female','Male','Male','Female','Male','Male','Male',
        'Female','Male','Male','Male','Female','Male','Male','Male','Female','Female',
        'Female','Male','Male','Female','Female','Female','Male','Male','Female','Female',
        'Male','Female','Female','Male','Female','Male','Male','Female','Female','Male',
        'Female','Male','Female','Male','Female','Female','Female','Female','Male','Female',
        'Male','Female','Female','Male','Female','Female','Male','Male','Male','Male',
        'Male','Female','Female','Male','Female','Female','Male','Female','Female','Female',
        'Female','Female','Male','Female','Female','Male','Female','Male','Male','Male',
        'Female','Male','Male','Male','Female','Female','Male','Male','Female','Male',
        'Female','Male','Female','Male','Male','Female','Male','Female','Female','Female',
        'Female','Female','Male','Male','Male','Female','Male','Male','Male','Male',
        'Male','Male','Female','Female','Female','Male','Female','Male','Female','Female',
        'Female','Male','Male','Female','Male','Male','Male','Male','Male','Male',
        'Female','Female','Female','Male','Female','Male','Male','Male','Male','Female',
    ],
    "Age": [
        19,21,20,23,31,22,35,23,64,30,67,35,58,24,37,22,35,20,52,35,
        35,25,46,31,54,29,45,35,40,23,60,21,53,18,49,21,42,30,36,20,
        65,24,48,31,49,24,50,27,29,31,49,33,31,59,50,47,51,69,27,53,
        70,19,67,19,66,23,46,27,63,24,50,28,45,27,49,23,48,40,60,36,
        34,33,50,42,57,23,48,40,65,23,54,38,47,30,60,23,46,36,58,30,
        66,34,50,34,66,21,55,47,62,38,68,27,53,30,52,30,55,24,46,32,
        56,32,60,32,60,23,53,46,43,32,42,34,62,21,59,19,62,22,60,23,
        38,22,37,22,57,22,40,25,45,23,28,29,50,28,66,28,53,34,39,28,
        33,50,28,56,27,44,30,39,39,44,45,35,40,58,32,45,66,36,51,39,
        45,39,26,50,42,55,38,39,47,40,33,47,38,49,26,55,48,30,37,24,
    ],
    "Annual Income (k$)": [
        15,15,16,16,17,17,18,18,19,19,20,20,20,21,21,23,24,25,28,28,
        28,29,30,30,30,30,30,30,31,31,31,31,31,31,31,33,33,33,33,33,
        37,37,38,38,39,39,39,39,40,40,40,40,40,42,42,43,43,43,44,44,
        46,46,46,46,46,47,47,48,48,48,49,49,50,50,50,50,51,54,54,54,
        54,54,54,54,54,54,57,58,58,59,60,60,60,60,60,61,61,62,62,62,
        62,62,62,63,63,63,63,64,64,65,65,65,67,67,69,70,70,70,71,71,
        71,71,71,72,72,72,72,73,73,74,75,76,76,77,77,77,78,78,78,78,
        79,79,79,80,81,85,86,87,87,87,88,88,88,93,97,98,99,99,101,101,
        25,25,26,26,27,27,28,28,29,30,30,31,31,32,32,33,33,34,35,35,
        36,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,
    ],
    "Spending Score (1-100)": [
        39,81,6,77,40,76,6,94,3,72,14,99,15,77,13,79,35,66,29,98,
        35,73,13,73,14,80,9,77,31,77,22,88,17,74,18,76,6,89,29,83,
        63,82,47,73,14,72,7,81,40,90,14,73,4,77,35,82,32,80,9,74,
        54,91,67,83,53,89,42,90,47,73,48,90,31,84,40,91,31,84,23,78,
        43,83,29,62,41,86,32,94,36,54,46,75,48,45,26,58,29,73,35,72,
        20,73,31,72,34,56,18,85,34,76,50,73,37,81,27,79,40,74,36,77,
        37,58,46,60,40,56,28,67,40,73,50,68,42,70,38,79,35,83,20,90,
        31,83,45,75,22,74,25,73,31,79,33,83,31,80,28,80,45,62,24,67,
        46,75,41,77,36,73,40,89,26,59,48,77,35,80,14,90,21,86,44,83,
        50,80,44,83,28,76,16,83,20,84,25,79,49,61,38,89,19,82,29,68,
    ],
}

def load_dataset():
    df = pd.DataFrame(MALL_CUSTOMERS_DATA)
    return df


# ════════════════════════════════════════════════════════════
# EDA
# ════════════════════════════════════════════════════════════
def run_eda(df):
    print("\n" + "="*60)
    print("  1. EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\n  Shape          : {df.shape}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Duplicates     : {df.duplicated().sum()}")
    print("\n── Descriptive Statistics ──")
    print(df.describe().round(2).to_string())
    print("\n── Gender Distribution ──")
    print(df["Gender"].value_counts().to_string())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Mall Customers – EDA", fontsize=18, fontweight="bold",
                 color="#e0e0e0", y=0.98)

    num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    colors   = ["#FF6B6B", "#4ECDC4", "#FFE66D"]

    for ax, col, c in zip(axes[0], num_cols, colors):
        ax.hist(df[col], bins=20, color=c, edgecolor="#0f1117", alpha=0.85)
        ax.set_title(col, fontsize=12); ax.set_xlabel(col); ax.set_ylabel("Count")
        ax.grid(True)

    axes[1][0].scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
                       color="#A29BFE", alpha=0.6, edgecolors="#0f1117", s=40)
    axes[1][0].set_title("Income vs Spending Score", fontsize=12)
    axes[1][0].set_xlabel("Annual Income (k$)"); axes[1][0].set_ylabel("Spending Score"); axes[1][0].grid(True)

    axes[1][1].scatter(df["Age"], df["Spending Score (1-100)"],
                       color="#FD79A8", alpha=0.6, edgecolors="#0f1117", s=40)
    axes[1][1].set_title("Age vs Spending Score", fontsize=12)
    axes[1][1].set_xlabel("Age"); axes[1][1].set_ylabel("Spending Score"); axes[1][1].grid(True)

    gcount = df["Gender"].value_counts()
    axes[1][2].bar(gcount.index, gcount.values, color=["#4ECDC4","#FF6B6B"],
                   edgecolor="#0f1117", width=0.5)
    axes[1][2].set_title("Gender Distribution", fontsize=12)
    axes[1][2].set_ylabel("Count"); axes[1][2].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(); print("\n  ✔  eda_plots.png")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax2,
                annot_kws={"color": "#e0e0e0"}, cbar_kws={"shrink": 0.8})
    ax2.set_xticklabels(ax2.get_xticklabels(), color="#e0e0e0")
    ax2.set_yticklabels(ax2.get_yticklabels(), color="#e0e0e0")
    fig2.suptitle("Correlation Heatmap", fontsize=14, color="#e0e0e0")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(); print("  ✔  correlation_heatmap.png")


# ════════════════════════════════════════════════════════════
# OPTIMAL K
# ════════════════════════════════════════════════════════════
def find_optimal_k(X_scaled):
    print("\n" + "="*60)
    print("  2. FINDING OPTIMAL K (Elbow + Silhouette)")
    print("="*60)
    inertias, sil_scores = [], []
    k_range = range(2, 11)
    for k in k_range:
        km    = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs  = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimal K Selection", fontsize=16, color="#e0e0e0")
    ax1.plot(k_range, inertias,  "o-", color="#4ECDC4", lw=2, ms=7)
    ax1.axvline(5, color="#FF6B6B", ls="--", lw=1.5, label="k=5 (chosen)")
    ax1.set_title("Elbow Method"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
    ax1.legend(); ax1.grid(True)
    ax2.plot(k_range, sil_scores, "s-", color="#FFE66D", lw=2, ms=7)
    ax2.axvline(5, color="#FF6B6B", ls="--", lw=1.5, label="k=5 (chosen)")
    ax2.set_title("Silhouette Score"); ax2.set_xlabel("k"); ax2.set_ylabel("Score")
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig("optimal_k.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(); print("\n  ✔  optimal_k.png")

    print(f"\n  Silhouette scores : { {k:round(s,3) for k,s in zip(k_range,sil_scores)} }")
    return 5


# ════════════════════════════════════════════════════════════
# K-MEANS
# ════════════════════════════════════════════════════════════
def apply_kmeans(df, X_scaled, n_clusters=5):
    print("\n" + "="*60)
    print(f"  3. K-MEANS CLUSTERING  (k={n_clusters})")
    print("="*60)
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels)
    print(f"\n  Silhouette Score : {score:.4f}")
    df = df.copy()
    df["Cluster"] = labels
    print("\n── Raw Cluster Summary ──")
    print(df.groupby("Cluster")[["Age","Annual Income (k$)","Spending Score (1-100)"]].mean().round(2).to_string())
    return df, km


def remap_clusters(df, km, scaler, features):
    """Assign semantic labels 0-4 (no collisions, rank-based)."""
    centres   = scaler.inverse_transform(km.cluster_centers_)
    inc_idx   = features.index("Annual Income (k$)")
    sp_idx    = features.index("Spending Score (1-100)")
    inc_ranks = np.argsort(np.argsort(centres[:, inc_idx]))
    sp_ranks  = np.argsort(np.argsort(centres[:, sp_idx]))
    combined  = inc_ranks + sp_ranks
    sorted_by = np.argsort(combined)
    semantic  = [None] * len(sorted_by)
    semantic[sorted_by[0]] = 1          # lowest  → Budget Conscious
    semantic[sorted_by[-1]] = 0         # highest → Premium Shoppers
    mid = sorted_by[1:-1]
    mid_inc = sorted([(centres[r, inc_idx], r) for r in mid])
    semantic[mid_inc[0][1]] = 4         # low income, high spend → Impulse Buyers
    semantic[mid_inc[-1][1]] = 3        # high income, low spend → Conservative Wealthy
    semantic[mid_inc[1][1]] = 2         # middle → Balanced Buyers
    df = df.copy()
    df["Cluster"] = df["Cluster"].map({r: s for r, s in enumerate(semantic)})
    df["Segment"] = df["Cluster"].map(SEGMENT_NAMES)
    return df


# ════════════════════════════════════════════════════════════
# CLUSTER PLOT  (Income vs Spending 2-D)
# ════════════════════════════════════════════════════════════
def plot_clusters_2d(df, km, scaler, features):
    fig, ax = plt.subplots(figsize=(10, 7))
    for cid in range(5):
        mask = df["Cluster"] == cid
        ax.scatter(df.loc[mask, "Annual Income (k$)"],
                   df.loc[mask, "Spending Score (1-100)"],
                   color=PALETTE[cid], alpha=0.75, edgecolors="#0f1117", s=55,
                   label=f"C{cid}: {SEGMENT_NAMES[cid]}")
    centres = scaler.inverse_transform(km.cluster_centers_)
    ax.scatter(centres[:, features.index("Annual Income (k$)")],
               centres[:, features.index("Spending Score (1-100)")],
               marker="X", s=220, color="white", edgecolors="#0f1117",
               linewidths=1.5, zorder=5, label="Centroids")
    ax.set_title("K-Means Customer Segments\n(Annual Income vs Spending Score)",
                 fontsize=14, color="#e0e0e0")
    ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score (1-100)")
    ax.legend(fontsize=9, framealpha=0.3); ax.grid(True)
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(); print("  ✔  kmeans_clusters.png")


# ════════════════════════════════════════════════════════════
# PCA
# ════════════════════════════════════════════════════════════
def plot_pca(df, X_scaled):
    print("\n" + "="*60)
    print("  4. PCA VISUALISATION")
    print("="*60)
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    ev     = pca.explained_variance_ratio_
    print(f"\n  PC1 explained variance : {ev[0]:.2%}")
    print(f"  PC2 explained variance : {ev[1]:.2%}")
    print(f"  Total (2 PCs)          : {ev.sum():.2%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("PCA – Dimensionality Reduction", fontsize=15, color="#e0e0e0")
    for cid in range(5):
        mask = df["Cluster"] == cid
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    color=PALETTE[cid], alpha=0.75, edgecolors="#0f1117", s=50,
                    label=f"C{cid}: {SEGMENT_NAMES[cid]}")
    ax1.set_title(f"PCA Scatter  (PC1={ev[0]:.1%}, PC2={ev[1]:.1%})", fontsize=12)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.legend(fontsize=8, framealpha=0.3); ax1.grid(True)

    pca_full = PCA(random_state=42).fit(X_scaled)
    evr = pca_full.explained_variance_ratio_
    ax2.bar(range(1, len(evr)+1), evr, color="#4ECDC4", edgecolor="#0f1117")
    ax2.plot(range(1, len(evr)+1), np.cumsum(evr), "o--",
             color="#FFE66D", lw=2, ms=6, label="Cumulative")
    ax2.axhline(0.95, color="#FF6B6B", ls="--", lw=1.5, label="95% threshold")
    ax2.set_title("Explained Variance per Component", fontsize=12)
    ax2.set_xlabel("Principal Component"); ax2.set_ylabel("Explained Variance Ratio")
    ax2.legend(fontsize=9, framealpha=0.3); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("pca_visualization.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(); print("\n  ✔  pca_visualization.png")


# ════════════════════════════════════════════════════════════
# t-SNE
# ════════════════════════════════════════════════════════════
def plot_tsne(df, X_scaled):
    print("\n" + "="*60)
    print("  5. t-SNE VISUALISATION")
    print("="*60)
    print("\n  Running t-SNE (perplexity=30) …")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cid in range(5):
        mask = df["Cluster"] == cid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=PALETTE[cid], alpha=0.75, edgecolors="#0f1117", s=55,
                   label=f"C{cid}: {SEGMENT_NAMES[cid]}")
    ax.set_title("t-SNE Customer Segments (perplexity=30)", fontsize=14, color="#e0e0e0")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9, framealpha=0.3); ax.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_visualization.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(); print("  ✔  tsne_visualization.png")


# ════════════════════════════════════════════════════════════
# MARKETING DASHBOARD
# ════════════════════════════════════════════════════════════
def plot_marketing_dashboard(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Customer Segment Dashboard  |  Marketing Strategy Overview",
                 fontsize=17, fontweight="bold", color="#e0e0e0", y=0.98)

    counts = pd.Series({i: (df["Cluster"]==i).sum() for i in range(5)})
    names  = [SEGMENT_NAMES[i] for i in range(5)]
    bcolors = [PALETTE[i] for i in range(5)]

    # Pie
    wedges, _, auto = axes[0][0].pie(
        counts.values, colors=PALETTE, autopct="%1.1f%%", startangle=140,
        pctdistance=0.75, wedgeprops=dict(edgecolor="#0f1117", linewidth=1.5),
        textprops=dict(color="#e0e0e0", fontsize=9))
    axes[0][0].legend(wedges, names, loc="upper left",
                      bbox_to_anchor=(-0.35, 1.05), fontsize=7.5, framealpha=0.2)
    axes[0][0].set_title("Segment Size", fontsize=12, color="#e0e0e0")

    # Avg income
    avg_inc = pd.Series({i: df.loc[df["Cluster"]==i, "Annual Income (k$)"].mean() for i in range(5)})
    axes[0][1].bar(range(5), avg_inc.values, color=bcolors, edgecolor="#0f1117")
    axes[0][1].set_xticks(range(5)); axes[0][1].set_xticklabels([f"C{i}" for i in range(5)])
    axes[0][1].set_title("Avg Annual Income (k$)", fontsize=12, color="#e0e0e0")
    axes[0][1].set_ylabel("Income (k$)"); axes[0][1].grid(True, axis="y")

    # Avg spending
    avg_sp = pd.Series({i: df.loc[df["Cluster"]==i, "Spending Score (1-100)"].mean() for i in range(5)})
    axes[0][2].bar(range(5), avg_sp.values, color=bcolors, edgecolor="#0f1117")
    axes[0][2].set_xticks(range(5)); axes[0][2].set_xticklabels([f"C{i}" for i in range(5)])
    axes[0][2].set_title("Avg Spending Score", fontsize=12, color="#e0e0e0")
    axes[0][2].set_ylabel("Spending Score"); axes[0][2].grid(True, axis="y")

    # Age distribution per segment
    for cid in range(5):
        axes[1][0].hist(df.loc[df["Cluster"]==cid, "Age"], bins=12,
                        color=PALETTE[cid], alpha=0.6, edgecolor="#0f1117",
                        label=f"C{cid}")
    axes[1][0].set_title("Age Distribution by Segment", fontsize=12, color="#e0e0e0")
    axes[1][0].set_xlabel("Age"); axes[1][0].set_ylabel("Count")
    axes[1][0].legend(fontsize=8, framealpha=0.3); axes[1][0].grid(True)

    # Income vs spending coloured
    for cid in range(5):
        mask = df["Cluster"] == cid
        axes[1][1].scatter(df.loc[mask, "Annual Income (k$)"],
                           df.loc[mask, "Spending Score (1-100)"],
                           color=PALETTE[cid], alpha=0.6, s=50,
                           edgecolors="#0f1117", label=f"C{cid}")
    axes[1][1].set_title("Income vs Spending (Coloured)", fontsize=12, color="#e0e0e0")
    axes[1][1].set_xlabel("Annual Income (k$)"); axes[1][1].set_ylabel("Spending Score")
    axes[1][1].legend(fontsize=8, framealpha=0.3); axes[1][1].grid(True)

    # Gender stacked bar
    bottom = np.zeros(5)
    for gender, c in zip(["Female", "Male"], ["#4ECDC4", "#FF6B6B"]):
        vals = np.array([(df[(df["Cluster"]==i) & (df["Gender"]==gender)].shape[0]) for i in range(5)], dtype=float)
        axes[1][2].bar(range(5), vals, bottom=bottom, color=c, edgecolor="#0f1117", label=gender)
        bottom += vals
    axes[1][2].set_xticks(range(5)); axes[1][2].set_xticklabels([f"C{i}" for i in range(5)])
    axes[1][2].set_title("Gender Split per Segment", fontsize=12, color="#e0e0e0")
    axes[1][2].set_ylabel("Count")
    axes[1][2].legend(fontsize=9, framealpha=0.3); axes[1][2].grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("marketing_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(); print("  ✔  marketing_dashboard.png")


# ════════════════════════════════════════════════════════════
# MARKETING STRATEGIES
# ════════════════════════════════════════════════════════════
STRATEGIES = {
    0: {
        "profile":    "High Income · High Spending · Age ~30-45",
        "size":       "~22%",
        "strategies": [
            " VIP loyalty programme with exclusive rewards & early access",
            " Promote luxury / premium product lines",
            " Personalised concierge service & private shopping events",
            " Targeted email & app push for new high-end arrivals",
            " Referral incentives to attract similar high-value friends",
        ],
    },
    1: {
        "profile":    "Low Income · Low Spending · Age ~30-60",
        "size":       "~18%",
        "strategies": [
            "  Flash sales, deep discounts & clearance promotions",
            " Bundle deals that maximise perceived value",
            " Instalment / buy-now-pay-later payment options",
            " Highlight everyday essentials and value brands",
            " SMS/WhatsApp campaigns for time-limited offers",
        ],
    },
    2: {
        "profile":    "Medium Income · Medium Spending · Age ~25-55",
        "size":       "~24%",
        "strategies": [
            " Mid-tier loyalty points programme with redeemable rewards",
            " Cross-sell & upsell with 'you may also like' suggestions",
            " Seasonal promotions aligned with lifestyle events",
            " Educational email content building brand trust",
            " 'Complete the look' or curated bundle recommendations",
        ],
    },
    3: {
        "profile":    "High Income · Low Spending · Age ~40-65",
        "size":       "~20%",
        "strategies": [
            " Showcase product quality, durability & heritage story",
            " Address hesitation with free trials, demos & guarantees",
            " Content marketing: case studies, reviews, expert opinions",
            " Build long-term trust with consistent post-purchase follow-up",
            " Invite to exclusive in-store or online brand experiences",
        ],
    },
    4: {
        "profile":    "Low Income · High Spending · Age ~18-35",
        "size":       "~16%",
        "strategies": [
            " Flash sales & countdown timers to drive urgency",
            " Heavy social-media presence (Instagram, TikTok, Reels)",
            " Gamification: spin-the-wheel, mystery boxes, surprise discounts",
            "Real-time push notifications for trending / limited items",
            "Flexible micro-payment plans to remove purchase friction",
        ],
    },
}

def print_marketing_strategies(df):
    print("\n" + "="*60)
    print("  6. MARKETING STRATEGIES BY SEGMENT")
    print("="*60)
    total = len(df)
    for cid, info in STRATEGIES.items():
        n = (df["Cluster"] == cid).sum()
        print(f"\n  ── Cluster {cid}: {SEGMENT_NAMES[cid]}  ({n} customers, {n/total:.0%}) ──")
        print(f"     Profile : {info['profile']}")
        print("     Strategies:")
        for s in info["strategies"]:
            print(f"       {s}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("  CUSTOMER SEGMENTATION — MALL CUSTOMERS DATASET")
    print("="*60)

    # 1. Load dataset
    df = load_dataset()
    df.to_csv("mall_customers.csv", index=False)
    print(f"\n  Dataset : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Saved   → mall_customers.csv")

    # 2. EDA
    run_eda(df)

    # 3. Preprocess
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X        = df[features].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Optimal K
    optimal_k = find_optimal_k(X_scaled)

    # 5. K-Means
    df, km = apply_kmeans(df, X_scaled, n_clusters=optimal_k)

    # 6. Remap to semantic labels
    df = remap_clusters(df, km, scaler, features)
    print("\n── Final Segment Summary ──")
    print(df.groupby(["Cluster","Segment"])[["Age","Annual Income (k$)","Spending Score (1-100)"]].mean().round(2).to_string())

    # 7. Visualisations
    print("\n" + "="*60)
    print("  SAVING PLOTS")
    print("="*60)
    plot_clusters_2d(df, km, scaler, features)
    plot_pca(df, X_scaled)
    plot_tsne(df, X_scaled)
    plot_marketing_dashboard(df)

    # 8. Marketing strategies
    print_marketing_strategies(df)

    # 9. Summary
    print("\n" + "="*60)
    print("  ALL OUTPUT FILES")
    print("="*60)
    for f in [
        "mall_customers.csv         – dataset (200 customers)",
        "eda_plots.png              – histograms, scatter plots",
        "correlation_heatmap.png    – feature correlations",
        "optimal_k.png              – elbow & silhouette curves",
        "kmeans_clusters.png        – K-Means 2-D cluster plot",
        "pca_visualization.png      – PCA scatter + variance",
        "tsne_visualization.png     – t-SNE scatter",
        "marketing_dashboard.png    – full segment dashboard",
    ]:
        print(f"    {f}")
    print("\n  Done! ")


if __name__ == "__main__":
    main()
