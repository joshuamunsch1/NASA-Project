import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import scanpy as sc
import anndata as ad

# -------------------------
# 1. Read data
# -------------------------
counts = pd.read_csv("normalized_counts.csv", index_col=0)   # genes x samples
coldata = pd.read_csv("metadata.csv")             # expects 'sample' and 'study' columns (optional: 'group')

# Sanity checks & align
if 'sample' not in coldata.columns:
    raise ValueError("coldata must contain a column named 'sample' matching counts column names.")
samples_counts = list(counts.columns)
samples_meta = list(coldata['sample'])

missing_in_meta = set(samples_counts) - set(samples_meta)

missing_in_counts = set(samples_meta) - set(samples_counts)
if missing_in_meta or missing_in_counts:
    raise ValueError(f"Mismatch between counts columns and coldata 'sample' column. "
                     f"Missing in meta: {missing_in_meta}; missing in counts: {missing_in_counts}")

# Reorder metadata to match counts columns
coldata = coldata.set_index('sample').loc[samples_counts].reset_index()

# Ensure study column exists
if 'study' not in coldata.columns:
    raise ValueError("coldata must contain a 'study' column for batch labels.")

# Optional group column for marker shapes
use_group = ('group' in coldata.columns)

# -------------------------
# 2. PCA helper function
# -------------------------
def pca_on_matrix(matrix_samples_by_genes, n_components=10):
    """
    matrix_samples_by_genes: numpy array shape (n_samples, n_genes)
    Returns: pcs (n_samples, n_components), explained_variance_ratio_
    """
    # Standardize features (genes) before PCA for comparability
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(matrix_samples_by_genes)
    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(Xs)
    return pcs, pca.explained_variance_ratio_

# -------------------------
# 2B. Batch effect evaluation helpers
# -------------------------
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

def evaluate_batch_effect_clustering(X, batch_labels, dataset_name="Dataset"):
    """
    Evaluate batch effect using silhouette score (no ML training needed).
    Better for very small sample sizes.
    
    Silhouette score ranges from -1 to 1:
    - Close to 1: batches are well-separated (bad - strong batch effect)
    - Close to 0: batches are overlapping (good - no batch effect)
    - Negative: batches are mixed together (excellent!)
    """
    from sklearn.preprocessing import LabelEncoder
    
    n_samples = X.shape[0]
    le = LabelEncoder()
    y_encoded = le.fit_transform(batch_labels)
    n_batches = len(np.unique(y_encoded))
    
    # Compute silhouette score
    sil_score = silhouette_score(X, y_encoded, metric='euclidean')
    
    print(f"\n{'='*60}")
    print(f"Batch Separation (Silhouette Score): {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {n_samples}, Batches: {n_batches}")
    print(f"Silhouette score: {sil_score:.3f}")
    print(f"  > 0.5: Strong batch separation (BAD)")
    print(f"  0.2-0.5: Moderate batch separation")
    print(f"  0-0.2: Weak batch separation (GOOD)")
    print(f"  < 0: Batches well-mixed (EXCELLENT)")
    
    if sil_score > 0.5:
        interpretation = "POOR - strong batch effects"
    elif sil_score > 0.2:
        interpretation = "MODERATE - noticeable batch effects"
    elif sil_score > 0:
        interpretation = "GOOD - weak batch effects"
    else:
        interpretation = "EXCELLENT - batches well-mixed"
    
    print(f"Interpretation: {interpretation}")
    print(f"{'='*60}\n")
    
    return {
        'silhouette_score': sil_score,
        'n_samples': n_samples,
        'n_batches': n_batches,
        'interpretation': interpretation
    }


def evaluate_batch_effect_knn(X, batch_labels, dataset_name="Dataset", n_components=5):
    """
    Evaluate batch effect using k-NN on PCA-reduced data.
    Uses PCA to avoid curse of dimensionality with small N.
    
    For small sample sizes, this is more reliable than logistic regression.
    """
    from sklearn.preprocessing import LabelEncoder
    
    n_samples, n_features = X.shape
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(batch_labels)
    unique_batches = le.classes_
    n_batches = len(unique_batches)
    random_accuracy = 1.0 / n_batches
    
    print(f"\nDebug {dataset_name}:")
    print(f"  Sample count: {n_samples}, Feature count: {n_features}")
    print(f"  Batches: {unique_batches}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # PCA dimensionality reduction (critical for small N!)
    n_components = min(n_components, n_samples - 1, n_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_features} features -> {n_components} PCs (explaining {var_explained:.1%} variance)")
    
    # Use Leave-One-Out CV for small samples
    from sklearn.model_selection import LeaveOneOut
    cv = LeaveOneOut()
    
    # k-NN with k=3 (simple, works well with small samples)
    k = min(3, n_samples - 1)
    clf = KNeighborsClassifier(n_neighbors=k)
    
    cv_scores = cross_val_score(clf, X_pca, y, cv=cv, scoring='accuracy')
    mean_acc = cv_scores.mean()
    std_acc = cv_scores.std()
    
    print(f"\n{'='*60}")
    print(f"Batch Effect Assessment (k-NN on PCs): {dataset_name}")
    print(f"{'='*60}")
    print(f"Method: {k}-NN on {n_components} PCs with Leave-One-Out CV")
    print(f"Number of batches: {n_batches}")
    print(f"Random baseline accuracy: {random_accuracy:.3f}")
    print(f"Cross-val accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Accuracy above random: {(mean_acc - random_accuracy):.3f}")
    
    # Interpretation
    if mean_acc < random_accuracy + 0.15:
        interpretation = "EXCELLENT - batches well-mixed"
    elif mean_acc < random_accuracy + 0.25:
        interpretation = "GOOD - minor batch effects"
    elif mean_acc < random_accuracy + 0.40:
        interpretation = "MODERATE - noticeable batch effects"
    else:
        interpretation = "POOR - strong batch effects"
    
    print(f"Interpretation: {interpretation}")
    print(f"{'='*60}\n")
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'random_baseline': random_accuracy,
        'n_batches': n_batches,
        'interpretation': interpretation,
        'n_pcs': n_components,
        'variance_explained': var_explained
    }

# -------------------------
# 3. PCA BEFORE normalization (log1p raw counts)
# -------------------------
# Use log1p on raw counts (no library-size correction)
log_raw = np.log1p(counts)            # genes x samples
X_before = log_raw.T.values           # samples x genes

pcs_before, evr_before = pca_on_matrix(X_before, n_components=10)
pca_before_df = pd.DataFrame(pcs_before[:, :2], index=counts.columns, columns=["PC1","PC2"])
pca_before_df = pd.concat([pca_before_df, coldata.set_index('sample')], axis=1)

# -------------------------
# 3B. Evaluate batch effect BEFORE correction
# -------------------------
# Ensure labels match the sample order in X_before (which is counts.columns)
batch_labels_before = coldata.set_index('sample').loc[counts.columns, 'study'].values

# Method 1: Silhouette score (no ML needed, works best for small N)
sil_results_before = evaluate_batch_effect_clustering(X_before, batch_labels_before, 
                                                       dataset_name="BEFORE Harmonization")

# Method 2: k-NN on PCA (more interpretable as accuracy)
knn_results_before = evaluate_batch_effect_knn(X_before, batch_labels_before, 
                                               dataset_name="BEFORE Harmonization",
                                               n_components=5)

# -------------------------
# 4. Harmonisation: CPM -> log1p -> ComBat
# -------------------------
lib_sizes = counts.sum(axis=0)
cpm = counts.divide(lib_sizes, axis=1) * 1e6
#log_cpm = np.log1p(counts)   # genes x samples
log_cpm = cpm

# Build AnnData (samples x genes)
adata = ad.AnnData(X=log_cpm.T.values, obs=coldata.set_index('sample'), var=pd.DataFrame(index=log_cpm.index))
# scanpy's combat will correct adata.X in-place using adata.obs['study']
sc.pp.combat(adata, key='study')   # performs ComBat on log-transformed data

# Extract corrected matrix (samples x genes)
X_corrected = adata.X                # samples x genes, numpy array

OUT_CORRECTED = "dataset_C_corrected_logCPM_full.csv"   # genes x samples
OUT_CONCAT = "dataset_C_ml_concat.csv"            # samples x genes + label (last col)
OUT_COLDATA = "dataset_C_coldata_aligned.csv"
OUT_GENES = "dataset_C_genes.txt"

coldata_aligned = coldata.set_index('sample').loc[samples_counts].reset_index()
#coldata_aligned.to_csv(OUT_COLDATA, index=False)

if hasattr(X_corrected, "toarray"):
    X_corrected = X_corrected.toarray()

corrected_df = pd.DataFrame(X_corrected.T, index=log_cpm.index, columns=adata.obs_names)  # genes x samples

# Save corrected genes x samples matrix
corrected_df.to_csv(OUT_CORRECTED, float_format="%.6g")

# --- Build ML-ready DataFrame (samples x genes + label last) ---
genes = list(corrected_df.index)
ml_features = corrected_df.T.copy()    # now samples x genes

# Create label column numeric: map 'group' to 0/1 (alphabetical stable map)
group_series = coldata_aligned.set_index('sample').loc[ml_features.index, 'group'].astype(str)
unique_groups = sorted(group_series.unique())
label_map = {g: i for i, g in enumerate(unique_groups)}
labels_numeric = group_series.map(label_map)

ml_concat = ml_features.copy()
ml_concat['label'] = labels_numeric.values  # label placed as last column

# Save the concat dataframe (samples x genes + label)
#ml_concat.to_csv(OUT_CONCAT, index=True)

# Save genes list (ordered)
#with open(OUT_GENES, "w") as fh:
#    for g in genes:
#        fh.write(f"{g}\n")

print("\nSaved:")
print(" - corrected expression (genes x samples):", OUT_CORRECTED)
print(" - ML concat (samples x genes + label):", OUT_CONCAT)
print(" - aligned coldata:", OUT_COLDATA)
print(" - genes list:", OUT_GENES)
print("Label mapping (group -> numeric):", label_map)

# -------------------------
# 4B. Evaluate batch effect AFTER correction
# -------------------------
# Ensure labels match the sample order in X_corrected (which is adata.obs_names)
batch_labels_after = adata.obs.loc[adata.obs_names, 'study'].values

# Diagnostic check
print(f"\nDiagnostic: X_corrected shape: {X_corrected.shape}")
print(f"Diagnostic: batch_labels_after shape: {batch_labels_after.shape}")
print(f"Diagnostic: Unique batches: {np.unique(batch_labels_after)}")
print(f"Diagnostic: Batch counts: {pd.Series(batch_labels_after).value_counts().to_dict()}")

# Method 1: Silhouette score (no ML needed, works best for small N)
sil_results_after = evaluate_batch_effect_clustering(X_corrected, batch_labels_after, 
                                                      dataset_name="AFTER Harmonization")

# Method 2: k-NN on PCA (more interpretable as accuracy)
knn_results_after = evaluate_batch_effect_knn(X_corrected, batch_labels_after, 
                                              dataset_name="AFTER Harmonization",
                                              n_components=5)

# -------------------------
# 4C. Summary comparison
# -------------------------
print(f"\n{'#'*70}")
print("BATCH CORRECTION EFFECTIVENESS SUMMARY")
print(f"{'#'*70}")
print("\nMETHOD 1: Silhouette Score (clustering-based, no ML)")
print(f"  Silhouette BEFORE: {sil_results_before['silhouette_score']:+.3f} ")
print(f"  Silhouette AFTER:  {sil_results_after['silhouette_score']:+.3f} ")
print(f"  Reduction:         {sil_results_before['silhouette_score'] - sil_results_after['silhouette_score']:+.3f}")

print("\nMETHOD 2: k-NN Classification Accuracy")
print(f"  Accuracy BEFORE: {knn_results_before['mean_accuracy']:.3f} ")
print(f"  Accuracy AFTER:  {knn_results_after['mean_accuracy']:.3f} ")
print(f"  Reduction:       {knn_results_before['mean_accuracy'] - knn_results_after['mean_accuracy']:.3f}")

print(f"{'#'*70}\n")

# Save batch prediction results
batch_results = pd.DataFrame({
    'Metric': ['Silhouette_Score', 'kNN_Accuracy', 'kNN_Std', 'Random_Baseline'],
    'Before_Correction': [
        sil_results_before['silhouette_score'],
        knn_results_before['mean_accuracy'],
        knn_results_before['std_accuracy'],
        knn_results_before['random_baseline']
    ],
    'After_Correction': [
        sil_results_after['silhouette_score'],
        knn_results_after['mean_accuracy'],
        knn_results_after['std_accuracy'],
        knn_results_after['random_baseline']
    ]
})
#batch_results.to_csv("batch_effect_metrics.csv", index=False)

# PCA on corrected data
pcs_after, evr_after = pca_on_matrix(X_corrected, n_components=10)
pca_after_df = pd.DataFrame(pcs_after[:, :2], index=adata.obs_names, columns=["PC1","PC2"])
pca_after_df = pd.concat([pca_after_df, adata.obs.reset_index(drop=True).set_index(adata.obs_names)], axis=1)

# Save coordinates
#pca_after_df.to_csv("pca_after_coords.csv")

# -------------------------
# 5. Plot side-by-side PCAs with batch accuracy annotation
# -------------------------
studies = coldata['study'].astype(str)
unique_studies = sorted(coldata['study'].unique())
n_studies = len(unique_studies)

# Color palette
if n_studies <= 10:
    cmap = plt.get_cmap('tab10')
else:
    cmap = plt.get_cmap('tab20')
color_map = {study: cmap(i % cmap.N) for i, study in enumerate(unique_studies)}

# Marker shapes for group if present
marker_list = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
group_map = {}
if use_group:
    unique_groups = sorted(coldata['group'].astype(str).unique())
    for i, grp in enumerate(unique_groups):
        group_map[grp] = marker_list[i % len(marker_list)]

fig, axes = plt.subplots(1, 2, figsize=(16,6), sharex=False, sharey=False)
panels = [
    ("Before Integration", pca_before_df, evr_before, sil_results_before, knn_results_before),
    ("After Integration", pca_after_df, evr_after, sil_results_after, knn_results_after)
]

for ax, (title, pca_df, evr, sil_res, knn_res) in zip(axes, panels):
    for study in unique_studies:
        mask = (pca_df['study'] == study)
        if use_group:
            # plot each group separately for shapes
            for grp, marker in group_map.items():
                mask2 = mask & (pca_df['group'].astype(str) == grp)
                if mask2.sum() == 0:
                    continue
                ax.scatter(pca_df.loc[mask2, 'PC1'],
                           pca_df.loc[mask2, 'PC2'],
                           label=f"{study} | {grp}",
                           s=60, alpha=0.9, edgecolor='k', marker=marker, color=color_map[study])
        else:
            ax.scatter(pca_df.loc[mask, 'PC1'],
                       pca_df.loc[mask, 'PC2'],
                       label=str(study),
                       s=70, alpha=0.9, edgecolor='k', color=color_map[study])

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
    
    # Add both metrics to title
    sil = sil_res['silhouette_score']
    acc = knn_res['mean_accuracy']
    ax.set_title(f"{title}\nSilhouette: {sil:+.3f} | k-NN accuracy: {acc:.3f}", 
                 fontsize=11)
    ax.grid(False)
    ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
#plt.savefig("pca_before_after_integration1.png", dpi=300, bbox_inches='tight')
plt.show()

print("Saved PCA coords and figure:")
print(" - pca_before_coords.csv")
print(" - pca_after_coords.csv") 
print(" - pca_before_after_harmonization.png")
print(" - batch_effect_metrics.csv")