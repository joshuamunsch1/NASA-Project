import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import anndata as ad

counts = pd.read_csv("counts.csv", index_col=0)
coldata = pd.read_csv("metadat.csv")

if 'sample' not in coldata.columns:
    raise ValueError
samples_counts = list(counts.columns)
samples_meta = list(coldata['sample'])
missing_in_meta = set(samples_counts) - set(samples_meta)
missing_in_counts = set(samples_meta) - set(samples_counts)
if missing_in_meta or missing_in_counts:
    raise ValueError
coldata = coldata.set_index('sample').loc[samples_counts].reset_index()

if 'study' not in coldata.columns:
    raise ValueError

use_group = ('group' in coldata.columns)

def pca_on_matrix(matrix_samples_by_genes, n_components=10):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(matrix_samples_by_genes)
    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(Xs)
    return pcs, pca.explained_variance_ratio_

log_raw = np.log1p(counts)
X_before = log_raw.T.values

pcs_before, evr_before = pca_on_matrix(X_before, n_components=10)
pca_before_df = pd.DataFrame(pcs_before[:, :2], index=counts.columns, columns=["PC1","PC2"])
pca_before_df = pd.concat([pca_before_df, coldata.set_index('sample')], axis=1)

lib_sizes = counts.sum(axis=0)
cpm = counts.divide(lib_sizes, axis=1) * 1e6
log_cpm = np.log1p(cpm)

adata = ad.AnnData(X=log_cpm.T.values, obs=coldata.set_index('sample'), var=pd.DataFrame(index=log_cpm.index))
sc.pp.combat(adata, key='study')

X_corrected = adata.X

OUT_CORRECTED = "dataset_C_corrected_logCPM.csv"
OUT_CONCAT = "dataset_C_ml_concat.csv"
OUT_COLDATA = "dataset_C_coldata_aligned.csv"
OUT_GENES = "dataset_C_genes.txt"

coldata_aligned = coldata.set_index('sample').loc[samples_counts].reset_index()
coldata_aligned.to_csv(OUT_COLDATA, index=False)

if hasattr(X_corrected, "toarray"):
    X_corrected = X_corrected.toarray()

corrected_df = pd.DataFrame(X_corrected.T, index=log_cpm.index, columns=adata.obs_names)

corrected_df.to_csv(OUT_CORRECTED, float_format="%.6g")

genes = list(corrected_df.index)
ml_features = corrected_df.T.copy()

group_series = coldata_aligned.set_index('sample').loc[ml_features.index, 'group'].astype(str)
unique_groups = sorted(group_series.unique())
label_map = {g: i for i, g in enumerate(unique_groups)}
labels_numeric = group_series.map(label_map)

ml_concat = ml_features.copy()
ml_concat['label'] = labels_numeric.values

ml_concat.to_csv(OUT_CONCAT, index=True)

with open(OUT_GENES, "w") as fh:
    for g in genes:
        fh.write(f"{g}\n")

print("Saved:")
print(" - corrected expression (genes x samples):", OUT_CORRECTED)
print(" - ML concat (samples x genes + label):", OUT_CONCAT)
print(" - aligned coldata:", OUT_COLDATA)
print(" - genes list:", OUT_GENES)
print("Label mapping (group -> numeric):", label_map)

pcs_after, evr_after = pca_on_matrix(X_corrected, n_components=10)
pca_after_df = pd.DataFrame(pcs_after[:, :2], index=adata.obs_names, columns=["PC1","PC2"])
pca_after_df = pd.concat([pca_after_df, adata.obs.reset_index(drop=True).set_index(adata.obs_names)], axis=1)

pca_after_df.to_csv("pca_after_coords.csv")

studies = coldata['study'].astype(str)
unique_studies = sorted(coldata['study'].unique())
n_studies = len(unique_studies)

if n_studies <= 10:
    cmap = plt.get_cmap('tab10')
else:
    cmap = plt.get_cmap('tab20')
color_map = {study: cmap(i % cmap.N) for i, study in enumerate(unique_studies)}

marker_list = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
group_map = {}
if use_group:
    unique_groups = sorted(coldata['group'].astype(str).unique())
    for i, grp in enumerate(unique_groups):
        group_map[grp] = marker_list[i % len(marker_list)]

fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=False, sharey=False)
panels = [("Before normalisation\n(log1p raw counts)", pca_before_df, evr_before),
          ("After normalisation\n(CPM -> log1p -> ComBat)", pca_after_df, evr_after)]

for ax, (title, pca_df, evr) in zip(axes, panels):
    for study in unique_studies:
        mask = (pca_df['study'] == study)
        if use_group:
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
    ax.set_title(title)
    ax.grid(False)
    ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("pca_before_after_unification.png", dpi=300, bbox_inches='tight')
plt.show()

