
import pandas as pd
import numpy as np
import os
import itertools
import json
from pathlib import PurePath, Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from random import sample, seed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.covariance import GraphicalLassoCV
from sklearn.metrics import (RocCurveDisplay, roc_curve,
                              roc_auc_score, accuracy_score,
                              silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

import shap
import argparse
from platform import python_version

try:
    from utility.py_mrmr import mrmr_classif
    from utility import figure_generator
    MRMR_AVAILABLE = True
except ImportError:
    raise
    MRMR_AVAILABLE = False
    print("Note: utility.py_mrmr not found. MRMR feature selection will be skipped.")

plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 200})

METADATA_PATH = "meta.csv"

# ─────────────────────────────────────────────────────────────────────────────
class RNASeqModel:
    def __init__(self):
        self.count_dict = {}
        self.roc_dict   = {}

    def add_study(self, study_id):
        self.count_dict[study_id] = {}

    def add_study_counts(self, study_id, study_counts):
        self.count_dict[study_id]['counts'] = study_counts

    def add_study_metadata(self, study_id, study_metadata):
        self.count_dict[study_id]['metadata'] = study_metadata

    def build_combined_df(self, custom_concat=None, verbose=True):
        if custom_concat is None:
            if verbose:
                print("Order that counts are concatenated: {}".format(
                    self.count_dict.keys()))
            self.concat_df = pd.concat(
                [x['counts'] for x in self.count_dict.values()])
        else:
            print("Overwriting concatenated dataframe with file {}".format(
                custom_concat))
            self.concat_df = pd.read_csv(custom_concat, index_col=0).transpose()

        # Remove genes with NaN in any sample (ERCC spikes or missing genes)
        self.concat_df = self.concat_df.loc[
            :, self.concat_df.isna().sum(axis=0) == 0]

        metadata = pd.read_csv(METADATA_PATH)
        group_lookup = metadata.set_index('sample')['group']

        # Assign targets using metadata 'group' column
        self.concat_df['target'] = -1                       # default: Ground Control
        for sample_name in self.concat_df.index:
            if sample_name in group_lookup.index:
                g = group_lookup[sample_name]
                self.concat_df.loc[sample_name, 'target'] = (
                    1 if g == 0 else -1)                    # 0=SF→1, 1=GC→-1

    def load_filenames(self, data_dir, ftype='', verbose=False):
        p = Path.cwd()
        for d in data_dir:
            p = p / d
        fpaths = sorted([x for x in p.iterdir()])
        fpaths = list(filter(lambda x: (x.name[-4] == '.'), fpaths))
        if verbose:
            for f in fpaths:
                print("Loading the {} file: {}".format(ftype, f.name))
        else:
            print("Loaded files:", ",".join([f.name for f in fpaths]))
        return fpaths

    def feature_subset(self, count_dfs, feature_list, verbose=False):
        features = pd.read_csv(feature_list).iloc[:, -1]
        print('before: ', len(count_dfs))
        count_dfs = [x.loc[:, features] for x in count_dfs]
        print('after: ', len(count_dfs))
        if verbose:
            print("Subset dimensions:", [x.shape for x in count_dfs])
        return count_dfs

    # Fits StandardScaler on log2(training counts + 1), transforms all samples.
    # StandardScaler uses ddof=0 (sklearn default).
    def scale_data(self, count_dfs, exclude_test):
        for i in range(len(count_dfs)):
            scaler = StandardScaler()
            df = count_dfs[i]
            if exclude_test:
                test_ids = []
                for k in exclude_test.keys():
                    test_ids.append(exclude_test[k]['id']['SF'])
                    test_ids.append(exclude_test[k]['id']['GC'])
                test_ids = list(itertools.chain(*test_ids))
                valid_ids  = count_dfs[i].index.isin(test_ids)
                train_ids  = np.arange(df.index.values.shape[0])[~valid_ids]
                df         = df.iloc[train_ids]

            scaler.fit(np.log2(df + 1))
            count_dfs[i] = pd.DataFrame(
                scaler.transform(np.log2(count_dfs[i] + 1)),
                columns=count_dfs[i].columns,
                index=count_dfs[i].index)
        return count_dfs

    # ── load_counts ───────────────────────────────────────────────────────────
    # loads the single combined metadata.csv, splits by study,
    # and produces per-study DataFrames that the rest of the class can use.
    def load_counts(self, data_dir=['data', 'norm_counts'],
                    verbose=False, scale=False,
                    exclude_test=None, feature_list=None):

        count_files = self.load_filenames(data_dir, ftype='counts',
                                          verbose=verbose)

        metadata_all = pd.read_csv(METADATA_PATH)
        # Map filename stem (e.g. "A") to per-study metadata DataFrames.
        # Column 'sample' replaces 'Sample Name' from the original.
        study_metadata_map = {
            study: metadata_all[metadata_all['study'] == study].reset_index(drop=True)
            for study in metadata_all['study'].unique()
        }

        # Filter count files to those with a matching study in metadata
        keep_files, rem_files = [], []
        for x in count_files:
            stem = x.name.split('.')[0]         
            if stem in study_metadata_map:
                keep_files.append(x)
            else:
                rem_files.append(x.name)
        if rem_files:
            print("Removed following counts (no metadata match):",
                  ",".join(rem_files))
        count_files = keep_files

        count_dfs    = [pd.read_csv(x, index_col=0).transpose()
                        for x in count_files]
        metadata_dfs = [study_metadata_map[x.name.split('.')[0]]
                        for x in count_files]

        if feature_list:
            print('subset features')
            count_dfs = self.feature_subset(count_dfs, feature_list,
                                            verbose=verbose)

        if scale:
            count_dfs = self.scale_data(count_dfs,
                                        exclude_test=exclude_test)

        fnames = [x.name.split('.')[0] for x in count_files]

        for id, cdf, mdf in zip(fnames, count_dfs, metadata_dfs):
            self.add_study(id)
            self.add_study_counts(id, cdf)
            self.add_study_metadata(id, mdf)

    def treatment_filter(self, id_name, factor, valid_levels, verbose=False):
        for k, v in self.count_dict.items():
            if verbose:
                print("Handling filtering of study {}".format(k))
            counts_df   = v['counts']
            metadata_df = v['metadata']
            valid_ids = metadata_df.loc[
                metadata_df[factor].isin(valid_levels), id_name]
            valid_ids = np.intersect1d(counts_df.index.values,
                                       valid_ids.tolist())
            self.count_dict[k]['counts'] = counts_df.loc[valid_ids, :]

    def profile_data(self):
        profile_dict = {}
        kwargs = dict(axis=0)
        for id in self.count_dict.keys():
            study_counts = self.count_dict[id]['counts']
            profile_dict[id] = {
                'mean': study_counts.mean(**kwargs),
                'std':  study_counts.std(**kwargs),
                'var':  study_counts.var(**kwargs),
                'kurt': study_counts.kurt(**kwargs)
            }
        self.profile_dict = profile_dict

    def plot_profile(self, metric, generate_csv=False):
        stats = [[k, v[metric]] for k, v in self.profile_dict.items()]
        if generate_csv:
            pd.concat([v[metric] for v in self.profile_dict.values()],
                      axis=1).dropna(axis=0).to_csv(
                './data/stats/{}.csv'.format(metric))
        kwargs = dict(alpha=0.33, bins=250)
        for i in stats:
            x = i[1].to_numpy()
            plt.hist(x[np.isfinite(x)], **kwargs, label=i[0])
        plt.title('Frequency Histogram of {} at Gene Level'.format(metric))
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('{}.png'.format(metric))
        plt.close()

    def compute_mrmr(self, X, y, K):
        if not MRMR_AVAILABLE:
            raise ImportError("utility.py_mrmr not available. "
                              "Clone the original repository to use MRMR.")
        results = mrmr_classif(X, y, K)
        return results

    def update_roc(self, trues, scores, model_name):
        fpr, tpr, thresholds = roc_curve(trues, scores)
        auc = roc_auc_score(trues, scores)
        self.roc_dict[model_name] = {
            'fpr': fpr, 'tpr': tpr,
            'thresholds': thresholds, 'auc': auc
        }
        return self.roc_dict[model_name]

    def fit_model(self, models, mrmr=False, roc=False, pfi=False,
                  test_set=None, accuracy_block=False, verbose=False):

        model_params = {
            "rf": {
                "n_estimators": 100, "max_depth": 5, "random_state": 0,
                "min_samples_leaf": 1, "min_samples_split": 2,
                "oob_score": True, "class_weight": "balanced"
            },
            "svm": {
                "penalty": 'l2', "loss": 'squared_hinge', "C": 1.0,
                "class_weight": 'balanced', "random_state": 12345
            },
            "lda": {
                "solver": "eigen", "shrinkage": "auto"
            },
            "lda-svd": {"solver": "svd"},
            "LDA_lsqr_params": {
                "solver": "lsqr",
                "covariance_estimator": GraphicalLassoCV(
                    assume_centered=True, n_jobs=-1)
            },
            "glm": {"random_state": 0}
        }

        model_list, model_names = [], []
        if "rf"  in models:
            model_list.append(RandomForestClassifier(**model_params["rf"]))
            model_names.append('rf')
        if "svm" in models:
            model_list.append(LinearSVC(**model_params["svm"]))
            model_names.append('svm')
        if "lda" in models:
            model_list.append(LDA(**model_params["lda"]))
            model_names.append('lda')
        if "glm" in models:
            model_list.append(LogisticRegression(**model_params["glm"]))
            model_names.append('glm')

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)

        X = self.concat_df.iloc[:, :-1]
        y = self.concat_df.iloc[:,  -1]

        test_ids = []
        if test_set:
            for k in test_set.keys():
                test_ids.append(test_set[k]['id']['SF'])
                test_ids.append(test_set[k]['id']['GC'])
            test_ids  = list(itertools.chain(*test_ids))
            valid_ids = X.index.isin(test_ids)
            train_ids = np.arange(X.index.values.shape[0])[~valid_ids]
            test_ids  = np.arange(X.index.values.shape[0])[valid_ids]

        if mrmr:
            X = X.loc[:, self.Genes['Gene']]

        dict_frame = {"accuracy": [], "model_coefs": [], "importance": []}
        results    = {m: dict_frame.copy() for m in models}

        acc_list = []; model_coef = []; roc_y_scores = []
        roc_y_trues = []; imp_arr = []

        for (model, model_name) in zip(model_list, model_names):
            train_test = skf.split(X, y)
            if test_set:
                train_test = zip([train_ids], [test_ids])
            if accuracy_block:
                train_test = skf.split(X.iloc[train_ids, :],
                                       y.iloc[train_ids])

            for train, test in train_test:
                model    = model.fit(X.iloc[train, :], y.iloc[train])
                accuracy = accuracy_score(
                    model.predict(X.iloc[test, :]), y.iloc[test])
                acc_list.append(accuracy)

                if "rf" == model_name:
                    model_coef.append(model.feature_importances_)
                else:
                    model_coef.append(model.coef_.T)

                if roc:
                    if "rf"  == model_name:
                        y_scores = model.predict_proba(X.iloc[test])[:, 1]
                    elif "lda" == model_name:
                        y_scores = model.predict_proba(X.iloc[test])[:, 1]
                    else:
                        y_scores = model.decision_function(X.iloc[test])
                    roc_y_scores.append(y_scores)
                    roc_y_trues.append(y.iloc[test])

                if pfi:
                    X_test = X.iloc[test]

                    # Choose explainer based on model type
                    if model_name == "rf":
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)[1]  # class 1 (SF)
                    else:
                        explainer = shap.LinearExplainer(model, X.iloc[train], feature_dependence="independent")
                        shap_values = explainer.shap_values(X_test)

                    # Convert to DataFrame (same shape as X_test)
                    shap_df = pd.DataFrame(shap_values, columns=X.columns)

                    # Store (mean absolute SHAP per feature for this fold)
                    imp_arr.append(shap_df.abs())

                if verbose:
                    print("Average accuracy: {:.3f}".format(
                        np.mean(acc_list)))
                    if "rf" == model_name:
                        print('train {} | test {} | acc {:.3f}'.format(
                            np.bincount(y.iloc[train] + 1),
                            np.bincount(y.iloc[test]  + 1), accuracy))
                        print("OOB Score:", model.oob_score_)
                        print("Features seen:", model.n_features_in_)
                    if "lda" == model_name:
                        print("Features seen:", model.n_features_in_)
                        print("Coef shape:", model.coef_.shape)
                        
                
            if roc:
                roc_dict = self.update_roc(
                    np.concatenate(roc_y_trues,  axis=0),
                    np.concatenate(roc_y_scores, axis=0),
                    model_name)
                results[model_name]['roc'] = pd.DataFrame(roc_dict)
                results[model_name]['predictions'] = pd.concat([
                    pd.DataFrame(np.concatenate(roc_y_trues,  axis=0),
                                 columns=['truth']),
                    pd.DataFrame(np.concatenate(roc_y_scores, axis=0),
                                 columns=['probability'])
                ], axis=1)
                roc_y_scores.clear(); roc_y_trues.clear()

            if pfi:
                results[model_name]['importance'] = pd.concat(
                    imp_arr, axis=0, ignore_index=True)
                imp_arr.clear()

            results[model_name]['accuracy'] = pd.DataFrame(
                acc_list, columns=["Accuracy"])
            acc_list.clear()
            results[model_name]['model_coefs'] = pd.DataFrame(
                np.mean(model_coef, axis=0),
                columns=['Score'],
                index=X.columns.values)
            model_coef.clear()

            if shap_df is not None:
                    print("  Top 100 SHAP features:")
                    print(shap_df)
                   # for name in results.sort_values("importance", ascending=False).index[:100]:
                   #     print(name)

        return results

    def permutation_scoring(self, estimator, X, Y, sort=True, seed=12345):
        if python_version()[:3] == "3.9":
            result = permutation_importance(
                estimator, X.to_numpy(), Y.to_numpy(),
                n_repeats=100, random_state=seed,
                scoring='roc_auc', n_jobs=-1)
        else:
            result = permutation_importance(
                estimator, X, Y,
                n_repeats=100, random_state=seed, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()
        if sort:
            importances = pd.DataFrame(
                result.importances[sorted_idx].T,
                columns=X.columns[sorted_idx])
        else:
            importances = pd.DataFrame(
                result.importances.T, columns=X.columns)
        return importances

    def generate_directionality(self):
        data   = self.concat_df.iloc[:, :-1]
        labels = self.concat_df.iloc[:,  -1]
        sf_mask = labels ==  1
        gc_mask = labels == -1
        sf = data.loc[sf_mask, :]
        gc = data.loc[gc_mask, :]
        dir_mask   = np.greater(gc.mean(axis=0), sf.mean(axis=0))
        dir_vector = np.ones(dir_mask.shape[0])
        dir_vector[dir_mask] = -1
        dir_df = pd.DataFrame(dir_vector, columns=["Direction"],
                               index=data.columns)
        dir_df.to_csv("directional.csv", header=True)

    def plot_pca(self, mrmr=False):
        pca = PCA(n_components=2)

        X = self.concat_df.iloc[:, :-1]
        y = self.concat_df.iloc[:,  -1]
        target_names = ['GC', 'SF']

        labels = np.empty(X.shape[0], dtype=object)
        for k in self.count_dict.keys():
            # CHANGED: 'sample' instead of 'Sample Name'
            study_mask = np.isin(X.index.values,
                                 self.count_dict[k]['metadata']['sample'])
            labels[study_mask] = k

        if mrmr:
            X = X.loc[:, self.Genes['Gene']]

        X_r    = pca.fit(X).transform(X)
        colors = ["navy", "darkorange"]

        if MRMR_AVAILABLE:
            figure_generator.plot_pca(pca, X_r, y, target_names,
                                      colors, labels, mrmr)
        else:
            # Inline PCA plot when figure_generator is not available
            fig, ax = plt.subplots(figsize=(7, 5))
            unique_studies  = np.unique(labels)
            study_colors    = plt.cm.tab10(
                np.linspace(0, 1, len(unique_studies)))
            marker_map = {1: '^', -1: 'o'}
            for study, col in zip(unique_studies, study_colors):
                for target, marker in marker_map.items():
                    mask = (labels == study) & (y.values == target)
                    ax.scatter(X_r[mask, 0], X_r[mask, 1],
                               color=col, marker=marker,
                               label=f"{study} {'SF' if target==1 else 'GC'}",
                               s=60, edgecolors='k', linewidths=0.4)
            v = pca.explained_variance_ratio_ * 100
            ax.set_xlabel(f"PC1 ({v[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({v[1]:.1f}%)")
            title = "PCA — harmonised data (MRMR genes)" if mrmr \
                    else "PCA — harmonised data (all genes)"
            ax.set_title(title)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, linestyle='--', alpha=0.4)
            fig.tight_layout()
            fname = "pca_harmonised_mrmr.png" if mrmr else "pca_harmonised.png"
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            print(f"PCA saved to {fname}")
            print(f"  PC1={v[0]:.1f}%  PC2={v[1]:.1f}%")


    # ── evaluate_batch_effect ─────────────────────────────────────────────────
    def evaluate_batch_effect(self, X, study_labels, condition_labels=None,
                               title="Batch Effect Evaluation",
                               save_prefix="batch_eval", n_neighbors=15):

        study_arr      = np.asarray(study_labels)
        unique_studies = np.unique(study_arr)
        n_total        = len(study_arr)

        # ── 1. PCA ────────────────────────────────────────────────────────────
        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(X.values if hasattr(X, 'values') else X)
        v     = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=(7, 5))
        study_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_studies)))
        marker_cycle = ['o', '^', 's', 'D', 'v', '<', '>']

        if condition_labels is not None:
            cond_arr     = np.asarray(condition_labels)
            unique_conds = np.unique(cond_arr)
            marker_map   = {c: marker_cycle[i % len(marker_cycle)]
                            for i, c in enumerate(unique_conds)}
            for study, col in zip(unique_studies, study_colors):
                for cond in unique_conds:
                    mask = (study_arr == study) & (cond_arr == cond)
                    if mask.any():
                        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                                   color=col, marker=marker_map[cond],
                                   label=f"{study} / {cond}",
                                   s=60, edgecolors='k',
                                   linewidths=0.4, alpha=0.8)
        else:
            for study, col in zip(unique_studies, study_colors):
                mask = study_arr == study
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           color=col, label=study,
                           s=60, edgecolors='k', linewidths=0.4, alpha=0.8)

        ax.set_xlabel(f"PC1 ({v[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({v[1]:.1f}%)")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        pca_fname = f"{save_prefix}_pca.png"
        fig.savefig(pca_fname, dpi=200)
        plt.close(fig)
        print(f"  PCA plot saved → {pca_fname}  "
              f"(PC1={v[0]:.1f}%  PC2={v[1]:.1f}%)")

        # ── 2. Silhouette score (study / batch labels) ────────────────────────
        if len(unique_studies) > 1 and n_total > len(unique_studies):
            study_int = np.array(
                [np.where(unique_studies == s)[0][0] for s in study_arr])
            sil = silhouette_score(X_pca, study_int, metric='euclidean')
            print(f"  Silhouette score (study labels, PCA space): {sil:+.4f}  "
                  f"[range −1 → +1;  +1 = strong batch separation  "
                  f"0 = well-mixed]")
        else:
            sil = np.nan
            print("  Silhouette score: N/A "
                  "(only one study present or too few samples)")

        # ── 3. kNN batch purity ───────────────────────────────────────────────
        k  = min(n_neighbors, n_total - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(X_pca)
        # [:, 1:] drops the sample itself from its own neighbour list
        indices = nn.kneighbors(X_pca, return_distance=False)[:, 1:]

        same_batch = np.mean([
            np.mean(study_arr[indices[i]] == study_arr[i])
            for i in range(n_total)
        ])
        expected_purity = sum(
            (np.sum(study_arr == s) / n_total) ** 2
            for s in unique_studies
        )
        print(f"  kNN batch purity (k={k}): "
              f"observed={same_batch:.4f}  "
              f"expected (random)={expected_purity:.4f}  "
              f"[obs ≈ expected → well-mixed;  "
              f"obs >> expected → batch structure remains]")

        return {
            'silhouette':   sil,
            'knn_purity':   same_batch,
            'knn_expected': expected_purity,
            'pca_variance': v[:2].tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for MRMR run')
    parser.add_argument("-K", "--K",
        help="Number of MRMR features to select.", default=100, type=int)
    parser.add_argument("-O", "--outfile_name",
        help="Name for output MRMR csv.", default="mrmr")
    parser.add_argument("-F", "--feature_list",
        help="csv with gene set filtering for concatenated matrix.",
        default=None)
    parser.add_argument("-C", "--custom_concat",
        help="Provide custom concatenated data file.", default=None)
    parser.add_argument("-G", "--gene_list",
        help="csv with gene lists for model fitting.", default='mrmr.csv')
    parser.add_argument("-B", "--block_flags",
        help="json with flag specifications", default=None)
    parser.add_argument("-S", "--save_concat",
        help="Save the concatenated dataframe to concat_df.csv",
        action="store_true")
    parser.add_argument("-V", "--verbose",
        help="Toggle output from functions", action="store_true")

    args = parser.parse_args()

    run_flags = {}
    if args.block_flags is not None:
        with open(args.block_flags, 'r') as f:
            run_flags = json.loads(f.read())

    data_process_block             = run_flags.get("data_process_block",             True)
    batch_eval_block               = run_flags.get("batch_eval_block",               True)
    batch_eval_k                   = run_flags.get("batch_eval_k",                   15)
    save_concat_df_block           = run_flags.get("save_concat_df_block",           False)
    load_concat_df_to_clipboard_block = run_flags.get("load_concat_df_to_clipboard_block", False)
    profile_data_block             = run_flags.get("profile_data_block",             False)
    mrmr_fitting_block             = run_flags.get("mrmr_fitting_block",             True)
    rf_fitting_block               = run_flags.get("rf_fitting_block",               False)
    svm_fitting_block              = run_flags.get("svm_fitting_block",              False)
    lda_fitting_block              = run_flags.get("lda_fitting_block",              False)
    accuracy_block                 = run_flags.get("accuracy_block",                 False)
    pca_block                      = run_flags.get("pca_block",                      False)
    roc_block                      = run_flags.get("roc_block",                      False)
    pfi_block                      = run_flags.get("pfi_block",                      False)
    random_block                   = run_flags.get("random_block",                   False)
    test_block                     = run_flags.get("test_block",                     True)

    gene_list_flag = args.gene_list is not None

   
    if test_block:
        test_labels = {}  
    else:
        test_labels = None

    # ── Data loading and preprocessing ───────────────────────────────────────
    model = RNASeqModel()

    kwargs = {
        "data_dir":    ['data', 'norm_counts'],  # reads R output CSVs
        "feature_list": 'prefiltered_pseudogenes.csv'
    }

    # ── Batch effect evaluation BEFORE harmonisation ─────────────────────────
    # Load raw (unscaled) counts into a throw-away model so that the main
    # model instance is not affected.  log₂(x+1) is applied for comparability
    # with the post-harmonisation scaled matrix.
    if batch_eval_block and data_process_block:
        print("\n── Batch effect: BEFORE harmonisation "
              "──────────────────────────────────────")
        _raw = RNASeqModel()
        _raw.load_counts(verbose=False, scale=False,
                         exclude_test=None, **kwargs)
        _raw.treatment_filter(id_name='sample', factor='group',
                              valid_levels=[0, 1], verbose=False)
        _raw.build_combined_df(custom_concat=None)

        # log₂(x+1) transform — mirrors the first step inside scale_data()
        _X_before = np.log2(_raw.concat_df.iloc[:, :-1] + 1)

        _meta_be         = pd.read_csv(METADATA_PATH)
        _study_lkp_be    = _meta_be.set_index('sample')['study']
        _study_labels_be = np.array([_study_lkp_be.get(s, 'unknown')
                                     for s in _X_before.index])
        _cond_labels_be  = _raw.concat_df['target'].map(
            {1: 'SF', -1: 'GC'}).values
        del _raw  # free memory

        batch_metrics_before = model.evaluate_batch_effect(
            _X_before, _study_labels_be, _cond_labels_be,
            title="Batch Effect — Before Harmonisation (log₂(x+1), unscaled)",
            save_prefix="batch_before",
            n_neighbors=batch_eval_k
        )
    else:
        batch_metrics_before = None

    if data_process_block:
        model.load_counts(verbose=args.verbose, scale=True,
                          exclude_test=test_labels, **kwargs)

        model.treatment_filter(id_name='sample', factor='group',
                               valid_levels=[0, 1], verbose=args.verbose)
        model.build_combined_df(args.custom_concat)

    if batch_eval_block:
        print("\n── Batch effect: AFTER harmonisation "
              "───────────────────────────────────────")
        _meta_af         = pd.read_csv(METADATA_PATH)
        _study_lkp_af    = _meta_af.set_index('sample')['study']
        _X_after         = model.concat_df.iloc[:, :-1]
        _study_labels_af = np.array([_study_lkp_af.get(s, 'unknown')
                                     for s in _X_after.index])
        _cond_labels_af  = model.concat_df['target'].map(
            {1: 'SF', -1: 'GC'}).values

        batch_metrics_after = model.evaluate_batch_effect(
            _X_after, _study_labels_af, _cond_labels_af,
            title="Batch Effect — After Harmonisation (log₂(x+1), per-study scaled)",
            save_prefix="batch_after",
            n_neighbors=batch_eval_k
        )

        if batch_metrics_before is not None:
            print("\n── Batch effect summary  (before → after harmonisation) "
                  "─────────────────")
            sil_b = batch_metrics_before.get('silhouette', np.nan)
            sil_a = batch_metrics_after .get('silhouette', np.nan)
            knn_b = batch_metrics_before.get('knn_purity', np.nan)
            knn_a = batch_metrics_after .get('knn_purity', np.nan)
            exp_a = batch_metrics_after .get('knn_expected', np.nan)
            print(f"  Metric                    Before      After       Δ (after−before)")
            print(f"  {'─'*62}")
            print(f"  Silhouette (↓ better)    {sil_b:+8.4f}   {sil_a:+8.4f}   "
                  f"{sil_a - sil_b:+8.4f}")
            print(f"  kNN purity  (↓ better)    {knn_b:8.4f}   {knn_a:8.4f}   "
                  f"{knn_a - knn_b:+8.4f}   (random baseline: {exp_a:.4f})")
            print(f"  {'─'*62}")
            sil_improved = (not np.isnan(sil_b)) and (sil_a < sil_b)
            knn_improved = knn_a < knn_b
            print(f"  Silhouette improved after harmonisation: {sil_improved}")
            print(f"  kNN purity  improved after harmonisation: {knn_improved}")
            print(f"  PNGs saved: batch_before_pca.png  batch_after_pca.png")

    
    
    print("Concatenated matrix shape:", model.concat_df.shape)
    print("Target distribution:\n",
          model.concat_df['target'].value_counts().rename({1: 'Spaceflight',
                                                           -1: 'Ground Control'}))

    model.concat_df.T.to_csv('combined_df.csv')

    if random_block:
        seed(12345)
        ids = sample(list(np.arange(model.concat_df.shape[1] - 1)), 60)
        ids.append(-1)
        model.concat_df = model.concat_df.iloc[:, ids]

    if args.gene_list is not None:
        gene_list  = pd.read_csv(args.gene_list, index_col=0)
        model.Genes = gene_list

    if pca_block:
        model.plot_pca(mrmr=gene_list_flag)

    if save_concat_df_block or args.save_concat:
        pd.DataFrame(model.concat_df).transpose().to_csv(
            "concat_df.csv", header=True)
        model.generate_directionality()

    if load_concat_df_to_clipboard_block:
        model.concat_df.transpose().to_clipboard()

    if profile_data_block:
        model.profile_data()
        model.plot_profile('mean')

    if mrmr_fitting_block:
        if not MRMR_AVAILABLE:
            print("Skipping MRMR: utility.py_mrmr not available.")
        else:
            mrmr = model.compute_mrmr(
                model.concat_df.iloc[:, :-1],
                model.concat_df.iloc[:,  -1],
                K=args.K)
            mrmr_df = pd.DataFrame(mrmr, columns=["Gene"])
            mrmr_df.to_csv("{}.csv".format(args.outfile_name), header=True)

    model_names   = ['svm']
    chosen_models = [x for x, y in zip(model_names,
                     [rf_fitting_block, svm_fitting_block, lda_fitting_block])
                     if y]
    print("Models chosen:", chosen_models)

    if args.gene_list is not None:
        gene_list   = pd.read_csv(args.gene_list, index_col=0)
        model.Genes = gene_list

    results = model.fit_model(chosen_models, mrmr=gene_list_flag,
                              roc=roc_block, pfi=pfi_block,
                              test_set=test_labels)

    if accuracy_block and gene_list_flag:
        dim = gene_list.shape[0]
        acc_array = np.zeros([dim, 3])
        for i in range(2, dim + 1):
            model.Genes = gene_list.iloc[:i, ]
            results = model.fit_model(model_names, mrmr=gene_list_flag,
                                      roc=False, pfi=False,
                                      test_set=test_labels,
                                      accuracy_block=accuracy_block)
            for k, m in enumerate(model_names):
                acc_array[i - 1, k] = results[m]['accuracy'].mean()
        pd.DataFrame(acc_array, columns=model_names).to_csv(
            "{}.csv".format(args.outfile_name))

   
    print("\nSaving feature importances ...")
    for model_name in chosen_models:
        if 'model_coefs' not in results[model_name]:
            continue
        coef_df = results[model_name]['model_coefs']
        pfi = results[model_name]['importance']
        print(pfi)

        if model_name != 'rf':
            coef_df = coef_df.abs()

        pfi_sorted = pfi.sort()
        coef_sorted = coef_df.sort_values('Score', ascending=False)
        out_path = f"feature_importance_{model_name}.csv"
        coef_sorted.to_csv(out_path)
        print(pfi)

        print(f"  {out_path}  ({len(coef_sorted):,} genes ranked)")
        print(f"  Top 10 genes ({model_name}):")
        print(coef_sorted.head(100).to_string())
        print()

        # Also save accuracy summary
        acc_df = results[model_name]['accuracy']
        print(f"  Mean CV accuracy ({model_name}): "
              f"{acc_df['Accuracy'].mean():.3f} "
              f"± {acc_df['Accuracy'].std():.3f}")

    print("\nDone.")