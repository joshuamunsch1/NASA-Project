# Minimal LIVER pipeline (derived from RNASeqModel_mod.py).
#
# Does three things only:
#   1. builds/loads the concatenated SF-vs-control matrix,
#   2. fits the requested classifier(s) and extracts per-gene attributions
#      (signed coefficients / gain importances + SHAP),
#   3. saves the concatenated dataset.
#
# Stripped relative to RNASeqModel_mod.py: MRMR, PCA/plot_pca, data profiling,
# clipboard dump, random-gene subsetting, the accuracy gene-sweep, permutation
# importance, ROC, directionality, the gene-list/feature subsetting, and the
# JSON block-flag interface (model selection is now a direct --models flag).

import os
import itertools
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import shap


class RNASeqModel:
    def __init__(self):
        self.count_dict = {}

    # ---- dataset curation -------------------------------------------------
    def add_study(self, study_id):
        self.count_dict[study_id] = {}

    def add_study_counts(self, study_id, study_counts):
        self.count_dict[study_id]['counts'] = study_counts

    def add_study_metadata(self, study_id, study_metadata):
        self.count_dict[study_id]['metadata'] = study_metadata

    def _spaceflight_targets(self, metadata_dir=('data', 'single_study_metadata'),
                             id_col="Sample Name", factor_col="Factor Value[Spaceflight]"):
        """Map merged Sample Name -> +1 (Space Flight) / -1 (other arms) from the
        single-study ISA metadata, so studies whose ids carry no 'F' token
        (e.g. OSD-599 GSM accessions) are not silently mislabelled."""
        p = Path.cwd()
        for d in metadata_dir:
            p = p / d
        targets = {}
        if not p.exists():
            print("WARNING: metadata dir {} not found; target labels will rely on "
                  "the legacy 'F'-in-name fallback.".format(p))
            return targets
        for f in sorted(p.glob("*.txt")):
            md = pd.read_csv(f, delimiter="\t")
            if id_col not in md.columns or factor_col not in md.columns:
                continue
            is_sf = md[factor_col].astype(str).str.contains("Space", case=False, na=False)
            for name, sf in zip(md[id_col].astype(str), is_sf):
                targets[name] = 1 if sf else -1
        return targets

    def build_combined_df(self, custom_concat=None, prep='none', exclude_test=None, verbose=True):
        """Concatenate loaded studies (native path) or read an externally
        harmonised matrix (--custom_concat). Drops ERCC/NA columns, applies the
        optional downstream prep, then appends the {-1,+1} target column."""
        if custom_concat is None:
            if verbose:
                print("Order that counts are concatenated: {}".format(self.count_dict.keys()))
            self.concat_df = pd.concat([x['counts'] for x in self.count_dict.values()])
        else:
            print("Overwriting concatenated dataframe with file {}".format(custom_concat))
            # custom matrices are stored genes x samples; transpose to samples x genes
            self.concat_df = pd.read_csv(custom_concat, index_col=0).transpose()

        # drop ERCC spike-ins / genes absent from a subset of studies
        self.concat_df = self.concat_df.loc[:, self.concat_df.isna().sum(axis=0) == 0]
        if custom_concat is not None and prep in ('count', 'log'):
            self.apply_common_prep(prep, exclude_test)

        # --- target encoding: Space Flight (+1) vs ground/other controls (-1) ---
        sf_map = self._spaceflight_targets()
        idx = self.concat_df.index.to_series()
        mapped = idx.map(sf_map)
        unmapped = mapped.isna()
        if unmapped.any():
            print("WARNING: {} sample(s) had no Factor Value[Spaceflight] match; "
                  "falling back to the 'F'-in-name rule for: {}".format(
                      int(unmapped.sum()), list(idx[unmapped][:10])))
            mapped.loc[unmapped] = np.where(idx[unmapped].str.contains("F"), 1, -1)
        self.concat_df['target'] = mapped.astype(int).values

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

    def scale_data(self, count_dfs, exclude_test):
        """Per-study log2(x+1) + standardise. Scaler fit on training samples
        only (held-out pairs excluded) to avoid leakage."""
        for i in range(len(count_dfs)):
            scaler = StandardScaler()
            df = count_dfs[i]
            if exclude_test:
                test_ids = []
                for k in exclude_test.keys():
                    test_ids.append(exclude_test[k]['id']['SF'])
                    test_ids.append(exclude_test[k]['id']['GC'])
                test_ids = list(itertools.chain(*test_ids))
                valid_ids = count_dfs[i].index.isin(test_ids)
                train_ids = np.arange(df.index.values.shape[0])[~valid_ids]
                df = df.iloc[train_ids]
            scaler.fit(np.log2(df + 1))
            count_dfs[i] = pd.DataFrame(scaler.transform(np.log2(count_dfs[i] + 1)),
                                        columns=count_dfs[i].columns, index=count_dfs[i].index)
        return count_dfs

    def apply_common_prep(self, prep, exclude_test):
        """Downstream prep for externally harmonised matrices.
        'count' -> log2(x+1) then global per-gene standardise.
        'log'   -> global per-gene standardise only (already log-scale).
        Standardiser fit on TRAINING samples only (test held out)."""
        X = self.concat_df.astype(float)
        if prep == 'count':
            X = np.log2(X + 1)
        test_ids = []
        if exclude_test:
            for k in exclude_test.keys():
                test_ids.extend(exclude_test[k]['id']['SF'])
                test_ids.extend(exclude_test[k]['id']['GC'])
        train_mask = ~X.index.isin(test_ids)
        scaler = StandardScaler().fit(X.loc[train_mask])
        self.concat_df = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

    def load_counts(self, data_dir=['data', 'norm_counts'],
                    metadata_dir=['data', 'single_study_metadata'],
                    verbose=False, scale=False, exclude_test=None):
        count_files = self.load_filenames(data_dir, ftype='counts', verbose=verbose)
        metadata_files = self.load_filenames(metadata_dir, ftype='metadata', verbose=verbose)

        # keep only counts that have a matching metadata file
        keep_files, rem_files = [], []
        for x in count_files:
            if x.name[:-4] in [y.name[:-4] for y in metadata_files]:
                keep_files.append(x)
            else:
                if verbose:
                    print("Removing {} counts due to lack of corresponding metadata".format(x.name))
                else:
                    rem_files.append(x.name)
        if not verbose:
            print("Removed following counts:", ",".join(rem_files))
        count_files = keep_files

        count_dfs = [pd.read_csv(x, index_col=0).transpose() for x in count_files]
        metadata_dfs = [pd.read_csv(x, delimiter="\t") for x in metadata_files]

        if scale:
            count_dfs = self.scale_data(count_dfs, exclude_test=exclude_test)

        fnames = [x.name.split('.')[0] for x in count_files]
        for sid, cdf, mdf in zip(fnames, count_dfs, metadata_dfs):
            self.add_study(sid)
            self.add_study_counts(sid, cdf)
            self.add_study_metadata(sid, mdf)

    def treatment_filter(self, id_name, factor, valid_levels, verbose=False):
        for k, v in self.count_dict.items():
            if verbose:
                print("Handling filtering of study {}".format(k))
            counts_df = v['counts']
            metadata_df = v['metadata']
            re = "|".join(valid_levels)
            valid_ids = metadata_df.loc[metadata_df[factor].str.contains(re), id_name]
            valid_ids = np.intersect1d(counts_df.index.values, valid_ids.tolist())
            self.count_dict[k]['counts'] = counts_df.loc[valid_ids, :]

    # ---- model fitting ----------------------------------------------------
    def _build_estimator(self, model_name, model_params):
        if model_name == "svm":
            return LinearSVC(**model_params["svm"])
        if model_name == "glm":
            return LogisticRegression(**model_params["glm"])
        if model_name == "elasticnet":
            return make_pipeline(StandardScaler(), LogisticRegression(**model_params["elasticnet"]))
        if model_name == "xgboost":
            return XGBClassifier(**model_params["xgboost"])
        raise ValueError("Unknown model: {}".format(model_name))

    def _final_estimator(self, model):
        """Terminal estimator, so coef_/feature_importances_ read uniformly
        whether or not the model is wrapped in a Pipeline (elasticnet)."""
        if hasattr(model, "named_steps"):
            return list(model.named_steps.values())[-1]
        return model

    def _shap_fold(self, model, X_train, X_test, verbose=False):
        """Per-fold SHAP -> (2, p): row 0 signed mean (direction of effect),
        row 1 mean |SHAP| (magnitude, the ranking metric).

        A LinearExplainer must see features in the SAME space the linear model
        was trained in. The elasticnet pipeline lives in standardised space, so
        its StandardScaler is applied before explaining; passing raw values
        would rescale each gene's attribution by its raw SD and corrupt the
        ranking. The bare linear models (svm/glm) already operate on the
        pre-scaled matrix."""
        final = self._final_estimator(model)
        if hasattr(model, "named_steps") and len(model.named_steps) > 1:
            pre = model[:-1]
            Xtr_in = pre.transform(X_train.values)
            Xte_in = pre.transform(X_test.values)
        else:
            Xtr_in = X_train.values
            Xte_in = X_test.values

        try:
            if hasattr(final, "feature_importances_"):          # xgboost
                explainer = shap.TreeExplainer(final)
                sv = explainer.shap_values(Xte_in)
            elif hasattr(final, "coef_"):                       # svm / glm / elasticnet
                background = Xtr_in if Xtr_in.shape[0] <= 100 else \
                    Xtr_in[np.random.choice(Xtr_in.shape[0], 100, replace=False)]
                explainer = shap.LinearExplainer(final, background)
                sv = explainer.shap_values(Xte_in)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) == 2 else np.mean(sv, axis=0)
            else:
                if verbose:
                    print("  SHAP: unsupported model type; skipping fold.")
                return None
        except Exception as e:
            if verbose:
                print("  SHAP failed for this fold: {}".format(e))
            return None

        # normalise SHAP output shape across shap versions / class conventions
        if isinstance(sv, list):
            arr = sv[1] if len(sv) == 2 else np.mean(np.stack(sv, axis=0), axis=0)
        else:
            arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[..., 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)

        signed = arr.mean(axis=0)            # direction, averaged over samples
        magnitude = np.abs(arr).mean(axis=0)  # mean |SHAP|, used for ranking
        return np.vstack([signed, magnitude])

    def fit_model(self, models, test_set=None, shap_values=True, verbose=False):
        """Fit each requested classifier and collect per-gene attributions.

        With test_set: a single split (train = all minus the held-out SF+GC
        pairs, evaluate on the held-out pairs). Without test_set: 5-fold
        stratified CV, coefficients/SHAP averaged across folds.
        Returns {model: {accuracy, model_coefs, shap}}."""
        model_params = {
            # svm: dense signed coefficients (GSEA-arm partner to glm)
            "svm": {"penalty": 'l2', "loss": 'squared_hinge', "C": 1.0,
                    "class_weight": 'balanced', "random_state": 12345, "max_iter": 50000},
            # glm: L2 (ridge) logistic regression, dense signed coefficients
            "glm": {"penalty": "l2", "C": 1.0, "class_weight": "balanced",
                    "max_iter": 10000, "random_state": 0},
            # elasticnet: L1+L2 logistic regression (sparse selector -> ORA arm)
            "elasticnet": {"penalty": "elasticnet", "solver": "saga", "C": 1.0,
                           "l1_ratio": 0.5, "max_iter": 10000, "tol": 1e-4,
                           "random_state": 12345},
            # xgboost: shallow trees, heavy column subsampling, L1/L2 leaf reg
            "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
                        "subsample": 0.8, "colsample_bytree": 0.3, "reg_alpha": 0.1,
                        "reg_lambda": 1.0, "min_child_weight": 1,
                        "objective": "binary:logistic", "eval_metric": "logloss",
                        "tree_method": "hist", "n_jobs": -1, "random_state": 12345},
        }

        supported = ["svm", "glm", "elasticnet", "xgboost"]
        model_list, model_names = [], []
        for name in supported:
            if name in models:
                model_list.append(self._build_estimator(name, model_params))
                model_names.append(name)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
        X = self.concat_df.iloc[:, :-1]
        # target stored as {-1,+1}; re-encode to {0,1} (xgboost rejects {-1,+1};
        # accuracy and decision_function sign conventions are unchanged, the
        # positive class stays the SF / +1 group).
        y = (self.concat_df.iloc[:, -1] > 0).astype(int)

        if test_set:
            test_ids = []
            for k in test_set.keys():
                test_ids.append(test_set[k]['id']['SF'])
                test_ids.append(test_set[k]['id']['GC'])
            test_ids = list(itertools.chain(*test_ids))
            valid_ids = X.index.isin(test_ids)
            train_ids = np.arange(X.index.values.shape[0])[~valid_ids]
            test_ids = np.arange(X.index.values.shape[0])[valid_ids]

        results = {m: {} for m in models}
        acc_list, model_coef, shap_arr = [], [], []
        for (model, model_name) in zip(model_list, model_names):
            train_test = skf.split(X, y)
            if test_set:
                train_test = zip([train_ids], [test_ids])

            for train, test in train_test:
                model = model.fit(X.iloc[train, :], y.iloc[train])
                accuracy = accuracy_score(model.predict(X.iloc[test, :]), y.iloc[test])
                acc_list.append(accuracy)

                # native importance / coefficients from the terminal estimator:
                #   tree models  -> feature_importances_ (unsigned, gain-based)
                #   linear models -> coef_ (signed)
                final = self._final_estimator(model)
                if hasattr(final, "feature_importances_"):
                    model_coef.append(final.feature_importances_)
                else:
                    model_coef.append(final.coef_.T)

                if shap_values:
                    sv = self._shap_fold(model, X.iloc[train], X.iloc[test], verbose=verbose)
                    if sv is not None:
                        shap_arr.append(sv)

                if verbose:
                    print("The average accuracy was {:.3f}".format(np.mean(acc_list)))

            if shap_values and shap_arr:
                # each entry is (2, p): row 0 = signed mean, row 1 = mean |SHAP|
                signed_stack = np.vstack([s[0] for s in shap_arr])
                abs_stack = np.vstack([s[1] for s in shap_arr])
                results[model_name]['shap'] = pd.DataFrame({
                    "shap_mean": signed_stack.mean(axis=0),       # direction of effect
                    "shap_abs_mean": abs_stack.mean(axis=0),      # magnitude (rank by this)
                    "shap_std": signed_stack.std(axis=0),         # cross-fold stability
                }, index=X.columns.values)
                shap_arr.clear()

            results[model_name]["accuracy"] = pd.DataFrame(acc_list, columns=["Accuracy"]); acc_list.clear()
            results[model_name]["model_coefs"] = pd.DataFrame(
                np.mean(model_coef, axis=0), columns=['Score'], index=X.columns.values); model_coef.clear()

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal LIVER SF-vs-control classifier + gene attribution + concat export.")
    parser.add_argument("-C", "--custom_concat", default=None,
                        help="Externally harmonised concatenated matrix (genes x samples). "
                             "If omitted, the native z-score path (load + scale + filter) is used.")
    parser.add_argument("-P", "--prep", default="none", choices=["count", "log", "none"],
                        help="Downstream prep for --custom_concat matrices: "
                             "'count' (log2+global standardise), 'log' (global standardise only), 'none'.")
    parser.add_argument("-T", "--tag", default="",
                        help="Suffix appended to output filenames to keep per-arm results separate.")
    parser.add_argument("-M", "--models", nargs="+", default=["svm", "glm", "elasticnet", "xgboost"],
                        choices=["svm", "glm", "elasticnet", "xgboost"],
                        help="Classifier(s) to fit and attribute.")
    parser.add_argument("--no_test", action="store_true",
                        help="Disable the held-out test set and use 5-fold stratified CV instead.")
    parser.add_argument("--no_shap", action="store_true",
                        help="Skip SHAP attribution (export coefficients/importances + accuracy only).")
    parser.add_argument("-V", "--verbose", action="store_true")
    args = parser.parse_args()

    # held-out SF + GC pair(s) per liver study (merged ids written by 00_merge.R)
    test_labels = {
        '47': {'id': {'SF': ["Mmus_C57-6T_LVR_FLT_Rep1_F1"],
                      'GC': ["Mmus_C57-6T_LVR_GC_Rep3_G5"]}},
        '168_rr1': {'id': {'SF': ["Mmus_C57-6J_LVR_RR1_FLT_wERCC_Rep1_M25"],
                           'GC': ["Mmus_C57-6J_LVR_RR1_GC_wERCC_Rep5_M40"]}},
        '168_rr3': {'id': {'SF': ["Mmus_BAL-TAL_LVR_RR3_FLT_wERCC_Rep1_F1"],
                           'GC': ["Mmus_BAL-TAL_LVR_RR3_GC_wERCC_Rep4_G5"]}},
        '242': {'id': {'SF': ["Mmus_C57-6J_LVR_FLT_C1_Rep1_F1"],
                       'GC': ["Mmus_C57-6J_LVR_GC_C2_Rep1_G1",
                              "Mmus_C57-6J_LVR_CC_C1_Rep1_C1-1",
                              "Mmus_C57-6J_LVR_CC_C2_Rep1_C2-1"]}},
        '245': {'id': {'SF': ["Mmus_C57-6T_LVR_FLT_LAR_Rep1_F1",
                              "Mmus_C57-6T_LVR_FLT_ISS-T_Rep4_F10",
                              "Mmus_C57-6T_LVR_FLT_ISS-T_Rep2_F8"],
                       'GC': ["Mmus_C57-6T_LVR_GC_ISS-T_Rep1_G4",
                              "Mmus_C57-6T_LVR_GC_ISS-T_Rep2_G9",
                              "Mmus_C57-6T_LVR_GC_LAR_Rep1_G6",
                              "Mmus_C57-6T_LVR_GC_LAR_Rep2_G4"]}},
        '379': {'id': {'SF': ["RR8_LVR_FLT_ISS-T_YNG_FI7", "RR8_LVR_FLT_ISS-T_YNG_FI8",
                              "RR8_LVR_FLT_ISS-T_YNG_FI9", "RR8_LVR_FLT_ISS-T_OLD_FI11",
                              "RR8_LVR_FLT_ISS-T_OLD_FI12", "RR8_LVR_FLT_ISS-T_OLD_FI13"],
                       'GC': ["RR8_LVR_GC_ISS-T_YNG_GI8", "RR8_LVR_GC_ISS-T_YNG_GI9",
                              "RR8_LVR_GC_ISS-T_YNG_GI10", "RR8_LVR_GC_ISS-T_OLD_GI11",
                              "RR8_LVR_GC_ISS-T_OLD_GI12", "RR8_LVR_GC_ISS-T_OLD_GI13"]}},
    }
    if args.no_test:
        test_labels = None

    model = RNASeqModel()

    # --- build the concatenated matrix ---
    if args.custom_concat is None:                      # native z-score path
        model.load_counts(data_dir=['data', 'norm_counts'], verbose=args.verbose,
                          scale=True, exclude_test=test_labels)
        model.treatment_filter(id_name="Sample Name", factor='Factor Value[Spaceflight]',
                               valid_levels=['Space Flight', 'Ground Control',
                                             'Cohort Control #1', 'Cohort Control #2'],
                               verbose=args.verbose)
    model.build_combined_df(args.custom_concat, prep=args.prep, exclude_test=test_labels)
    print(model.concat_df.shape)

    suffix = ('_' + args.tag) if args.tag else ''

    # --- save the concatenated dataset (genes x samples, 'target' as last row) ---
    model.concat_df.transpose().to_csv("concat_df{}.csv".format(suffix), header=True)

    # --- classify + attribute ---
    results = model.fit_model(args.models, test_set=test_labels,
                              shap_values=not args.no_shap, verbose=args.verbose)

    # --- export attributions ---
    os.makedirs("feature_importance", exist_ok=True)
    for m in args.models:
        rd = results[m]
        if rd.get('model_coefs') is not None:
            coef_name = 'svm_importances' if m == 'svm' else '{}_importances'.format(m)
            rd['model_coefs'].sort_values('Score', ascending=False).to_csv(
                'feature_importance/{}{}.csv'.format(coef_name, suffix), header=True)
        if rd.get('shap') is not None:
            rd['shap'].to_csv('feature_importance/{}_shap_importance{}.csv'.format(m, suffix), header=True)
        if rd.get('accuracy') is not None:
            rd['accuracy'].to_csv('feature_importance/{}_accuracy{}.csv'.format(m, suffix), index=False)

