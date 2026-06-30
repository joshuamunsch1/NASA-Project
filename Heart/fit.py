# imports <- data handling
import pandas as pd
import numpy as np

# imports <- path navigation and data accession
import RNASeqUtility
import os
import itertools
import json
from pathlib import PurePath, Path

# imports <- visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# imports <- data set partioning
from random import sample
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import GraphicalLassoCV

# imports <- performance metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# imports <- data preprocessing
from sklearn.preprocessing import StandardScaler

# imports <- models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
#from utility.py_mrmr import mrmr_classif
#from utility.py_mrmr import random_forest_classif
#from utility import figure_generator

# imports <- optional models / explainers (added: XGBoost + SHAP feature importance)
# xgboost and shap are optional so the rest of the module still imports when they
# are not installed; the relevant code paths raise a clear error only if invoked.
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    raise
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# imports <- runtime
import argparse

from platform import python_version


# set the default plotting dimensions 7" by 5" with 200 dots per inch 
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':200})

"""RNASeqModel
Object representation of actions necessary to load, concatenate, and train on
transcriptomics data sets extracted from NASA GeneLab repository for the 
purposes of the Transalational Radiation Research and Countermeasures (TRRaC)
program for the NASA Space Radiation Element (SRE).
"""
class RNASeqModel:
    def __init__(self):
        # empty dictionary to hold RNA Seq counts data and respective metadata
        self.count_dict = {}
        self.roc_dict = {}
        

    ### DATA SET CURATION ###
    """add_study
    adds GeneLab data set (GLDS) id to the counts dictionary
    
    study_id <- string containing the GLDS ID
    """
    def add_study(self, study_id):
        self.count_dict[study_id] = {}
    
    """add_study_counts
    adds GLDS counts associated with GLDS ID
    
    study_id <- string containing the GLDS ID
    study_counts <- DataFrame containing RNA-Seq counts
    """
    def add_study_counts(self, study_id, study_counts):
        self.count_dict[study_id]['counts'] = study_counts
    
    """add_study_metadata
    adds GLDS study metadata associated with GLDS ID
    
    study_id <- string containing the GLDS ID
    study_metadata <- DataFrame containing RNA-Seq metadata
    """    
    def add_study_metadata(self, study_id, study_metadata):
        self.count_dict[study_id]['metadata'] = study_metadata

    """_spaceflight_targets
    Build {Sample Name: +1 (Space Flight) / -1 (all other arms)} from the
    single-study ISA metadata files in `metadata_dir`.

    This implements the metadata match the legacy index.str.contains("F")
    encoding only approximated. It is needed because some studies (e.g.
    OSD-599) use sample IDs that carry no "F" token (GSM accessions), so the
    name heuristic silently labels their flight samples as controls. The join
    key is the ISA "Sample Name", which is what 00_merge.R uses as the merged
    column id and therefore what the integration matrices carry as sample ids.
    """
    def _spaceflight_targets(self, metadata_dir=('data', 'single_study_metadata'),
                             id_col="Sample Name", factor_col="Factor Value[Spaceflight]"):
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

    """build_combined_df
    concatenates all GLDS studies into a single DataFrame
    
    custom_concat <- string filepath for concatenated DataFrame
    """
    def build_combined_df(self, custom_concat=None, prep='none', exclude_test=None, verbose=True):
        # create concatenated DataFrame if one is not provided
        if custom_concat is None:
            # inform the user the order that counts are concatenated
            # this is included to ensure consistent matching with metadata
            if verbose:
                print("Order that counts are concatenated: {}".format(self.count_dict.keys()))
            # iterate over counts dictionary to produce concatenated counts
            self.concat_df = pd.concat([x['counts'] for x in self.count_dict.values()])
        # if externally defined concat dataframe is provided, do not use loaded counts
        else:
            # inform user that concatenated dataframe will not be based off loaded counts
            # in this iteration
            print("Overwriting concatenated dataframe with file {}".format(custom_concat))
            # counts need to be transposed to match orientation of samples along 0-axis (rows)
            self.concat_df = pd.read_csv(custom_concat, index_col=0).transpose()
        
        # filter the DataFrame to remove ERCC spike ins which are only present in a subset of studies
        self.concat_df = self.concat_df.loc[:,self.concat_df.isna().sum(axis=0) == 0]
        # shared downstream prep for externally harmonized matrices (RUVg / ComBat-ref / DESeq2)
        if custom_concat is not None and prep in ('count', 'log'):
            self.apply_common_prep(prep, exclude_test)
        # --- target encoding: Space Flight (+1) vs Ground/other controls (-1) ---
        # Use an explicit Factor Value[Spaceflight] match from the study metadata
        # (the documented fix for the old index.str.contains("F") heuristic). Only
        # samples with no metadata entry fall back to the legacy name rule, and a
        # warning is printed so nothing is silently mislabeled. This corrects
        # studies whose sample ids carry no "F" (e.g. OSD-599 GSM accessions)
        # while leaving every existing study's labels unchanged.
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
    
    """load_filenames
    list directory contents from provided directory
    
    data_dir <- string filepath with data contents as flat files
    ftype <- string indicating the type of file being loaded
    verbose <- boolean for printing and troubleshooting
    """    
    def load_filenames(self, data_dir, ftype='', verbose=False):
        p = Path.cwd()
        # generate full file paths
        for d in data_dir:
            p = p / d
        fpaths = sorted([x for x in p.iterdir()])
        fpaths = list(filter(lambda x: (x.name[-4] == '.'), fpaths))
        # long form versus short hand file load outputting
        if verbose:
            for f in fpaths:
                print("Loading the {} file: {}".format(ftype, f.name))
        else:
            print("Loaded files:", ",".join([f.name for f in fpaths]))
        return fpaths
    """feature_subset
    filter counts dataframe by provided list of features
    
    count_dfs <- list of counts data
    feature_list <- string filename of features
    """
    def feature_subset(self, count_dfs, feature_list, verbose=False):
        features = pd.read_csv(feature_list).iloc[:,-1] 
        count_dfs = [x.loc[:,features] for x in count_dfs]
        if verbose:
            print("The subset dimensions of the count dataframes are: ", [x.shape for x in count_dfs])
        return count_dfs
    
    """scale_data
    apply the standard scale to a log transformed input
    
    count_dfs <- list of count data
    """
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
                valid_ids = count_dfs[i].index.isin(test_ids)
                train_ids = np.arange(df.index.values.shape[0])[~valid_ids]
                df = df.iloc[train_ids]
                
            scaler.fit(np.log2(df+1))
            
            count_dfs[i] = pd.DataFrame(scaler.transform(np.log2(count_dfs[i]+1)), columns=count_dfs[i].columns, index=count_dfs[i].index)
        return count_dfs   

    def apply_common_prep(self, prep, exclude_test):
        """Shared downstream prep for externally harmonized matrices.
        'count' -> log2(x+1) then global per-gene standardize.
        'log'   -> global per-gene standardize only (already log-scale).
        Standardizer is fit on TRAINING samples only (test held out)."""
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
    
    """load_counts
    load in the unnormalizard counts data
    
    data_dir <- array of strings with filepath to data contents as flar files
    metadata_dir <- array of strings with filepath to metadata contents as flat file
    verbose <- boolean for printing and troubleshooting
    scale <- boolean for applying transformation to loaded counts data
    feature_list <- string filepath to feature variables for subsetting
    """
    def load_counts(self, data_dir=['data', 'raw_counts'], metadata_dir=['data', 'single_study_metadata'], verbose=False, scale=False, exclude_test=None, feature_list=None):
        # call load_filenames for import of flat text files
        count_files = self.load_filenames(data_dir, ftype='counts', verbose=verbose)
        metadata_files = self.load_filenames(metadata_dir, ftype='metadata', verbose=verbose)
        
        # remove count files that do not have corresponding metadata file
        keep_files = []

        # perform check of counts against metadata files
        # do not include counts if metadata is not available
        rem_files = []
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
        
        # read in the files 
        count_dfs = [pd.read_csv(x, index_col=0).transpose() for x in count_files]
        metadata_dfs = [pd.read_csv(x, delimiter="\t") for x in metadata_files]
        
        ### APPLY OPTIONAL BLOCK PARAMETERS ###
        
        # apply subsetting to feature list if provided
        if feature_list:
            count_dfs = self.feature_subset(count_dfs, feature_list, verbose=verbose)
        
        # fit scale and transform data
        if scale:
            count_dfs = self.scale_data(count_dfs, exclude_test=exclude_test)

        fnames = [x.name.split('.')[0] for x in count_files]
       
       # save counts into object model
        for id, cdf, mdf in zip(fnames, count_dfs, metadata_dfs):
            self.add_study(id)
            self.add_study_counts(id, cdf)
            self.add_study_metadata(id, mdf)
    
    """treatment_filter
    filter the counts data by the treatments per the metadata file
    
    id_name <- string indicating the study id
    factor <- string indicating the column used for filtering
    valid_levels <- array of strings indicating valid values for factor column
    """
    def treatment_filter(self, id_name, factor, valid_levels, verbose=False):
        for k, v in self.count_dict.items():
            if verbose:
                print("Handling filtering of study {}".format(k))
            counts_df = v['counts']
            metadata_df = v['metadata']
            # get valid sample names 
            re = "|".join(valid_levels)
            valid_ids = metadata_df.loc[metadata_df[factor].str.contains(re), id_name]
            valid_ids = np.intersect1d(counts_df.index.values, valid_ids.tolist())
            self.count_dict[k]['counts'] = counts_df.loc[valid_ids, :]

    ### DATA SET PROFILING ###

    """profile_data
    generate summary statistics for counts data
    """
    def profile_data(self):
        profile_dict = {}
        kwargs = dict(axis=0)
        for id in self.count_dict.keys():
            study_counts = self.count_dict[id]['counts']
            means = study_counts.mean(**kwargs)
            stdevs = study_counts.std(**kwargs)
            vars = study_counts.var(**kwargs)
            kurts = study_counts.kurt(**kwargs)
            profile_dict[id] = {'mean':means, 
                                'std':stdevs, 
                                'var':vars,
                                'kurt':kurts}
        self.profile_dict = profile_dict
        
    """plot_profile
    metric <- string key for profile dict
    generate_csv <- boolean that saves output
    
    generates plots from data profiling metrics
    """
    def plot_profile(self, metric, generate_csv=False):
        # statistics generated from profile_data function call
        stats = [[k, v[metric]] for k, v in self.profile_dict.items()]
        # optional save statistics to csv
        if generate_csv:
            pd.concat([v[metric] for v in self.profile_dict.values()], axis=1).dropna(axis=0).to_csv('./data/stats/{}.csv'.format(metric))

        # define histogram parameters
        kwargs = dict(alpha=0.33, bins=250)
        
        # generate histogram of output        
        for i in stats:
            x = i[1].to_numpy()
            plt.hist(x[np.isfinite(x)], **kwargs, label=i[0]) # kurts
        
        # set plotting params and save histogram output            
        plt.title('Frequency Histogram of {} at Gene Level'.format(metric)); plt.ylabel('Frequency'); plt.legend()
        plt.savefig('{}.png'.format(metric))
    
    ### FEATURE SELECTION ### 
    """compute_mrmr
    X <- dataframe counts matrix 
    y <- series label vector
    K <- integer hyperparameter with number of iterations
    
    executes the minimum redundancy maximum relevance framework
    """
    def compute_mrmr(self, X, y, K):
        results = mrmr_classif(X, y, K)
        return results
    
    ### MODEL FITTING ###
    
    """update_roc
    trues <- array of labels 
    scores <- array of scores
    model_name <- name for fitted model
    
    saves the receiver operator characteristic curve fit results into a dictionary
    """
    def update_roc(self, trues, scores, model_name):
        fpr, tpr, thresholds = roc_curve(trues, scores)
        auc = roc_auc_score(trues, scores)
        self.roc_dict[model_name] = {}
        self.roc_dict[model_name]['fpr'] = fpr
        self.roc_dict[model_name]['tpr'] = tpr
        self.roc_dict[model_name]['thresholds'] = thresholds
        self.roc_dict[model_name]['auc'] = auc
        return self.roc_dict[model_name]
    
    """_build_estimator
    model_name <- string key identifying the model to construct
    model_params <- dictionary of per-model hyperparameters

    helper that returns an (unfitted) estimator for the requested model. New
    models added alongside the original rf / svm / lda / glm baselines:
      - 'elasticnet' : LogisticRegression with an elastic-net penalty (L1+L2 mix).
                       Wrapped in a StandardScaler pipeline because the 'saga'
                       solver is first-order and converges much faster on
                       standardized features. Scaling happens *inside* each CV
                       fold (fit on the training split only), so there is no
                       leakage even when the matrix has not been pre-scaled. On
                       the already-z-scored arm this re-standardization is
                       effectively a no-op.
      - 'xgboost'    : gradient-boosted trees. Shallow trees + heavy column
                       subsampling + L1/L2 leaf regularization for small-n /
                       high-p expression data. No scaler (trees are invariant
                       to monotonic feature transforms).
    """
    def _build_estimator(self, model_name, model_params):
        if model_name == "rf":
            return RandomForestClassifier(**model_params["rf"])
        if model_name == "svm":
            return LinearSVC(**model_params["svm"])
        if model_name == "lda":
            return LDA(**model_params["lda"])
        if model_name == "glm":
            return LogisticRegression(**model_params["glm"])
        if model_name == "elasticnet":
            return make_pipeline(StandardScaler(), LogisticRegression(**model_params["elasticnet"]))
        if model_name == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("xgboost is not installed. Run: pip install xgboost")
            return XGBClassifier(**model_params["xgboost"])
        raise ValueError("Unknown model: {}".format(model_name))

    """_final_estimator
    model <- a fitted estimator, possibly a Pipeline

    returns the terminal estimator so that .coef_ / .feature_importances_ can be
    read uniformly whether or not the model is wrapped in a Pipeline (elasticnet).
    """
    def _final_estimator(self, model):
        if hasattr(model, "named_steps"):
            return list(model.named_steps.values())[-1]
        return model

    """_shap_fold
    model <- fitted estimator (Pipeline or bare estimator)
    X_train <- training-fold design matrix (DataFrame)
    X_test  <- test-fold design matrix (DataFrame) on which SHAP is evaluated

    Computes per-fold SHAP values and returns a (2, n_features) array whose first
    row is the signed mean SHAP across test samples (direction of effect) and
    whose second row is the mean |SHAP| (magnitude, the usual ranking metric).

    Explainer selection:
      - tree models (xgboost)        -> TreeExplainer (exact, fast)
      - linear models (svm/elasticnet/lda/glm) -> LinearExplainer (closed form)

    NOTE on correctness: a LinearExplainer must see features in the *same space*
    the linear model was trained in. For the elasticnet pipeline the model lives
    in standardized space, so the StandardScaler is applied to the data before
    explaining. Passing raw (unscaled) values here would scale each gene's
    attribution by its raw standard deviation and silently corrupt the ranking
    (a low-variance signal gene gets demoted, a high-variance noise gene
    promoted). The bare linear models (svm/glm/lda) operate directly on the
    pre-scaled concat_df, so X is already in their input space.
    """
    def _shap_fold(self, model, X_train, X_test, verbose=False):
        if not SHAP_AVAILABLE:
            if verbose:
                print("  SHAP requested but the 'shap' package is not installed; skipping.")
            return None

        final = self._final_estimator(model)

        # Map data into the terminal estimator's input space (apply any pre-steps,
        # e.g. the StandardScaler in the elasticnet pipeline).
        if hasattr(model, "named_steps") and len(model.named_steps) > 1:
            pre = model[:-1]
            print(pre)
            Xtr_in = pre.transform(X_train.values)
            Xte_in = pre.transform(X_test.values)
        else:
            Xtr_in = X_train.values
            Xte_in = X_test.values

        try:
            if hasattr(final, "feature_importances_"):
                explainer = shap.TreeExplainer(final)
                sv = explainer.shap_values(Xte_in)
            elif hasattr(final, "coef_"):
                # cap the background at 100 samples for speed/stability
              #  if Xtr_in.shape[0] > 100:
              #      sel = np.random.RandomState(12345).choice(Xtr_in.shape[0], 100, replace=False)
              #      background = Xtr_in[sel]
              #  else:
              #      background = Xtr_in
              #  explainer = shap.LinearExplainer(final, background)
              #  sv = explainer.shap_values(Xte_in)
                background = Xtr_in if Xtr_in.shape[0] <= 100 else Xtr_in[np.random.choice(Xtr_in.shape[0], 100, replace=False)]
                explainer = shap.LinearExplainer(final, background)
                sv = explainer.shap_values(Xte_in)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) == 2 else np.mean(sv, axis=0)
            else:
                if verbose:
                    print("  SHAP: model type not supported by available explainers; skipping fold.")
                return None
        except Exception as e:
            if verbose:
                print("  SHAP failed for this fold: {}".format(e))
            return None

        # Normalize SHAP output shape across shap versions / class conventions:
        #   - list of per-class arrays (older shap)
        #   - (n, p) directly (binary, recent shap)
        #   - (n, p, n_classes) (multiclass)
        if isinstance(sv, list):
            arr = sv[1] if len(sv) == 2 else np.mean(np.stack(sv, axis=0), axis=0)
        else:
            arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[..., 1] if arr.shape[-1] == 2 else arr.mean(axis=-1)

        signed = arr.mean(axis=0)          # direction of effect, averaged over samples
        magnitude = np.abs(arr).mean(axis=0)  # mean |SHAP|, used for ranking
        return np.vstack([signed, magnitude])

    """fit_model
    models <- list of strings with models selected
              (supported: 'rf', 'svm', 'lda', 'glm', 'elasticnet', 'xgboost')
    mrmr <- boolean flag that indicates mrmr feature subset
    roc <- boolean flag that indicates roc curve fitting
    pfi <- boolean flag that indicates permutation feature importance calculation
    shap_values <- boolean flag that indicates SHAP feature importance calculation
    test_set <- dictionary with test set samples
    accuracy_block <- boolean flag that indicates accuracy testing calculation
    verbose <- boolean flag for function monitoring
    
    fits statistical models to data
    """
    def fit_model(self, models, mrmr=False, roc=False, pfi=False, shap_values=False, test_set=None, accuracy_block=False, verbose=False):
        model_params = {
            "rf": {
                    "n_estimators": 100, 
                    "max_depth": 5, 
                    "random_state" : 0, 
                    "min_samples_leaf": 1,
                    "min_samples_split":2,
                    "oob_score":True,
                    "class_weight":"balanced"
                },
            "svm": {
                    "penalty":'l2',
                    "loss":'squared_hinge',
                    "C":1.0,
                    "class_weight":'balanced',
                    "random_state":12345,
                    "max_iter":50000
            },
            "lda": {
                "solver":"eigen",
                "shrinkage":"auto"
            },
            "lda-svd" : {
                "solver":"svd"
                },
            "LDA_lsqr_params" :  {"solver":"lsqr", "covariance_estimator":GraphicalLassoCV(assume_centered=True, n_jobs=-1)},
            # glm = L2-penalized (ridge) logistic regression. Dense, signed
            # coefficients -> a probabilistic GSEA-arm partner to the SVM. Class
            # balancing + explicit C match the svm baseline for comparability.
            "glm" : {
                "penalty":"l2",
                "C":1.0,
                "class_weight":"balanced",
                "max_iter":10000,
                "random_state":0
            },
            # --- added models (hyperparameters ported from train_models.py) ---
            # elastic-net logistic regression: l1_ratio in [0,1] (0=ridge, 1=lasso);
            # C is the *inverse* regularization strength. 'saga' is required for an
            # elastic-net penalty in scikit-learn. For a more principled fit you can
            # swap LogisticRegression for LogisticRegressionCV (nested CV over Cs and
            # l1_ratios), which is the standard glmnet-style genomics workflow.
            "elasticnet": {
                "penalty": "elasticnet",
                "solver": "saga",
                "C": 1.0,
                "l1_ratio": 0.5,
                "max_iter": 10000,
                "tol": 1e-4,
                "random_state": 12345
            },
            # xgboost tuned for small-n / high-p: shallow trees, heavy feature
            # subsampling (lower colsample_bytree -> stronger implicit selection),
            # L1/L2 leaf regularization. objective is binary (SF vs GC).
            "xgboost": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.3,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "min_child_weight": 1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_jobs": -1,
                "random_state": 12345
            }
        }

        # Build the requested estimators, preserving the original ordering for the
        # baselines and appending the new models after them.
        supported = ["rf", "svm", "lda", "glm", "elasticnet", "xgboost"]
        model_list = []
        model_names = []
        for name in supported:
            if name in models:
                model_list.append(self._build_estimator(name, model_params))
                model_names.append(name)

        
        # perform stratified kfold splitting of the data sets
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
        
        X = self.concat_df.iloc[:,:-1]
        # The target is stored as {-1, +1} in concat_df. Re-encode to {0, 1}:
        #   - XGBoost requires labels in [0, num_class) and rejects {-1, +1}.
        #   - accuracy and ROC-AUC are invariant to this relabeling, and the
        #     decision_function / predict_proba[:, 1] sign conventions are
        #     unchanged (the positive class stays the SF / +1 group).
        # As a result the 'truth' column written to predictions is now {0, 1}.
        y = (self.concat_df.iloc[:,-1] > 0).astype(int)
        test_ids = []
        if test_set:
            for k in test_set.keys():
                test_ids.append(test_set[k]['id']['SF'])
                test_ids.append(test_set[k]['id']['GC'])
            test_ids = list(itertools.chain(*test_ids))
            valid_ids = X.index.isin(test_ids)
            train_ids = np.arange(X.index.values.shape[0])[~valid_ids]
            test_ids = np.arange(X.index.values.shape[0])[valid_ids]            
        
        if mrmr:
            X = X.loc[:,self.Genes['Gene']]
        
        results = {m : {} for m in models}
        acc_list = []; model_coef = []; roc_y_scores = []; roc_y_trues = []; 
        imp_arr = []; shap_arr = []
        for (model, model_name) in zip(model_list, model_names):
            train_test = skf.split(X, y)
            if test_set:
                train_test = zip([train_ids], [test_ids])
            if accuracy_block:
                train_test=skf.split(X.iloc[train_ids,:], y.iloc[train_ids])
                
            for train, test in train_test:
                model = model.fit(X.iloc[train,:], y.iloc[train])
                accuracy = accuracy_score(model.predict(X.iloc[test,:]), y.iloc[test])
                acc_list.append(accuracy)
                ### MODEL SPECIFIC ACTIONS ###
                # Native importance / coefficients, read from the terminal estimator:
                #   - tree models (rf, xgboost): feature_importances_ (unsigned, gain-based)
                #   - linear models (svm, lda, glm, elasticnet): coef_ (signed)
                final = self._final_estimator(model)
                if hasattr(final, "feature_importances_"):
                    model_coef.append(final.feature_importances_)
                else:
                    model_coef.append(final.coef_.T)
                
                if roc:
                    if model_name in ("rf", "lda", "xgboost"):
                        y_scores = model.predict_proba(X.iloc[test])[:, 1]
                    else:  # svm, glm, elasticnet expose decision_function
                        y_scores = model.decision_function(X.iloc[test])
                    roc_y_scores.append(y_scores); roc_y_trues.append(y.iloc[test])
                            
                if pfi:
                    importances = self.permutation_scoring(model, X.iloc[test], y.iloc[test], sort=False)
                    imp_arr.append(importances)

                if shap_values:
                    sv = self._shap_fold(model, X.iloc[train], X.iloc[test], verbose=verbose)
                    if sv is not None:
                        shap_arr.append(sv)
                
                ### MODEL SPECIFIC VERBOSITY ###
                if verbose:
                    print("The average accuracy was {:.3f}".format(np.mean(acc_list)))
                    if "rf" == model_name:
                        print('train -  {}   |   test -  {}   |   accuracy - {:.3f}'.format(np.bincount(y.iloc[train]), np.bincount(y.iloc[test]), accuracy))
                        print("OOB Score is {}".format(model.oob_score_))
                        print("Number of features seen is {}".format(model.n_features_in_))
                    if "lda" == model_name:
                        print("The number of features seen during model fitting: {}".format(model.n_features_in_))
                        print("The shape of the coefficient matrix from the model: {}".format(model.coef_.shape))

            if roc: 
                roc_dict = self.update_roc(np.concatenate(roc_y_trues, axis=0), np.concatenate(roc_y_scores, axis=0), model_name)
                results[model_name]['roc'] = pd.DataFrame(roc_dict)
                results[model_name]['predictions'] = pd.concat([pd.DataFrame(np.concatenate(roc_y_trues, axis=0), columns=['truth']), pd.DataFrame(np.concatenate(roc_y_scores, axis=0), columns=['probability'])], axis=1)
                roc_y_scores.clear(); roc_y_trues.clear()
                
            if pfi:
                results[model_name]['importance'] = pd.concat(imp_arr, axis=0, ignore_index=True); imp_arr.clear()

            if shap_values and shap_arr:
                # each entry is (2, p): row 0 = signed mean, row 1 = mean |SHAP|
                signed_stack = np.vstack([s[0] for s in shap_arr])
                abs_stack    = np.vstack([s[1] for s in shap_arr])
                results[model_name]['shap'] = pd.DataFrame({
                    "shap_mean":     signed_stack.mean(axis=0),   # direction of effect
                    "shap_abs_mean": abs_stack.mean(axis=0),      # magnitude (rank by this)
                    "shap_std":      signed_stack.std(axis=0),    # cross-fold stability
                }, index=X.columns.values)
                shap_arr.clear()
                
            results[model_name]["accuracy"] = pd.DataFrame(acc_list, columns=["Accuracy"]); acc_list.clear()
            results[model_name]["model_coefs"] = pd.DataFrame(np.mean(model_coef, axis=0), columns=['Score'], index=X.columns.values); model_coef.clear()
        
        return results
    
    """permutation_scoring
    estimator <- fitted model object
    X <- dataframe matrix of counts data
    Y <- Series vector of predicted labels
    sort <- boolean indicator to sort values by importance
    seed <- integer to constrain randomness
    
    helper function to run permutation feature importance scoring 
    """
    def permutation_scoring(self, estimator, X, Y, sort=True, seed=12345):
        # run the permutation feature importance
        if python_version()[:3] == "3.9":
            result = permutation_importance(estimator, X.to_numpy(), Y.to_numpy(), n_repeats=100, random_state=seed, scoring='roc_auc', n_jobs=-1)#, max_samples=10)
        else:
            result = permutation_importance(estimator, X, Y, n_repeats=100, random_state=seed, n_jobs=-1)
        # run a sort on the features
        sorted_importances_idx = result.importances_mean.argsort()
        # save to table
        # print(result.importances_mean[sorted_importances_idx])
        if sort:
            importances = pd.DataFrame(
                result.importances[sorted_importances_idx].T,
                columns=X.columns[sorted_importances_idx]
            )
        else:
            importances = pd.DataFrame(
                result.importances.T,
                columns=X.columns
            )
        return importances
                
    """generate_directionality
    uses the direction of average counts to create a directionality vector for features
    """
    def generate_directionality(self):
        data = self.concat_df.iloc[:,:-1]
        labels = self.concat_df.iloc[:,-1]
        
        sf_mask = labels == 1
        gc_mask = labels == -1
        
        sf = data.loc[sf_mask,:]
        gc = data.loc[gc_mask,:]
        
        print(sf.mean(axis=0))
        print(gc.mean(axis=0))
        print(np.greater(sf.mean(axis=0), gc.mean(axis=0)))
        
        dir_mask = np.greater(gc.mean(axis=0), sf.mean(axis=0))
        
        dir_vector = np.ones(dir_mask.shape[0])
        dir_vector[dir_mask] = -1
        
        print(dir_vector)
        
        dir_df = pd.DataFrame(dir_vector, columns=["Direction"], index=data.columns)
        dir_df.to_csv("directional.csv", header=True)
    
    """plot_pca
    mrmr <- boolean flag that indicates if the mrmr feature subset should be used
    
    plots a PCA of the data based on classification target
    """
    def plot_pca(self, mrmr=False):
        pca = PCA(n_components=2)
        
        X = self.concat_df.iloc[:,:-1]
        y = self.concat_df.iloc[:,-1]
        target_names = ['Ground Control', 'Space Flight']
        target_names = ['GC', 'SF']

        labels = np.empty(X.shape[0], dtype=object)
        for k in self.count_dict.keys():
            study_mask = np.isin(X.index.values, self.count_dict[k]['metadata']['Sample Name'])
            labels[study_mask] = k
        
        if mrmr:
            X = X.loc[:,self.Genes['Gene']]
        
        X_r = pca.fit(X).transform(X)
        colors = ["navy", "darkorange"]


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for MRMR run')

    parser.add_argument("-K", "--K", help="Specifies the number of MRMR features to select.", default=60, type=int)
    parser.add_argument("-O", "--outfile_name", help="Name for output MRMR csv.", default="mrmr")
    parser.add_argument("-F", "--feature_list", help="csv with gene set filtering for concatenated matrix.", default=None)
    parser.add_argument("-C", "--custom_concat", help="Provide custom concatenated data file.", default=None)
    parser.add_argument("-G", "--gene_list", help="csv with gene lists for model fitting.", default=None)
    parser.add_argument("-B", "--block_flags", help="json with flag specifications", default=None)
    parser.add_argument("-V", "--verbose", help="toggle output from functions", action="store_true")
    parser.add_argument("-P", "--prep", help="Downstream prep for externally corrected matrices: 'count' (log2+global standardize), 'log' (global standardize only), or 'none'.", default="none", choices=["count", "log", "none"])
    parser.add_argument("-T", "--tag", help="Tag appended to output filenames to keep per-arm results separate.", default="")
    
    args = parser.parse_args()
        
    kwargs = { 
              "data_dir":['data', 'norm_counts'],
              "feature_list": args.feature_list 
              }
    
    run_flags = {}
    if args.block_flags is not None:
        with open(args.block_flags, 'r') as f:
            json_str = f.read()
            run_flags = json.loads(json_str)    
    
    data_process_block = run_flags.get("data_process_block", True)
    save_concat_df_block = run_flags.get("save_concat_df_block", False)
    load_concat_df_to_clipboard_block = run_flags.get("load_concat_df_to_clipboard_block", False)
    profile_data_block = run_flags.get("profile_data_block", False)
    generate_pickle_block = run_flags.get("generate_pickle_block", False)
    mrmr_fitting_block = run_flags.get("mrmr_fitting_block", False)
    rf_fitting_block = run_flags.get("rf_fitting_block", False)
    svm_fitting_block = run_flags.get("svm_fitting_block", False)
    lda_fitting_block = run_flags.get("lda_fitting_block", False)
    glm_fitting_block = run_flags.get("glm_fitting_block", True)   # ridge logistic regression (GSEA arm)
    elasticnet_fitting_block = run_flags.get("elasticnet_fitting_block", False)
    xgboost_fitting_block = run_flags.get("xgboost_fitting_block", False)
    accuracy_block = run_flags.get("accuracy_block", False)
    pca_block = run_flags.get("pca_block", False)
    roc_block = run_flags.get("roc_block", True)
    pfi_block = run_flags.get("pfi_block", False)
    shap_block = run_flags.get("shap_block", True)
    random_block = run_flags.get("random_block", False)
    test_block = run_flags.get("test_block", True)

    gene_list_flag = args.gene_list is not None
    
    if test_block:
        test_labels = {
        # --- new studies (OSD-270 / 580 / 599). One held-out SF + GC pair each,
        #     matching the merged sample ids written by 00_merge.R. Remove a block
        #     to keep that study's samples entirely in training instead. NB: OSD-270
        #     has only 2 GC / 3 SF total, so holding one of each out leaves it thin. ---
        '270': {
            'id': {
                'SF':["RR3_HRT_FLT_F1"],
                'GC':["RR3_HRT_GC_G7"]
                }
            },
        '580': {
            'id': {
                'SF':["RRRM2_HRT_FLT_ISS-T_YNG_FY1"],
                'GC':["RRRM2_HRT_GC_ISS-T_YNG_GY1"]
                }
            },
        '599': {
            'id': {
                'SF':["GSM6996080"],
                'GC':["GSM6996077"]
                }
            },
    }
    else:
        test_labels = None
    
    model = RNASeqModel()

    # this block creates the combined dataframe 
    if data_process_block:
        if args.custom_concat is None:                       # z-score arm: native paper path
            model.load_counts(verbose=args.verbose, scale=True, exclude_test=test_labels, **kwargs)
            model.treatment_filter(id_name="Sample Name", factor='Factor Value[Spaceflight]', valid_levels=['Space Flight', 'Ground Control', 'Cohort Control #1', 'Cohort Control #2'], verbose=args.verbose)
        model.build_combined_df(args.custom_concat, prep=args.prep, exclude_test=test_labels)
    
    print(model.concat_df.shape)
    
    if random_block:
        seed(12345)
        ids = sample(list(np.arange(model.concat_df.shape[1]-1)),60)
        ids.append(-1)
        print(f"Random gene IDs {ids[:5]}")
        model.concat_df = model.concat_df.iloc[:,ids]

    if args.gene_list is not None:
        gene_list = pd.read_csv(args.gene_list, index_col=0)#.iloc[:100,]
        model.Genes = gene_list
    
    if pca_block:
        model.plot_pca(mrmr=args.gene_list)
        

    # this block is used to export the concatenated data set
    if save_concat_df_block:
        pd.DataFrame(model.concat_df).transpose().to_csv("concat_df.csv", header=True)
        model.generate_directionality()
    # this block saves the concatenated dataframe to the clipboard for troubleshooting    
    if load_concat_df_to_clipboard_block:
        model.concat_df.transpose().to_clipboard()
        
    # this block profiles the mean and std of the data 
    if profile_data_block:
        model.profile_data()
        model.plot_profile('mean')
    
    # # this block computes MRMR and exports 
    if mrmr_fitting_block:
        mrmr = model.compute_mrmr(model.concat_df.iloc[:,:-1], model.concat_df.iloc[:,-1], K=args.K)
        mrmr_df = pd.DataFrame(mrmr, columns=["Gene"])
        mrmr_df.to_csv("{}.csv".format(args.outfile_name), header=True)
    
    model_names = ['rf', 'svm', 'lda', 'glm', 'elasticnet', 'xgboost']
    _flags = [rf_fitting_block, svm_fitting_block, lda_fitting_block,
              glm_fitting_block, elasticnet_fitting_block, xgboost_fitting_block]
    chosen_models = [x for x, y in zip(model_names, _flags) if y]
    print("Models chosen are: ", chosen_models)    
    # this block loads a gene list and computes performance
    if args.gene_list is not None:
        gene_list = pd.read_csv(args.gene_list, index_col=0)#.iloc[:100,]
        model.Genes = gene_list
    

    results = model.fit_model(chosen_models, mrmr=gene_list_flag, roc=roc_block, pfi=pfi_block, shap_values=shap_block, test_set=test_labels)
    _suffix = ('_' + args.tag) if args.tag else ''

    # Per-model exports: coefficients/importances for every fitted model, plus
    # SHAP importances when shap_block is enabled. (The original code wrote only
    # svm_importances; this generalizes it without breaking that filename.)
    for m in chosen_models:
        rd = results[m]
        if rd.get('model_coefs') is not None:
            coef_name = 'svm_importances' if m == 'svm' else '{}_importances'.format(m)
            rd['model_coefs'].sort_values('Score', ascending=False).to_csv('feature_importance/{}{}.csv'.format(coef_name, _suffix), header=True)
        if rd.get('shap') is not None:
            rd['shap'].to_csv('feature_importance/{}_shap_importance{}.csv'.format(m, _suffix), header=True)
        if rd.get('accuracy') is not None:
            rd['accuracy'].to_csv('feature_importance/{}_accuracy{}.csv'.format(m, _suffix), index=False)

    if accuracy_block and gene_list_flag:
        dim = gene_list.shape[0]
        print(f"dim is {dim}")
        acc_array = np.zeros([dim, len(chosen_models)])
        for i in range(2,dim+1):
            model.Genes = gene_list.iloc[:i,]
            print("Gene {} added: ID {}".format(i, model.Genes.iloc[i-1,0]))
            results = model.fit_model(chosen_models, mrmr=gene_list_flag, roc=False, pfi=False, test_set=test_labels, accuracy_block=accuracy_block)
            for k, m in enumerate(chosen_models):
                acc_array[i-1,k] = results[m]['accuracy'].mean()
        col_names = chosen_models
        pd.DataFrame(acc_array, columns=col_names).to_csv("{}.csv".format(args.outfile_name))
        
    print("Done.")