import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from typing import List, Optional, Dict, Any

MODEL_PARAMS = {
    "rf": {
        "n_estimators": 200,
        "max_depth": 7,
        "random_state": 0,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "svm": {
        "C": 1.0,
        "loss": "squared_hinge",
        "penalty": "l2",
        "random_state": 12345,
    },
    "lda": {
        "solver": "eigen",
        "shrinkage": "auto"
    },
    "glm": {
        "C": 1.0,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": 0
    }
}


class MLTrainer:
    def __init__(self, ml_concat_csv: str, genes_txt: Optional[str] = None):
        self.ml_concat_csv = ml_concat_csv
        self.genes_txt = genes_txt
        self.concat_df = pd.read_csv(self.ml_concat_csv, index_col=0)
        self.label_col_name = self.concat_df.columns[-1]
        self.feature_names = list(self.concat_df.columns[:-1])
        self.X = self.concat_df.iloc[:, :-1]
        self.y = self.concat_df.iloc[:, -1].astype(int)
        self.Genes = {}
        if self.genes_txt is not None and os.path.exists(self.genes_txt):
            with open(self.genes_txt, 'r') as fh:
                genes = [line.strip() for line in fh if line.strip()]
            self.Genes['Gene'] = [g for g in genes if g in self.feature_names]
        self.random_state = 12345

    def _build_model(self, name: str):
        if name == "rf":
            model = RandomForestClassifier(**MODEL_PARAMS["rf"])
        elif name == "svm":
            svm_params = MODEL_PARAMS["svm"].copy()
            model = make_pipeline(StandardScaler(), LinearSVC(**svm_params))
        elif name == "lda":
            model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(**MODEL_PARAMS["lda"]))
        elif name == "glm":
            glm_params = MODEL_PARAMS["glm"].copy()
            model = make_pipeline(StandardScaler(), LogisticRegression(**glm_params))
        else:
            raise ValueError(f"Unknown model name: {name}")
        return model

    def train(self,
              models: List[str],
              roc: bool = False,
              pfi: bool = False,
              test_set: Optional[List[str]] = None,
              accuracy_block: bool = False,
              n_splits: int = 5,
              verbose: bool = False) -> Dict[str, Dict[str, Any]]:

        y = self.y.copy()
        X = self.X

        if test_set:
            test_mask = X.index.isin(test_set)
            if test_mask.sum() == 0:
                raise ValueError
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
        else:
            train_idx = None
            test_idx = None

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        results: Dict[str, Dict[str, Any]] = {}
        for model_name in models:
            results[model_name] = {
                "accuracy": [],
                "model_coefs": None,
                "importance": None,
                "roc": None,
                "predictions": None,
            }

        for model_name in models:
            if verbose:
                print(f"\nTraining model: {model_name}")

            estimator = self._build_model(model_name)

            acc_scores = []
            coef_list = []
            pfi_list = []
            roc_records = []
            pred_records = []

            if test_set is not None:
                if accuracy_block:
                    splits = skf.split(X.iloc[train_idx, :], y.iloc[train_idx])
                    wrapped_splits = []
                    for tr, te in splits:
                        wrapped_splits.append((train_idx[tr], train_idx[te]))
                    iterator = wrapped_splits
                else:
                    iterator = [(train_idx, test_idx)]
            else:
                iterator = skf.split(X, y)

            for split_num, (train_inds, test_inds) in enumerate(iterator):
                X_train = X.iloc[train_inds, :].values
                y_train = y.iloc[train_inds].values
                X_test = X.iloc[test_inds, :].values
                y_test = y.iloc[test_inds].values

                estimator.fit(X_train, y_train)

                y_pred = estimator.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                acc_scores.append(acc)

                if model_name == "rf":
                    base = estimator
                    if hasattr(base, 'named_steps'):
                        final = list(base.named_steps.values())[-1]
                    else:
                        final = base
                    feat_imp = getattr(final, "feature_importances_", None)
                    if feat_imp is not None:
                        coef_list.append(feat_imp)
                else:
                    base = estimator
                    if hasattr(base, 'named_steps'):
                        steps = list(base.named_steps.items())
                        final = steps[-1][1]
                    else:
                        final = base
                    if hasattr(final, "coef_"):
                        coefs = final.coef_
                        if coefs.ndim == 2:
                            avg_coef = np.mean(coefs, axis=0)
                        else:
                            avg_coef = coefs
                        coef_list.append(avg_coef)

                if roc:
                    prob_scores = None
                    if hasattr(estimator, "predict_proba"):
                        prob_scores = estimator.predict_proba(X_test)[:, 1]
                    else:
                        if hasattr(estimator, "decision_function"):
                            prob_scores = estimator.decision_function(X_test)
                        else:
                            prob_scores = y_pred
                    try:
                        auc = roc_auc_score(y_test, prob_scores)
                    except Exception:
                        auc = np.nan
                    roc_records.append({
                        "y_true": y_test,
                        "y_score": prob_scores,
                        "auc": auc,
                        "split": split_num
                    })
                    sample_names = X.index[test_inds]
                    pred_df = pd.DataFrame({
                        "sample": sample_names,
                        "truth": y_test,
                        "score": prob_scores
                    })
                    pred_records.append(pred_df)

                if pfi:
                    try:
                        imp = permutation_importance(estimator, X_test, y_test, n_repeats=30, random_state=self.random_state, n_jobs=-1, scoring='accuracy')
                        pfi_list.append(imp.importances_mean)
                    except Exception as e:
                        if verbose:
                            print("Permutation importance failed:", e)

            results_model = results[model_name]
            results_model["accuracy"] = pd.DataFrame({"Accuracy": acc_scores})

            if coef_list:
                coef_arr = np.vstack(coef_list)
                coef_mean = coef_arr.mean(axis=0)
                coef_df = pd.DataFrame(coef_mean, index=X.columns, columns=["Score"])
                results_model["model_coefs"] = coef_df
            else:
                results_model["model_coefs"] = None

            if pfi_list:
                pfi_arr = np.vstack(pfi_list)
                pfi_mean = pfi_arr.mean(axis=0)
                pfi_df = pd.DataFrame(pfi_mean, index=X.columns, columns=["pfi_mean"])
                pfi_df["pfi_std"] = pfi_arr.std(axis=0)
                results_model["importance"] = pfi_df
            else:
                results_model["importance"] = None

            if roc:
                aucs = [r["auc"] for r in roc_records]
                results_model["roc"] = {"per_split_auc": aucs, "mean_auc": np.nanmean(aucs) if len(aucs) else np.nan}
                if pred_records:
                    results_model["predictions"] = pd.concat(pred_records, axis=0, ignore_index=True)
                else:
                    results_model["predictions"] = None
            else:
                results_model["roc"] = None
                results_model["predictions"] = None

            if verbose:
                acc_mean = np.mean(results_model["accuracy"]["Accuracy"].values) if not results_model["accuracy"].empty else None
                print(f"Model {model_name} mean accuracy: {acc_mean:.4f}" if acc_mean is not None else f"Model {model_name} no accuracies recorded")
                if results_model["model_coefs" ] is not None:
                    top = results_model["model_coefs"].abs().sort_values("Score", ascending=False).head(10)
                    print("Top features (by absolute coefficient):")
                    print(top)
                if results_model["importance"] is not None:
                    print("Top features (by permutation importance):")
                    print(results_model["importance"].sort_values("pfi_mean", ascending=False).head(10))

        return results


if __name__ == "__main__":
    ML_CONCAT_F = "dataset_C_ml_concat.csv"
    GENES_F = "dataset_C_genes.txt"

    trainer = MLTrainer(ml_concat_csv=ML_CONCAT_F, genes_txt=GENES_F)

    rng = np.random.RandomState(0)
    all_samples = list(trainer.concat_df.index)
    n_holdout = max(1, int(0.2 * len(all_samples)))
    holdout_samples = rng.choice(all_samples, size=n_holdout, replace=False).tolist()

    models_to_run = ["lda"]
    results = trainer.train(models=models_to_run,
                            mrmr=False,
                            roc=True,
                            pfi=True,
                            test_set=holdout_samples,
                            accuracy_block=False,
                            n_splits=5,
                            verbose=True)

    out_dir = "training_results"
    os.makedirs(out_dir, exist_ok=True)
    for mname, rd in results.items():
        if rd["accuracy"] is not None:
            rd["accuracy"].to_csv(os.path.join(out_dir, f"{mname}_accuracy_per_fold.csv"), index=False)
        if rd["model_coefs"] is not None:
            rd["model_coefs"].to_csv(os.path.join(out_dir, f"{mname}_model_coefs.csv"))
        if rd["importance"] is not None:
            rd["importance"].to_csv(os.path.join(out_dir, f"{mname}_permutation_importance.csv"))
        if rd["predictions"] is not None:
            rd["predictions"].to_csv(os.path.join(out_dir, f"{mname}_predictions.csv"), index=False)
        if rd["roc"] is not None:
            pd.DataFrame([rd["roc"]]).to_csv(os.path.join(out_dir, f"{mname}_roc_summary.csv"), index=False)

    print(f"Training finished. Results saved in: {out_dir}")
