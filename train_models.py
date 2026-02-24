import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
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
        "max_iter": 5000,
        "random_state": 12345,
    },
    "lda": {
        "solver": "eigen",
        "shrinkage": "auto",
    },
    "glm": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 0,
    },
}



class MLTrainer:
    def __init__(
        self,
        feature_csv: str,
        metadata_csv: str,
        label_col: str = "group",
        sample_id_col: Optional[str] = None,
        genes_txt: Optional[str] = None,
    ):
        self.random_state = 12345

        feature_df = pd.read_csv(feature_csv, index_col=0)
        feature_df.index   = feature_df.index.astype(str)
        feature_df.columns = feature_df.columns.astype(str)

        meta_df = pd.read_csv(metadata_csv)
        id_col = sample_id_col if sample_id_col is not None else meta_df.columns[0]
        meta_df = meta_df.set_index(id_col)
        meta_df.index = meta_df.index.astype(str)

        if label_col not in meta_df.columns:
            raise ValueError

        overlap_rows = len(set(feature_df.index)   & set(meta_df.index))
        overlap_cols = len(set(feature_df.columns) & set(meta_df.index))

        if overlap_rows == 0 and overlap_cols == 0:
            raise ValueError

        if overlap_cols > overlap_rows:
            feature_df = feature_df.T   

        combined = feature_df.join(meta_df[[label_col]], how="inner")

        if combined.shape[0] == 0:
            raise ValueError

        print(f"Loaded {combined.shape[0]} with {combined.shape[1] - 1}")
        print(f"Label distribution:\n{combined[label_col].value_counts()}\n")

        self.label_encoder = LabelEncoder()
        combined["_label_enc"] = self.label_encoder.fit_transform(combined[label_col].astype(str))

        self.feature_names = [c for c in combined.columns if c not in [label_col, "_label_enc"]]
        self.X = combined[self.feature_names]
        self.y = combined["_label_enc"].astype(int)

       
        self.concat_df = combined.rename(columns={label_col: "target_label", "_label_enc": "target_label_encoded"})

        self.Genes: Dict[str, List[str]] = {}
        if genes_txt is not None and os.path.exists(genes_txt):
            with open(genes_txt) as fh:
                genes = [line.strip() for line in fh if line.strip()]
            self.Genes["Gene"] = [g for g in genes if g in self.feature_names]
            print(f"Gene list loaded: {len(self.Genes['Gene'])} genes overlap with features.")


    def _build_model(self, name: str):
        if name == "rf":
            return RandomForestClassifier(**MODEL_PARAMS["rf"])
        elif name == "svm":
            return make_pipeline(StandardScaler(), LinearSVC(**MODEL_PARAMS["svm"]))
        elif name == "lda":
            return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(**MODEL_PARAMS["lda"]))
        elif name == "glm":
            return make_pipeline(StandardScaler(), LogisticRegression(**MODEL_PARAMS["glm"]))
        else:
            raise ValueError

    # ---------------------------
    def train(
        self,
        models: List[str],
        mrmr: bool = False,
        roc: bool = False,
        pfi: bool = False,
        test_set: Optional[List[str]] = None,
        n_splits: int = 5,
        verbose: bool = False,
    ) -> Dict[str, Dict[str, Any]]:

        X = self.X.copy()
        y = self.y.copy()

        if test_set is not None:
            test_mask  = X.index.isin(test_set)
            train_idx  = np.where(~test_mask)[0]
            test_idx   = np.where(test_mask)[0]
            if len(test_idx) == 0:
                raise ValueError
        else:
            train_idx = test_idx = None

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        results: Dict[str, Dict[str, Any]] = {}

        for model_name in models:
            if verbose:
                print(f"\n--- Training: {model_name} ---")

            estimator  = self._build_model(model_name)
            acc_scores = []
            coef_list  = []
            pfi_list   = []
            roc_records   = []
            pred_records  = []

            if test_set is not None:
                iterator = [(train_idx, test_idx)]
            else:
                iterator = list(skf.split(X, y))

           # print(iterator)
            for split_num, (train_inds, test_inds) in enumerate(iterator):
              #  print(split_num)
              #  print(train_inds)
              #  print(test_inds)
                X_train = X.iloc[train_inds].values
                y_train = y.iloc[train_inds].values
                X_test  = X.iloc[test_inds].values
                y_test  = y.iloc[test_inds].values

                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                acc_scores.append(acc)

                if verbose:
                    print(f"  Fold {split_num + 1} accuracy: {acc:.4f}")
                
                final = list(estimator.named_steps.values())[-1] if hasattr(estimator, "named_steps") else estimator
                if hasattr(final, "feature_importances_"):
                    coef_list.append(final.feature_importances_)
                elif hasattr(final, "coef_"):
                    coefs = final.coef_
                    coef_list.append(np.mean(coefs, axis=0) if coefs.ndim == 2 else coefs)

                if roc:
                    if hasattr(estimator, "predict_proba"):
                        scores = estimator.predict_proba(X_test)[:, 1]
                    elif hasattr(estimator, "decision_function"):
                        scores = estimator.decision_function(X_test)
                    else:
                        scores = y_pred
                    try:
                        auc = roc_auc_score(y_test, scores)
                    except Exception:
                        auc = np.nan
                    roc_records.append(auc)
                    pred_records.append(pd.DataFrame({
                        "sample": X.index[test_inds],
                        "truth":  y_test,
                        "score":  scores,
                    }))

                if pfi:
                    try:
                        imp = permutation_importance(
                            estimator, X_test, y_test,
                            n_repeats=1, random_state=self.random_state,
                            n_jobs=-1, scoring="accuracy",
                        )
                        pfi_list.append(imp.importances_mean)
                        print(imp)
                        print(imp.importances_mean)
                        print(pfi_list)
                    except Exception as e:
                        if verbose:
                            print(f"  Permutation importance failed: {e}")

            accuracy_df = pd.DataFrame({"Accuracy": acc_scores})

            coef_df = None
            if coef_list:
                coef_mean = np.vstack(coef_list).mean(axis=0)
                coef_df = pd.DataFrame(coef_mean, index=X.columns, columns=["Score"])

            pfi_df = None
            if pfi_list:
                pfi_arr = np.vstack(pfi_list)
                pfi_df  = pd.DataFrame({
                    "pfi_mean": pfi_arr.mean(axis=0),
                    "pfi_std":  pfi_arr.std(axis=0),
                }, index=X.columns)

            roc_summary  = {"per_split_auc": roc_records, "mean_auc": np.nanmean(roc_records)} if roc else None
            predictions  = pd.concat(pred_records, ignore_index=True) if pred_records else None

            results[model_name] = {
                "accuracy":    accuracy_df,
                "model_coefs": coef_df,
                "importance":  pfi_df,
                "roc":         roc_summary,
                "predictions": predictions,
            }

            if verbose:
                mean_acc = accuracy_df["Accuracy"].mean()
                print(f"  Mean accuracy: {mean_acc:.4f}")
                if roc_summary:
                    print(f"  Mean AUC:      {roc_summary['mean_auc']:.4f}")
                if coef_df is not None:
                    print("  Top 10 features:")
                    print(coef_df.abs().sort_values("Score", ascending=False).head(10))

        return results


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    FEATURE_CSV   = "dataset_C_corrected_logCPM.csv"
    METADATA_CSV  = "metadata.csv"
    LABEL_COL     = "group"   
    SAMPLE_ID_COL = "sample"      
    GENES_F       = None     
    trainer = MLTrainer(
        feature_csv   = FEATURE_CSV,
        metadata_csv  = METADATA_CSV,
        label_col     = LABEL_COL,
        sample_id_col = SAMPLE_ID_COL,
        genes_txt     = GENES_F,
    )

    results = trainer.train(
        models   = ["rf"],
        roc      = True,
        pfi      = False,
        n_splits = 5,
        verbose  = True,
    )

    # Save outputs
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


    print(f"\nDone. Results saved to: {out_dir}/")
