"""second_benchmark_pipeline.py - the Capstone 1 leakage-free pipeline on a second dataset (Section 4.8).

Usage:   python second_benchmark_pipeline.py HR_comma_sep.csv second_benchmark_results.json
Needs:   pip install pandas numpy scikit-learn imbalanced-learn
Time:    about 10 minutes on a laptop.

This version uses imbalanced-learn (SMOTENC + imblearn Pipeline) exactly as the Kaggle notebook does,
so SMOTE-NC is fitted inside every training fold and never sees validation or test rows.

Steps (same protocol as the IBM experiments, Table 13 of the report):
  1. Load HR_comma_sep.csv and REMOVE exact duplicate rows (3,008), so no employee can appear in both
     the training and the test set.
  2. Split off the test set first: stratified 80/20, seed 42.
  3. For each model (LR, RF, GB) and each imbalance option (SMOTE-NC, or class weights / none):
     grid search with 5-fold stratified CV on the training set, scored by PR-AUC.
  4. Select the pipeline with the best CV PR-AUC; choose the threshold that maximises F1 on its
     out-of-fold training predictions.
  5. Evaluate ONCE on the test set at that threshold.
  6. Robustness and extra checks: permutation importance, 20 repeated splits (seeds 100-119),
     ablation without satisfaction_level (CV only), duplicates-kept sensitivity, and the controlled
     leakage comparison of Experiment A2 (seeds 42-46).
"""
import sys, json, time, warnings
import numpy as np, pandas as pd
import sklearn, imblearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix,
                             precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
SEED = 42
T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

CSV = sys.argv[1] if len(sys.argv) > 1 else "HR_comma_sep.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "second_benchmark_results.json"
log("scikit-learn", sklearn.__version__, "| imbalanced-learn", imblearn.__version__)

# ---------------------------------------------------------------- 1. data
raw = pd.read_csv(CSV).rename(columns={"sales": "Department"})
TARGET = "left"
CAT = ["Department", "salary"]
NUM = [c for c in raw.columns if c not in CAT + [TARGET]]
dups = int(raw.duplicated().sum())
data = raw.drop_duplicates().reset_index(drop=True)
y = data[TARGET].to_numpy()
X = data.drop(columns=TARGET)
X[CAT] = X[CAT].astype(object)
R = {"versions": {"scikit-learn": sklearn.__version__, "imbalanced-learn": imblearn.__version__},
     "data": {"rows_raw": len(raw), "duplicates": dups, "rows_unique": len(data), "leavers_unique": int(y.sum()),
              "leaver_rate_unique": float(y.mean()), "leaver_rate_raw": float(raw[TARGET].mean()),
              "numeric": NUM, "categorical": CAT}}
log("data", R["data"])

# ---------------------------------------------------------------- 2. held-out test set first
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
R["split"] = {"train": len(ytr), "train_leavers": int(ytr.sum()), "test": len(yte), "test_leavers": int(yte.sum())}

# ---------------------------------------------------------------- pipeline builder (same as the notebook's pipe())
def pipe(model, balance="smotenc", num=NUM, cat=CAT):
    """scale numeric -> SMOTE-NC (categoricals still raw) -> one-hot encode -> model, all inside one pipeline."""
    scale = ColumnTransformer([("num", StandardScaler(), num)], remainder="passthrough",
                              verbose_feature_names_out=False).set_output(transform="pandas")
    enc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                            remainder="passthrough")
    steps = [("scale", scale)]
    if balance == "smotenc":
        steps.append(("smote", SMOTENC(categorical_features=cat, random_state=SEED)))
    return ImbPipeline(steps + [("ohe", enc), ("model", model)])

def metrics(yt, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
    return {"pr_auc": average_precision_score(yt, p), "roc_auc": roc_auc_score(yt, p),
            "precision": precision_score(yt, yh, zero_division=0), "recall": recall_score(yt, yh),
            "f1": f1_score(yt, yh), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def oof_threshold(yt, oof):
    pr, rc, th = precision_recall_curve(yt, oof)
    f1s = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.nanargmax(f1s[:-1])])

# ---------------------------------------------------------------- 3-4. grid search for every model x imbalance option
grids = {
    "LR": (LogisticRegression(max_iter=3000), {"model__C": [0.01, 0.1, 1, 10]}),
    "RF": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
           {"model__n_estimators": [300], "model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 5]}),
    "GB": (GradientBoostingClassifier(random_state=SEED),
           {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}),
}
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
comp, fitted = [], {}
for name, (m, g) in grids.items():
    for bal in ["smotenc", "weights"]:
        mm = clone(m)
        if bal == "weights" and name != "GB":
            mm.set_params(class_weight="balanced")
        gs = GridSearchCV(pipe(mm, "smotenc" if bal == "smotenc" else None), g, scoring="average_precision",
                          cv=cv5).fit(Xtr, ytr)
        oof = cross_val_predict(gs.best_estimator_, Xtr, ytr, cv=cv5, method="predict_proba")[:, 1]
        thr = oof_threshold(ytr, oof); mo = metrics(ytr, oof, thr)
        imb = "smote" if bal == "smotenc" else ("none" if name == "GB" else "cw")
        comp.append({"model": name, "imb": imb, "hp": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
                     "cv_pr_auc": gs.best_score_, "threshold": thr, "oof_recall": mo["recall"],
                     "oof_precision": mo["precision"], "oof_f1": mo["f1"]})
        fitted[(name, imb)] = (gs.best_estimator_, thr, gs.best_params_)
        log("combo", name, imb, comp[-1]["hp"], round(gs.best_score_, 4), round(thr, 3))
comp.sort(key=lambda r: -r["cv_pr_auc"])
R["cv_comparison"] = comp
sel = comp[0]
est, thr, best_params = fitted[(sel["model"], sel["imb"])]
log("selected", sel["model"], sel["imb"], sel["hp"], "threshold", round(thr, 3))

# ---------------------------------------------------------------- 5. single test evaluation
pte = est.predict_proba(Xte)[:, 1]
R["test"] = {"selected": sel, "at_threshold": metrics(yte, pte, thr), "at_0.5": metrics(yte, pte, 0.5)}
log("test", {k: round(v, 3) for k, v in R["test"]["at_threshold"].items()})

# ---------------------------------------------------------------- 6a. permutation importance (drop in test PR-AUC)
pi = permutation_importance(est, Xte, yte, scoring="average_precision", n_repeats=20, random_state=SEED)
R["perm_importance"] = dict(sorted(zip(X.columns, map(float, pi.importances_mean)), key=lambda kv: -kv[1]))
log("perm", {k: round(v, 3) for k, v in R["perm_importance"].items()})

# ---------------------------------------------------------------- 6b. 20 repeated splits (same hyperparameters and threshold)
rep = []
for sd in range(100, 120):
    a, b, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=sd)
    rep.append(metrics(yb, clone(est).fit(a, ya).predict_proba(b)[:, 1], thr))
R["repeated"] = {k: {"mean": float(np.mean([r[k] for r in rep])), "sd": float(np.std([r[k] for r in rep], ddof=1)),
                     "min": float(np.min([r[k] for r in rep])), "max": float(np.max([r[k] for r in rep]))}
                 for k in ["f1", "recall", "precision", "pr_auc", "roc_auc"]}
log("repeated F1", round(R["repeated"]["f1"]["min"], 3), "-", round(R["repeated"]["f1"]["max"], 3))

# ---------------------------------------------------------------- 6c. ablation without satisfaction_level (CV only)
num_wo = [c for c in NUM if c != "satisfaction_level"]
m, g = grids[sel["model"]]
mm = clone(m)
if sel["imb"] == "cw":
    mm.set_params(class_weight="balanced")
gs_wo = GridSearchCV(pipe(mm, "smotenc" if sel["imb"] == "smote" else None, num=num_wo), g,
                     scoring="average_precision", cv=cv5).fit(Xtr.drop(columns="satisfaction_level"), ytr)
R["ablation_no_satisfaction"] = {"model": sel["model"], "imb": sel["imb"], "cv_pr_auc": gs_wo.best_score_,
                                 "hp": {k.replace("model__", ""): v for k, v in gs_wo.best_params_.items()},
                                 "cv_pr_auc_with": sel["cv_pr_auc"]}
log("ablation", round(sel["cv_pr_auc"], 3), "->", round(gs_wo.best_score_, 3))

# ---------------------------------------------------------------- 6d. sensitivity: what if duplicates were kept?
Xr = raw.drop(columns=TARGET).copy(); Xr[CAT] = Xr[CAT].astype(object); yr = raw[TARGET].to_numpy()
a, b, ya, yb = train_test_split(Xr, yr, test_size=0.2, stratify=yr, random_state=SEED)
key = lambda d: d.astype(str).agg("|".join, axis=1)
overlap = int(key(b).isin(set(key(a))).sum())
R["duplicates_kept"] = {"test_rows": len(yb), "test_rows_with_copy_in_train": overlap,
                        **metrics(yb, clone(est).fit(a, ya).predict_proba(b)[:, 1], thr)}
log("duplicates kept:", overlap, "of", len(yb), "test rows have a copy in training")

# ---------------------------------------------------------------- 6e. controlled leakage comparison (as experiment_A2_smotenc.py)
cols = NUM + CAT
cat_idx = list(range(len(NUM), len(cols)))
Xa = X[cols].reset_index(drop=True)
def scaler():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", "passthrough", CAT)])
def encoder():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx)],
                             remainder="passthrough")
def forest(s):
    return RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
arms = {"leaky": [], "inside": [], "none": []}
for s in range(42, 47):
    cv = StratifiedKFold(5, shuffle=True, random_state=s)
    # leaky: SMOTE-NC on the whole dataset first, then CV; scored on the original records only
    Xs = pd.DataFrame(scaler().fit_transform(Xa), columns=cols)
    Xres, yres = SMOTENC(categorical_features=cat_idx, random_state=s).fit_resample(Xs, y)
    Xres = pd.DataFrame(Xres, columns=cols); Xres[NUM] = Xres[NUM].astype(float)
    p_leak = cross_val_predict(Pipeline([("ohe", encoder()), ("rf", forest(s))]), Xres, yres, cv=cv,
                               method="predict_proba")[:, 1]
    arms["leaky"].append(metrics(y, p_leak[:len(y)], 0.5))
    # SMOTE-NC inside each training fold
    inside = ImbPipeline([("scale", scaler()), ("smote", SMOTENC(categorical_features=cat_idx, random_state=s)),
                          ("ohe", encoder()), ("rf", forest(s))])
    arms["inside"].append(metrics(y, cross_val_predict(inside, Xa, y, cv=cv, method="predict_proba")[:, 1], 0.5))
    # no resampling
    plain = Pipeline([("scale", scaler()), ("ohe", encoder()), ("rf", forest(s))])
    arms["none"].append(metrics(y, cross_val_predict(plain, Xa, y, cv=cv, method="predict_proba")[:, 1], 0.5))
    log("A2 seed", s, "F1 leaky/inside/none:", *[round(arms[k][-1]["f1"], 3) for k in arms])
R["a2"] = {arm: {k: {"mean": float(np.mean([r[k] for r in v])), "sd": float(np.std([r[k] for r in v], ddof=1)),
                     "min": float(np.min([r[k] for r in v])), "max": float(np.max([r[k] for r in v]))}
                 for k in ["f1", "recall", "precision", "pr_auc", "roc_auc"]}
           for arm, v in arms.items()}

json.dump(R, open(OUT, "w"), indent=2, default=str)
log("done ->", OUT)
