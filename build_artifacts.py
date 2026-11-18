"""build_artifacts.py - offline pipeline that prepares everything the Streamlit dashboard shows.

Run once (about 1 minute):
    python build_artifacts.py path/to/WA_Fn-UseC_-HR-Employee-Attrition.csv

What it does (same steps and settings as the Kaggle notebook, Sections 1, 6-9):
  1. Clean the IBM data: drop the constant fields and the ID, keep Gender and MaritalStatus OUT of the model.
  2. Split off the held-out test set first (stratified 80/20, seed 42).
  3. Re-create the selected pipeline: scale numeric -> one-hot encode categorical -> logistic regression
     with class weights. Tune C by 5-fold stratified CV scored by PR-AUC on the training set only.
  4. Choose the threshold that maximises F1 on out-of-fold training predictions (0.60).
  5. Evaluate once on the test set; compute permutation importance, SHAP contributions and the fairness audit.
  6. Save the artefacts to ./artifacts. The dashboard only READS these files - it never retrains,
     so it cannot change any reported result.

The selected pipeline uses class weights, not SMOTE-NC, so it needs only scikit-learn.
"""
import sys, json, os
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)

SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. clean
raw = pd.read_csv(CSV, encoding="utf-8-sig")
ids = raw["EmployeeNumber"].values
df = raw.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])
y = (df.pop("Attrition") == "Yes").astype(int).values
sensitive = df[["Gender", "Age"]].copy()
X = df.drop(columns=["Gender", "MaritalStatus"])
CAT = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
X[CAT] = X[CAT].astype(object)
NUM = [c for c in X.columns if c not in CAT]
print(f"{len(df)} employees, {y.sum()} leavers ({y.mean():.1%}); {X.shape[1]} predictors")

# ---------------------------------------------------------------- 2. held-out test set first
Xtr, Xte, ytr, yte, Str, Ste, idtr, idte = train_test_split(
    X, y, sensitive, ids, test_size=0.2, stratify=y, random_state=SEED)

# ---------------------------------------------------------------- 3. selected pipeline (as pipe(model, None) in the notebook)
def make_pipeline(model):
    scale = ColumnTransformer([("num", StandardScaler(), NUM)], remainder="passthrough",
                              verbose_feature_names_out=False).set_output(transform="pandas")
    enc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
                            remainder="passthrough")
    return Pipeline([("scale", scale), ("ohe", enc), ("model", model)])

cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
gs = GridSearchCV(make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced")),
                  {"model__C": [0.01, 0.1, 1, 10]}, scoring="average_precision", cv=cv5).fit(Xtr, ytr)
est = gs.best_estimator_
print("best C:", gs.best_params_["model__C"], "| CV PR-AUC:", round(gs.best_score_, 3))

# ---------------------------------------------------------------- 4. threshold from out-of-fold predictions
oof = cross_val_predict(est, Xtr, ytr, cv=cv5, method="predict_proba")[:, 1]
pr, rc, th = precision_recall_curve(ytr, oof)
f1s = 2 * pr * rc / (pr + rc + 1e-12)
thr = float(th[np.nanargmax(f1s[:-1])])
print("threshold from OOF F1:", round(thr, 3))

# ---------------------------------------------------------------- 5. single test evaluation
pte = est.predict_proba(Xte)[:, 1]

def metrics(yt, p, t):
    yh = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
    return dict(pr_auc=average_precision_score(yt, p), roc_auc=roc_auc_score(yt, p),
                precision=precision_score(yt, yh, zero_division=0), recall=recall_score(yt, yh),
                f1=f1_score(yt, yh), accuracy=accuracy_score(yt, yh), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

m_test = metrics(yte, pte, thr)
m_test_05 = metrics(yte, pte, 0.5)
m_oof = metrics(ytr, oof, thr)
print("test at threshold:", {k: round(v, 3) for k, v in m_test.items()})

# threshold sweep on the test set (what-if view for the slider; the threshold itself comes from training data)
sweep = []
for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
    m = metrics(yte, pte, t)
    sweep.append(dict(threshold=t, flagged=m["tp"] + m["fp"], recall=m["recall"], precision=m["precision"],
                      fpr=m["fp"] / max(m["fp"] + m["tn"], 1)))
prc_p, prc_r, _ = precision_recall_curve(yte, pte)

# permutation importance (drop in test PR-AUC), as in the notebook
pi = permutation_importance(est, Xte, yte, scoring="average_precision", n_repeats=20, random_state=SEED)
imp = pd.DataFrame({"feature": X.columns, "perm_importance": pi.importances_mean})

# SHAP for a linear model (interventional, background = transformed training set):
# contribution_j = coef_j * (x_j - mean_j) in log-odds; one-hot columns are summed back to their feature.
pre = Pipeline(est.steps[:-1])
Ztr = np.asarray(pre.transform(Xtr), dtype=float); Zte = np.asarray(pre.transform(Xte), dtype=float)
names = [n.split("__")[-1] for n in est.named_steps["ohe"].get_feature_names_out()]
lr = est.named_steps["model"]; coef = lr.coef_[0]; mu = Ztr.mean(axis=0)
contrib = (Zte - mu) * coef
base_logodds = float(lr.intercept_[0] + coef @ mu)

def source_feature(col):
    for c in CAT:
        if col.startswith(c + "_"):
            return c
    return col
src = [source_feature(n) for n in names]
shap_df = pd.DataFrame(contrib, columns=names).T.groupby(src).sum().T
shap_df = shap_df[list(X.columns)]
shap_df.insert(0, "EmployeeNumber", idte)
# check: contributions + baseline reproduce each employee's log-odds
logit = np.log(pte / (1 - pte))
assert np.allclose(shap_df[list(X.columns)].sum(axis=1).values + base_logodds, logit, atol=1e-6)
imp["mean_abs_shap"] = imp["feature"].map(shap_df[list(X.columns)].abs().mean())
imp = imp.sort_values("perm_importance", ascending=False)

# ---------------------------------------------------------------- fairness audit (as in the notebook) + Wilson intervals
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def audit(yt, p, t, S, partition):
    groups = {"Female": S.Gender.values == "Female", "Male": S.Gender.values == "Male",
              "Under 30": S.Age.values < 30, "30 to 45": (S.Age.values >= 30) & (S.Age.values <= 45),
              "Over 45": S.Age.values > 45}
    out = []
    for g, mk in groups.items():
        tn, fp, fn, tp = confusion_matrix(yt[mk], (p[mk] >= t).astype(int), labels=[0, 1]).ravel()
        lo, hi = wilson(tp, tp + fn)
        out.append(dict(partition=partition, group=g, n=int(mk.sum()), leavers=int(yt[mk].sum()),
                        base_rate=float(yt[mk].mean()), recall=tp / max(tp + fn, 1), recall_ci_low=lo, recall_ci_high=hi,
                        precision=tp / max(tp + fp, 1), fpr=fp / max(fp + tn, 1), flagged=float((p[mk] >= t).mean())))
    return out

fair = pd.DataFrame(audit(ytr, oof, thr, Str, "Training (out-of-fold)") + audit(yte, pte, thr, Ste, "Test"))

# ---------------------------------------------------------------- 6. save artefacts
joblib.dump({"pipeline": est, "threshold": thr, "features": list(X.columns), "categorical": CAT,
             "shap_background_mean": mu, "base_log_odds": base_logodds},
            os.path.join(OUT, "attrition_pipeline.joblib"))

emp = raw.drop(columns=["EmployeeCount", "Over18", "StandardHours"]).copy()
emp["Left"] = (emp["Attrition"] == "Yes").astype(int)
emp["Partition"] = np.where(np.isin(emp["EmployeeNumber"], idte), "Test", "Training")
emp.to_csv(os.path.join(OUT, "employees.csv"), index=False)

scored = Xte.copy()
scored.insert(0, "EmployeeNumber", idte)
scored["Gender"] = Ste["Gender"].values
scored["score"] = pte
scored["actual_left"] = yte
scored.to_csv(os.path.join(OUT, "scored_test_employees.csv"), index=False)
shap_df.to_csv(os.path.join(OUT, "shap_test_contributions.csv"), index=False)
imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
fair.to_csv(os.path.join(OUT, "fairness_audit.csv"), index=False)
pd.DataFrame(sweep).to_csv(os.path.join(OUT, "threshold_sweep_test.csv"), index=False)
pd.DataFrame({"precision": prc_p, "recall": prc_r}).to_csv(os.path.join(OUT, "pr_curve_test.csv"), index=False)
json.dump({
    "model": "Logistic regression, class weights", "C": gs.best_params_["model__C"],
    "cv_pr_auc": gs.best_score_, "threshold": thr, "base_log_odds": base_logodds,
    "test": m_test, "test_at_0.5": m_test_05, "oof": m_oof,
    "n_train": int(len(ytr)), "n_test": int(len(yte)), "test_leavers": int(yte.sum()),
    "no_skill_pr_auc": float(yte.mean()), "dataset": "IBM HR Analytics Employee Attrition & Performance (fictional)",
}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
print("Artefacts saved to", OUT)
