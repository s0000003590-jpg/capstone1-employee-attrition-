"""shap_stability.py - is the SHAP ranking of the selected model stable across data splits? (report Section 4.6)

Usage:  python shap_stability.py ../WA_Fn-UseC_-HR-Employee-Attrition.csv
Needs:  pandas, numpy, scikit-learn, scipy (no SHAP library needed: for logistic regression the
        interventional SHAP value is coefficient x (value - training mean), summed over one-hot columns).
Time:   under a minute.

Steps:
  1. Main split (seed 42) and the 20 repeated splits used in Section 4.5 (seeds 100-119).
  2. On each split, fit the selected pipeline (scale -> one-hot -> class-weighted logistic regression,
     C = 0.01, the hyperparameter chosen by cross-validation) on the training part only.
  3. Compute mean |SHAP| per original feature on that split's test part (background = its training part).
  4. Compare each split's ranking with the main split: Spearman rank correlation over all 28 features,
     overlap of the top 5 and top 10, and the rank of OverTime.
Writes shap_stability_results.csv (one row per split) and prints a summary.
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

CSV = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-HR-Employee-Attrition.csv"
raw = pd.read_csv(CSV, encoding="utf-8-sig")
df = raw.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])
y = (df.pop("Attrition") == "Yes").astype(int).values
X = df.drop(columns=["Gender", "MaritalStatus"])
CAT = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
X[CAT] = X[CAT].astype(object)
NUM = [c for c in X.columns if c not in CAT]

def pipeline():
    scale = ColumnTransformer([("num", StandardScaler(), NUM)], remainder="passthrough",
                              verbose_feature_names_out=False).set_output(transform="pandas")
    enc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
                            remainder="passthrough")
    return Pipeline([("scale", scale), ("ohe", enc),
                     ("model", LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"))])

def mean_abs_shap(seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    est = pipeline().fit(Xtr, ytr)
    pre = Pipeline(est.steps[:-1])
    Ztr = np.asarray(pre.transform(Xtr), float); Zte = np.asarray(pre.transform(Xte), float)
    coef = est.named_steps["model"].coef_[0]
    contrib = (Zte - Ztr.mean(axis=0)) * coef
    names = [n.split("__")[-1] for n in est.named_steps["ohe"].get_feature_names_out()]
    src = [next((c for c in CAT if n.startswith(c + "_")), n) for n in names]
    per_feature = pd.DataFrame(contrib, columns=names).T.groupby(src).sum().T.abs()   # keeps its own column labels
    return per_feature.mean()[list(X.columns)]

main = mean_abs_shap(42)
main_rank = main.rank(ascending=False)
top5, top10 = set(main.nlargest(5).index), set(main.nlargest(10).index)
rows, ranks = [], {}
for s in range(100, 120):
    m = mean_abs_shap(s)
    r = m.rank(ascending=False); ranks[s] = r
    rows.append({"seed": s, "spearman_vs_main": spearmanr(main, m).correlation,
                 "top5_overlap": len(top5 & set(m.nlargest(5).index)),
                 "top10_overlap": len(top10 & set(m.nlargest(10).index)),
                 "overtime_rank": int(r["OverTime"]), "top_feature": m.idxmax(),
                 "overtime_mean_abs_shap": m["OverTime"]})
res = pd.DataFrame(rows)
res.to_csv("shap_stability_results.csv", index=False)
R = pd.DataFrame(ranks)
print("Main split (seed 42) top 5:", ", ".join(main.nlargest(5).index))
print(f"Spearman rank correlation with the main split (20 splits): mean {res.spearman_vs_main.mean():.2f}, "
      f"min {res.spearman_vs_main.min():.2f}, max {res.spearman_vs_main.max():.2f}")
print(f"Top-5 overlap: mean {res.top5_overlap.mean():.1f} of 5 (min {res.top5_overlap.min()}); "
      f"top-10 overlap: mean {res.top10_overlap.mean():.1f} of 10 (min {res.top10_overlap.min()})")
print(f"OverTime ranked first in {(res.overtime_rank == 1).sum()} of 20 splits (worst rank {res.overtime_rank.max()}); "
      f"mean |SHAP| {res.overtime_mean_abs_shap.min():.3f}-{res.overtime_mean_abs_shap.max():.3f}")
print("Median rank over the 20 splits of the main top 5:",
      {f: int(R.loc[f].median()) for f in main.nlargest(5).index})
print("Saved shap_stability_results.csv")
