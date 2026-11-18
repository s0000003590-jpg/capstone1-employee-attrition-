"""experiment_A2_smotenc.py - controlled Experiment A2 with SMOTE-NC (5 seeds)
Usage (terminal):  python experiment_A2_smotenc.py path/to/WA_Fn-UseC_-HR-Employee-Attrition.csv
Usage (Spyder):    set DATA_PATH below, then press F5.
Needs: pip install scikit-learn imbalanced-learn pandas numpy
Arms (same SMOTE-NC settings, same random forest, same folds):
  1. SMOTE-NC before CV, scored on resampled data (original + synthetic)
  2. SMOTE-NC before CV, scored on original records only
  3. SMOTE-NC inside each training fold, scored on original records
  4. No resampling, scored on original records
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = r"C:\Users\ragha\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv"   # used when no path is given
if len(sys.argv) > 1:
    DATA_PATH = sys.argv[1]
SEEDS = [42, 43, 44, 45, 46]   # five seeds, as in the report (Table 23)

raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = raw.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])
y = (df.pop("Attrition") == "Yes").astype(int).values
X = df.drop(columns=["Gender", "MaritalStatus"]).reset_index(drop=True)
CAT = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
NUM = [c for c in X.columns if c not in CAT]
X[CAT] = X[CAT].astype(object)
cols = NUM + CAT                       # scaled numeric first, categorical after (passthrough)
cat_idx = list(range(len(NUM), len(cols)))

def scaler():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", "passthrough", CAT)])
def encoder():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                               list(range(len(NUM), len(cols))))], remainder="passthrough")
def forest(s):
    return RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
def metrics(yt, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(precision=precision_score(yt, yh, zero_division=0), recall=recall_score(yt, yh),
                f1=f1_score(yt, yh), roc_auc=roc_auc_score(yt, p), pr_auc=average_precision_score(yt, p))

rows = []
for s in SEEDS:
    cv = StratifiedKFold(5, shuffle=True, random_state=s)
    # Arm 1/2: SMOTE-NC on the whole dataset first (leaky), original rows come first in the output
    Xs = pd.DataFrame(scaler().fit_transform(X), columns=cols)
    Xr, yr = SMOTENC(categorical_features=cat_idx, random_state=s).fit_resample(Xs, y)
    Xr = pd.DataFrame(Xr, columns=cols)
    Xr[NUM] = Xr[NUM].astype(float)
    p_leak = cross_val_predict(Pipeline([("ohe", encoder()), ("rf", forest(s))]), Xr, yr, cv=cv,
                               method="predict_proba")[:, 1]
    orig = np.arange(len(yr)) < len(y)
    rows.append(("1 SMOTE-NC before CV, scored on resampled data", s, metrics(yr, p_leak)))
    rows.append(("2 SMOTE-NC before CV, scored on original records", s, metrics(y, p_leak[orig])))
    # Arm 3: SMOTE-NC inside each training fold
    inside = ImbPipeline([("scale", scaler()), ("smote", SMOTENC(categorical_features=cat_idx, random_state=s)),
                          ("ohe", encoder()), ("rf", forest(s))])
    p_in = cross_val_predict(inside, X, y, cv=cv, method="predict_proba")[:, 1]
    rows.append(("3 SMOTE-NC inside each training fold, original records", s, metrics(y, p_in)))
    # Arm 4: no resampling
    plain = Pipeline([("scale", scaler()), ("ohe", encoder()), ("rf", forest(s))])
    p_no = cross_val_predict(plain, X, y, cv=cv, method="predict_proba")[:, 1]
    rows.append(("4 No resampling, original records", s, metrics(y, p_no)))
    print("seed", s, "done", flush=True)

res = pd.DataFrame([dict(arm=a, seed=s, **m) for a, s, m in rows])
print("\n=== Seed 42 (single run, comparable to Table 23) ===")
print(res[res.seed == 42].drop(columns="seed").set_index("arm").round(3).to_string())
g = res.groupby("arm")[["precision", "recall", "f1", "roc_auc", "pr_auc"]]
print("\n=== Mean over seeds", SEEDS, "===")
print(g.mean().round(3).to_string())
print("\n=== SD over seeds ===")
print(g.std().round(3).to_string())
print("\n=== Min / max F1 ===")
print(res.groupby("arm")["f1"].agg(["min", "max"]).round(3).to_string())
res.to_csv("experiment_A2_smotenc_results.csv", index=False)
