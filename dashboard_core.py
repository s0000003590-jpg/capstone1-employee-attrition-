"""dashboard_core.py - data logic and charts for the HR attrition dashboard.

Kept separate from app.py so every calculation can be checked without starting Streamlit
(see test_dashboard_core.py). Nothing here trains a model: it only reads the artefacts
written by build_artifacts.py and uses the saved pipeline to score and explain employees.
"""
import os, json
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, CORAL, AMBER, GREY = "#1F3B63", "#2A9D8F", "#E76F51", "#E09A2B", "#9AA3B2"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#5B6472", "axes.titleweight": "bold", "axes.titlesize": 10})

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
AGE_BANDS = ["Under 30", "30 to 45", "Over 45"]


# ---------------------------------------------------------------- loading
def load_artifacts(folder=ART):
    """Read every artefact the dashboard needs. Raises FileNotFoundError with a clear message if missing."""
    need = ["attrition_pipeline.joblib", "metrics.json", "employees.csv", "scored_test_employees.csv",
            "feature_importance.csv", "fairness_audit.csv", "threshold_sweep_test.csv", "pr_curve_test.csv"]
    missing = [f for f in need if not os.path.exists(os.path.join(folder, f))]
    if missing:
        raise FileNotFoundError("Missing artefacts: " + ", ".join(missing) +
                                ". Run: python build_artifacts.py path/to/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    A = {"bundle": joblib.load(os.path.join(folder, "attrition_pipeline.joblib")),
         "metrics": json.load(open(os.path.join(folder, "metrics.json")))}
    for key, f in [("employees", "employees.csv"), ("test", "scored_test_employees.csv"),
                   ("importance", "feature_importance.csv"), ("fairness", "fairness_audit.csv"),
                   ("sweep", "threshold_sweep_test.csv"), ("prc", "pr_curve_test.csv")]:
        A[key] = pd.read_csv(os.path.join(folder, f))
    A["employees"]["AgeBand"] = age_band(A["employees"]["Age"])
    return A


def age_band(age):
    age = np.asarray(age)
    return np.where(age < 30, "Under 30", np.where(age <= 45, "30 to 45", "Over 45"))


# ---------------------------------------------------------------- overview
def filter_employees(df, departments=None, roles=None, overtime=None, tenure=None, age_bands=None):
    m = pd.Series(True, index=df.index)
    if departments: m &= df["Department"].isin(departments)
    if roles: m &= df["JobRole"].isin(roles)
    if overtime: m &= df["OverTime"].isin(overtime)
    if tenure: m &= df["YearsAtCompany"].between(tenure[0], tenure[1])
    if age_bands: m &= df["AgeBand"].isin(age_bands)
    return df[m]


def headline(df):
    n = len(df); left = int(df["Left"].sum())
    return {"employees": n, "leavers": left, "rate": left / n if n else float("nan")}


def rates_by(df, col, min_n=1):
    g = df.groupby(col)["Left"].agg(["mean", "size", "sum"]).rename(columns={"mean": "rate", "size": "n", "sum": "leavers"})
    return g[g["n"] >= min_n].sort_values("rate")


# ---------------------------------------------------------------- scoring and explanations
def score(bundle, X):
    """Score employees with the saved pipeline (no retraining)."""
    return bundle["pipeline"].predict_proba(X[bundle["features"]])[:, 1]


def explain(bundle, X):
    """Per-employee contributions (log-odds) for the logistic regression, summed back to the original features.
    Same definition as shap.LinearExplainer with the training set as background: coef * (x - mean)."""
    pipe = bundle["pipeline"]
    pre_steps = pipe.steps[:-1]
    Z = X[bundle["features"]]
    for _, step in pre_steps:
        Z = step.transform(Z)
    Z = np.asarray(Z, dtype=float)
    lr = pipe.steps[-1][1]
    contrib = (Z - bundle["shap_background_mean"]) * lr.coef_[0]
    names = [n.split("__")[-1] for n in pipe.named_steps["ohe"].get_feature_names_out()]
    src = []
    for n in names:
        src.append(next((c for c in bundle["categorical"] if n.startswith(c + "_")), n))
    out = pd.DataFrame(contrib, columns=names, index=X.index).T.groupby(src).sum().T
    return out[bundle["features"]]


def risk_tier(scores, thr):
    """Display tiers: High = flagged (score >= threshold); Medium = within 0.20 below it; Low = the rest."""
    s = np.asarray(scores)
    return np.where(s >= thr, "High", np.where(s >= thr - 0.20, "Medium", "Low"))


def ranked_list(test_df, scores, thr, departments=None, tiers=None):
    d = test_df.copy()
    d["Score"] = scores
    d["Risk tier"] = risk_tier(scores, thr)
    if departments: d = d[d["Department"].isin(departments)]
    if tiers: d = d[d["Risk tier"].isin(tiers)]
    return d.sort_values("Score", ascending=False)


def plain_value(row, feat):
    v = row[feat]
    return f"{v:,.0f}" if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer() else str(v)


# ---------------------------------------------------------------- charts (matplotlib figures)
def _fig(w=5.2, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    return fig, ax


def bar_rates(rates, title, overall=None):
    fig, ax = _fig(5.2, 0.32 * len(rates) + 0.9)
    ax.barh(rates.index.astype(str), rates["rate"] * 100, color=TEAL)
    for i, (r, n) in enumerate(zip(rates["rate"], rates["n"])):
        ax.text(r * 100 + 0.5, i, f"{r:.0%}  (n={n})", va="center", fontsize=8)
    if overall is not None:
        ax.axvline(overall * 100, color=NAVY, ls="--", lw=0.9, alpha=0.5, zorder=0)
    ax.set_xlabel("attrition rate (%)" + (f"   (dashed line = overall {overall:.0%})" if overall is not None else ""))
    ax.set_title(title, loc="left")
    ax.set_xlim(0, max(rates["rate"].max() * 100 * 1.35, 5))
    fig.tight_layout(); return fig


def distribution(df, col, title, bins=20):
    fig, ax = _fig()
    stay, left = df[df["Left"] == 0][col], df[df["Left"] == 1][col]
    rng = (df[col].min(), df[col].max())
    ax.hist(stay, bins=bins, range=rng, density=True, alpha=0.55, color=GREY, label=f"Stayed (n={len(stay)})")
    ax.hist(left, bins=bins, range=rng, density=True, alpha=0.65, color=CORAL, label=f"Left (n={len(left)})")
    ax.set_title(title, loc="left"); ax.set_yticks([]); ax.set_xlabel(col); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); return fig


def reasons_chart(contrib_row, row, k=8):
    s = contrib_row.reindex(contrib_row.abs().sort_values(ascending=False).index)[:k][::-1]
    fig, ax = _fig(5.2, 0.34 * k + 0.8)
    ax.barh([f"{f} = {plain_value(row, f)}" for f in s.index], s.values, color=[CORAL if v > 0 else "#4A76A8" for v in s.values])
    ax.axvline(0, color="#5B6472", lw=0.8)
    ax.set_xlabel("log-odds contribution  (← stay | leave →)")
    ax.set_title("Reasons behind this employee's score", loc="left")
    fig.tight_layout(); return fig


def drivers_chart(imp, k=10):
    d = imp.sort_values("perm_importance", ascending=False).head(k)[::-1]
    fig, ax = _fig(5.2, 0.3 * k + 1.5)
    y = np.arange(len(d)); h = 0.38
    ax.barh(y + h / 2, d["perm_importance"], h, color=NAVY, label="Permutation importance (drop in test PR-AUC)")
    ax.barh(y - h / 2, d["mean_abs_shap"] / d["mean_abs_shap"].max() * d["perm_importance"].max(), h, color=TEAL,
            label="Mean |SHAP| (rescaled)")
    ax.set_yticks(y, d["feature"]); ax.set_xlabel("drop in test PR-AUC")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=1)
    ax.set_title("Global drivers: what the model relies on", loc="left")
    fig.tight_layout(); return fig


def confusion_chart(m):
    fig, ax = _fig(3.6, 3.0)
    cells = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    labels = np.array([["TP\nleavers flagged", "FN\nleavers missed"], ["FP\nfalse alarms", "TN\nstayers not flagged"]])
    colors = np.array([[TEAL, "#F6D5CC"], ["#F6D5CC", "#E8EAF5"]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.97, 0.97, color=colors[i, j]))
            ax.text(j + 0.485, 1 - i + 0.6, f"{cells[i, j]:,}", ha="center", va="center", fontsize=15, fontweight="bold",
                    color="white" if (i, j) == (0, 0) else NAVY)
            ax.text(j + 0.485, 1 - i + 0.27, labels[i, j], ha="center", va="center", fontsize=7.5,
                    color="white" if (i, j) == (0, 0) else "#1F2937")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_xticks([0.485, 1.485], ["Flagged", "Not flagged"])
    ax.set_yticks([1.485, 0.485], ["Left", "Stayed"]); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Confusion matrix (test set)", loc="left")
    fig.tight_layout(); return fig


def pr_chart(prc, no_skill, ap, thr_point=None):
    fig, ax = _fig(4.4, 3.0)
    ax.plot(prc["recall"], prc["precision"], color=NAVY, lw=1.6, label=f"Model (PR-AUC {ap:.2f})")
    ax.axhline(no_skill, color=GREY, ls="--", lw=1, label=f"No skill ({no_skill:.2f})")
    if thr_point is not None:
        ax.scatter([thr_point[0]], [thr_point[1]], color=CORAL, zorder=3, label="Chosen threshold")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8); ax.set_title("Precision-recall curve (test set)", loc="left")
    fig.tight_layout(); return fig


def sweep_chart(sweep, thr):
    fig, ax = _fig(5.2, 3.1)
    ax.plot(sweep["threshold"], sweep["recall"], color=TEAL, label="recall")
    ax.plot(sweep["threshold"], sweep["precision"].where(sweep["flagged"] > 0), color=NAVY, label="precision")
    ax.plot(sweep["threshold"], sweep["fpr"], color=CORAL, label="false-positive rate")
    ax.axvline(thr, color="#5B6472", ls="--", lw=1)
    ax.text(thr + 0.01, 0.95, f"selected {thr:.2f}", fontsize=8)
    ax.set_xlabel("threshold"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title("Effect of the threshold (test set, what-if)", loc="left")
    fig.tight_layout(); return fig


def fairness_chart(fair):
    d = fair.copy()
    groups = ["Female", "Male", "Under 30", "30 to 45", "Over 45"]
    fig, ax = _fig(5.4, 3.0)
    x = np.arange(len(groups)); w = 0.38
    for k, (part, col) in enumerate([("Training (out-of-fold)", NAVY), ("Test", TEAL)]):
        s = d[d["partition"] == part].set_index("group").loc[groups]
        err = np.vstack([s["recall"] - s["recall_ci_low"], s["recall_ci_high"] - s["recall"]])
        ax.bar(x + (k - 0.5) * w, s["recall"], w, color=col, label=part, yerr=err, capsize=3,
               error_kw={"elinewidth": 0.8, "ecolor": "#5B6472"})
    ax.set_xticks(x, groups); ax.set_ylim(0, 1.05); ax.set_ylabel("recall (share of leavers caught)")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Recall by group with 95% intervals", loc="left")
    fig.tight_layout(); return fig
