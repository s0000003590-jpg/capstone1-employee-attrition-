"""test_dashboard_core.py - checks that the dashboard shows the same numbers as the report.

Run:  python test_dashboard_core.py      (after build_artifacts.py)
Every check prints PASS or stops with an error explaining what differs.
"""
import numpy as np
import dashboard_core as core

A = core.load_artifacts()
B, M = A["bundle"], A["metrics"]
X = A["test"].copy()
for c in B["categorical"]:
    X[c] = X[c].astype(object)

def check(name, cond):
    assert cond, "FAIL: " + name
    print("PASS:", name)

# 1. the dashboard scores employees with the saved pipeline and gets the saved test scores
s = core.score(B, X)
check("scores from the saved pipeline match the offline scores", np.allclose(s, X["score"].values, atol=1e-9))

# 2. headline test results match the Intermediate Report (Section 4.5)
t = M["test"]
check("threshold 0.60 (0.604 before rounding)", round(M["threshold"], 2) == 0.60)
check("CV PR-AUC 0.627", round(M["cv_pr_auc"], 3) == 0.627)
check("confusion matrix TP 26, FN 21, FP 25, TN 222", (t["tp"], t["fn"], t["fp"], t["tn"]) == (26, 21, 25, 222))
check("test PR-AUC 0.58, recall 0.55, precision 0.51, F1 0.53",
      (round(t["pr_auc"], 2), round(t["recall"], 2), round(t["precision"], 2), round(t["f1"], 2)) == (0.58, 0.55, 0.51, 0.53))

# 3. explanations add up to each employee's score (log-odds) from the baseline
contrib = core.explain(B, X)
logit = np.log(s / (1 - s))
check("explanations + baseline = score for every test employee",
      np.allclose(contrib.sum(axis=1).values + B["base_log_odds"], logit, atol=1e-6))
check("SHAP baseline log-odds -0.42 (report Section 4.6)", round(B["base_log_odds"], 2) == -0.42)
top = A["importance"].iloc[0]
check("OverTime is the top driver (permutation 0.189, SHAP 0.269)",
      top["feature"] == "OverTime" and round(top["perm_importance"], 3) == 0.189 and round(top["mean_abs_shap"], 3) == 0.269)

# 4. fairness table matches Tables 30-31
f = A["fairness"].set_index(["partition", "group"])
check("over-45 recall 24% (6/25) out-of-fold and 11.1% (1/9) on test",
      round(f.loc[("Training (out-of-fold)", "Over 45"), "recall"], 3) == 0.24 and
      round(f.loc[("Test", "Over 45"), "recall"], 3) == 0.111)

# 5. overview numbers and filters
h = core.headline(A["employees"])
check("overview: 1,470 employees, 237 leavers, 16.1%", (h["employees"], h["leavers"], round(h["rate"], 3)) == (1470, 237, 0.161))
ot = core.rates_by(A["employees"], "OverTime")
check("overtime raises the attrition rate", ot.loc["Yes", "rate"] > ot.loc["No", "rate"])
d = core.filter_employees(A["employees"], departments=["Sales"], overtime=["Yes"])
check("filters return only matching employees", set(d["Department"]) == {"Sales"} and set(d["OverTime"]) == {"Yes"})

# 6. risk tiers follow the threshold
tiers = core.risk_tier(s, M["threshold"])
check("High tier = exactly the flagged employees", ((tiers == "High") == (s >= M["threshold"])).all())
print("\nAll checks passed.")
