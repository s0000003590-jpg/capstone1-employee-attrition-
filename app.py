"""app.py - HR Attrition Analytics dashboard (Capstone 1 proof of concept).

Start it with:   streamlit run app.py
First run build_artifacts.py once; the dashboard only reads the saved artefacts and never retrains.
Pages follow the report's wireframes (Figures 9-11): Overview, Risk list, Model, Fairness.
"""
import numpy as np, pandas as pd
import streamlit as st
import dashboard_core as core

st.set_page_config(page_title="HR Attrition Analytics", page_icon="\U0001F4CA", layout="wide")
NOTICE = ("**Decision support, not decisions.** Scores rank employees for review by a person; explanations show what "
          "the model relies on, not why an employee leaves. Group results rest on small samples. "
          "Data: IBM HR Analytics (fictional).")


@st.cache_resource
def get_artifacts():
    return core.load_artifacts()


try:
    A = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

B, M = A["bundle"], A["metrics"]
THR = float(B["threshold"])
emp = A["employees"]

st.sidebar.title("HR Attrition Analytics")
page = st.sidebar.radio("Navigation", ["Overview", "Risk list", "Model", "Fairness"])
st.sidebar.caption("Capstone 1 proof of concept · University of Dubai")


def notice():
    st.divider()
    st.warning(NOTICE)


# ================================================================ Overview
if page == "Overview":
    st.title("Overview and attrition patterns")
    st.sidebar.subheader("Filters")
    deps = st.sidebar.multiselect("Department", sorted(emp["Department"].unique()))
    roles = st.sidebar.multiselect("Job role", sorted(emp["JobRole"].unique()))
    ot = st.sidebar.multiselect("Overtime", ["Yes", "No"])
    lo, hi = int(emp["YearsAtCompany"].min()), int(emp["YearsAtCompany"].max())
    ten = st.sidebar.slider("Tenure (years at company)", lo, hi, (lo, hi))
    ages = st.sidebar.multiselect("Age band", core.AGE_BANDS)
    d = core.filter_employees(emp, deps, roles, ot, ten, ages)

    if len(d) == 0:
        st.info("No employees match these filters."); notice(); st.stop()
    h = core.headline(d)
    c1, c2, c3 = st.columns(3)
    c1.metric("Employees", f"{h['employees']:,}")
    c2.metric("Leavers", f"{h['leavers']:,}")
    c3.metric("Attrition rate", f"{h['rate']:.1%}",
              delta=f"{(h['rate'] - emp['Left'].mean()) * 100:+.1f} pts vs all" if len(d) < len(emp) else None,
              delta_color="inverse")
    if len(d) < 30:
        st.caption(f"Only {len(d)} employees match: rates below are unstable.")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(core.bar_rates(core.rates_by(d, "Department"), "Attrition by department", h["rate"]))
        st.pyplot(core.bar_rates(core.rates_by(d, "JobRole"), "Attrition by job role", h["rate"]))
    with c2:
        st.pyplot(core.bar_rates(core.rates_by(d, "OverTime"), "Attrition by overtime", h["rate"]))
        st.pyplot(core.bar_rates(core.rates_by(d, "BusinessTravel"), "Attrition by business travel", h["rate"]))
        st.pyplot(core.bar_rates(core.rates_by(d, "JobSatisfaction"), "Attrition by job satisfaction (1 = low)", h["rate"]))
    c1, c2 = st.columns(2)
    c1.pyplot(core.distribution(d, "Age", "Age: leavers vs stayers"))
    c2.pyplot(core.distribution(d, "MonthlyIncome", "Monthly income: leavers vs stayers"))
    st.caption("Descriptive patterns in all 1,470 records; they are associations, not causes.")
    notice()

# ================================================================ Risk list
elif page == "Risk list":
    st.title("Risk list and explanations")
    st.sidebar.subheader("Decision threshold")
    thr = st.sidebar.slider("Flag employees with a score of at least", 0.05, 0.95, round(THR, 2), 0.01)
    st.sidebar.caption(f"Default {THR:.2f} was chosen on training data (maximum out-of-fold F1).")
    deps = st.sidebar.multiselect("Department", sorted(emp["Department"].unique()))
    tiers = st.sidebar.multiselect("Risk tier", ["High", "Medium", "Low"])

    st.sidebar.subheader("Score another file (optional)")
    up = st.sidebar.file_uploader("CSV with the IBM columns", type="csv")
    if up is not None:
        X = pd.read_csv(up, encoding="utf-8-sig")
        missing = [f for f in B["features"] if f not in X.columns]
        if missing:
            st.error("The file is missing columns: " + ", ".join(missing)); st.stop()
        if "EmployeeNumber" not in X.columns:
            X.insert(0, "EmployeeNumber", np.arange(1, len(X) + 1))
        source = f"uploaded file ({len(X)} employees)"
    else:
        X = A["test"].copy()
        source = f"held-out test employees ({len(X)}), not used to train the model"
    X = X.reset_index(drop=True)
    for c in B["categorical"]:
        X[c] = X[c].astype(object)
    scores = core.score(B, X)
    contrib = core.explain(B, X)

    ranked = core.ranked_list(X, scores, thr, deps, tiers)
    n_flag = int((scores >= thr).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Employees scored", f"{len(X):,}")
    c2.metric(f"Flagged at {thr:.2f}", f"{n_flag:,}")
    c3.metric("Share flagged", f"{n_flag / len(X):.1%}")
    st.caption(f"Scoring {source} with the saved pipeline. Scores are ranking scores, not calibrated probabilities.")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Ranked employees")
        show = ranked[["EmployeeNumber", "Score", "Risk tier", "Department", "JobRole", "OverTime", "YearsAtCompany"]].copy()
        show["Score"] = show["Score"].round(2)
        st.dataframe(show, hide_index=True, height=420)
        st.download_button("Download this list (CSV)", show.to_csv(index=False).encode(), "risk_list.csv", "text/csv")
    with right:
        st.subheader("Reasons (selected employee)")
        if len(ranked) == 0:
            st.info("No employees match these filters.")
        else:
            pick = st.selectbox("Employee", ranked["EmployeeNumber"].tolist(),
                                format_func=lambda e: f"ID {e} — score {ranked.set_index('EmployeeNumber').loc[e, 'Score']:.2f}")
            i = X.index[X["EmployeeNumber"] == pick][0]
            st.pyplot(core.reasons_chart(contrib.loc[i], X.loc[i]))
            st.caption(f"Bars start from the average training score (log-odds {B['base_log_odds']:.2f}) and add up to this "
                       "employee's score. Orange pushes towards leave, blue towards stay.")
    c1, c2 = st.columns([1.05, 1])
    c1.pyplot(core.drivers_chart(A["importance"]))
    with c2:
        st.subheader("How to read this")
        st.info("Bars show what the model relies on. They do not show what would make an employee stay: "
                "changing a factor (for example, pay) would not necessarily change the outcome. "
                "Review each case with a person before acting.")
    notice()

# ================================================================ Model
elif page == "Model":
    st.title("Model performance")
    t = M["test"]
    st.caption(f"{M['model']} (C = {M['C']}), selected by 5-fold cross-validated PR-AUC {M['cv_pr_auc']:.3f} on "
               f"{M['n_train']:,} training employees. Evaluated once on {M['n_test']} held-out employees "
               f"({M['test_leavers']} leavers) at threshold {M['threshold']:.2f}.")
    c = st.columns(4)
    c[0].metric("PR-AUC", f"{t['pr_auc']:.2f}", help=f"No-skill baseline = leaver rate {M['no_skill_pr_auc']:.2f}")
    c[1].metric("Recall", f"{t['recall']:.2f}", help="Share of real leavers the model flags")
    c[2].metric("Precision", f"{t['precision']:.2f}", help="Share of flagged employees who really left")
    c[3].metric("F1", f"{t['f1']:.2f}")
    c1, c2 = st.columns(2)
    c1.pyplot(core.confusion_chart(t))
    c2.pyplot(core.pr_chart(A["prc"], M["no_skill_pr_auc"], t["pr_auc"], (t["recall"], t["precision"])))
    c1, c2 = st.columns(2)
    c1.pyplot(core.sweep_chart(A["sweep"], M["threshold"]))
    with c2:
        st.subheader("Reading the results")
        t5 = M["test_at_0.5"]
        st.markdown(
            f"- **{t['tp']} of {t['tp'] + t['fn']} leavers** were flagged; **{t['fp']} false alarms** among "
            f"{t['fp'] + t['tn']} stayers.\n"
            f"- At the default 0.50 threshold the model would catch more leavers (recall {t5['recall']:.2f}) "
            f"but raise more false alarms (precision {t5['precision']:.2f}).\n"
            f"- Accuracy ({t['accuracy']:.2f}) is not used: always predicting “stay” scores "
            f"{1 - M['no_skill_pr_auc']:.2f}.\n"
            "- The what-if chart uses test data and is for illustration; the threshold itself comes from training data.")
    notice()

# ================================================================ Fairness
else:
    st.title("Fairness audit by gender and age")
    st.caption(f"Recall, precision and false-positive rate per group at threshold {THR:.2f}. Gender was not a model input; "
               "age was a model input and is also audited.")
    part = st.radio("Partition", ["Training (out-of-fold)", "Test"], horizontal=True)
    f = A["fairness"][A["fairness"]["partition"] == part].copy()
    tbl = pd.DataFrame({
        "Group": f["group"], "n": f["n"], "Leavers": f["leavers"],
        "Recall": f["recall"].map("{:.0%}".format),
        "Recall 95% interval": [f"{a:.0%}–{b:.0%}" for a, b in zip(f["recall_ci_low"], f["recall_ci_high"])],
        "Precision": f["precision"].map("{:.0%}".format), "FPR": f["fpr"].map("{:.1%}".format),
        "Flagged": f["flagged"].map("{:.0%}".format)})
    st.dataframe(tbl, hide_index=True)
    small = f[f["leavers"] < 30]["group"].tolist()
    if small:
        st.error("Small groups: " + ", ".join(small) + " have fewer than 30 leavers in this partition, so their recall "
                 "is uncertain (see the wide intervals). Treat gaps as a reason for caution, not proof of bias.")
    c1, c2 = st.columns([1.2, 1])
    c1.pyplot(core.fairness_chart(A["fairness"]))
    with c2:
        st.subheader("What this shows")
        oo = A["fairness"].set_index(["partition", "group"])
        st.markdown(
            f"- Over-45 leavers are flagged least often: {oo.loc[('Training (out-of-fold)', 'Over 45'), 'recall']:.0%} "
            f"of {int(oo.loc[('Training (out-of-fold)', 'Over 45'), 'leavers'])} in training and "
            f"{oo.loc[('Test', 'Over 45'), 'recall']:.0%} of {int(oo.loc[('Test', 'Over 45'), 'leavers'])} in the test set.\n"
            "- Under-30 employees have the highest recall but also the most false alarms.\n"
            "- Differences may reflect base rates and correlated features (job level, working years); they have not been tested further.")
    notice()
