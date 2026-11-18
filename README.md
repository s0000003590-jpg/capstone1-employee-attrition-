# Employee Attrition Prediction and HR Analytics Dashboard — project code

Capstone 1 · University of Dubai · Mahra Alhammadi, Raghd Ghanim, Eman Aboraya · Supervisor: Dr. Hamzah Alkhazaleh

Every team member should be able to run each script below and explain its main steps. This guide covers both.

---

## 1. Folder contents

| File | What it is | Report section |
|---|---|---|
| `attrition_leakage_free_kaggle_corrected.ipynb` (your notebook) | Main IBM analysis: Experiments A and B, model comparison, test evaluation, fairness, SHAP, repeated splits | 3.3–3.9, 4.2–4.7 |
| `experiments/experiment_A2_smotenc.py` | Controlled Experiment A2 (SMOTE-NC before vs inside CV, 5 seeds) — **fixed copy**, see §4 | 4.3 |
| `experiments/shap_stability.py` | SHAP stability check: ranking of the selected model over the 20 repeated splits | 4.6 |
| `second_benchmark/second_benchmark_pipeline.py` | Same pipeline on the Kaggle HR Analytics data (HR_comma_sep.csv), now with **imbalanced-learn** | 3.2, 4.8 |
| `dashboard/build_artifacts.py` | Offline step: rebuilds the selected IBM pipeline and saves the files the dashboard reads | 3.10 |
| `dashboard/dashboard_core.py` | Dashboard calculations and charts (no Streamlit inside, so they can be tested) | 3.10 |
| `dashboard/app.py` | The Streamlit dashboard: Overview, Risk list, Model, Fairness | 3.10 |
| `dashboard/test_dashboard_core.py` | 13 checks that the dashboard shows the same numbers as the report | 3.10.4 |
| `requirements.txt` | Libraries to install | Appendix B |

Data files needed (both are public, not stored in this repository): download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from Kaggle ("IBM HR Analytics Employee Attrition & Performance") and `HR_comma_sep.csv` ("HR Analytics", giripujar), and put both in this main folder.

---

## 2. Set up once

```bash
pip install -r requirements.txt
```
(Anaconda users can run this in the Anaconda Prompt. Python 3.10 or newer.)

---

## 3. Run order and what you should see

| Step | Command (run inside the folder shown) | Time | Check that you get |
|---|---|---|---|
| 1 | `cd dashboard` → `python build_artifacts.py ../WA_Fn-UseC_-HR-Employee-Attrition.csv` | ~10 s | best C 0.01, CV PR-AUC 0.627, threshold 0.604, test TP 26 / FP 25 / FN 21 / TN 222 |
| 2 | `python test_dashboard_core.py` | ~5 s | `All checks passed.` |
| 3 | `streamlit run app.py` | opens a browser tab | four pages; stop with Ctrl+C |
| 4 | `cd ../second_benchmark` → `python second_benchmark_pipeline.py ../HR_comma_sep.csv second_benchmark_results.json` | ~10 min | first line shows the imbalanced-learn version; ends with `done` |
| 5 | `cd ../experiments` → `python experiment_A2_smotenc.py ../WA_Fn-UseC_-HR-Employee-Attrition.csv` | ~2 min | mean F1 0.636 (before CV, original records) vs 0.448 (inside folds) |
| 6 | `python shap_stability.py ../WA_Fn-UseC_-HR-Employee-Attrition.csv` | <1 min | OverTime ranked first in 20 of 20 splits; Spearman mean 0.89 |

Steps 4–6 write their results next to the script (`*_results.json` / `*_results.csv`); these generated files are not stored in the repository.

---

## 4. Explaining the main steps

### 4.1 The leakage-free pipeline (all scripts)
1. **Clean** — drop constant columns and the ID; keep Gender and MaritalStatus out of the model (used only to audit fairness).
2. **Split first** — stratified 80/20, seed 42. The test set is not touched until the final evaluation.
3. **One pipeline** — `scale numeric → SMOTE-NC (or class weights) → one-hot encode → model`. Because it is a single
   `imblearn` Pipeline, cross-validation refits every step on the training folds only, so the validation fold is never
   scaled, oversampled or encoded with information from itself. *This is what prevents leakage.*
4. **Select by CV PR-AUC** — 5-fold stratified grid search on the training set. PR-AUC is used because only 16% leave;
   accuracy is misleading (always "stay" = 84%).
5. **Threshold from training data** — the threshold that maximises F1 on out-of-fold training predictions (0.604 ≈ 0.60).
6. **Test once** — apply the chosen pipeline and threshold to the test set a single time.
7. **Check stability** — 20 repeated splits; report ranges, not one number.

### 4.2 `experiment_A2_smotenc.py` (H1)
Same data, same random forest, same SMOTE-NC settings, same folds, same seeds. Only *when* SMOTE-NC is applied changes:
before CV on the whole dataset (leaky) vs inside each training fold. Both are scored on the original employees.
Leaky F1 ≈ 0.63 vs 0.45 → oversampling before CV inflates scores (H1 supported).
**Fix made:** the original file used `SEEDS` without defining it (the script stopped at once) and ignored the CSV path
given on the command line. Both are fixed; nothing else changed.

### 4.3 `second_benchmark_pipeline.py` (Section 4.8)
1. Removes 3,008 duplicate rows first — otherwise an identical employee could sit in both training and test (31% of test rows would).
2. Runs the same `pipe()` and grid as the notebook for LR, RF and GB with SMOTE-NC or class weights / none.
3. Gradient boosting without resampling wins (CV PR-AUC ≈ 0.97); the test set is used once.
4. Extra checks: permutation importance (satisfaction_level matters most), ablation without satisfaction_level
   (PR-AUC stays ≈ 0.95, so the dataset is easy overall), duplicates-kept sensitivity, and the A2 leakage comparison.
5. Message: the pipeline transfers unchanged; high scores reflect an easier dataset, **not** a better model.
   The IBM model is not applied to this data because the features differ.

### 4.4 The dashboard (`build_artifacts.py` + `app.py`)
- **Architecture (report Figure 8):** `build_artifacts.py` runs offline and saves the fitted pipeline, threshold, metrics,
  importance, explanations and fairness table to `dashboard/artifacts/`. `app.py` only reads them — it never retrains,
  so it cannot change a reported result.
- **Overview page:** headline numbers and attrition rates by department, role, overtime, travel and satisfaction, with filters.
- **Risk list page:** scores the 294 held-out test employees with the saved pipeline (or a CSV you upload), ranks them,
  lets HR move the threshold, and explains one employee's score. Explanations are SHAP values for the logistic
  regression: `coefficient × (employee value − training average)`, which add up exactly to the score (checked by the tests).
- **Model page:** PR-AUC, recall, precision, F1, confusion matrix, precision–recall curve and a threshold what-if chart.
- **Fairness page:** recall, precision and false-positive rate by gender and age band with 95% intervals and a
  small-group warning (over-45: only 25 and 9 leavers).
- Every page says **"decision support, not decisions"** and that explanations are not causes.

### 4.5 Questions the examiner may ask
| Question | Short answer |
|---|---|
| Why not use accuracy? | 84% of employees stay, so predicting "stay" for everyone already scores 84%. PR-AUC focuses on finding leavers. |
| How do you know there is no leakage? | All preprocessing and SMOTE-NC are inside one pipeline that is refitted on each training fold; the test set is used once; duplicates are removed in the second dataset. |
| Why logistic regression on IBM but gradient boosting on HR_comma_sep? | The same selection rule (best CV PR-AUC) was applied to each dataset; the rule picks what suits each one. |
| Does a high SHAP value mean overtime causes leaving? | No. It shows what the model relies on. Changing it would not necessarily change the outcome. |
| Is the model biased against older employees? | Recall is lowest for over-45s, but on 25 and 9 leavers with wide intervals — a reason for caution, not proof of bias. |
| Why are the second-dataset results so high? | The data is easy: satisfaction, evaluation, projects and hours separate leavers almost perfectly. Without satisfaction_level, CV PR-AUC is still 0.947. It shows the pipeline transfers, not a better model. |
| Why gradient boosting there? | The same rule (best CV PR-AUC) picked it: the patterns are non-linear (e.g. leavers at both very low and very high hours), so logistic regression reaches only 0.38. |
| Why remove duplicates? What if kept? | 3,008 rows are exact copies. If kept, 934 of 3,000 test rows (31%) have a copy in training, so the model is partly tested on employees it has seen; test PR-AUC rises from 0.966 to 0.983 and F1 from 0.944 to 0.949. Removing them keeps the test honest. |
| How do you know the dashboard shows the report's results? | The app never trains; it reads files saved by the offline script, and 13 automated checks compare its numbers with the report (CV PR-AUC, threshold, confusion matrix, SHAP sums, fairness tables). |
| How would results change with real UAE data? | Unknown until tested: different features, base rate and culture (e.g. visas, nationality mix, end-of-service rules). The pipeline would be re-run and re-selected, the threshold re-chosen, fairness re-audited, with ethics approval. Capstone 2. |
| Why is the threshold 0.60 and not 0.50? | It maximises F1 on training predictions; at 0.50 recall rises (0.66) but precision falls (0.34). |

---

## 5. Uploading to GitHub (repository for the report)
1. Create a new **public** repository on github.com (e.g. `capstone1-employee-attrition`), without a README.
2. Easiest without Git: on the repository page choose **Add file → Upload files**, drag in everything in this folder
   **except** the CSV files and any `artifacts`/`results` folders, and commit.
3. Or with Git, inside this folder: `git init`, `git add .`, `git commit -m "Capstone 1 code"`,
   `git branch -M main`, `git remote add origin <repository URL>`, `git push -u origin main`
   (the `.gitignore` already keeps data and generated outputs out).
4. Send the repository link so it can be added to the report (Appendix A) and the slides.

## 6. Troubleshooting
- `ModuleNotFoundError: imblearn` → `pip install imbalanced-learn`.
- Dashboard says artefacts are missing → run step 1 first (inside the `dashboard` folder).
- A warning about scikit-learn versions when loading the pipeline → re-run step 1 on your own machine.
- `streamlit` is not recognised → `python -m streamlit run app.py`.
