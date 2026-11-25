# RL Credit Scoring

Reinforcement learning applied to loan approval and interest rate pricing, framed as a sequential decision-making problem. An RL agent learns a policy that balances portfolio profitability against credit risk, targeting a non-performing loan (NPL) ratio below 5%.

Built on the [LendingClub Loan Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (~2.26M loan records, 2007–2018).

## Summary
A two-track credit-risk project on the LendingClub loan data (2007–2018). Track 1 benchmarks four calibrated default classifiers (Logistic Regression, Random Forest, XGBoost, CatBoost) on a strict temporal split, using PR-AUC-driven Optuna tuning, target-aware encoders and isotonic calibration on a held-out validation set. Track 2 treats lending as a sequential decision problem: an agent chooses to reject a loan or approve it at one of three rate tiers while the portfolio's non-performing-loan (NPL) ratio is tracked. We first proposed a constrained DQN, but its value estimates were unstable under sparse, high-variance rewards and could not guarantee a hard risk budget. We replaced it with an XGBoost policy that uses a reject region (p_default > 0.20, FICO < 700) to hold test NPL at 4.01%, against 12.35% for a greedy bandit baseline and 22.42% for the unconstrained ablation.
---

## Methods

| Method | Type | Description |
|--------|------|-------------|
| **Greedy Contextual Bandit** | RL baseline | One logistic regression per rate tier, ε-greedy action selection |
| **Constrained DQN** | RL proposed | Portfolio NPL ratio in state + explicit NPL penalty in reward |
| **Unconstrained DQN** | RL ablation | Same DQN without penalty term — isolates contribution of constraint |
| **Static Classifier Policy** | Non-RL baseline | Best of LR / RF / XGBoost converted to a lending rule: approve if P(default) < threshold, assign rate tier by risk score |

The RL action space: `{Reject, Approve@LowRate (8%), Approve@MidRate (13%), Approve@HighRate (19%)}`. All four methods are evaluated on the same `LendingEnv` for a direct profit/NPL comparison.

---

## Project Structure

```
RL-CREDIT-SCORING/
├── data/
│   ├── raw_data/               # Downloaded from Kaggle (auto-created by load_data.py)
│   └── processed_data/         # Cleaned outputs used by all models
├── notebook/
│   ├── eda_accepted.ipynb
│   └── eda_rejected.ipynb
├── src/
│   ├── bandit/
│   │   └── baseline.ipynb      # Greedy contextual bandit (Task 3)
│   ├── dqn/                    # Constrained DQN (Task 4)
│   ├── unconstrained_dqn/      # Ablation — no NPL penalty (Task 4)
│   └── classification_models.ipynb  # LR, RF, XGBoost + static policy (Task 5)
├── clean_accepted.ipynb        # Cleans accepted loans → processed_data/
├── clean_rejected.ipynb        # Cleans rejected loans → processed_data/
├── load_data.py                # Downloads raw data from Kaggle
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the data

```bash
python load_data.py
```

This creates `data/raw_data/` and downloads the LendingClub dataset from Kaggle into it.

### 3. Clean data
```bash
cd data
python clean_accepted.py
```
Output:

### 4. Extract data for ML
Run notebook/raw_data/feat_for_ML.ipynb, then make directory and write their outputs to `data/ml_data/`.

The cleaning steps include: dropping features with >10% null values, removing correlated redundant features (Pearson r > 0.70), encoding categoricals, and engineering eight derived features (e.g. `loan_to_income`, `monthly_debt_burden`, `poor_fico_flag`).

### 5. Explore (optional)

```
notebook/eda_accepted.ipynb   # Distribution analysis, default rate by feature
notebook/eda_rejected.ipynb   # Rejected loan characteristics
```

### 6. Run the models

| Notebook / folder | Task | What it runs |
|---|---|---|
| `src/bandit/baseline.ipynb` | Task 3 | Greedy contextual bandit |
| `src/dqn/` | Task 4 | Constrained DQN with NPL penalty |
| `src/unconstrained_dqn/` | Task 4 | Unconstrained DQN (ablation) |
| `src/classification_models.ipynb` | Task 5 | LR, Random Forest, XGBoost + static lending policy |

> **Task 5 depends on Tasks 3 & 4.** The unified evaluation (Task 6) collects metrics from all four methods, so run the RL notebooks before running the final comparison cells in `classification_models.ipynb`.

All models read from `data/processed_data/`. A strict temporal split is applied: **2007–2015 → train**, **2016–2018 → test**.

---

## State & Reward

**State vector (6 features):**

| Feature | Description |
|---|---|
| `fico_score` | Midpoint of FICO range, normalized to [0, 1] |
| `dti` | Debt-to-income ratio, log-transformed |
| `annual_inc` | Annual income, log-transformed |
| `emp_length` | Employment length, ordinal encoded (0–10) |
| `purpose` | Loan purpose, integer encoded |
| `npl_ratio` | Current portfolio NPL ratio *(DQN methods only)* |

**Reward function:**

```
R = rate × loan_amnt   if fully repaid
R = −loan_amnt         if defaulted
R = 0                  if rejected
```

The constrained DQN adds an NPL penalty: `R̃ = R − λ · max(0, NPL − 0.05)`.

---

## Evaluation Metrics

### Classification metrics (Task 5)

Evaluated on the default (minority) class across LR, RF, and XGBoost:

| Metric | Description |
|---|---|
| F1 | Harmonic mean of precision and recall for the default class |
| Precision | Of predicted defaults, how many actually defaulted |
| Recall | Of actual defaults, how many were caught |
| AUC-ROC | Area under the ROC curve |

Class imbalance is handled via `class_weight='balanced'` or SMOTE. Hyperparameters for RF and XGBoost (`n_estimators`, `max_depth`, `learning_rate`) are tuned with Bayesian optimization over cross-validation.

### Portfolio metrics (all 4 methods, Task 6)

All methods are run through `LendingEnv` and compared on the same metrics:

| Metric | Description |
|---|---|
| Cumulative portfolio profit | Total interest income minus default losses — primary metric |
| NPL ratio over time | Must stay below 5% regulatory threshold |
| Profit per approved loan | Pricing efficiency, independent of approval volume |
| Approval rate | Tracks conservative/permissive drift over the simulation |
| Constraint violation rate | % of episodes where NPL exceeds 5% |

The final deliverable is a unified results table (all 4 methods × all metrics) and a profit-risk tradeoff chart plotting cumulative profit against NPL ratio.

---

## Requirements

See `requirements.txt`. Key dependencies:

```
torch
scikit-learn
xgboost
imbalanced-learn    # for SMOTE
scikit-optimize     # for Bayesian hyperparameter tuning
pandas
numpy
matplotlib
seaborn
kaggle
```

---

## References

S. Kiatsupaibul, P. Chansiripas, P. Manopanjasiri, K. Visantavarakul, and Z. Wen, "Reinforcement learning in credit scoring and underwriting," *arXiv preprint arXiv:2212.07632*, 2022.
