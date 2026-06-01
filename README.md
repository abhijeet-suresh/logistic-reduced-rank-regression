# Logistic Reduced-Rank Regression
### MTech Data Science — Computational Mathematics (DSC25003) | Sem 1

**Abhijeet S | CB.AI.P2DSC25003**

Implementation and analysis of the paper:
> *"A New Algorithm and Visualization Framework for Logistic Reduced Rank Regression"* — Mark de Rooij (2024, Behaviormetrika)

---

## What is Logistic Reduced-Rank Regression?

Standard multivariate logistic regression models multiple binary outcomes independently — ignoring the correlations between them and leading to a large number of parameters.

**LRRR** addresses this by imposing a rank constraint on the coefficient matrix:

```
A = B × Vᵀ
```

Where **B** (P×S) captures predictor loadings and **V** (R×S) captures response loadings, with S << min(P, R). For the Drug Consumption dataset (9 predictors, 11 responses), this reduces parameters from 99 to 40 at rank S=2.

---

## Project Overview

This project:
- Re-implements the **MM (Majorization-Minimization) algorithm** from the paper
- Compares MM with the classical **IRLS (Iteratively Reweighted Least Squares)** method
- Applies both on two real-world datasets: **UCI Drug Consumption** and **Student Performance**
- Produces **triplots** for interpreting the latent space

---

## Repository Structure

```
├── WORKS/
│   ├── DSC25003.ipynb                               # Main submission notebook
│   ├── Logistic_RRR_MM.ipynb                        # MM algorithm — clean implementation
│   ├── Full_Logistic_RRR_MM_triplots.ipynb          # MM + triplot visualizations
│   ├── Logistic_Reduced_Rank_Regression_Paper_Reproduction.ipynb  # Full paper reproduction
│   ├── logistic_reduced_rank_regression.ipynb       # Earlier implementation
│   ├── clad.ipynb                                   # Scratch implementation
│   ├── triplotstandalone.ipynb                      # Standalone triplot
│   ├── final_code.py                                # Complete Python implementation
│   ├── logistic_reduced_rank_regression.py          # Core module
│   ├── lrrr_convergence.png                         # Convergence plot output
│   ├── lrrr_triplot.png                             # Triplot output
│   ├── project_timeline.png                         # 7-day dev timeline
│   ├── LRRR_Complete_Guide.md                       # 7-day learning guide with toy problems
│   └── 7DAY_TRACKER.md                              # Development log
│
├── MMALGO.ipynb                                     # MM algorithm scratch work
├── gp.ipynb                                         # Walkthrough notebook
├── LRRR_code.py                                     # Earlier implementation (367 lines)
├── SVD GoRank (1).ipynb                             # SVD-based ranking project
├── convergence.png                                  # Convergence output
└── object_scores.png                                # Object scores plot
```

---

## Algorithm: MM (Majorization-Minimization)

The MM algorithm avoids the heavy matrix inversions required by IRLS by using a quadratic upper bound on the log-likelihood:

```
L(η) ≤ (1/8)(η − z)² + c
```

Each iteration solves a weighted least-squares problem via **Generalized SVD**, giving closed-form updates for B, V, and intercepts m.

**Per iteration:**
1. Compute Θ = 1mᵀ + XBVᵀ and Π = sigmoid(Θ)
2. Compute working responses: Z = Θ + 4(Y − Π)
3. Update intercepts: m⁺ = mean(Z − XBVᵀ)
4. Center: Z̃ = Z − 1m⁺ᵀ
5. GSVD of (XᵀX)^(-1/2) Xᵀ Z̃ → update B and V

---

## Datasets

Data files are excluded from this repo (`.gitignore`). Download from UCI:

- **Drug Consumption**: https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified  
  1885 participants, 9 personality predictors, 11 binary drug-use responses

- **Student Performance**: https://archive.ics.uci.edu/dataset/320/student+performance  
  Binary pass/fail outcomes from Math and Portuguese courses

---

## Key Results

- MM algorithm converges monotonically; faster per-iteration than IRLS
- Rank S=2 captures the dominant structure in both datasets
- Triplots reveal interpretable latent dimensions (e.g. sensation-seeking vs. conformity axes for drug use)

---

## Requirements

```bash
pip install numpy scipy pandas matplotlib scikit-learn
```

---

## Reference

de Rooij, M. (2024). A new algorithm and a discussion about visualization for logistic reduced rank regression. *Behaviormetrika*, 51, 237–267.
