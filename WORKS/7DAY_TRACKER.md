# 7-DAY PROJECT TRACKER & QUICK REFERENCE

## YOUR DEADLINE: November 15, 2025 (PRESENTATION DAY)
## TODAY: November 7, 2025 (7 days remaining)

---

## WHAT YOU NEED TO DELIVER

✅ **SINGLE JUPYTER NOTEBOOK** containing:
1. Complete working code (with line-by-line comments)
2. Conceptual explanations (simple, with toy examples)
3. Algorithm implementation (MM algorithm)
4. Testing on synthetic data
5. Real data analysis (if possible)
6. Visualizations & inferences
7. No separate report needed!

---

## YOUR 7-DAY TIMELINE

### **DAY 1 (Today, Nov 7)**
**Focus: CONCEPTUAL UNDERSTANDING - Part 1**
- [ ] Read "LRRR_Complete_Guide.md" - Sections 1.1 to 1.2
- [ ] Work through Toy Problems 1-3 (understand matrix decomposition)
- [ ] Key takeaways:
  - Why RRR? (Parameter reduction from 99→40)
  - What is rank constraint? (A = B × V^T)
  - How does logistic work for multiple responses?

**Time commitment**: 2-3 hours

---

### **DAY 2 (Nov 8)**
**Focus: CONCEPTUAL UNDERSTANDING - Part 2**
- [ ] Read "LRRR_Complete_Guide.md" - Sections 1.3 to 2.4
- [ ] Work through Toy Problems 4-6 (understand algorithm)
- [ ] Key takeaways:
  - How MM algorithm works (majorization)
  - What is GSVD and why we need it
  - Algorithm steps 1-7 in detail

**Time commitment**: 2-3 hours
**End of today**: You should understand EVERY algorithm step, even without code

---

### **DAYS 3-4 (Nov 9-10)**
**Focus: PYTHON IMPLEMENTATION**
- [ ] Copy helper functions from Section 3.1 into your notebook
  - sigmoid()
  - negative_loglikelihood()
  - compute_theta()
  - compute_working_responses()
  - generalized_svd()
- [ ] Copy the LogisticReducedRankRegression class
- [ ] Test each function individually
- [ ] Create synthetic data and run full fit
- [ ] Plot convergence graphs

**Notebook structure so far**: Cells 1-9

**Time commitment**: 4-6 hours (coding is faster once you understand concepts)

---

### **DAY 5 (Nov 11)**
**Focus: SYNTHETIC DATA EXPERIMENTS**
- [ ] Run model on synthetic data (known ground truth)
- [ ] Verify:
  - Loss decreases monotonically ✓
  - Algorithm converges ✓
  - Accuracy is reasonable ✓
- [ ] Create visualizations:
  - Convergence plot (loss vs iteration)
  - Delta plot (change vs iteration)
  - Object scores plot (U = XB in 2D)
- [ ] Compare estimated B,V with true B_true, V_true

**Debug notes**:
- If loss increases: check majorization
- If slow convergence: try different tolerance
- If NaN: check for numerical issues (clip values)

**Time commitment**: 2-3 hours

---

### **DAY 6 (Nov 12)**
**Focus: REAL DATA APPLICATION**
- [ ] Download drug consumption data from UCI
  ```
  https://archive.ics.uci.edu/ml/datasets/Drug+consumption
  ```
- [ ] Preprocess:
  - Select 9 predictors (age, gender, Big5, SS)
  - Select 11 binary responses (drugs with 10-90% prevalence)
  - Create X (n×9) and Y (n×11) matrices
- [ ] Train/test split (80/20)
- [ ] Fit model on training data
- [ ] Evaluate on test data (accuracy, etc.)
- [ ] Compute quality of representation (Q_r)

**Note**: If you can't download data, use larger synthetic dataset (works fine!)

**Time commitment**: 2-3 hours

---

### **DAY 7 (Nov 13-14)**
**Focus: FINALIZATION & PRESENTATION PREP**
- [ ] Create hybrid triplot (if time)
- [ ] Write summary interpretations
- [ ] Add markdown explanations throughout notebook
- [ ] Double-check all code runs without errors
- [ ] Test on fresh notebook from scratch (no hidden state)
- [ ] Add section headers and nice formatting
- [ ] Practice explaining each section (for presentation)

**Final notebook sections**: 1-16 (see structure below)

**Time commitment**: 3-4 hours

**PRESENT**: Nov 15 - You're ready!

---

## JUPYTER NOTEBOOK STRUCTURE (16 CELLS)

```
CELL 1:  Title & Overview
         Paper info, project goals

CELL 2:  Imports & Setup
         numpy, pandas, scipy, matplotlib

CELL 3:  Conceptual Introduction
         Simple explanation with toy example

CELL 4:  Mathematical Foundations
         Logistic regression, multivariate extension, rank constraint

CELL 5:  Helper Functions
         sigmoid, negative_loglikelihood, compute_theta, 
         compute_working_responses, generalized_svd

CELL 6:  MM Algorithm Class
         LogisticReducedRankRegression (complete fit method)

CELL 7:  Create Synthetic Data
         Generate data from known model, fit, evaluate

CELL 8:  Convergence Visualization
         Plot loss and delta vs iteration

CELL 9:  Object Scores Plot
         Visualize U = XB in 2D

CELL 10: Load Real Data
         Drug consumption or NCD dataset

CELL 11: Data Preprocessing
         Create X and Y matrices, train/test split

CELL 12: Fit on Real Data
         model.fit(X_train, Y_train)

CELL 13: Quality of Representation
         Compute Q_r for each response

CELL 14: Hybrid Triplot
         Visualize predictor-response relationships

CELL 15: Results & Inferences
         Interpretation, advantages, conclusions

CELL 16: References
         Paper citations and additional resources
```

---

## CRITICAL CODE SNIPPETS (Copy-Paste Ready)

### Snippet 1: Import Everything
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
```

### Snippet 2: Test Sigmoid
```python
def sigmoid(theta):
    theta_clipped = np.clip(theta, -500, 500)
    return 1.0 / (1.0 + np.exp(-theta_clipped))

# Test
print(sigmoid(0))      # Should be 0.5
print(sigmoid(10))     # Should be close to 1
print(sigmoid(-10))    # Should be close to 0
```

### Snippet 3: Quick Model Test
```python
# Create small synthetic data
N, P, R, S = 50, 3, 4, 2
X = np.random.randn(N, P)
B_true = np.random.randn(P, S) * 0.5
V_true = np.random.randn(R, S) * 0.5
m_true = np.zeros(R)
theta_true = X @ B_true @ V_true.T + m_true
Y = (np.random.random((N, R)) < sigmoid(theta_true)).astype(int)

# Fit
model = LogisticReducedRankRegression(rank=2, max_iterations=50)
model.fit(X, Y)

# Check accuracy
Y_pred = model.predict_class(X)
print(f"Accuracy: {np.mean(Y_pred == Y):.4f}")
```

---

## COMMON MISTAKES TO AVOID

❌ **DON'T**: Skip the conceptual part and jump to code
   ✓ **DO**: Understand algorithm thoroughly first

❌ **DON'T**: Test on real data directly
   ✓ **DO**: Test on synthetic data where you know ground truth

❌ **DON'T**: Use separate notebook cells for data/model/visualization
   ✓ **DO**: Keep everything organized in logical flow

❌ **DON'T**: Forget to clip theta values (causes overflow)
   ✓ **DO**: Use np.clip(theta, -500, 500)

❌ **DON'T**: Use random initialization without setting seed
   ✓ **DO**: Use np.random.seed(42) for reproducibility

❌ **DON'T**: Assume convergence without checking
   ✓ **DO**: Plot loss and verify monotonic decrease

---

## IF YOU GET STUCK

### Issue: Algorithm doesn't converge
- [ ] Check if loss is increasing (bad majorization?)
- [ ] Reduce tolerance (maybe 1e-4 instead of 1e-5)
- [ ] Check data preprocessing (NaN values?)
- [ ] Try longer iterations (200 instead of 100)

### Issue: NaN in predictions
- [ ] Add epsilon to avoid log(0): `clip(Pi, 1e-10, 1-1e-10)`
- [ ] Check for overflow: `clip(theta, -500, 500)`
- [ ] Verify Y is binary (0 or 1)

### Issue: Very slow fitting
- [ ] Normal if N is large (>5000)
- [ ] Check if you're recomputing (X^T X)^(-1/2) each iteration (should compute once!)
- [ ] Try smaller data subset for testing

### Issue: Can't download drug data
- [ ] Use synthetic data instead (works perfectly!)
- [ ] Or manually download from: https://archive.ics.uci.edu/ml/datasets/Drug+consumption

---

## PRESENTATION TIPS (Nov 15)

### Structure Your Talk (15 minutes)
1. **Problem (2 min)**: Why multivariate binary responses? Why not separate models?
2. **Solution (3 min)**: RRR approach, rank constraint, hidden dimensions
3. **Algorithm (4 min)**: MM algorithm, GSVD, convergence
4. **Implementation (3 min)**: Code structure, key functions
5. **Results (2 min)**: Synthetic experiment, real data results, visualizations
6. **Conclusion (1 min)**: Advantages, what you learned

### What to Show
- ✓ Conceptual slide (with toy example)
- ✓ Algorithm pseudocode with explanation
- ✓ Convergence plot (SHOWS monotonicity!)
- ✓ Object scores plot (looks cool, shows structure)
- ✓ Triplot if time (shows predictor-response relationships)
- ✓ Accuracy/quality metrics (quantitative results)

### What to Say
- "This paper solves the problem of predicting multiple related binary outcomes"
- "The key insight is decomposing A = BV^T to reduce parameters by 60%"
- "MM algorithm guarantees monotonic convergence, ~10x faster than alternatives"
- "We implemented from scratch and validated on synthetic data first"
- "Results show excellent reconstruction with both algorithms and visualizations"

---

## FINAL CHECKLIST (Before Nov 15)

**Code Quality**
- [ ] All functions have docstrings
- [ ] All for-loops have comments
- [ ] No hardcoded numbers (use variables)
- [ ] Test runs without errors
- [ ] Restart kernel and run all cells (no hidden state)

**Content**
- [ ] Clear explanation for each section
- [ ] Toy examples with concrete numbers
- [ ] Synthetic experiment with verification
- [ ] Real data (or large synthetic data)
- [ ] At least 3 visualizations
- [ ] Final summary/inferences

**Presentation**
- [ ] Time your talk (must be <20 minutes)
- [ ] Practice explaining algorithm
- [ ] Prepare backup answers to likely questions
- [ ] Know what each visualization shows
- [ ] Bring printed copy as backup

**Technical**
- [ ] Notebook file format: .ipynb
- [ ] Filename: `LRRR_Reproduction_[YourName].ipynb`
- [ ] Upload to drive/submit 24 hours before presentation

---

## YOU'VE GOT THIS! 🚀

**Key Success Factors**:
1. ✓ Understand concepts first (don't rush!)
2. ✓ Implement step-by-step (test each piece)
3. ✓ Start with synthetic data (verify correctness)
4. ✓ Visualize results clearly (plots tell story)
5. ✓ Practice presentation (confidence matters)

**Timeline**: 7 days is PLENTY if you stay focused. You're learning a research paper implementation - that's advanced stuff. Be proud!

---

**Questions? Issues?**
- Email: Try to reach the paper author or check arXiv preprints
- Stack Overflow: Post with tags [python] [numpy] [optimization]
- Consult professor/TA if stuck on concepts

**Good luck! You can do this!** 🎯

---
Last Updated: November 7, 2025
Total Time Estimate: 20-25 hours across 7 days
