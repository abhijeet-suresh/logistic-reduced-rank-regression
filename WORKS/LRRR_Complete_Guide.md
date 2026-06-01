 # Complete Guide: Logistic Reduced Rank Regression (7-Day Crash Course)

## TABLE OF CONTENTS
1. **CONCEPTUAL UNDERSTANDING** (Days 1-2)
   - Basic Concepts with Toy Problems
   - Mathematical Foundations
   
2. **ALGORITHM DEEP DIVE** (Days 3-4)
   - MM Algorithm Explained
   - GSVD Explained
   
3. **PYTHON IMPLEMENTATION** (Days 5-6)
   - Line-by-line code with comments
   - Testing on synthetic data
   
4. **REAL DATA & VISUALIZATION** (Day 7)
   - Hybrid Triplot
   - Final Results

---

# PART 1: CONCEPTUAL UNDERSTANDING (Days 1-2)

## Section 1.1: Why Do We Need Logistic Reduced Rank Regression?

### Real-World Problem (Drug Use Example)

Imagine you have:
- **1,885 people** answering surveys
- **9 predictors**: age, gender, Big Five personality traits, sensation seeking
- **11 binary responses**: Did you use each of these drugs? (Yes/No for each)

**The Challenge**: How do we predict multiple related binary outcomes from multiple predictors?

### Why Not Just Use Separate Logistic Regressions?

**Option 1 (NAIVE)**: Fit 11 separate logistic regressions
- Pro: Simple, easy to interpret
- Con: IGNORES that drug use is correlated (person using cocaine likely to use other drugs)
- Con: We have 9 × 11 = **99 parameters** to estimate (overfitting risk!)

**Option 2 (BETTER)**: Use Logistic Reduced Rank Regression
- Pro: Models correlations between responses
- Pro: Reduces parameters from 99 to maybe 18 (if rank S=2: S×(P+R) = 2×(9+11) = 40)
- Pro: Better predictions and interpretability
- Pro: Provides visualization showing relationships

### Toy Problem 1: Understanding Reduced Rank Decomposition

```
FULL MODEL:
Matrix A (9 × 11) - one coefficient for each (predictor, response) pair
A = [a11  a12  ... a1,11 ]  (9 × 11 coefficients)
    [a21  a22  ... a2,11 ]
    [... (total 99 coefficients)]

REDUCED RANK MODEL (Rank 2):
Instead of A directly, we have A = B × V^T where:

B (9 × 2):           V (11 × 2):
[b11  b12]          [v11  v12]
[b21  b22]          [v21  v22]
[...  ...]          [...  ...]
[b9,1 b9,2]         [v11,1 v11,2]

Total coefficients: 9×2 + 11×2 = 40 (instead of 99!)

Each response depends on only 2 "latent dimensions":
A = B × V^T reconstructs the full coefficient matrix
```

**Intuition**: Imagine 2 hidden factors like "Hard Drugs Index" and "Party Drugs Index"
- B tells us how each person's traits relate to these hidden factors
- V tells us how these hidden factors relate to each drug

---

## Section 1.2: Core Mathematical Concepts (Simplified)

### Concept 1: Logistic Regression Basics

**Goal**: Predict binary outcome Y given predictors X

**Probability Model**:
```
P(Y=1) = π = 1 / (1 + e^(-θ))

where θ = m + x₁a₁ + x₂a₂ + ... + xₚaₚ (the "log-odds")
      m = intercept
      aₚ = coefficient for predictor p
```

**Toy Example**:
- Want to predict: Will person use cannabis? (Y = 0 or 1)
- Predictors: age (X₁), extraversion (X₂)
- Model: θ = -2 + 0.5×age + 1.2×extraversion

For a 30-year-old with extraversion score 5:
- θ = -2 + 0.5×30 + 1.2×5 = -2 + 15 + 6 = 19
- π = 1 / (1 + e^(-19)) ≈ 0.9999999... (almost certain they use cannabis)

### Concept 2: From Single Response to Multiple Responses

**Single Response** (Standard Logistic Regression):
```
θᵢ = m + Σ(xᵢₚ × aₚ)  for person i
```

**Multiple Responses** (What we want to do):
```
θᵢᵣ = mᵣ + Σ(xᵢₚ × aₚᵣ)  for person i and response r

Example with 3 drugs and 2 predictors:
θᵢ₁ (cannabis) = m₁ + xᵢ₁×a₁₁ + xᵢ₂×a₂₁
θᵢ₂ (cocaine)  = m₂ + xᵢ₁×a₁₂ + xᵢ₂×a₂₂
θᵢ₃ (LSD)      = m₃ + xᵢ₁×a₁₃ + xᵢ₂×a₂₃

Now we have a 3×2 coefficient matrix A:
A = [a₁₁ a₁₂ a₁₃]  (2 predictors, 3 responses)
    [a₂₁ a₂₂ a₂₃]
```

### Concept 3: The RANK CONSTRAINT (The Key Idea!)

**Standard multivariate logistic**:
- A is P × R (unrestricted)
- Has P×R coefficients to estimate

**Reduced Rank Logistic**:
- A = B × V^T where B is P×S and V is R×S
- S < min(P, R) (reduced rank)
- Has only S×(P+R) coefficients

**Why S=2 is powerful**:
```
For Drug data: P=9, R=11, S=2
- Full model: 9×11 = 99 parameters
- Reduced rank: 2×(9+11) = 40 parameters
- Reduction: 60% fewer parameters!
- Benefit: Less overfitting, more stable estimates, better interpretability
```

### Toy Problem 2: Matrix Decomposition

Let's say we have true coefficient matrix A (3×2):

```
A = [0.5  1.0]  (3 responses, 2 predictors)
    [1.5  0.2]
    [2.0  0.8]

We want: A = B × V^T where B is 2×S and V is 2×S, with S=1

For rank 1:
B = [1.0]  (2×1)
    [2.0]

V = [0.3  0.4]  (2×1 transposed for visualization)

Check: B × V^T = 
[1.0] × [0.3  0.4] = [0.3  0.4]     Reconstruction
[2.0]               = [0.6  0.8]     of A
                     [1.2  1.6]
                     
This doesn't match exactly, but rank-1 gives us the best low-rank approximation!
```

---

## Section 1.3: Two Different Mathematical Perspectives

### Perspective 1: Regression View (Predict Responses from Predictors)

```
Y = 1m^T + X×A + E
  = 1m^T + X×B×V^T + E

where:
- Y: N×R matrix of binary responses (N people, R drugs)
- X: N×P matrix of predictors (N people, P traits)
- B: P×S predictor loadings
- V: R×S response loadings
- m: R intercepts
- E: N×R residuals
```

### Perspective 2: PCA View (Dimension Reduction)

```
Y = 1m^T + U×V^T + E

where:
- U = X×B is the N×S matrix of "object scores"
  (each person's position on S latent dimensions)
- V: R×S "loadings" (how drugs relate to dimensions)

Interpretation: Each person gets scored on S hidden dimensions,
then each drug's probability depends on these scores.
```

Both perspectives are equivalent! The paper uses both concepts.

---

## Section 1.4: Toy Problem 3: Understanding Data Dimensions

```
Drug Data:
- N = 1,885 people (observations/rows)
- P = 9 predictors (age, gender, Big5, SS)
- R = 11 responses (drugs)
- S = 2 (rank - number of hidden dimensions)

Matrices:
X (1885 × 9):     predictors - each person's scores on 9 traits
Y (1885 × 11):    responses - binary (0/1) for drug use
B (9 × 2):        predictor loadings - how traits load onto 2 dimensions
V (11 × 2):       response loadings - how drugs load onto 2 dimensions
U (1885 × 2):     U = X×B - each person's score on 2 dimensions
m (11):           intercepts - baseline probability for each drug

Model:
For person i and drug r:
θ[i,r] = m[r] + U[i,1]×V[r,1] + U[i,2]×V[r,2]
π[i,r] = exp(θ[i,r]) / (1 + exp(θ[i,r]))
```

---

# PART 2: ALGORITHM DEEP DIVE (Days 3-4)

## Section 2.1: Why We Need MM Algorithm

**The Problem**: How do we estimate B and V?

**Why Not Direct Optimization?**
- The likelihood function is complex and non-linear
- Hard to get closed-form solutions
- Need iterative optimization

**Why MM Algorithm?**
- Converts hard problem into easy-to-solve least squares problems
- Each iteration has closed-form solution via SVD
- Guaranteed monotonic convergence
- ~10x faster than alternative IRLS algorithm

---

## Section 2.2: MM Algorithm Concept (Simple Explanation)

### The Big Idea: "Majorization"

Think of two functions:
```
Original function:  L(θ) - our loss (negative log-likelihood)
                    [complex, nonlinear, hard to minimize]

Majorizing function: M(θ|θₙ) - our surrogate
                     [simple, quadratic, easy to minimize]

Key properties:
1. M(θ|θₙ) lies ABOVE L(θ) (majorizes it)
2. M(θₙ|θₙ) = L(θₙ) (they touch at support point)

How to use it:
- Start at θₙ
- Find surrogate M(θ|θₙ)
- Minimize surrogate: θₙ₊₁ = argmin M(θ|θₙ)
- Repeat until convergence

Guarantee: L(θₙ₊₁) ≤ L(θₙ) (monotonic improvement!)
```

### Toy Problem 4: Understanding Majorization

```
Imagine L(θ) = |θ - 3| (absolute value function, hard to minimize)

At support point θₙ = 1:
- L(1) = |1 - 3| = 2
- Derivative sign at θ=1 is negative

We create quadratic majorizer:
M(θ|1) = 2 + (derivative) × (θ-1) + λ(θ-1)²

Minimizing quadratic is easy!
Get θₙ₊₁, then repeat with new support point.

This is the core idea of MM algorithm.
```

---

## Section 2.3: MM Algorithm for Logistic Reduced Rank Regression (Step by Step)

### The Mathematical Setup

**Objective**: Minimize negative log-likelihood
```
L(θ) = - Σᵢ Σᵣ [yᵢᵣ log(πᵢᵣ) + (1-yᵢᵣ) log(1-πᵢᵣ)]

where θᵢᵣ = mᵣ + xᵢ^T B vᵣ
      πᵢᵣ = 1/(1+exp(-θᵢᵣ))
```

### Key Inequality (The Magic!)

The paper uses Böhning & Lindsay (1988):
```
For each element, the log-likelihood can be bounded by:

1/8 (θᵢᵣ - zᵢᵣ)² + constant

where zᵢᵣ = θᵢᵣ - 4ξᵢᵣ
      ξᵢᵣ = -(yᵢᵣ - πᵢᵣ)  [the "residual" term]

This is the majorizer!
```

### Algorithm: The Complete Iteration

**INPUT**: X (predictors), Y (binary responses), S (desired rank)

**INITIALIZATION**:
```
B ← small random matrix (P × S)
V ← small random matrix (R × S)
m ← mean of each response variable (R × 1)
```

**EACH ITERATION**:

```
Step 1: Compute predictions
  For all i,r: θᵢᵣ = mᵣ + xᵢ^T B vᵣ  (matrix form: Θ = 1m^T + XBV^T)
  
Step 2: Compute probabilities
  For all i,r: πᵢᵣ = 1/(1+exp(-θᵢᵣ))

Step 3: Compute residuals for working responses
  ξᵢᵣ = yᵢᵣ - πᵢᵣ  (prediction error)
  zᵢᵣ = θᵢᵣ - 4ξᵢᵣ  (working response)
  In matrix form: Z = Θ + 4(Y - Π)

Step 4: Update intercepts
  m⁺ = N⁻¹ (Z - XBV^T)^T 1
  (where 1 is vector of ones)

Step 5: Center responses
  Z_centered = Z - 1m⁺^T

Step 6: GENERALIZED SVD
  Compute: (X^T X)^(-1/2) X^T Z_centered = P Σ Q^T
  (Details of GSVD in Section 2.4)

Step 7: Update parameters
  B⁺ = √N (X^T X)^(-1/2) P_S
  V⁺ = (1/√N) Q_S Σ_S
  where P_S, Q_S, Σ_S are first S columns/values

Step 8: Check convergence
  Compute new loss L_new
  If |L_new - L_old| < tolerance: STOP
  Otherwise: L_old = L_new, go to Step 1
```

### Toy Problem 5: Complete Mini Example

**Data**:
```
N=4 people, P=2 predictors, R=3 responses, S=1 (rank)

X = [1.0  0.5]  (person 1: pred1=1.0, pred2=0.5)
    [2.0  1.0]
    [-1.0 0.5]
    [0.5  -1.0]

Y = [1  0  1]  (person 1: response 1=yes, response 2=no, response 3=yes)
    [0  1  1]
    [1  1  0]
    [0  0  1]
```

**Initialization (Iteration 0)**:
```
B = [0.01]  (2×1)
    [0.02]

V = [0.01  0.02  0.03]^T → stored as V = [0.01]  (3×1)
                                           [0.02]
                                           [0.03]

m = [0.5   0.5   0.75]  (mean of each response)
```

**Iteration 1, Step 1-2: Compute predictions**
```
θ = 1m^T + XBV^T
  = [1] [0.5 0.5 0.75] + [1.0 0.5] [0.01] [0.01 0.02 0.03]
    [1]                  [2.0 1.0] [0.02]
    [1]                  [-1.0 0.5]
    [1]                  [0.5 -1.0]

For person 1:
  θ₁ᵣ = m + X₁^T B V_r
  θ₁₁ = 0.5 + [1.0 0.5] [0.01] × 0.01 ≈ 0.5 + 0.012×0.01 ≈ 0.5
              [0.02]
  (etc. for other responses and people)

π = 1/(1+exp(-θ))  (apply logistic to each θ)
```

**Iteration 1, Step 3-4: Work responses and intercepts**
```
ξ = Y - π  (element-wise residuals)
Z = θ + 4ξ  (working responses)
m⁺ = mean(Z - XBV^T)
```

**Iteration 1, Step 5-7: GSVD and parameter updates**
```
(X^T X)^(-1/2) X^T Z_centered = P Σ Q^T  (GSVD)
B⁺ = √4 × (X^T X)^(-1/2) × P_1
V⁺ = (1/√4) × Q_1 × Σ_1
```

This is tedious to compute by hand, but computer does it automatically!

---

## Section 2.4: Generalized SVD (GSVD) Explained

### What is Standard SVD?

```
For any matrix A (m × n):
A = U Σ V^T

where:
- U (m × m): left singular vectors
- Σ (m × n): diagonal singular values
- V (n × n): right singular vectors

Properties:
- U^T U = I (orthonormal columns)
- V^T V = I (orthonormal columns)
```

### What is GSVD?

**Standard SVD is orthonormal with respect to identity matrix**

**GSVD is orthonormal with respect to metric matrices G and H**

```
GSVD of Y in metrics G and H:
Y = P Σ Q^T

where:
- P: left generalized singular vectors
- Q: right generalized singular vectors
- Σ: singular values
- Satisfy: P^T G P = I and Q^T H Q = I (orthonormal w.r.t. G, H)
```

### How to Compute GSVD

**Algorithm** (from paper Appendix A):

```
Step 1: Transform to standard SVD space
  Y* = G^(1/2) Y H^(1/2)

Step 2: Compute standard SVD of Y*
  Y* = P* Σ* Q*^T

Step 3: Transform back
  P = G^(-1/2) P*
  Q = H^(-1/2) Q*
  Σ = Σ*  (unchanged)
```

### In Our Application

For logistic reduced rank regression:
```
We need GSVD of: (X^T X)^(-1/2) X^T Z_centered

Metrics are:
- G = X^T X (metric for left side)
- H = I (identity for right side)

Algorithm:
1. Compute M = (X^T X)^(-1/2) X^T Z_centered
2. Do standard SVD: M = P* Σ* Q*^T
3. Transform: P = (X^T X)^(-1/2) P*
4. Already orthonormal w.r.t. H=I, so Q = Q*
5. Take first S columns for rank S constraint
```

### Toy Problem 6: GSVD Example

```
Simple example with G = [2 0; 0 1], H = I

Y = [1 2]
    [3 4]

Step 1: Transform
G^(1/2) = [√2  0]   H^(1/2) = [1 0]
          [0   1]           [0 1]

Y* = G^(1/2) Y H^(1/2) = [√2  0] [1 2] [1 0]
                          [0   1] [3 4] [0 1]
                        = [√2  2√2]
                          [3   4]

Step 2: Standard SVD of Y* gives P*, Σ*, Q*^T

Step 3: Transform back
P = G^(-1/2) P* = [1/√2  0] P*
                  [0     1]
```

---

# PART 3: PYTHON IMPLEMENTATION (Days 5-6)

## Section 3.1: Helper Functions (Build Blocks)

I'll now provide complete, well-commented Python code.

```python
# ==============================================================================
# LOGISTIC REDUCED RANK REGRESSION - COMPLETE IMPLEMENTATION
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.linalg import sqrtm, inv
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# SECTION 1: BASIC HELPER FUNCTIONS
# ==============================================================================

def sigmoid(theta):
    """
    Compute sigmoid/logistic function
    π = 1 / (1 + exp(-θ))
    
    Safe version to avoid overflow: use scipy.special.expit
    
    Args:
        theta: ndarray of log-odds values
        
    Returns:
        π: ndarray of probabilities in [0,1]
    """
    # Clip to avoid overflow
    theta_clipped = np.clip(theta, -500, 500)
    return 1.0 / (1.0 + np.exp(-theta_clipped))


def negative_loglikelihood(Y, Pi, epsilon=1e-10):
    """
    Compute negative log-likelihood: -Σ[y*log(π) + (1-y)*log(1-π)]
    
    Args:
        Y: ndarray (N × R) of binary responses {0, 1}
        Pi: ndarray (N × R) of predicted probabilities (0 to 1)
        epsilon: small value to avoid log(0)
        
    Returns:
        loss: scalar negative log-likelihood
        
    Formula:
        L(θ) = -Σᵢ Σᵣ [yᵢᵣ log(πᵢᵣ) + (1-yᵢᵣ) log(1-πᵢᵣ)]
    """
    # Clip probabilities to avoid log(0)
    Pi_clipped = np.clip(Pi, epsilon, 1 - epsilon)
    
    # Compute loss
    loss = -np.sum(Y * np.log(Pi_clipped) + (1 - Y) * np.log(1 - Pi_clipped))
    
    return loss


def compute_theta(X, B, V, m):
    """
    Compute log-odds: θ = 1m^T + XBV^T
    
    Args:
        X: ndarray (N × P) predictor matrix
        B: ndarray (P × S) predictor loadings
        V: ndarray (R × S) response loadings
        m: ndarray (R,) intercepts
        
    Returns:
        theta: ndarray (N × R) log-odds
        
    Formula:
        θᵢᵣ = mᵣ + Σₛ xᵢ^T bₛ vᵣₛ
    """
    N, R = X.shape[0], V.shape[0]
    
    # θ = XBV^T  (matrix multiplication)
    theta = X @ B @ V.T  # (N×P) @ (P×S) @ (S×R) = (N×R)
    
    # Add intercepts: θ + 1m^T where 1 is column of ones
    theta = theta + m.reshape(1, -1)  # broadcast intercepts
    
    return theta


def compute_working_responses(Y, theta, Pi):
    """
    Compute working responses for MM algorithm: Z = θ - 4ξ
    where ξ = -(y - π)
    
    In matrix form: Z = θ + 4(Y - Π)
    
    Args:
        Y: ndarray (N × R) binary responses
        theta: ndarray (N × R) current log-odds
        Pi: ndarray (N × R) current probabilities
        
    Returns:
        Z: ndarray (N × R) working responses
        
    Intuition:
        - If y=1 and π≈1: Z ≈ θ (no change, already predicted well)
        - If y=1 and π≈0: Z >> θ (push θ higher to match y=1)
        - If y=0 and π≈0: Z ≈ θ (no change, already predicted well)
        - If y=0 and π≈1: Z << θ (push θ lower to match y=0)
    """
    # ξ = -(y - π) = π - y  [residuals]
    residuals = Y - Pi
    
    # Z = θ + 4(Y - Π) = θ + 4*(-ξ) = θ - 4ξ
    Z = theta + 4 * residuals
    
    return Z


def generalized_svd(X, Z_centered):
    """
    Compute generalized SVD of M = (X^T X)^(-1/2) X^T Z_centered
    in metrics G = X^T X and H = I
    
    Steps:
    1. Compute metric G = X^T X
    2. Compute G^(-1/2) (inverse square root of G)
    3. Form M = G^(-1/2) X^T Z_centered
    4. SVD: M = P* Σ Q*^T
    5. Return P, Σ, Q (which are already orthonormal for our purpose)
    
    Args:
        X: ndarray (N × P) predictor matrix
        Z_centered: ndarray (N × R) centered working responses
        
    Returns:
        P: ndarray (P × S) left singular vectors
        Sigma: ndarray (S,) singular values (sorted descending)
        Q: ndarray (R × S) right singular vectors
    """
    # Step 1: Compute X^T X
    XTX = X.T @ X  # (P × N) @ (N × P) = (P × P)
    
    # Step 2: Compute (X^T X)^(-1/2)
    try:
        # Try to use matrix square root
        XTX_inv_sqrt = inv(sqrtm(XTX))
    except:
        # If singular, use pseudo-inverse and regularization
        XTX_reg = XTX + np.eye(XTX.shape[0]) * 1e-8
        XTX_sqrt = sqrtm(XTX_reg)
        XTX_inv_sqrt = inv(XTX_sqrt)
    
    # Step 3: Form M = (X^T X)^(-1/2) X^T Z_centered
    M = XTX_inv_sqrt @ X.T @ Z_centered  # (P×P) @ (P×N) @ (N×R) = (P×R)
    
    # Step 4: Standard SVD
    P_star, Sigma, Qt_star = np.linalg.svd(M, full_matrices=False)
    # Note: SVD returns Q^T, so transposed
    
    # Step 5: For our case (H = I), P and Q are already in right form
    P = P_star
    Q = Qt_star.T  # transpose back to get Q (not Q^T)
    
    return P, Sigma, Q


# ==============================================================================
# SECTION 2: MAIN MM ALGORITHM
# ==============================================================================

class LogisticReducedRankRegression:
    """
    Logistic Reduced Rank Regression model using MM algorithm
    
    Model: θᵢᵣ = mᵣ + xᵢ^T B vᵣ
           πᵢᵣ = 1 / (1 + exp(-θᵢᵣ))
           
    where:
    - X (N×P): predictor matrix
    - Y (N×R): binary response matrix
    - B (P×S): predictor loadings
    - V (R×S): response loadings
    - m (R,): intercepts
    - S: rank constraint
    """
    
    def __init__(self, rank=2, max_iterations=100, tolerance=1e-5, 
                 random_state=42):
        """
        Initialize the model
        
        Args:
            rank: desired rank S (default 2)
            max_iterations: maximum MM iterations (default 100)
            tolerance: convergence tolerance for deviance change
            random_state: for reproducibility
        """
        self.rank = rank
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Will be filled after fitting
        self.B = None
        self.V = None
        self.m = None
        self.convergence_history = {'loss': [], 'delta': []}
        self.n_iterations = 0
        
    def fit(self, X, Y):
        """
        Fit the model using MM algorithm
        
        Args:
            X: ndarray (N × P) predictors
            Y: ndarray (N × R) binary responses (0 or 1)
            
        Returns:
            self (for chaining)
        """
        np.random.seed(self.random_state)
        
        N, P = X.shape
        N_check, R = Y.shape
        assert N == N_check, "X and Y must have same number of rows"
        
        S = self.rank
        
        # =====================================================================
        # INITIALIZATION
        # =====================================================================
        
        # Initialize B and V with small random values
        self.B = np.random.randn(P, S) * 0.01
        self.V = np.random.randn(R, S) * 0.01
        
        # Initialize intercepts as empirical proportions
        self.m = np.log(np.mean(Y, axis=0) / (1 - np.mean(Y, axis=0)))
        
        # For computing (X^T X)^(-1/2) only once (efficiency)
        XTX = X.T @ X
        try:
            XTX_inv_sqrt = inv(sqrtm(XTX))
        except:
            XTX_reg = XTX + np.eye(P) * 1e-8
            XTX_inv_sqrt = inv(sqrtm(XTX_reg))
        
        print(f"Starting MM algorithm: N={N}, P={P}, R={R}, S={S}")
        print(f"Parameters to estimate: {S*(P+R)} (vs full model {P*R})")
        print("-" * 70)
        
        # =====================================================================
        # MM ITERATIONS
        # =====================================================================
        
        loss_old = np.inf
        
        for iteration in range(self.max_iterations):
            # -----------------------------------------------------------------
            # STEP 1: Compute current predictions
            # -----------------------------------------------------------------
            theta = compute_theta(X, self.B, self.V, self.m)
            Pi = sigmoid(theta)
            
            # -----------------------------------------------------------------
            # STEP 2: Compute loss for monitoring convergence
            # -----------------------------------------------------------------
            loss_new = negative_loglikelihood(Y, Pi)
            delta = abs(loss_new - loss_old)
            
            # Store history
            self.convergence_history['loss'].append(loss_new)
            self.convergence_history['delta'].append(delta)
            
            # Print progress
            if iteration % 5 == 0:
                print(f"Iteration {iteration:3d}: Loss = {loss_new:.6f}, "
                      f"Delta = {delta:.2e}")
            
            # Check convergence
            if delta < self.tolerance:
                print(f"\nConverged at iteration {iteration}")
                self.n_iterations = iteration
                break
            
            # -----------------------------------------------------------------
            # STEP 3: Compute working responses (majorization step)
            # -----------------------------------------------------------------
            # Z = θ + 4(Y - Π)
            Z = compute_working_responses(Y, theta, Pi)
            
            # -----------------------------------------------------------------
            # STEP 4: Update intercepts
            # -----------------------------------------------------------------
            # m⁺ = N⁻¹(Z - XBV^T)^T 1
            m_plus = np.mean(Z - X @ self.B @ self.V.T, axis=0)
            
            # -----------------------------------------------------------------
            # STEP 5: Center working responses
            # -----------------------------------------------------------------
            Z_centered = Z - m_plus.reshape(1, -1)
            
            # -----------------------------------------------------------------
            # STEP 6: Generalized SVD
            # -----------------------------------------------------------------
            # Decompose (X^T X)^(-1/2) X^T Z_centered
            P_all, Sigma_all, Q_all = generalized_svd(X, Z_centered)
            
            # -----------------------------------------------------------------
            # STEP 7: Update B and V using top S components
            # -----------------------------------------------------------------
            P_S = P_all[:, :S]
            Q_S = Q_all[:, :S]
            Sigma_S = Sigma_all[:S]
            
            # B⁺ = √N (X^T X)^(-1/2) P_S
            self.B = np.sqrt(N) * (XTX_inv_sqrt @ P_S)
            
            # V⁺ = (1/√N) Q_S Σ_S
            self.V = (1.0 / np.sqrt(N)) * (Q_S @ np.diag(Sigma_S))
            
            # Update intercepts
            self.m = m_plus
            
            # Update for next iteration
            loss_old = loss_new
        
        # If didn't converge
        if iteration == self.max_iterations - 1:
            print(f"\nDid not converge after {self.max_iterations} iterations")
            self.n_iterations = self.max_iterations
        
        print("-" * 70)
        print(f"Final Loss: {loss_old:.6f}")
        
        return self
    
    def predict_probabilities(self, X_new):
        """
        Predict probabilities for new data
        
        Args:
            X_new: ndarray (M × P) new predictor values
            
        Returns:
            Pi_pred: ndarray (M × R) predicted probabilities
        """
        if self.B is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        theta_new = compute_theta(X_new, self.B, self.V, self.m)
        Pi_pred = sigmoid(theta_new)
        
        return Pi_pred
    
    def predict_class(self, X_new, threshold=0.5):
        """
        Predict binary classes for new data
        
        Args:
            X_new: ndarray (M × P) new predictor values
            threshold: probability threshold for class 1 (default 0.5)
            
        Returns:
            Y_pred: ndarray (M × R) predicted classes {0, 1}
        """
        Pi_pred = self.predict_probabilities(X_new)
        return (Pi_pred >= threshold).astype(int)


# ==============================================================================
# SECTION 3: VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_convergence(model):
    """
    Plot convergence history of MM algorithm
    
    Args:
        model: fitted LogisticReducedRankRegression object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Loss over iterations
    ax1.plot(model.convergence_history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax1.set_title('MM Algorithm Convergence: Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Delta (change) over iterations
    ax2.plot(model.convergence_history['delta'], 'r-', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('|Loss Change|', fontsize=12)
    ax2.set_title('MM Algorithm Convergence: Change', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_object_scores(model, X, Y_true, labels=None):
    """
    Plot object scores (U = XB) in 2D
    
    Args:
        model: fitted LogisticReducedRankRegression object
        X: ndarray (N × P) predictor matrix
        Y_true: ndarray (N × R) or (N,) responses for coloring (optional)
        labels: list of sample labels (optional)
    """
    if model.rank < 2:
        print("Rank must be at least 2 for 2D visualization")
        return None
    
    # Compute object scores
    U = X @ model.B  # (N × S)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by first response (or by class)
    if Y_true.ndim > 1:
        colors = Y_true[:, 0]
        label_text = "Response 1"
    else:
        colors = Y_true
        label_text = "Class"
    
    scatter = ax.scatter(U[:, 0], U[:, 1], c=colors, cmap='RdYlBu', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'Dimension 1 (B₁)', fontsize=12)
    ax.set_ylabel(f'Dimension 2 (B₂)', fontsize=12)
    ax.set_title(f'Object Scores (U = XB)', fontsize=14)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label_text, fontsize=11)
    
    return fig


# ==============================================================================
# SECTION 4: SYNTHETIC TOY EXAMPLE
# ==============================================================================

def create_synthetic_data(n_samples=100, n_predictors=3, n_responses=4,
                         rank=2, noise=0.1, seed=42):
    """
    Create synthetic data from known reduced rank model
    
    X ~ N(0, I)  (random predictors)
    θ = XBV^T + m  (log-odds from model)
    Y ~ Bernoulli(sigmoid(θ))  (binary responses)
    
    Args:
        n_samples: number of samples
        n_predictors: P
        n_responses: R
        rank: S
        noise: noise level
        seed: random seed
        
    Returns:
        X, Y, B_true, V_true, m_true
    """
    np.random.seed(seed)
    
    # Create true B and V
    B_true = np.random.randn(n_predictors, rank) * 0.5
    V_true = np.random.randn(n_responses, rank) * 0.5
    m_true = np.random.randn(n_responses) * 0.2
    
    # Generate predictors
    X = np.random.randn(n_samples, n_predictors)
    
    # Generate true log-odds
    theta_true = X @ B_true @ V_true.T + m_true.reshape(1, -1)
    
    # Add small noise
    theta_noisy = theta_true + np.random.randn(*theta_true.shape) * noise
    
    # Generate binary responses
    pi = sigmoid(theta_noisy)
    Y = (np.random.random(pi.shape) < pi).astype(int)
    
    return X, Y, B_true, V_true, m_true, theta_true


# ==============================================================================
# SECTION 5: EXAMPLE USAGE AND TESTING
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("LOGISTIC REDUCED RANK REGRESSION - DEMONSTRATION")
    print("="*70 + "\n")
    
    # Create synthetic toy data
    print("1. Creating synthetic data...")
    X, Y, B_true, V_true, m_true, theta_true = create_synthetic_data(
        n_samples=200, n_predictors=4, n_responses=5, rank=2, 
        noise=0.2, seed=42
    )
    print(f"   Data shape: X={X.shape}, Y={Y.shape}")
    print(f"   True model: B={B_true.shape}, V={V_true.shape}, m={m_true.shape}")
    
    # Fit model
    print("\n2. Fitting Logistic Reduced Rank Regression model...")
    model = LogisticReducedRankRegression(rank=2, max_iterations=100, 
                                          tolerance=1e-5, random_state=42)
    model.fit(X, Y)
    
    # Make predictions
    print("\n3. Making predictions on training data...")
    Pi_pred = model.predict_probabilities(X)
    Y_pred = model.predict_class(X, threshold=0.5)
    
    # Compute accuracy
    accuracy = np.mean(Y_pred == Y)
    print(f"   Classification accuracy: {accuracy:.4f}")
    
    # Visualize convergence
    print("\n4. Creating visualizations...")
    fig1 = plot_convergence(model)
    plt.savefig('convergence.png', dpi=100, bbox_inches='tight')
    print("   Saved: convergence.png")
    
    fig2 = plot_object_scores(model, X, Y)
    plt.savefig('object_scores.png', dpi=100, bbox_inches='tight')
    print("   Saved: object_scores.png")
    
    # Print parameter estimates
    print("\n5. Estimated Parameters:")
    print(f"   B shape: {model.B.shape}")
    print(f"   V shape: {model.V.shape}")
    print(f"   m shape: {model.m.shape}")
    print(f"   Iterations to convergence: {model.n_iterations}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")
```

---

## Section 3.2: Testing on Synthetic Data

[Complete test code with visualization provided in above implementation]

---

# PART 4: REAL DATA & VISUALIZATION (Day 7)

## Section 4.1: Loading Drug Consumption Data

```python
import pandas as pd

# Load drug consumption dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"
data = pd.read_csv(url, header=None)

# The dataset has specific columns - see paper for details
# For this exercise, we'll load and prepare it

column_names = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
                'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsivity',
                'SS', 'Amphetamine', 'Amyl', 'Benzodiazepine', 'Cannabis',
                'Chocolate', 'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
                'Legal.Highs', 'LSD', 'Methadone', 'Mushrooms', 'Nicotine', 'Solvents']

# ... (data preprocessing code)

# Final matrices:
# X: predictor matrix (1885 × 9)
# Y: response matrix (1885 × 11) for selected drugs
```

---

## Section 4.2: Hybrid Triplot Visualization

```python
def create_hybrid_triplot(model, X, predictor_names, response_names):
    """
    Create hybrid triplot combining Type I and Type D visualization
    
    Args:
        model: fitted LogisticReducedRankRegression
        X: predictor matrix
        predictor_names: list of predictor names
        response_names: list of response names
    """
    if model.rank < 2:
        print("Rank must be >= 2 for 2D triplot")
        return
    
    # Compute scores
    U = X @ model.B  # object scores
    B = model.B
    V = model.V
    m = model.m
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 1. Plot object points (participants)
    ax.scatter(U[:, 0], U[:, 1], alpha=0.4, s=30, c='gray', 
              label='Participants', edgecolors='none')
    
    # 2. Plot predictor variable axes (blue)
    arrow_scale = 3
    for p, pred_name in enumerate(predictor_names):
        bp = B[p, :]  # (2,) for rank 2
        ax.arrow(0, 0, bp[0]*arrow_scale, bp[1]*arrow_scale,
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', 
                alpha=0.6, linewidth=2)
        # Add label
        ax.text(bp[0]*arrow_scale*1.15, bp[1]*arrow_scale*1.15, pred_name,
               fontsize=10, color='blue', fontweight='bold')
    
    # 3. Plot response variable axes (green) with markers
    for r, resp_name in enumerate(response_names):
        vr = V[r, :]  # (2,) for rank 2
        mr = m[r]
        
        # Markers for probabilities 0.1 to 0.9
        probs = np.linspace(0.1, 0.9, 9)
        
        for prob in probs:
            lambda_val = np.log(prob / (1 - prob)) - mr
            marker_pos = lambda_val * vr / np.dot(vr, vr)
            
            ax.plot(marker_pos[0], marker_pos[1], 'g.', markersize=4, alpha=0.5)
        
        # Draw response axis
        ax.arrow(0, 0, vr[0]*arrow_scale, vr[1]*arrow_scale,
                head_width=0.1, head_length=0.1, fc='green', ec='green',
                alpha=0.6, linewidth=2)
        
        # Add label
        ax.text(vr[0]*arrow_scale*1.15, vr[1]*arrow_scale*1.15, resp_name,
               fontsize=10, color='green', fontweight='bold')
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Hybrid Triplot: Logistic Reduced Rank Regression', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    return fig
```

---

# 7-DAY TIMELINE & DELIVERABLES

## Day 1 (Nov 8): Conceptual Understanding - Part 1
- Read Sections 1.1-1.2 of this guide
- Work through Toy Problems 1-3
- Understand: RRR concept, logistic basics, rank constraint

## Day 2 (Nov 9): Conceptual Understanding - Part 2
- Read Section 1.3-1.4
- Work through Toy Problems 4-6
- Understand: Majorization, GSVD, algorithm steps

## Day 3-4 (Nov 10-11): Implementation & Testing
- Code up Section 3.1-3.2
- Run synthetic examples
- Verify convergence, accuracy
- Create visualizations

## Day 5-6 (Nov 12-13): Real Data Analysis
- Load drug consumption data (or NCD data if available)
- Preprocess and prepare matrices X, Y
- Fit model on real data
- Compute quality metrics

## Day 7 (Nov 14): Visualization & Final Notebook
- Create hybrid tripplot
- Interpret results
- Finalize Jupyter notebook
- Prepare for presentation (Nov 15)

---

# FINAL JUPYTER NOTEBOOK STRUCTURE (for Nov 15 presentation)

```
1. TITLE & OVERVIEW
   - Paper title, authors, date
   - Your name, date

2. INTRODUCTION (markdown + code)
   - Problem statement
   - Why RRR matters
   - Paper contributions

3. CONCEPTUAL EXPLANATION
   - Simple explanations with toy problems
   - Figures/diagrams
   - Example calculations

4. ALGORITHM WALKTHROUGH
   - MM algorithm steps
   - GSVD explanation
   - Pseudocode with comments

5. IMPLEMENTATION
   - Helper functions with detailed comments
   - Main MM algorithm class
   - Convergence monitoring

6. SYNTHETIC DATA EXPERIMENT
   - Create synthetic data
   - Fit model
   - Evaluate accuracy
   - Plot convergence & scores

7. REAL DATA APPLICATION
   - Load drug consumption data
   - Preprocess
   - Fit model (rank 2)
   - Display results in table

8. VISUALIZATION
   - Convergence plots
   - Object scores plot
   - Hybrid triplot (if possible)
   - Interpretation

9. RESULTS & INFERENCES
   - Key findings
   - Interpretation of loadings
   - Comparison to baseline

10. CONCLUSION
    - What you learned
    - Strengths of the paper
    - Limitations/future work
```

---

Good luck! You have all the code and concepts needed. Focus on understanding step-by-step, implement carefully with comments, and present your findings clearly.

