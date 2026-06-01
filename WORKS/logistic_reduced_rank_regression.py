
'''
LOGISTIC REDUCED RANK REGRESSION - COMPLETE IMPLEMENTATION
Based on de Rooij (2024) - Behaviormetrika

This implementation includes:
1. MM Algorithm for parameter estimation
2. Type I, Type D, and Hybrid visualization methods
3. Quality metrics (Qr, deviance, timing)
4. Testing on UCI Drug Consumption dataset
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from scipy.stats import logistic
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

#============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
#============================================================================

def load_drug_consumption_data():
    '''
    Load and preprocess the UCI Drug Consumption dataset

    Returns:
    --------
    X : array-like, shape (n_samples, n_predictors)
        Predictor variables (age, gender, personality traits)
    Y : array-like, shape (n_samples, n_responses)
        Binary response variables (drug use: 0=no, 1=yes)
    predictor_names : list
        Names of predictor variables
    response_names : list
        Names of response variables
    '''
    try:
        from ucimlrepo import fetch_ucirepo

        # Fetch dataset from UCI repository
        drug_consumption = fetch_ucirepo(id=373)

        # Get features (X) and targets (y)
        X_raw = drug_consumption.data.features
        y_raw = drug_consumption.data.targets

        print("Dataset loaded successfully from UCI repository")
        print(f"Original shape - X: {X_raw.shape}, y: {y_raw.shape}")

    except:
        print("Could not load from UCI repository. Loading from URL...")
        # Alternative: load from URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"

        # Column names based on UCI dataset description
        col_names = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
                     'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
                     'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis',
                     'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
                     'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

        data = pd.read_csv(url, names=col_names)

        # Select predictor variables (demographics + personality)
        predictor_cols = ['Age', 'Gender', 'Nscore', 'Escore', 'Oscore', 
                          'Ascore', 'Cscore', 'SS']
        X_raw = data[predictor_cols]

        # Select response variables (drugs with prevalence between 10-90%)
        # As per paper: Amphet, Benzos, Cannabis, Coke, Ecstasy, Ketamine, Legalh, LSD, Meth, Mushrooms, Nicotine
        response_cols = ['Amphet', 'Benzos', 'Cannabis', 'Coke', 'Ecstasy', 
                         'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine']
        y_raw = data[response_cols]

        print(f"Dataset shape - X: {X_raw.shape}, y: {y_raw.shape}")

    # Convert predictors to numpy array
    X = X_raw.values.astype(float)

    # Convert responses to binary (0/1)
    # Original data has categories: CL0-CL6 representing usage frequency
    # We'll convert to binary: 0 = Never/Over decade ago, 1 = Used recently
    def convert_to_binary(val):
        if isinstance(val, str):
            if val in ['CL0', 'CL1']:  # Never used or used over decade ago
                return 0
            else:  # Used in last decade/year/month/week/day
                return 1
        return val

    if isinstance(y_raw, pd.DataFrame):
        Y = y_raw.applymap(convert_to_binary).values.astype(int)
        response_names = y_raw.columns.tolist()
    else:
        Y = y_raw.astype(int)
        response_names = [f'Drug_{i}' for i in range(Y.shape[1])]

    if isinstance(X_raw, pd.DataFrame):
        predictor_names = X_raw.columns.tolist()
    else:
        predictor_names = [f'Pred_{i}' for i in range(X.shape[1])]

    print(f"\nProcessed data shapes:")
    print(f"  X (predictors): {X.shape}")
    print(f"  Y (responses): {Y.shape}")
    print(f"  Predictor names: {predictor_names}")
    print(f"  Response names: {response_names}")

    return X, Y, predictor_names, response_names


def preprocess_data(X, Y, standardize=True):
    '''
    Preprocess the data for logistic reduced rank regression

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_predictors)
        Predictor variables
    Y : array-like, shape (n_samples, n_responses)
        Binary response variables
    standardize : bool
        Whether to standardize predictors

    Returns:
    --------
    X_processed : array-like
        Processed predictors
    Y_processed : array-like
        Processed responses
    scaler : StandardScaler object or None
    '''
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)

    # Check for missing values
    n_missing_X = np.sum(np.isnan(X))
    n_missing_Y = np.sum(np.isnan(Y))

    print(f"Missing values in X: {n_missing_X}")
    print(f"Missing values in Y: {n_missing_Y}")

    # Handle missing values (if any)
    if n_missing_X > 0:
        print("Imputing missing values in X with column means...")
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_means[i]

    if n_missing_Y > 0:
        print("Removing rows with missing response values...")
        valid_rows = ~np.any(np.isnan(Y), axis=1)
        X = X[valid_rows]
        Y = Y[valid_rows]

    # Standardize predictors
    scaler = None
    if standardize:
        print("Standardizing predictors (mean=0, std=1)...")
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    else:
        X_processed = X.copy()

    Y_processed = Y.copy()

    print(f"\nFinal data shapes:")
    print(f"  X: {X_processed.shape}")
    print(f"  Y: {Y_processed.shape}")

    # Summary statistics
    print(f"\nResponse variable statistics:")
    for r in range(Y_processed.shape[1]):
        prevalence = np.mean(Y_processed[:, r])
        print(f"  Response {r}: prevalence = {prevalence:.3f}")

    return X_processed, Y_processed, scaler


#============================================================================
# PART 2: MM ALGORITHM FOR LOGISTIC REDUCED RANK REGRESSION
#============================================================================

class LogisticReducedRankRegression:
    '''
    Logistic Reduced Rank Regression using MM Algorithm

    Based on de Rooij (2024), Behaviormetrika

    Parameters:
    -----------
    rank : int
        Reduced rank (number of dimensions), typically 2 for visualization
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance based on deviance change
    verbose : bool
        Whether to print iteration details
    '''

    def __init__(self, rank=2, max_iter=100, tol=1e-6, verbose=True):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Parameters to be estimated
        self.m = None  # Intercepts, shape (R,)
        self.B = None  # Predictor coefficients, shape (P, S)
        self.V = None  # Response loadings, shape (R, S)
        self.A = None  # Full coefficient matrix, shape (P, R)

        # Training history
        self.deviances = []
        self.n_iter = 0
        self.converged = False

    def _logistic_prob(self, theta):
        '''
        Compute logistic probabilities from log-odds

        π = exp(θ) / (1 + exp(θ)) = 1 / (1 + exp(-θ))

        Uses numerically stable computation
        '''
        # Clip to avoid overflow
        theta_clipped = np.clip(theta, -500, 500)
        return 1.0 / (1.0 + np.exp(-theta_clipped))

    def _compute_deviance(self, Y, Pi):
        '''
        Compute deviance (negative log-likelihood × 2)

        Deviance = -2 * Σ_i Σ_r [y_ir log(π_ir) + (1-y_ir) log(1-π_ir)]
        '''
        # Clip probabilities to avoid log(0)
        Pi_clipped = np.clip(Pi, 1e-10, 1 - 1e-10)

        # Compute log-likelihood
        loglik = np.sum(Y * np.log(Pi_clipped) + (1 - Y) * np.log(1 - Pi_clipped))

        # Deviance = -2 * log-likelihood
        deviance = -2 * loglik

        return deviance

    def _generalized_svd(self, Z_centered, X, rank):
        '''
        Generalized Singular Value Decomposition

        Decomposes (X'X)^(-1/2) X' Z_centered

        Returns B and V such that Z_centered ≈ X B V'

        Parameters:
        -----------
        Z_centered : array, shape (N, R)
            Centered working responses
        X : array, shape (N, P)
            Predictor matrix
        rank : int
            Number of dimensions to keep

        Returns:
        --------
        B : array, shape (P, rank)
            Predictor coefficients
        V : array, shape (R, rank)
            Response loadings
        '''
        N, P = X.shape
        N, R = Z_centered.shape

        # Compute X'X
        XtX = X.T @ X

        # Compute (X'X)^(-1/2) using eigendecomposition
        # For numerical stability
        eigvals, eigvecs = linalg.eigh(XtX)
        eigvals_inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, 1e-10))
        XtX_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

        # Compute matrix to decompose: (X'X)^(-1/2) X' Z_centered
        M = XtX_inv_sqrt @ X.T @ Z_centered

        # Standard SVD
        U, s, Vt = linalg.svd(M, full_matrices=False)

        # Keep only rank dimensions
        U_rank = U[:, :rank]
        s_rank = s[:rank]
        V_rank = Vt[:rank, :].T

        # Compute B and V
        # B = (X'X)^(-1/2) U_rank
        B = XtX_inv_sqrt @ U_rank

        # V = V_rank * sqrt(N) * diag(s_rank) / sqrt(N) = V_rank * diag(s_rank)
        V = V_rank * s_rank

        return B, V

    def fit(self, X, Y):
        '''
        Fit logistic reduced rank regression model using MM algorithm

        Parameters:
        -----------
        X : array, shape (N, P)
            Predictor variables
        Y : array, shape (N, R)
            Binary response variables (0 or 1)

        Algorithm (from de Rooij, 2024):
        ---------------------------------
        1. Initialize m, B, V
        2. Repeat until convergence:
           a. Compute θ = 1 m' + X B V'
           b. Compute π = logistic(θ)
           c. Compute working response: Z = 1 m' + X B V' + 4(Y - π)
           d. Update m = (Z - X B V')' 1 / N
           e. Perform generalized SVD to update B and V
           f. Check convergence
        '''
        print("\n" + "="*60)
        print("FITTING LOGISTIC REDUCED RANK REGRESSION")
        print("="*60)

        N, P = X.shape
        N, R = Y.shape
        S = self.rank

        print(f"Data dimensions: N={N}, P={P}, R={R}")
        print(f"Reduced rank: S={S}")
        print(f"Parameters to estimate: {P*S + R*S + R} = {P*S}(B) + {R*S}(V) + {R}(m)")

        # Step 1: Initialize parameters
        print("\nInitializing parameters...")

        # Initialize intercepts with logit of marginal probabilities
        prevalences = np.mean(Y, axis=0)
        prevalences_clipped = np.clip(prevalences, 0.01, 0.99)
        self.m = np.log(prevalences_clipped / (1 - prevalences_clipped))

        # Initialize B and V with small random values
        self.B = np.random.randn(P, S) * 0.1
        self.V = np.random.randn(R, S) * 0.1

        print(f"  m shape: {self.m.shape}")
        print(f"  B shape: {self.B.shape}")
        print(f"  V shape: {self.V.shape}")

        # Initial deviance
        theta_init = np.outer(np.ones(N), self.m) + X @ self.B @ self.V.T
        Pi_init = self._logistic_prob(theta_init)
        deviance_init = self._compute_deviance(Y, Pi_init)
        self.deviances.append(deviance_init)

        print(f"  Initial deviance: {deviance_init:.4f}")

        # Step 2: MM algorithm iterations
        print("\nStarting MM algorithm iterations...")
        start_time = time.time()

        for iteration in range(self.max_iter):
            # (a) Compute current log-odds: θ = 1 m' + X B V'
            theta = np.outer(np.ones(N), self.m) + X @ self.B @ self.V.T

            # (b) Compute probabilities: π = 1 / (1 + exp(-θ))
            Pi = self._logistic_prob(theta)

            # (c) Compute working response: Z = 1 m' + X B V' + 4(Y - π)
            # This is the key majorization step
            Z = theta + 4 * (Y - Pi)

            # (d) Update intercepts: m = (Z - X B V')' 1 / N
            predicted_responses = X @ self.B @ self.V.T
            self.m = np.mean(Z - predicted_responses, axis=0)

            # (e) Center Z and perform generalized SVD
            Z_centered = Z - np.outer(np.ones(N), self.m)
            self.B, self.V = self._generalized_svd(Z_centered, X, S)

            # (f) Compute deviance and check convergence
            theta_new = np.outer(np.ones(N), self.m) + X @ self.B @ self.V.T
            Pi_new = self._logistic_prob(theta_new)
            deviance_new = self._compute_deviance(Y, Pi_new)
            self.deviances.append(deviance_new)

            # Check convergence
            deviance_change = self.deviances[-2] - deviance_new

            if self.verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"  Iteration {iteration+1:3d}: Deviance = {deviance_new:10.4f}, "
                      f"Change = {deviance_change:10.6f}")

            # Convergence check
            if deviance_change < self.tol and deviance_change >= 0:
                self.converged = True
                self.n_iter = iteration + 1
                print(f"\nConverged after {self.n_iter} iterations!")
                break

            if deviance_change < 0:
                print(f"\nWarning: Deviance increased at iteration {iteration+1}")

        if not self.converged:
            self.n_iter = self.max_iter
            print(f"\nWarning: Did not converge after {self.max_iter} iterations")

        elapsed_time = time.time() - start_time

        # Compute full coefficient matrix A = B V'
        self.A = self.B @ self.V.T

        # Final results
        print("\n" + "-"*60)
        print("FITTING COMPLETE")
        print("-"*60)
        print(f"Number of iterations: {self.n_iter}")
        print(f"Final deviance: {self.deviances[-1]:.4f}")
        print(f"Deviance reduction: {self.deviances[0] - self.deviances[-1]:.4f}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        return self

    def predict_proba(self, X):
        '''
        Predict probabilities for new data

        Parameters:
        -----------
        X : array, shape (N_new, P)
            New predictor data

        Returns:
        --------
        Pi : array, shape (N_new, R)
            Predicted probabilities
        '''
        N = X.shape[0]
        theta = np.outer(np.ones(N), self.m) + X @ self.B @ self.V.T
        Pi = self._logistic_prob(theta)
        return Pi

    def predict(self, X, threshold=0.5):
        '''
        Predict binary outcomes for new data

        Parameters:
        -----------
        X : array, shape (N_new, P)
            New predictor data
        threshold : float
            Classification threshold

        Returns:
        --------
        Y_pred : array, shape (N_new, R)
            Predicted binary outcomes
        '''
        Pi = self.predict_proba(X)
        Y_pred = (Pi >= threshold).astype(int)
        return Y_pred

    def get_object_scores(self, X):
        '''
        Compute object scores (positions in reduced space)

        U = X B

        Parameters:
        -----------
        X : array, shape (N, P)
            Predictor data

        Returns:
        --------
        U : array, shape (N, S)
            Object scores
        '''
        return X @ self.B




#============================================================================
# PART 3: VISUALIZATION METHODS - TYPE I, TYPE D, AND HYBRID TRIPLOTS
#============================================================================

class TriplotVisualizer:
    '''
    Visualization methods for logistic reduced rank regression

    Implements three types of triplots:
    - Type I: Inner product representation with variable axes
    - Type D: Distance representation with category points
    - Hybrid: Combination of Type I and Type D
    '''

    def __init__(self, model, X, Y, predictor_names, response_names):
        '''
        Parameters:
        -----------
        model : LogisticReducedRankRegression
            Fitted model
        X : array, shape (N, P)
            Predictor data
        Y : array, shape (N, R)
            Response data
        predictor_names : list
            Names of predictors
        response_names : list
            Names of responses
        '''
        self.model = model
        self.X = X
        self.Y = Y
        self.predictor_names = predictor_names
        self.response_names = response_names

        # Compute object scores
        self.U = model.get_object_scores(X)

        # For Type D: compute category points
        self._compute_category_points()

    def _compute_category_points(self):
        '''
        Compute category points for Type D representation

        Each response variable has two points: w_r0 (no) and w_r1 (yes)

        Based on: W = A_l L + A_k K
        where K = -V/2 and L_rs = m_r * K_rs / (2 * sum_s K_rs^2)
        '''
        R, S = self.model.V.shape

        # K matrix: discriminatory power
        self.K = -self.model.V / 2

        # L matrix: category point locations
        self.L = np.zeros((R, S))
        for r in range(R):
            k_squared_sum = np.sum(self.K[r, :]**2)
            if k_squared_sum > 1e-10:
                self.L[r, :] = self.model.m[r] * self.K[r, :] / (2 * k_squared_sum)

        # Category points for "no" (0) and "yes" (1)
        self.W0 = self.L + self.K  # Category 0 points
        self.W1 = self.L - self.K  # Category 1 points

    def plot_type_I_triplot(self, dims=[0, 1], figsize=(12, 10), 
                           n_points=200, prob_levels=[0.1, 0.3, 0.5, 0.7, 0.9]):
        '''
        Plot Type I triplot (inner product representation)

        Features:
        - Object points
        - Predictor variable axes with unit markers
        - Response variable axes with probability markers

        Parameters:
        -----------
        dims : list
            Which dimensions to plot [dim1, dim2]
        figsize : tuple
            Figure size
        n_points : int
            Number of object points to plot (for clarity)
        prob_levels : list
            Probability levels to mark on response axes
        '''
        dim1, dim2 = dims

        fig, ax = plt.subplots(figsize=figsize)

        # Plot object points (sample if too many)
        N = self.U.shape[0]
        if N > n_points:
            indices = np.random.choice(N, n_points, replace=False)
            U_plot = self.U[indices, :]
        else:
            U_plot = self.U

        ax.scatter(U_plot[:, dim1], U_plot[:, dim2], 
                  alpha=0.3, s=20, c='gray', label='Objects')

        # Plot predictor variable axes
        max_coord = np.max(np.abs(U_plot))
        scale_factor = max_coord * 1.2

        for p, name in enumerate(self.predictor_names):
            # Direction of axis
            direction = self.model.B[p, dims]
            direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

            # Draw axis
            ax.arrow(0, 0, direction_norm[0] * scale_factor, direction_norm[1] * scale_factor,
                    head_width=0.1, head_length=0.2, fc='blue', ec='blue', 
                    alpha=0.6, linestyle='--', linewidth=1.5)

            # Label at end of axis
            ax.text(direction_norm[0] * scale_factor * 1.1, 
                   direction_norm[1] * scale_factor * 1.1,
                   name, color='blue', fontsize=10, fontweight='bold',
                   ha='center', va='center')

        # Plot response variable axes with probability markers
        for r, name in enumerate(self.response_names):
            # Direction of axis
            direction = self.model.V[r, dims]
            direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

            # Draw axis
            ax.arrow(0, 0, direction_norm[0] * scale_factor, direction_norm[1] * scale_factor,
                    head_width=0.1, head_length=0.2, fc='green', ec='green',
                    alpha=0.6, linestyle='-', linewidth=2)

            # Add probability markers
            for prob in prob_levels:
                # Compute position of probability marker
                # λ = log(π/(1-π)) - m_r
                lambda_val = np.log(prob / (1 - prob)) - self.model.m[r]
                v_norm_sq = np.sum(self.model.V[r, :]**2)
                if v_norm_sq > 1e-10:
                    marker_pos = (lambda_val / v_norm_sq) * self.model.V[r, dims]

                    # Only plot if within bounds
                    if np.abs(marker_pos[0]) < scale_factor and np.abs(marker_pos[1]) < scale_factor:
                        ax.plot(marker_pos[0], marker_pos[1], 'o', 
                               color='green', markersize=4, alpha=0.7)
                        if prob == 0.5:  # Label the 0.5 marker
                            ax.text(marker_pos[0] * 1.05, marker_pos[1] * 1.05, 
                                   f'{prob}', color='green', fontsize=7)

            # Label at end of axis
            ax.text(direction_norm[0] * scale_factor * 1.15, 
                   direction_norm[1] * scale_factor * 1.15,
                   name, color='green', fontsize=10, fontweight='bold',
                   ha='center', va='center', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.7))

        # Formatting
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel(f'Dimension {dim1+1}', fontsize=12)
        ax.set_ylabel(f'Dimension {dim2+1}', fontsize=12)
        ax.set_title('Type I Triplot\n(Inner Product Representation)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

        plt.tight_layout()
        return fig, ax

    def plot_type_D_triplot(self, dims=[0, 1], figsize=(12, 10), n_points=200):
        '''
        Plot Type D triplot (distance representation)

        Features:
        - Object points
        - Predictor variable axes
        - Response category points (two points per response)

        Parameters:
        -----------
        dims : list
            Which dimensions to plot
        figsize : tuple
            Figure size
        n_points : int
            Number of object points to plot
        '''
        dim1, dim2 = dims

        fig, ax = plt.subplots(figsize=figsize)

        # Plot object points
        N = self.U.shape[0]
        if N > n_points:
            indices = np.random.choice(N, n_points, replace=False)
            U_plot = self.U[indices, :]
        else:
            U_plot = self.U

        ax.scatter(U_plot[:, dim1], U_plot[:, dim2], 
                  alpha=0.3, s=20, c='gray', label='Objects')

        # Plot predictor variable axes (same as Type I)
        max_coord = np.max(np.abs(np.vstack([U_plot, self.W0[:, dims], self.W1[:, dims]])))
        scale_factor = max_coord * 1.2

        for p, name in enumerate(self.predictor_names):
            direction = self.model.B[p, dims]
            direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

            ax.arrow(0, 0, direction_norm[0] * scale_factor, direction_norm[1] * scale_factor,
                    head_width=0.1, head_length=0.2, fc='blue', ec='blue',
                    alpha=0.6, linestyle='--', linewidth=1.5)

            ax.text(direction_norm[0] * scale_factor * 1.1,
                   direction_norm[1] * scale_factor * 1.1,
                   name, color='blue', fontsize=10, fontweight='bold',
                   ha='center', va='center')

        # Plot response category points
        for r, name in enumerate(self.response_names):
            # Plot category 0 point (no)
            ax.scatter(self.W0[r, dim1], self.W0[r, dim2], 
                      s=100, c='red', marker='s', edgecolors='black',
                      linewidths=2, alpha=0.8, label=f'{name}_no' if r == 0 else '')

            # Plot category 1 point (yes)
            ax.scatter(self.W1[r, dim1], self.W1[r, dim2],
                      s=100, c='lime', marker='o', edgecolors='black',
                      linewidths=2, alpha=0.8, label=f'{name}_yes' if r == 0 else '')

            # Draw line connecting category points
            ax.plot([self.W0[r, dim1], self.W1[r, dim2]],
                   [self.W0[r, dim2], self.W1[r, dim2]],
                   'g-', linewidth=2, alpha=0.5)

            # Label
            midpoint = (self.W0[r, dims] + self.W1[r, dims]) / 2
            ax.text(midpoint[0], midpoint[1], name, 
                   color='green', fontsize=9, fontweight='bold',
                   ha='center', va='bottom', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.7))

        # Formatting
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel(f'Dimension {dim1+1}', fontsize=12)
        ax.set_ylabel(f'Dimension {dim2+1}', fontsize=12)
        ax.set_title('Type D Triplot\n(Distance Representation)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

        # Legend
        ax.legend(['Objects', 'Category 0 (No)', 'Category 1 (Yes)'], 
                 loc='upper right', fontsize=10)

        plt.tight_layout()
        return fig, ax

    def plot_hybrid_triplot(self, dims=[0, 1], figsize=(14, 12), n_points=200):
        '''
        Plot Hybrid triplot (combining Type I and Type D)

        Features:
        - Object points
        - Predictor variable axes
        - Response variable axes with solid part showing category points
        - Length of solid part indicates discriminatory power

        Parameters:
        -----------
        dims : list
            Which dimensions to plot
        figsize : tuple
            Figure size
        n_points : int
            Number of object points to plot
        '''
        dim1, dim2 = dims

        fig, ax = plt.subplots(figsize=figsize)

        # Plot object points
        N = self.U.shape[0]
        if N > n_points:
            indices = np.random.choice(N, n_points, replace=False)
            U_plot = self.U[indices, :]
        else:
            U_plot = self.U

        ax.scatter(U_plot[:, dim1], U_plot[:, dim2],
                  alpha=0.3, s=20, c='gray', label='Objects', zorder=1)

        # Compute scale
        max_coord = np.max(np.abs(np.vstack([U_plot, self.W0[:, dims], self.W1[:, dims]])))
        scale_factor = max_coord * 1.2

        # Plot predictor variable axes
        for p, name in enumerate(self.predictor_names):
            direction = self.model.B[p, dims]
            direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

            # Dotted line
            ax.plot([0, direction_norm[0] * scale_factor],
                   [0, direction_norm[1] * scale_factor],
                   'b--', linewidth=1.5, alpha=0.6, zorder=2)

            # Label
            ax.text(direction_norm[0] * scale_factor * 1.1,
                   direction_norm[1] * scale_factor * 1.1,
                   name, color='blue', fontsize=10, fontweight='bold',
                   ha='center', va='center', bbox=dict(boxstyle='round',
                   facecolor='lightblue', alpha=0.7), zorder=5)

        # Plot response variable axes (hybrid style)
        for r, name in enumerate(self.response_names):
            # Endpoints are the category points
            w0 = self.W0[r, dims]
            w1 = self.W1[r, dims]
            midpoint = (w0 + w1) / 2

            # Direction
            direction = self.model.V[r, dims]
            direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

            # Dotted line from origin to w0
            ax.plot([0, w0[0]], [0, w0[1]], 
                   'g:', linewidth=1.5, alpha=0.5, zorder=2)

            # Solid line from w0 to w1 (discriminatory power)
            ax.plot([w0[0], w1[0]], [w0[1], w1[1]],
                   'g-', linewidth=3, alpha=0.8, zorder=3)

            # Dotted line from w1 to edge
            end_point = direction_norm * scale_factor
            ax.plot([w1[0], end_point[0]], [w1[1], end_point[1]],
                   'g:', linewidth=1.5, alpha=0.5, zorder=2)

            # Mark category points
            ax.scatter(w0[0], w0[1], s=80, c='red', marker='s',
                      edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)
            ax.scatter(w1[0], w1[1], s=80, c='lime', marker='o',
                      edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)

            # Label (on the positive side)
            label_pos = end_point * 1.15
            ax.text(label_pos[0], label_pos[1], name,
                   color='green', fontsize=10, fontweight='bold',
                   ha='center', va='center', bbox=dict(boxstyle='round',
                   facecolor='lightgreen', alpha=0.8), zorder=5)

        # Formatting
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel(f'Dimension {dim1+1}', fontsize=12)
        ax.set_ylabel(f'Dimension {dim2+1}', fontsize=12)
        ax.set_title('Hybrid Triplot\n(Combined Inner Product and Distance Representation)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

        # Legend
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='gray', label='Objects',
                  markerfacecolor='gray', markersize=6, linestyle=''),
            Line2D([0], [0], color='blue', linestyle='--', label='Predictors',
                  linewidth=2),
            Line2D([0], [0], color='green', linestyle='-', label='Responses (solid=discriminatory power)',
                  linewidth=3),
            Line2D([0], [0], marker='s', color='red', label='Category 0 (No)',
                  markerfacecolor='red', markersize=8, linestyle=''),
            Line2D([0], [0], marker='o', color='lime', label='Category 1 (Yes)',
                  markerfacecolor='lime', markersize=8, linestyle='')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.tight_layout()
        return fig, ax

    def plot_all_triplots(self, dims=[0, 1], n_points=200):
        '''
        Plot all three types of triplots side by side
        '''
        fig = plt.figure(figsize=(20, 6))

        # Type I
        ax1 = fig.add_subplot(131)
        self.plot_type_I_triplot(dims=dims, n_points=n_points)

        # Type D
        ax2 = fig.add_subplot(132)
        self.plot_type_D_triplot(dims=dims, n_points=n_points)

        # Hybrid
        ax3 = fig.add_subplot(133)
        self.plot_hybrid_triplot(dims=dims, n_points=n_points)

        plt.tight_layout()
        return fig




#============================================================================
# PART 4: EVALUATION METRICS
#============================================================================

class ModelEvaluator:
    '''
    Evaluation metrics for logistic reduced rank regression

    Implements:
    - Quality of representation (Qr) for each response variable
    - Deviance comparisons
    - Timing benchmarks
    - Classification accuracy
    '''

    def __init__(self, model, X, Y, response_names):
        '''
        Parameters:
        -----------
        model : LogisticReducedRankRegression
            Fitted model
        X : array, shape (N, P)
            Predictor data
        Y : array, shape (N, R)
            Response data
        response_names : list
            Names of responses
        '''
        self.model = model
        self.X = X
        self.Y = Y
        self.response_names = response_names
        self.N, self.P = X.shape
        self.N, self.R = Y.shape

    def compute_quality_of_representation(self, verbose=True):
        '''
        Compute quality of representation (Qr) for each response variable

        Qr = (D(0,r) - D_r) / (D(0,r) - D_lr)

        where:
        - D(0,r): deviance of intercept-only model for response r
        - D_r: deviance of reduced rank model for response r
        - D_lr: deviance of full logistic regression for response r

        Qr ranges from 0 (very bad fit) to 1 (no loss due to rank restriction)

        Returns:
        --------
        qr_values : array, shape (R,)
            Quality of representation for each response
        '''
        if verbose:
            print("\n" + "="*60)
            print("COMPUTING QUALITY OF REPRESENTATION (Qr)")
            print("="*60)

        qr_values = np.zeros(self.R)

        # Get predictions from reduced rank model
        Pi_reduced = self.model.predict_proba(self.X)

        for r in range(self.R):
            # D(0,r): Intercept-only model deviance
            prevalence = np.mean(self.Y[:, r])
            prevalence_clipped = np.clip(prevalence, 1e-10, 1 - 1e-10)
            pi_intercept_only = np.full(self.N, prevalence_clipped)

            loglik_0 = np.sum(self.Y[:, r] * np.log(pi_intercept_only) + 
                             (1 - self.Y[:, r]) * np.log(1 - pi_intercept_only))
            D_0_r = -2 * loglik_0

            # D_r: Reduced rank model deviance for response r
            pi_reduced_r = np.clip(Pi_reduced[:, r], 1e-10, 1 - 1e-10)
            loglik_reduced = np.sum(self.Y[:, r] * np.log(pi_reduced_r) + 
                                   (1 - self.Y[:, r]) * np.log(1 - pi_reduced_r))
            D_r = -2 * loglik_reduced

            # D_lr: Full logistic regression deviance for response r
            lr = LogisticRegression(max_iter=1000, penalty=None)
            lr.fit(self.X, self.Y[:, r])
            pi_full_r = np.clip(lr.predict_proba(self.X)[:, 1], 1e-10, 1 - 1e-10)
            loglik_full = np.sum(self.Y[:, r] * np.log(pi_full_r) + 
                                (1 - self.Y[:, r]) * np.log(1 - pi_full_r))
            D_lr = -2 * loglik_full

            # Compute Qr
            denominator = D_0_r - D_lr
            if abs(denominator) > 1e-10:
                qr_values[r] = (D_0_r - D_r) / denominator
            else:
                qr_values[r] = 1.0  # Perfect fit

            # Clip to [0, 1] range
            qr_values[r] = np.clip(qr_values[r], 0.0, 1.0)

            if verbose:
                print(f"  {self.response_names[r]:15s}: Qr = {qr_values[r]:.4f} "
                      f"(D_0={D_0_r:8.2f}, D_r={D_r:8.2f}, D_lr={D_lr:8.2f})")

        if verbose:
            print(f"\nMean Qr: {np.mean(qr_values):.4f}")
            print(f"Min Qr:  {np.min(qr_values):.4f} ({self.response_names[np.argmin(qr_values)]})")
            print(f"Max Qr:  {np.max(qr_values):.4f} ({self.response_names[np.argmax(qr_values)]})")

        return qr_values

    def compute_classification_accuracy(self, threshold=0.5, verbose=True):
        '''
        Compute classification accuracy for each response variable

        Returns:
        --------
        accuracies : array, shape (R,)
            Classification accuracy for each response
        '''
        if verbose:
            print("\n" + "="*60)
            print("COMPUTING CLASSIFICATION ACCURACY")
            print("="*60)

        Y_pred = self.model.predict(self.X, threshold=threshold)

        accuracies = np.zeros(self.R)
        for r in range(self.R):
            accuracies[r] = np.mean(Y_pred[:, r] == self.Y[:, r])
            if verbose:
                print(f"  {self.response_names[r]:15s}: {accuracies[r]:.4f}")

        if verbose:
            print(f"\nMean accuracy: {np.mean(accuracies):.4f}")

        return accuracies

    def plot_deviance_convergence(self, figsize=(10, 6)):
        '''
        Plot deviance convergence over iterations
        '''
        fig, ax = plt.subplots(figsize=figsize)

        iterations = range(len(self.model.deviances))
        ax.plot(iterations, self.model.deviances, 'b-', linewidth=2)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Deviance', fontsize=12)
        ax.set_title('Deviance Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add text with final deviance
        final_deviance = self.model.deviances[-1]
        ax.text(0.95, 0.95, f'Final Deviance: {final_deviance:.2f}\nIterations: {self.model.n_iter}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig, ax

    def compare_with_full_model(self, verbose=True):
        '''
        Compare reduced rank model with full (unrestricted) logistic regression

        Returns:
        --------
        comparison_dict : dict
            Dictionary with comparison metrics
        '''
        if verbose:
            print("\n" + "="*60)
            print("COMPARING WITH FULL MODEL")
            print("="*60)

        # Fit full model for each response
        deviance_full_total = 0
        n_params_full = self.P * self.R + self.R

        for r in range(self.R):
            lr = LogisticRegression(max_iter=1000, penalty=None)
            lr.fit(self.X, self.Y[:, r])
            pi_full = np.clip(lr.predict_proba(self.X)[:, 1], 1e-10, 1 - 1e-10)
            loglik_full = np.sum(self.Y[:, r] * np.log(pi_full) + 
                                (1 - self.Y[:, r]) * np.log(1 - pi_full))
            deviance_full_total += -2 * loglik_full

        # Reduced rank model
        deviance_reduced = self.model.deviances[-1]
        n_params_reduced = self.P * self.model.rank + self.R * self.model.rank + self.R

        # Compute AIC and BIC
        aic_full = deviance_full_total + 2 * n_params_full
        aic_reduced = deviance_reduced + 2 * n_params_reduced

        bic_full = deviance_full_total + np.log(self.N) * n_params_full
        bic_reduced = deviance_reduced + np.log(self.N) * n_params_reduced

        if verbose:
            print(f"\nFull Model (Unrestricted):")
            print(f"  Deviance: {deviance_full_total:.4f}")
            print(f"  Parameters: {n_params_full}")
            print(f"  AIC: {aic_full:.4f}")
            print(f"  BIC: {bic_full:.4f}")

            print(f"\nReduced Rank Model (Rank {self.model.rank}):")
            print(f"  Deviance: {deviance_reduced:.4f}")
            print(f"  Parameters: {n_params_reduced}")
            print(f"  AIC: {aic_reduced:.4f}")
            print(f"  BIC: {bic_reduced:.4f}")

            print(f"\nDifferences:")
            print(f"  Deviance increase: {deviance_reduced - deviance_full_total:.4f}")
            print(f"  Parameter reduction: {n_params_full - n_params_reduced} ({100*(1-n_params_reduced/n_params_full):.1f}%)")
            print(f"  AIC {'decrease' if aic_reduced < aic_full else 'increase'}: {abs(aic_reduced - aic_full):.4f}")
            print(f"  BIC {'decrease' if bic_reduced < bic_full else 'increase'}: {abs(bic_reduced - bic_full):.4f}")

        comparison = {
            'deviance_full': deviance_full_total,
            'deviance_reduced': deviance_reduced,
            'n_params_full': n_params_full,
            'n_params_reduced': n_params_reduced,
            'aic_full': aic_full,
            'aic_reduced': aic_reduced,
            'bic_full': bic_full,
            'bic_reduced': bic_reduced
        }

        return comparison


#============================================================================
# PART 5: EXAMPLE USAGE AND TESTING
#============================================================================

def main_analysis():
    '''
    Main analysis function demonstrating the complete workflow
    '''
    print("="*70)
    print(" LOGISTIC REDUCED RANK REGRESSION - COMPLETE ANALYSIS")
    print(" Based on de Rooij (2024), Behaviormetrika")
    print("="*70)

    # Step 1: Load data
    print("\nSTEP 1: LOADING DATA")
    print("-"*70)
    X, Y, predictor_names, response_names = load_drug_consumption_data()

    # Step 2: Preprocess data
    print("\nSTEP 2: PREPROCESSING")
    print("-"*70)
    X_processed, Y_processed, scaler = preprocess_data(X, Y, standardize=True)

    # Step 3: Fit model
    print("\nSTEP 3: FITTING MODEL")
    print("-"*70)
    model = LogisticReducedRankRegression(rank=2, max_iter=100, tol=1e-6, verbose=True)
    model.fit(X_processed, Y_processed)

    # Step 4: Evaluate model
    print("\nSTEP 4: EVALUATING MODEL")
    print("-"*70)
    evaluator = ModelEvaluator(model, X_processed, Y_processed, response_names)

    # Quality of representation
    qr_values = evaluator.compute_quality_of_representation(verbose=True)

    # Classification accuracy
    accuracies = evaluator.compute_classification_accuracy(verbose=True)

    # Compare with full model
    comparison = evaluator.compare_with_full_model(verbose=True)

    # Plot deviance convergence
    print("\nPlotting deviance convergence...")
    evaluator.plot_deviance_convergence()
    plt.savefig('deviance_convergence.png', dpi=300, bbox_inches='tight')
    print("Saved: deviance_convergence.png")

    # Step 5: Visualizations
    print("\nSTEP 5: CREATING VISUALIZATIONS")
    print("-"*70)
    visualizer = TriplotVisualizer(model, X_processed, Y_processed, 
                                   predictor_names, response_names)

    # Type I triplot
    print("Creating Type I triplot...")
    visualizer.plot_type_I_triplot()
    plt.savefig('triplot_type_I.png', dpi=300, bbox_inches='tight')
    print("Saved: triplot_type_I.png")

    # Type D triplot
    print("Creating Type D triplot...")
    visualizer.plot_type_D_triplot()
    plt.savefig('triplot_type_D.png', dpi=300, bbox_inches='tight')
    print("Saved: triplot_type_D.png")

    # Hybrid triplot
    print("Creating Hybrid triplot...")
    visualizer.plot_hybrid_triplot()
    plt.savefig('triplot_hybrid.png', dpi=300, bbox_inches='tight')
    print("Saved: triplot_hybrid.png")

    # Step 6: Summary
    print("\n" + "="*70)
    print(" ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nModel: Logistic Reduced Rank Regression (Rank {model.rank})")
    print(f"Data: {X_processed.shape[0]} observations, {X_processed.shape[1]} predictors, {Y_processed.shape[1]} responses")
    print(f"\nConvergence: {model.converged} (iterations: {model.n_iter})")
    print(f"Final deviance: {model.deviances[-1]:.4f}")
    print(f"Deviance reduction: {model.deviances[0] - model.deviances[-1]:.4f}")
    print(f"\nMean quality of representation: {np.mean(qr_values):.4f}")
    print(f"Mean classification accuracy: {np.mean(accuracies):.4f}")
    print(f"\nParameter reduction vs full model: {comparison['n_params_full'] - comparison['n_params_reduced']}")
    print(f"  ({100*(1-comparison['n_params_reduced']/comparison['n_params_full']):.1f}% reduction)")

    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)

    return model, evaluator, visualizer

# Run the analysis
if __name__ == '__main__':
    model, evaluator, visualizer = main_analysis()
