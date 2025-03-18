"""
UKF.py

This module implements an Unscented Kalman Filter (UKF) for state estimation in nonlinear systems.
The UKF uses the unscented transformation to propagate sigma points through a process model,
updating state estimates with noisy measurements.

Dependencies:
    - numpy: For numerical computations, matrix operations, and linear algebra.
"""

import numpy as np


class UnscentedKalmanFilter:
    """
    An Unscented Kalman Filter (UKF) for nonlinear state estimation.

    The UKF approximates the state distribution using sigma points, avoiding linearization.
    It predicts the next state using a provided process model and updates estimates with measurements.

    Attributes:
        n_dim (int): Number of state dimensions.
        Q (np.ndarray): Process noise covariance matrix, shape (n_dim, n_dim).
        R (np.ndarray): Measurement noise covariance matrix, shape (m, m) where m is measurement dimension.
        alpha (float): Sigma point spread parameter.
        beta (float): Parameter incorporating prior distribution knowledge (2 is optimal for Gaussian).
        kappa (float): Secondary scaling parameter.
        lambda_ (float): Scaling factor for sigma point generation.
        W_m (np.ndarray): Weights for mean calculation, shape (2 * n_dim + 1,).
        W_c (np.ndarray): Weights for covariance calculation, shape (2 * n_dim + 1,).
        x (np.ndarray): Current state estimate, shape (n_dim,).
        P (np.ndarray): Current state covariance, shape (n_dim, n_dim).
    """

    def __init__(self, n_dim, process_noise, measurement_noise, alpha=1e-3, beta=2, kappa=0):
        """
        Initialize the Unscented Kalman Filter.

        Args:
            n_dim (int): Number of state dimensions (e.g., 3 for Lorenz system: x, y, z).
            process_noise (np.ndarray): Process noise covariance matrix (Q), shape (n_dim, n_dim).
            measurement_noise (np.ndarray): Measurement noise covariance matrix (R), shape (m, m).
            alpha (float, optional): Sigma point spread parameter (default: 1e-3).
            beta (float, optional): Parameter for prior distribution (default: 2, optimal for Gaussian).
            kappa (float, optional): Secondary scaling parameter (default: 0).
        """
        # Store core parameters
        self.n_dim = n_dim
        self.Q = process_noise
        self.R = measurement_noise

        # UKF tuning parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (n_dim + kappa) - n_dim  # Scaling factor for sigma points

        # Initialize sigma point weights
        self.W_m = np.zeros(2 * n_dim + 1)  # Weights for mean
        self.W_c = np.zeros(2 * n_dim + 1)  # Weights for covariance

        # First weight (central sigma point)
        denom = n_dim + self.lambda_
        self.W_m[0] = self.lambda_ / denom
        self.W_c[0] = self.lambda_ / denom + (1 - alpha**2 + beta)

        # Remaining weights (symmetric sigma points)
        for i in range(1, 2 * n_dim + 1):
            self.W_m[i] = 1 / (2 * denom)
            self.W_c[i] = 1 / (2 * denom)

        # Initialize state and covariance
        self.x = np.zeros(n_dim)      # Initial state estimate
        self.P = np.eye(n_dim)        # Initial covariance estimate

    def generate_sigma_points(self):
        """
        Generate sigma points around the current state estimate using the unscented transformation.

        Returns:
            np.ndarray: Sigma points, shape (2 * n_dim + 1, n_dim).

        Raises:
            numpy.linalg.LinAlgError: If P is not positive definite (Cholesky fails).
        """
        sigma_points = np.zeros((2 * self.n_dim + 1, self.n_dim))
        sigma_points[0] = self.x  # Central point is the mean

        # Compute scaled covariance square root via Cholesky decomposition
        scale_factor = self.n_dim + self.lambda_
        sqrt_P = np.linalg.cholesky(scale_factor * self.P)  # Shape (n_dim, n_dim)

        # Generate symmetric sigma points
        for i in range(self.n_dim):
            sigma_points[i + 1] = self.x + sqrt_P[i]             # Positive offset
            sigma_points[self.n_dim + i + 1] = self.x - sqrt_P[i]  # Negative offset

        return sigma_points

    def predict(self, process_model):
        """
        Predict the next state using the provided process model.

        Propagates sigma points through the process model and updates the state and covariance.

        Args:
            process_model (callable): Function mapping state to next state, f(x) -> x_next.

        Returns:
            np.ndarray: Predicted sigma points, shape (2 * n_dim + 1, n_dim).
        """
        # Generate sigma points around current state
        sigma_points = self.generate_sigma_points()
        
        # Propagate sigma points through the process model
        sigma_points_pred = np.array([process_model(sp) for sp in sigma_points])

        # Update state estimate (weighted mean)
        self.x = np.sum(self.W_m[:, None] * sigma_points_pred, axis=0)  # Shape (n_dim,)

        # Update covariance estimate
        self.P = self.Q.copy()  # Start with process noise
        for i in range(2 * self.n_dim + 1):
            diff = sigma_points_pred[i] - self.x
            self.P += self.W_c[i] * np.outer(diff, diff)  # Shape (n_dim, n_dim)

        return sigma_points_pred

    def update(self, sigma_points_pred, measurement, measurement_model):
        """
        Update the state estimate with a new measurement.

        Uses predicted sigma points to compute the Kalman gain and refine state and covariance.

        Args:
            sigma_points_pred (np.ndarray): Predicted sigma points, shape (2 * n_dim + 1, n_dim).
            measurement (np.ndarray): Observed measurement, shape (measurement_dim,).
            measurement_model (callable): Function mapping state to measurement space, h(x) -> z.
        """
        # Transform predicted sigma points into measurement space
        sigma_points_meas = np.array([measurement_model(sp) for sp in sigma_points_pred])

        # Compute predicted measurement mean
        z_pred = np.sum(self.W_m[:, None] * sigma_points_meas, axis=0)  # Shape (measurement_dim,)

        # Compute measurement covariance
        P_zz = self.R.copy()  # Start with measurement noise
        for i in range(2 * self.n_dim + 1):
            diff = sigma_points_meas[i] - z_pred
            P_zz += self.W_c[i] * np.outer(diff, diff)  # Shape (measurement_dim, measurement_dim)

        # Compute cross-covariance between state and measurement
        P_xz = np.zeros((self.n_dim, len(measurement)))
        for i in range(2 * self.n_dim + 1):
            diff_x = sigma_points_pred[i] - self.x
            diff_z = sigma_points_meas[i] - z_pred
            P_xz += self.W_c[i] * np.outer(diff_x, diff_z)  # Shape (n_dim, measurement_dim)

        # Compute Kalman gain
        K = np.dot(P_xz, np.linalg.inv(P_zz))  # Shape (n_dim, measurement_dim)

        # Update state estimate
        innovation = measurement - z_pred
        self.x += np.dot(K, innovation)  # Shape (n_dim,)

        # Update covariance
        self.P -= np.dot(K, P_zz).dot(K.T)  # Shape (n_dim, n_dim)

    def get_state(self):
        """
        Retrieve the current state estimate.

        Returns:
            np.ndarray: Current state estimate, shape (n_dim,).
        """
        return self.x


if __name__ == "__main__":
    # Optional: Add a simple test or example usage here
    pass