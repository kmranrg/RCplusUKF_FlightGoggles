"""
RCplusUKF.py

This module defines the RC_UKF class, which integrates a single Reservoir Computer (RC)
with an Unscented Kalman Filter (UKF) for data-driven state estimation of nonlinear systems,
such as the Lorenz system. The RC serves as the process model, trained on time-series data,
while the UKF refines predictions using noisy measurements.

Dependencies:
    - numpy: For numerical computations and array operations.
    - RC: Custom ReservoirComputer class for reservoir computing.
    - UKF: Custom UnscentedKalmanFilter class for UKF implementation.
"""

import numpy as np
from RC import ReservoirComputer
from UKF import UnscentedKalmanFilter


class RC_UKF:
    """
    A hybrid framework combining a single Reservoir Computer (RC) with an Unscented Kalman Filter (UKF).

    The RC is trained on a time series to learn system dynamics, then used as the process model
    within the UKF to predict states. The UKF assimilates noisy measurements to refine these predictions.

    Attributes:
        rc (ReservoirComputer): The underlying RC instance for state prediction.
        ukf (UnscentedKalmanFilter): The UKF instance for filtering noisy measurements.
        reservoir_state (np.ndarray): Current internal state of the reservoir, shape (n_reservoir, 1).
    """

    def __init__(
        self,
        n_inputs,
        n_reservoir,
        process_noise,
        measurement_noise,
        rc_params=None,
        ukf_params=None
    ):
        """
        Initialize the RC_UKF framework.

        Args:
            n_inputs (int): Dimension of the system state (e.g., 3 for Lorenz system: x, y, z).
            n_reservoir (int): Number of neurons in the RC reservoir.
            process_noise (np.ndarray): Process noise covariance matrix (Q), shape (n_inputs, n_inputs).
            measurement_noise (np.ndarray): Measurement noise covariance matrix (R), shape (m, m) where m is measurement dimension.
            rc_params (dict, optional): Parameters for ReservoirComputer. Defaults to empty dict.
            ukf_params (dict, optional): Parameters for UnscentedKalmanFilter. Defaults to empty dict.
        """
        # Default to empty dictionaries if params are not provided
        if rc_params is None:
            rc_params = {}
        if ukf_params is None:
            ukf_params = {}

        # Initialize the Reservoir Computer
        self.rc = ReservoirComputer(
            n_inputs=n_inputs,
            n_reservoir=n_reservoir,
            **rc_params
        )

        # Initialize the Unscented Kalman Filter
        # n_dim matches the state dimension (e.g., 3 for Lorenz system)
        self.ukf = UnscentedKalmanFilter(
            n_dim=n_inputs,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            **ukf_params
        )

        # Initialize the reservoir state as a zero vector
        self.reservoir_state = np.zeros((self.rc.n_reservoir, 1))

        # Adding slight noise in measurement model
        self.measurement_model_noise_std = 0.00001 

    def train_reservoir(self, train_inputs, train_outputs):
        """
        Train the RC on a known time series to learn system dynamics.

        The RC generates internal states from the input sequence and trains its readout weights
        to predict the next state (train_outputs).

        Args:
            train_inputs (np.ndarray): Input time series, shape (T, n_inputs).
            train_outputs (np.ndarray): Target output time series, shape (T, n_inputs), typically next states.

        Note:
            The method accounts for an internal washout period defined in the RC, discarding initial states.
        """
        # Generate reservoir states from input sequence
        states = self.rc.run_reservoir(train_inputs)  # Shape (T - washout, n_reservoir)
        
        # Align outputs with states by removing washout period
        valid_outputs = train_outputs[self.rc.washout:]  # Shape (T - washout, n_inputs)
        
        # Train the RC readout weights (W_out) using ridge regression
        self.rc.train_readout(states, valid_outputs)

    def process_model(self, x):
        """
        Predict the next state using the trained RC as the process model.

        Updates the internal reservoir state and returns the predicted next state based on the current state.

        Args:
            x (np.ndarray): Current state, shape (n_inputs,).

        Returns:
            np.ndarray: Predicted next state, shape (n_inputs,).
        """
        # Reshape input state to column vector
        x_col = x.reshape(-1, 1)  # Shape (n_inputs, 1)
        
        # Update reservoir state using current state as input
        self.reservoir_state = self.rc.update_reservoir_state(
            x_prev=self.reservoir_state,  # Previous reservoir state
            u=x_col                       # Current state as input
        )  # Shape (n_reservoir, 1)
        
        # Compute next state using trained readout weights
        x_next = self.rc.W_out[:, :self.rc.n_reservoir].dot(self.reservoir_state)  # Shape (n_inputs, 1)
        if self.rc.use_bias:
            # Add bias term if enabled in RC
            bias_col = self.rc.W_out[:, -1].reshape(-1, 1)
            x_next += bias_col
        
        return x_next.ravel()  # Shape (n_inputs,)

    def reset_reservoir_state(self):
        """
        Reset the reservoir state to zeros.

        Useful for starting a new prediction sequence or testing phase independently of prior states.
        """
        self.reservoir_state = np.zeros((self.rc.n_reservoir, 1))

    def measurement_model(self, x):
        """
        Define the measurement model h(x) mapping state to observation space.

        Args:
            x (np.ndarray): Current state, shape (n_inputs,).

        Returns:
            np.ndarray: Observed measurement, shape (n_inputs,) for full observation or subset for partial.
        
        Example:
            For partial observation (e.g., only y in Lorenz system), use: return np.array([x[1]])
        """
        return x + np.random.normal(0, self.measurement_model_noise_std, x.shape)

    def filter_step(self, measurement):
        """
        Execute a single UKF predict-and-update cycle given a measurement.

        Args:
            measurement (np.ndarray): Observed measurement at current time step, shape (measurement_dim,).

        Returns:
            np.ndarray: Updated state estimate, shape (n_inputs,).
        """
        # Predict next state using RC as process model
        sigma_points_pred = self.ukf.predict(process_model=self.process_model)
        
        # Update state estimate with measurement
        self.ukf.update(
            sigma_points_pred=sigma_points_pred,
            measurement=measurement,
            measurement_model=self.measurement_model
        )
        
        # Return the current state estimate
        return self.ukf.get_state()

    def run_filter(self, measurements):
        """
        Run the UKF filter over a sequence of measurements.

        Args:
            measurements (np.ndarray): Time series of measurements, shape (T, measurement_dim).

        Returns:
            np.ndarray: Array of filtered state estimates, shape (T, n_inputs).
        """
        estimates = []
        for z in measurements:
            x_est = self.filter_step(z)
            estimates.append(x_est)
        return np.array(estimates)


if __name__ == "__main__":
    # Optional: Add a simple test or example usage here
    pass