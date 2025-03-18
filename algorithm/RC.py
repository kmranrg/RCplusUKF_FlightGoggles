"""
RC.py

This module implements a Reservoir Computer (RC), also known as an Echo State Network (ESN),
for modeling and predicting nonlinear dynamical systems. The RC uses a fixed, randomly initialized
reservoir with trainable readout weights to map reservoir states to outputs.

Dependencies:
    - numpy: For numerical computations and array operations.
    - matplotlib.pyplot: For plotting prediction results.
"""

import numpy as np
import matplotlib.pyplot as plt


class ReservoirComputer:
    """
    A Reservoir Computer (Echo State Network) for time-series prediction.

    The RC consists of a fixed, random reservoir with recurrent connections and a trainable
    linear readout layer. It processes input sequences to generate reservoir states, which
    are then mapped to outputs via trained weights.

    Attributes:
        n_inputs (int): Number of input dimensions.
        n_reservoir (int): Number of reservoir neurons.
        spectral_radius (float): Spectral radius of the reservoir weight matrix.
        sparsity (float): Fraction of nonzero weights in the reservoir matrix.
        reg (float): Regularization parameter for readout training.
        noise_std (float): Standard deviation of input noise.
        leak_rate (float): Leaking rate for reservoir updates (1.0 = standard ESN).
        washout (int): Number of initial steps to discard.
        use_bias (bool): Whether to include a bias term in the readout.
        input_scale (float): Scaling factor for input weights.
        W_in (np.ndarray): Input-to-reservoir weight matrix, shape (n_reservoir, n_inputs).
        W (np.ndarray): Reservoir weight matrix, shape (n_reservoir, n_reservoir).
        W_out (np.ndarray or None): Readout weights, shape (n_inputs, n_reservoir [+1 if bias]).
    """

    def __init__(
        self,
        n_inputs,
        n_reservoir,
        spectral_radius=0.9,
        sparsity=0.2,
        reg=1e-5,
        noise_std=0.1,
        random_seed=None,
        leak_rate=1.0,
        washout=100,
        use_bias=True,
        input_scale=0.1
    ):
        """
        Initialize the Reservoir Computer.

        Args:
            n_inputs (int): Number of input dimensions (e.g., 3 for Lorenz system).
            n_reservoir (int): Number of reservoir (hidden) neurons.
            spectral_radius (float, optional): Desired spectral radius of W (default: 0.9).
            sparsity (float, optional): Fraction of nonzero weights in W (0 to 1, default: 0.2).
            reg (float, optional): Ridge regression regularization parameter (default: 1e-5).
            noise_std (float, optional): Std dev of Gaussian noise added to inputs (default: 0.1).
            random_seed (int or None, optional): Seed for reproducibility (default: None).
            leak_rate (float, optional): Leaking rate for integration (1.0 = no leak, default: 1.0).
            washout (int, optional): Number of initial steps to discard (default: 100).
            use_bias (bool, optional): Include bias in readout (default: True).
            input_scale (float, optional): Scaling factor for W_in (default: 0.1).
        """
        # Set random seed for reproducibility if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Store initialization parameters
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.reg = reg
        self.noise_std = noise_std
        self.leak_rate = leak_rate
        self.washout = washout
        self.use_bias = use_bias
        self.input_scale = input_scale

        # Initialize fixed weight matrices
        self.W_in = self.initialize_input_weights()  # Input-to-reservoir weights
        self.W = self.initialize_reservoir()         # Reservoir-to-reservoir weights

        # Readout weights (to be trained later)
        self.W_out = None

    def initialize_input_weights(self):
        """
        Initialize the input-to-reservoir weight matrix W_in.

        Returns:
            np.ndarray: W_in matrix, shape (n_reservoir, n_inputs), with values scaled by input_scale.
        """
        # Generate random weights in [-1, 1], then scale
        return self.input_scale * (2 * np.random.rand(self.n_reservoir, self.n_inputs) - 1)

    def initialize_reservoir(self):
        """
        Initialize the reservoir weight matrix W with specified sparsity and spectral radius.

        The matrix is randomly initialized, sparsified, and rescaled to meet the spectral radius.

        Returns:
            np.ndarray: W matrix, shape (n_reservoir, n_reservoir).
        """
        # Random weights in [-0.5, 0.5]
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        
        # Apply sparsity by zeroing out entries based on sparsity parameter
        mask = np.random.rand(self.n_reservoir, self.n_reservoir)
        W[mask > self.sparsity] = 0.0

        # Rescale to desired spectral radius for echo state property
        eig_values = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eig_values))
        W *= (self.spectral_radius / max_eig)

        return W

    def add_noise(self, inputs):
        """
        Add optional Gaussian noise to the input sequence.

        Args:
            inputs (np.ndarray): Input time series, shape (T, n_inputs).

        Returns:
            np.ndarray: Noisy inputs, same shape as inputs, or unchanged if noise_std = 0.
        """
        if self.noise_std > 0:
            return inputs + np.random.normal(0, self.noise_std, inputs.shape)
        return inputs

    def update_reservoir_state(self, x_prev, u):
        """
        Update the reservoir state using leaky integration.

        The update rule is:
            x(t+1) = (1 - leak_rate) * x(t) + leak_rate * tanh(W_in * u(t+1) + W * x(t))

        Args:
            x_prev (np.ndarray): Previous reservoir state, shape (n_reservoir, 1).
            u (np.ndarray): Current input, shape (n_inputs, 1).

        Returns:
            np.ndarray: Updated reservoir state, shape (n_reservoir, 1).
        """
        pre_activation = np.dot(self.W_in, u) + np.dot(self.W, x_prev)  # Shape (n_reservoir, 1)
        return (1 - self.leak_rate) * x_prev + self.leak_rate * np.tanh(pre_activation)

    def run_reservoir(self, inputs):
        """
        Run the reservoir over an input sequence and return post-washout states.

        Args:
            inputs (np.ndarray): Input time series, shape (T, n_inputs).

        Returns:
            np.ndarray: Reservoir states after washout, shape (T - washout, n_reservoir).

        Raises:
            ValueError: If washout >= number of input steps.
        """
        # Add noise to inputs if specified
        inputs_noisy = self.add_noise(inputs)
        n_steps = inputs_noisy.shape[0]

        # Store all reservoir states
        states_all = np.zeros((n_steps, self.n_reservoir))
        
        # Initialize reservoir state to zeros
        x = np.zeros((self.n_reservoir, 1))

        # Iterate over input sequence to update reservoir states
        for t in range(n_steps):
            u = inputs_noisy[t].reshape(-1, 1)  # Shape (n_inputs, 1)
            x = self.update_reservoir_state(x, u)
            states_all[t] = x.ravel()

        # Check washout validity
        if self.washout >= n_steps:
            raise ValueError("washout is >= the total number of steps!")
        
        # Return states after washout period
        return states_all[self.washout:]

    def train_readout(self, states, outputs):
        """
        Train the readout layer (W_out) using ridge regression.

        Args:
            states (np.ndarray): Reservoir states, shape (N, n_reservoir).
            outputs (np.ndarray): Target outputs, shape (N, output_dim).

        Notes:
            If use_bias is True, states are augmented with a column of ones for bias term.
            W_out shape becomes (output_dim, n_reservoir + 1) with bias, else (output_dim, n_reservoir).
        """
        # Augment states with bias term if enabled
        if self.use_bias:
            ones = np.ones((states.shape[0], 1))
            X_aug = np.hstack([states, ones])  # Shape (N, n_reservoir + 1)
        else:
            X_aug = states  # Shape (N, n_reservoir)

        # Ridge regression: W_out = (Y^T X_aug) (X_aug^T X_aug + reg * I)^(-1)
        ridge_term = self.reg * np.eye(X_aug.shape[1])
        self.W_out = (outputs.T @ X_aug) @ np.linalg.inv(X_aug.T @ X_aug + ridge_term)

    def predict(self, states):
        """
        Predict outputs from reservoir states using trained readout weights.

        Args:
            states (np.ndarray): Reservoir states, shape (N, n_reservoir).

        Returns:
            np.ndarray: Predicted outputs, shape (N, output_dim).

        Raises:
            ValueError: If W_out is None (model not trained).
        """
        if self.W_out is None:
            raise ValueError("Model has not been trained. Call `train_readout` first.")
        
        # Augment states with bias term if enabled
        if self.use_bias:
            ones = np.ones((states.shape[0], 1))
            X_aug = np.hstack([states, ones])  # Shape (N, n_reservoir + 1)
        else:
            X_aug = states  # Shape (N, n_reservoir)

        # Compute predictions: Y = X_aug * W_out^T
        return X_aug @ self.W_out.T  # Shape (N, output_dim)

    def plot_results(self, true_values, predicted_values, title="Prediction Performance"):
        """
        Plot true vs. predicted values for visual comparison.

        Args:
            true_values (np.ndarray): True output values, shape (N,).
            predicted_values (np.ndarray): Predicted output values, shape (N,).
            title (str, optional): Plot title (default: "Prediction Performance").

        Note:
            Designed for 1D outputs; for multi-dimensional outputs, plot each dimension separately.
        """
        plt.figure(figsize=(8, 4))
        plt.plot(true_values, label="True Output", color='blue')
        plt.plot(predicted_values, label="Predicted Output", linestyle="--", alpha=0.8, color='orange')
        plt.xlabel("Time Step")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    # Optional: Add a simple test or example usage here
    pass