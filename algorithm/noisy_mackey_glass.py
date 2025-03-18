"""
noisy_mackey_glass.py

This module generates a noisy Mackey-Glass time series, a nonlinear delayed differential
equation exhibiting chaotic behavior. The data is simulated using Euler integration and
perturbed with Gaussian noise.

Dependencies:
    - numpy: For numerical computations and random number generation.
"""

import numpy as np


def generate_noisy_mackey_glass_data(dt=0.1, num_steps=5000, tau=17.0, beta=0.2, gamma=0.1, n=10.0, noise_std=0.01):
    """
    Generate noisy time-series data for the Mackey-Glass system.

    The Mackey-Glass equation is:
        dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)
    This function uses Euler integration with a delay buffer and adds Gaussian noise.

    Args:
        dt (float, optional): Time step for Euler integration (default: 0.1).
        num_steps (int, optional): Number of time steps to generate (default: 5000).
        tau (float, optional): Time delay parameter (default: 17.0 for chaotic behavior).
        beta (float, optional): Nonlinear term coefficient (default: 0.2).
        gamma (float, optional): Decay rate (default: 0.1).
        n (float, optional): Nonlinear exponent (default: 10.0).
        noise_std (float, optional): Standard deviation of Gaussian noise (default: 0.01).

    Returns:
        np.ndarray: Noisy Mackey-Glass time series, shape (num_steps, 1).

    Notes:
        - Initial history is set to x(t) = 1.2 for t <= 0.
        - Delay is discretized as tau_steps = int(tau / dt).
    """
    # Discretize delay
    tau_steps = int(tau / dt)
    
    # Total steps including initial history
    total_steps = num_steps + tau_steps
    data = np.zeros((total_steps, 1))
    
    # Initial condition: x(t) = 1.2 for t <= 0
    data[:tau_steps] = 1.2

    # Simulate Mackey-Glass system using Euler integration
    for i in range(tau_steps, total_steps):
        x_t = data[i - 1, 0]              # Current state
        x_delay = data[i - tau_steps, 0]  # Delayed state
        # Compute derivative
        dx = beta * x_delay / (1 + x_delay**n) - gamma * x_t + np.random.normal(0, noise_std)
        # Update state
        data[i, 0] = x_t + dx * dt
    
    # Trim initial history and keep only num_steps
    data = data[tau_steps:]

    return data


if __name__ == "__main__":
    # Example usage for testing
    noisy_data = generate_noisy_mackey_glass_data(num_steps=1000)
    print(f"Generated noisy Mackey-Glass data shape: {noisy_data.shape}")
    print(f"First 5 samples:\n{noisy_data[:5]}")