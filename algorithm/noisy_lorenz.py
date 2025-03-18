"""
noisy_lorenz.py

This module provides a function to generate noisy time-series data for the Lorenz system,
a classic chaotic dynamical system. The data is generated using Euler integration and
perturbed with Gaussian noise.

Dependencies:
    - numpy: For numerical computations and random number generation.
"""

import numpy as np


def generate_noisy_lorenz_data(dt=0.01, num_steps=5000, noise_std=0.1):
    """
    Generate noisy time-series data for the Lorenz system.

    The Lorenz system is defined by the differential equations:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z
    This function uses Euler integration to simulate the system and adds Gaussian noise.

    Args:
        dt (float, optional): Time step for Euler integration (default: 0.01).
        num_steps (int, optional): Number of time steps to generate (default: 5000).
        noise_std (float, optional): Standard deviation of Gaussian noise (default: 0.1).

    Returns:
        np.ndarray: Noisy Lorenz data, shape (num_steps, 3), with columns [x, y, z].

    Notes:
        - Parameters sigma=10, beta=8/3, rho=28 are fixed for chaotic behavior.
        - Initial conditions are set to (x, y, z) = (1.0, 1.0, 1.0).
    """
    # Lorenz system parameters for chaotic behavior
    sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
    
    # Initial conditions
    x, y, z = 1.0, 1.0, 1.0

    # Pre-allocate array for efficiency
    data = np.zeros((num_steps, 3))

    # Simulate the Lorenz system using Euler integration
    for i in range(num_steps):
        # Compute derivatives
        dx = sigma * (y - x) + np.random.normal(0, noise_std)
        dy = x * (rho - z) - y  + np.random.normal(0, noise_std)
        dz = x * y - beta * z  + np.random.normal(0, noise_std)

        # Update state using Euler method
        x += dx * dt 
        y += dy * dt
        z += dz * dt

        # Store current state
        data[i] = [x, y, z]

    return data


if __name__ == "__main__":
    # Example usage for testing
    noisy_data = generate_noisy_lorenz_data(dt=0.01, num_steps=1000, noise_std=0.1)
    print(f"Generated noisy Lorenz data shape: {noisy_data.shape}")
    print(f"First 5 samples:\n{noisy_data[:5]}")