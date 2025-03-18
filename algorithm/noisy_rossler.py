"""
noisy_rossler.py

This module provides a function to generate noisy time-series data for the Rössler system,
a chaotic dynamical system. The data is generated using Euler integration and perturbed
with Gaussian noise.

Dependencies:
    - numpy: For numerical computations and random number generation.
"""

import numpy as np


def generate_noisy_rossler_data(dt=0.01, num_steps=5000, noise_std=0.1, a=0.2, b=0.2, c=5.7):
    """
    Generate noisy time-series data for the Rössler system.

    The Rössler system is defined by the differential equations:
        dx/dt = -y - z
        dy/dt = x + a * y
        dz/dt = b + z * (x - c)
    This function uses Euler integration to simulate the system and adds Gaussian noise.

    Args:
        dt (float, optional): Time step for Euler integration (default: 0.01).
        num_steps (int, optional): Number of time steps to generate (default: 5000).
        noise_std (float, optional): Standard deviation of Gaussian noise (default: 0.1).
        a (float, optional): Rössler parameter a (default: 0.2).
        b (float, optional): Rössler parameter b (default: 0.2).
        c (float, optional): Rössler parameter c (default: 5.7).

    Returns:
        np.ndarray: Noisy Rössler data, shape (num_steps, 3), with columns [x, y, z].

    Notes:
        - Default parameters (a=0.2, b=0.2, c=5.7) produce chaotic behavior.
        - Initial conditions are set to (x, y, z) = (1.0, 1.0, 1.0).
    """
    # Rössler system parameters for chaotic behavior
    a, b, c = a, b, c
    
    # Initial conditions
    x, y, z = 1.0, 1.0, 1.0

    # Pre-allocate array for efficiency
    data = np.zeros((num_steps, 3))

    # Simulate the Rössler system using Euler integration
    for i in range(num_steps):
        # Compute derivatives
        dx = -y - z + np.random.normal(0, noise_std)
        dy = x + a * y + np.random.normal(0, noise_std)
        dz = b + z * (x - c) + np.random.normal(0, noise_std)

        # Update state using Euler method
        x += dx * dt
        y += dy * dt
        z += dz * dt

        # Store current state
        data[i] = [x, y, z]

    return data


if __name__ == "__main__":
    # Example usage for testing
    noisy_data = generate_noisy_rossler_data(dt=0.01, num_steps=1000, noise_std=0.1)
    print(f"Generated noisy Rössler data shape: {noisy_data.shape}")
    print(f"First 5 samples:\n{noisy_data[:5]}")