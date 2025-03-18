"""
compute_error.py

This module provides a function to calculate the Root Mean Squared Error (RMSE) between true
and predicted values for multi-dimensional data, such as states of the Lorenz system.

Dependencies:
    - numpy: For numerical computations and array operations.
"""

import numpy as np


def calculate_rmse(true_values, predicted_values):
    """
    Calculate the Root Mean Squared Error (RMSE) for each dimension of multi-dimensional data.

    Computes RMSE along each column (dimension) of the input arrays, returning results in a dictionary
    labeled for the x, y, and z dimensions (assuming 3D data, e.g., Lorenz system states).

    Args:
        true_values (np.ndarray): Array of true values, shape (n_samples, n_dims).
        predicted_values (np.ndarray): Array of predicted values, shape (n_samples, n_dims).

    Returns:
        dict: RMSE values for each dimension with keys "RMSE_X", "RMSE_Y", "RMSE_Z".

    Raises:
        AssertionError: If the shapes of true_values and predicted_values do not match.

    Example:
        >>> true = np.array([[1, 2, 3], [4, 5, 6]])
        >>> pred = np.array([[1.1, 2.2, 2.9], [3.8, 5.1, 6.2]])
        >>> calculate_rmse(true, pred)
        {'RMSE_X': 0.1414, 'RMSE_Y': 0.1414, 'RMSE_Z': 0.1414}
    """
    # Ensure input arrays have identical shapes
    assert true_values.shape == predicted_values.shape, "Shapes of true and predicted values must match."

    # Calculate RMSE for each dimension (column-wise)
    # - Difference: true_values - predicted_values, shape (n_samples, n_dims)
    # - Squared: (difference) ** 2
    # - Mean: np.mean(..., axis=0), shape (n_dims,)
    # - Square root: np.sqrt(...), shape (n_dims,)
    rmse_values = np.sqrt(np.mean((true_values - predicted_values) ** 2, axis=0))

    # Return RMSEs in a dictionary assuming 3D data (x, y, z)
    return {
        "RMSE_X": rmse_values[0],
        "RMSE_Y": rmse_values[1],
        "RMSE_Z": rmse_values[2]
    }


if __name__ == "__main__":
    # Example usage for testing
    true_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pred_data = np.array([[1.1, 2.2, 2.9], [3.8, 5.1, 6.2]])
    rmse = calculate_rmse(true_data, pred_data)
    print(f"RMSE Results: {rmse}")