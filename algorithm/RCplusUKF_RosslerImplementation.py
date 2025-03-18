"""
RCplusUKF_RosslerImplementation.py

Implementation of a Reservoir Computer (RC) with Unscented Kalman Filter (UKF) for state
estimation of the Rössler system.

This script generates noisy Rössler data, trains an RC on a portion of the data, and uses
an UKF to filter noisy measurements during testing. It computes the Root Mean Squared Error (RMSE)
and visualizes the results. Designed with flexibility for real-world applications.

Dependencies:
    - numpy: For numerical computations.
    - matplotlib.pyplot: For plotting results.
    - RCplusUKF: Custom RC_UKF class.
    - noisy_rossler: Custom function to generate noisy Rössler data.
    - compute_error: Custom function to calculate RMSE.
"""

import numpy as np
import matplotlib.pyplot as plt
from RCplusUKF import RC_UKF
from noisy_rossler import generate_noisy_rossler_data
from compute_error import calculate_rmse


def main():
    """
    Main function to execute the RC+UKF estimation pipeline for the Rössler system.
    
    Steps:
        1. Generate noisy Rössler system data.
        2. Split data into training and testing sets.
        3. Initialize and train the RC+UKF framework.
        4. Run the UKF filter on test data with noisy measurements.
        5. Compute RMSE and visualize results.
    """
    # Configurable parameters for generalization
    num_steps = 50000         # Total time steps for Rössler data
    split_ratio = 0.7         # Training data proportion (70%)
    measurement_noise_std = 0.1  # Std dev of noise added to measurements during filtering

    # ------------------------------
    # Step 1: Data Generation
    # ------------------------------
    # Generate noisy Rössler data (returns shape: (num_steps, 3))
    data = generate_noisy_rossler_data(num_steps=num_steps)
    
    # Prepare input-output pairs for RC training/next-state prediction
    inputs = data[:-1]  # Inputs: all but last step, shape (num_steps-1, 3)
    outputs = data[1:]  # Outputs: all but first step, shape (num_steps-1, 3)

    # ------------------------------
    # Step 2: Data Splitting
    # ------------------------------
    split_index = int(len(inputs) * split_ratio)
    
    # Training data
    train_inputs = inputs[:split_index]    # Shape (split_index, 3)
    train_outputs = outputs[:split_index]  # Shape (split_index, 3)
    
    # Testing data
    test_inputs = inputs[split_index:]    # Shape (num_steps-1-split_index, 3)
    test_outputs = outputs[split_index:]  # Shape (num_steps-1-split_index, 3)

    # ------------------------------
    # Step 3: Framework Setup and Training
    # ------------------------------
    # Initialize the RC+UKF framework
    framework = RC_UKF(
        n_inputs=3,                   # Rössler system has 3 states (x, y, z)
        n_reservoir=500,              # Number of reservoir neurons
        process_noise=np.eye(3) * 1e-3,  # Process noise covariance (Q)
        measurement_noise=np.eye(3) * measurement_noise_std**2,  # Measurement noise covariance (R)
        ukf_params={'alpha': 0.1}     # UKF sigma point spread parameter
    )
    
    # Train the reservoir using training data
    # Internal washout handled by RC_UKF.train_reservoir
    framework.train_reservoir(train_inputs, train_outputs)

    # ------------------------------
    # Step 4: UKF Filtering
    # ------------------------------
    # Initialize UKF state with the last training output
    initial_state = train_outputs[-1].copy()  # Shape (3,)
    framework.ukf.x = initial_state
    
    # Optional: Reset reservoir state (comment out to preserve training dynamics)
    # framework.reset_reservoir_state()
    
    # Run UKF filter over test data with noisy measurements
    predicted_states = []
    for i in range(len(test_inputs)):
        # Simulate noisy measurement of the current true state
        measurement = test_inputs[i] + np.random.normal(0, measurement_noise_std, test_inputs[i].shape)  # Shape (3,)
        x_est = framework.filter_step(measurement)  # UKF predict + update, shape (3,)
        predicted_states.append(x_est)
    
    predicted_states = np.array(predicted_states)  # Shape (len(test_outputs), 3)

    # ------------------------------
    # Step 5: Evaluation and Visualization
    # ------------------------------
    # Compute RMSE between true and estimated states
    rmse_results = calculate_rmse(test_outputs, predicted_states)
    print(f"RMSE for X: {rmse_results['RMSE_X']:.4f}")
    print(f"RMSE for Y: {rmse_results['RMSE_Y']:.4f}")
    print(f"RMSE for Z: {rmse_results['RMSE_Z']:.4f}")

    # Plot true vs estimated states
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    time = np.arange(len(test_outputs))
    labels = ['x', 'y', 'z']
    
    for i in range(3):
        axes[i].plot(time, test_outputs[:, i], label=f'True {labels[i]}', color='blue')
        axes[i].plot(time, predicted_states[:, i], label=f'Estimated {labels[i]}',
                     linestyle='--', color='orange')
        axes[i].set_ylabel(f'{labels[i]}')
        axes[i].legend(loc='best')
    
    axes[-1].set_xlabel('Time Step')
    plt.suptitle('Rössler System: True vs Estimated States (RCplusUKF)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()