"""
RCplusUKF_MackeyGlassImplementation.py

Implementation of a Reservoir Computer (RC) with Unscented Kalman Filter (UKF) for state
estimation of the Mackey-Glass time series.

This script generates noisy Mackey-Glass data, trains an RC to predict the next step,
and uses an UKF to filter noisy measurements during testing. It computes the Root Mean
Squared Error (RMSE) and visualizes the results. Designed with flexibility for real-world use.

Dependencies:
    - numpy: For numerical computations.
    - matplotlib.pyplot: For plotting results.
    - RCplusUKF: Custom RC_UKF class.
    - noisy_mackey_glass: Custom function to generate noisy Mackey-Glass data.
    - compute_error: Custom function to calculate RMSE (modified for 1D).
"""

import numpy as np
import matplotlib.pyplot as plt
from RCplusUKF import RC_UKF
from noisy_mackey_glass import generate_noisy_mackey_glass_data


def main():
    """
    Main function to execute the RC+UKF estimation pipeline for the Mackey-Glass time series.
    
    Steps:
        1. Generate noisy Mackey-Glass data.
        2. Split data into training and testing sets.
        3. Initialize and train the RC+UKF framework for scalar prediction.
        4. Run the UKF filter on test data with noisy measurements.
        5. Compute RMSE and visualize results.
    """
    # Configurable parameters
    num_steps = 5000          # Total time steps for Mackey-Glass data
    split_ratio = 0.7         # Training data proportion (70%)
    measurement_noise_std = 0.01  # Std dev of noise added during filtering

    # ------------------------------
    # Step 1: Data Generation
    # ------------------------------
    # Generate noisy Mackey-Glass data (returns shape: (num_steps, 1))
    data = generate_noisy_mackey_glass_data(num_steps=num_steps)
    
    # Prepare input-output pairs for RC training/next-step prediction
    inputs = data[:-1]  # Inputs: all but last step, shape (num_steps-1, 1)
    outputs = data[1:]  # Outputs: all but first step, shape (num_steps-1, 1)

    # ------------------------------
    # Step 2: Data Splitting
    # ------------------------------
    split_index = int(len(inputs) * split_ratio)
    
    # Training data
    train_inputs = inputs[:split_index]    # Shape (split_index, 1)
    train_outputs = outputs[:split_index]  # Shape (split_index, 1)
    
    # Testing data
    test_inputs = inputs[split_index:]    # Shape (num_steps-1-split_index, 1)
    test_outputs = outputs[split_index:]  # Shape (num_steps-1-split_index, 1)

    # ------------------------------
    # Step 3: Framework Setup and Training
    # ------------------------------
    # Initialize the RC+UKF framework for scalar time series
    framework = RC_UKF(
        n_inputs=1,                   # Mackey-Glass is a scalar time series
        n_reservoir=500,              # Number of reservoir neurons
        process_noise=np.array([[1e-3]]),  # Process noise covariance (Q), 1x1
        measurement_noise=np.array([[measurement_noise_std**2]]),  # Measurement noise (R), 1x1
        ukf_params={'alpha': 0.1}     # UKF sigma point spread parameter
    )
    
    # Train the reservoir to predict next step
    framework.train_reservoir(train_inputs, train_outputs)

    # ------------------------------
    # Step 4: UKF Filtering
    # ------------------------------
    # Initialize UKF state with the last training output
    initial_state = train_outputs[-1].copy()  # Shape (1,)
    framework.ukf.x = initial_state
    
    # Optional: Reset reservoir state (commented to preserve training dynamics)
    framework.reset_reservoir_state()
    
    # Run UKF filter over test data with noisy measurements
    predicted_states = []
    for i in range(len(test_inputs)):
        # Simulate noisy measurement of the current true state
        measurement = test_inputs[i] + np.random.normal(0, measurement_noise_std, test_inputs[i].shape)  # Shape (1,)
        x_est = framework.filter_step(measurement)  # UKF predict + update, shape (1,)
        predicted_states.append(x_est)
    
    predicted_states = np.array(predicted_states)  # Shape (len(test_outputs), 1)

    # ------------------------------
    # Step 5: Evaluation and Visualization
    # ------------------------------
    # Compute RMSE between true and estimated states
    rmse = np.sqrt(np.mean((test_outputs - predicted_states) ** 2))
    print(f"RMSE for Mackey-Glass: {rmse:.4f}")

    # Plot true vs estimated states
    plt.figure(figsize=(10, 4))
    time = np.arange(len(test_outputs))
    plt.plot(time, test_outputs[:, 0], label='True x', color='blue')
    plt.plot(time, predicted_states[:, 0], label='Estimated x', linestyle='--', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('x')
    plt.title('Mackey-Glass Time Series: True vs Estimated (RCplusUKF)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()