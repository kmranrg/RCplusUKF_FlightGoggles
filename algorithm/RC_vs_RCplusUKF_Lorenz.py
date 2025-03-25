"""
RC_vs_RCplusUKF_Lorenz.py

Compare RC-only vs. RC+UKF on the Lorenz system. Plots ground truth vs
both RC and RC+UKF predictions for x, y, and z. Prints RMSE for each.
"""

import numpy as np
import matplotlib.pyplot as plt

from RCplusUKF import RC_UKF
from RC import ReservoirComputer
from noisy_lorenz import generate_noisy_lorenz_data
from compute_error import calculate_rmse

def main():
    # 1. Generate noisy Lorenz data
    num_steps = 10000
    data = generate_noisy_lorenz_data(num_steps=num_steps)  # shape (num_steps, 3)

    # 2. Prepare input/outputs for next-step prediction
    inputs = data[:-1]    # (num_steps - 1, 3)
    outputs = data[1:]    # (num_steps - 1, 3)

    # 3. Split into train/test
    split_ratio = 0.7
    split_index = int(len(inputs) * split_ratio)

    train_inputs = inputs[:split_index]      # shape (split_index, 3)
    train_outputs = outputs[:split_index]    # shape (split_index, 3)

    test_inputs = inputs[split_index:]       # shape (N_test, 3)
    test_outputs = outputs[split_index:]     # shape (N_test, 3)
    N_test = len(test_outputs)

    # -------------------------------------------------------
    # A) RC-Only Approach (No UKF)
    # -------------------------------------------------------
    # 4A. Initialize and train a ReservoirComputer
    rc_params = {
        "n_inputs": 3,
        "n_reservoir": 100,
        "spectral_radius": 0.9,
        "reg": 1e-5,
        "leak_rate": 1.0,
        "washout": 100,
    }
    rc_model = ReservoirComputer(**rc_params)
    # Generate states from the training data
    train_states = rc_model.run_reservoir(train_inputs)   # shape (split_index - washout, n_reservoir)
    # Align training outputs
    valid_outputs = train_outputs[rc_model.washout:]      # shape (split_index - washout, 3)
    # Train readout
    rc_model.train_readout(train_states, valid_outputs)

    # 4B. RC-Only Prediction on Test set
    #  - typical approach: start from the last known state from training
    #  - We'll forcibly reset the RC to zero or to the last training state
    rc_model.plot_results = None  # not using the built-in plot in this code
    rc_model_state = np.zeros((rc_model.n_reservoir, 1))  # reservoir state

    # Optionally warm up the RC using the last few steps of train_inputs
    # so that it’s not starting from zero. We'll do a small "burn-in".
    # for i in range(30):  
    #     # feed the last 30 steps of training (some smaller number is typical)
    #     warm_inp = train_inputs[-30 + i].reshape(-1,1) 
    #     rc_model_state = rc_model.update_reservoir_state(rc_model_state, warm_inp)

    # Now do free-run for the test
    rc_predicted_states = []
    # We'll treat test_inputs as the "previous state" for the RC input.
    # Then the RC next-step is readout( reservoir_state ).
    for i in range(N_test):
        # Step 1: update reservoir with the current input (test_inputs[i])
        inp_col = test_inputs[i].reshape(-1,1)
        rc_model_state = rc_model.update_reservoir_state(rc_model_state, inp_col)
        # Step 2: predict next state from readout
        #   Y = W_out * reservoir_state (+ bias)
        next_state = rc_model.predict(rc_model_state.T)  # shape (1,3)
        rc_predicted_states.append(next_state.ravel())

    rc_predicted_states = np.array(rc_predicted_states)  # shape (N_test, 3)

    # B) Evaluate RC-Only RMSE
    rc_rmse = calculate_rmse(test_outputs, rc_predicted_states)
    print("==== RC-Only RMSE ====")
    print(f"RC-Only   RMSE for X: {rc_rmse['RMSE_X']:.4f}")
    print(f"RC-Only   RMSE for Y: {rc_rmse['RMSE_Y']:.4f}")
    print(f"RC-Only   RMSE for Z: {rc_rmse['RMSE_Z']:.4f}")
    print(f"Mean RMSE: {((rc_rmse['RMSE_X'] + rc_rmse['RMSE_Y'] + rc_rmse['RMSE_Z']) / 3):.4f}")


    # -------------------------------------------------------
    # B) RC+UKF Approach
    # -------------------------------------------------------
    # 5A. Initialize hybrid RC_UKF
    framework = RC_UKF(
        n_inputs=3,
        n_reservoir=100,
        process_noise=np.eye(3) * 1e-3,
        measurement_noise=np.eye(3) * 1e-2,
        ukf_params={"alpha": 0.1},
    )
    # 5B. Train the reservoir
    framework.train_reservoir(train_inputs, train_outputs)

    # 5C. Initialize UKF
    #   We'll set the UKF's initial state = last training output
    framework.ukf.x = train_outputs[-1].copy()
    framework.reset_reservoir_state()

    # 5D. Filter on test data
    rcukf_predicted_states = []
    for i in range(N_test):
        measurement = test_outputs[i]  # "synthetic: perfect measurement" or you can add noise
        x_est = framework.filter_step(measurement)
        rcukf_predicted_states.append(x_est)

    rcukf_predicted_states = np.array(rcukf_predicted_states)

    # 5E. Evaluate RC+UKF RMSE
    rcukf_rmse = calculate_rmse(test_outputs, rcukf_predicted_states)
    print("==== RCUKF RMSE ====")
    print(f"RCUKF    RMSE for X: {rcukf_rmse['RMSE_X']:.4f}")
    print(f"RCUKF    RMSE for Y: {rcukf_rmse['RMSE_Y']:.4f}")
    print(f"RCUKF    RMSE for Z: {rcukf_rmse['RMSE_Z']:.4f}")
    print(f"Mean RMSE: {((rcukf_rmse['RMSE_X'] + rcukf_rmse['RMSE_Y'] + rcukf_rmse['RMSE_Z']) / 3):.4f}")

    # -------------------------------------------------------
    # C) Plot Comparison: RC-Only vs RC+UKF vs Ground Truth
    # -------------------------------------------------------
    time = np.arange(N_test)
    labels = ['x', 'y', 'z']
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 7), sharex=True)

    for dim in range(3):
        axes[dim].plot(
            time, 
            test_outputs[:, dim], 
            label=f"True {labels[dim]}", 
            color='blue'
        )
        axes[dim].plot(
            time, 
            rc_predicted_states[:, dim], 
            label=f"RC-Only {labels[dim]}", 
            linestyle='--', 
            color='green'
        )
        axes[dim].plot(
            time, 
            rcukf_predicted_states[:, dim], 
            label=f"RCUKF {labels[dim]}",
            linestyle=':', 
            color='red'
        )
        axes[dim].set_ylabel(labels[dim])
        axes[dim].legend(loc='best')

    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Lorenz System: RC vs RCUKF vs True States")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
