"""
RC_vsRCplusUKF_Rossler.py

Compare RC-only vs. RC+UKF on the Rössler system. Plots ground truth
vs both RC-only and RC+UKF predictions for x, y, z, and prints RMSE
for both approaches.

Dependencies:
  - RCplusUKF.py
  - RC.py
  - UKF.py
  - compute_error.py
  - noisy_rossler.py
"""

import numpy as np
import matplotlib.pyplot as plt

from RCplusUKF import RC_UKF
from RC import ReservoirComputer
from noisy_rossler import generate_noisy_rossler_data
from compute_error import calculate_rmse

def main():
    # ----------------------------------------------------------------
    # 1) Generate Rössler Data
    # ----------------------------------------------------------------
    num_steps = 700
    data = generate_noisy_rossler_data(num_steps=num_steps)  # shape: (num_steps, 3)

    # Prepare next-step train data
    inputs = data[:-1]   # (num_steps - 1, 3)
    outputs = data[1:]   # (num_steps - 1, 3)

    # ----------------------------------------------------------------
    # 2) Split into Train/Test
    # ----------------------------------------------------------------
    split_ratio = 0.7
    split_index = int(len(inputs) * split_ratio)

    train_inputs = inputs[:split_index]      # shape (split_index, 3)
    train_outputs = outputs[:split_index]    # shape (split_index, 3)

    test_inputs = inputs[split_index:]       # shape (N_test, 3)
    test_outputs = outputs[split_index:]     # shape (N_test, 3)
    N_test = len(test_outputs)

    # ----------------------------------------------------------------
    # A) RC-Only Approach
    # ----------------------------------------------------------------
    # 3A. Create a bare ReservoirComputer
    rc_params = {
        "n_inputs": 3,
        "n_reservoir": 100,
        "spectral_radius": 0.9,
        "reg": 1e-5,
        "leak_rate": 1.0,
        "washout": 100,
    }
    rc_model = ReservoirComputer(**rc_params)

    # 3B. Train the RC
    train_states = rc_model.run_reservoir(train_inputs)   # shape (split_index - washout, n_reservoir)
    valid_outputs = train_outputs[rc_model.washout:]      # shape (split_index - washout, 3)
    rc_model.train_readout(train_states, valid_outputs)

    # 3C. RC-Only Free-run Prediction on Test set
    rc_model_state = np.zeros((rc_model.n_reservoir, 1))
    # Optionally warm up the reservoir with last few training steps
    # warm_len = 30
    # for i in range(warm_len):
    #     inp = train_inputs[-warm_len + i].reshape(-1,1)
    #     rc_model_state = rc_model.update_reservoir_state(rc_model_state, inp)

    rc_predicted = []
    for i in range(N_test):
        # 1) Update reservoir with test_inputs[i]
        inp = test_inputs[i].reshape(-1,1)
        rc_model_state = rc_model.update_reservoir_state(rc_model_state, inp)

        # 2) Predict next state
        rc_input = rc_model_state.T  # shape (1, n_reservoir)
        rc_next = rc_model.predict(rc_input)  # shape (1,3)
        rc_predicted.append(rc_next.ravel())

    rc_predicted = np.array(rc_predicted)  # shape (N_test, 3)

    # 3D. Evaluate RC-Only RMSE
    rc_rmse = calculate_rmse(test_outputs, rc_predicted)
    print("==== RC-Only RMSE (Rössler) ====")
    print(f"RC-Only   RMSE for X: {rc_rmse['RMSE_X']:.4f}")
    print(f"RC-Only   RMSE for Y: {rc_rmse['RMSE_Y']:.4f}")
    print(f"RC-Only   RMSE for Z: {rc_rmse['RMSE_Z']:.4f}")
    print(f"Mean RMSE: {((rc_rmse['RMSE_X'] + rc_rmse['RMSE_Y'] + rc_rmse['RMSE_Z']) / 3):.4f}")

    # ----------------------------------------------------------------
    # B) RC+UKF Approach
    # ----------------------------------------------------------------
    # 4A. Build RC+UKF framework
    measurement_noise_std = 0.1
    framework = RC_UKF(
        n_inputs=3,
        n_reservoir=100,
        process_noise=np.eye(3) * 1e-3,
        measurement_noise=np.eye(3) * measurement_noise_std**2,
        ukf_params={"alpha": 0.1}
    )
    # 4B. Train the RC portion inside RC+UKF
    framework.train_reservoir(train_inputs, train_outputs)

    # 4C. Initialize UKF state
    framework.ukf.x = train_outputs[-1].copy()  # shape (3,)
    framework.reset_reservoir_state()

    # 4D. Filter on test data
    rcukf_predicted = []
    for i in range(N_test):
        # We'll feed the true state as measurement (no additional noise)
        # or add random noise if desired:
        measurement = test_outputs[i]
        # if you want noise: measurement += np.random.normal(0, measurement_noise_std, size=3)

        x_est = framework.filter_step(measurement)
        rcukf_predicted.append(x_est)

    rcukf_predicted = np.array(rcukf_predicted)  # shape (N_test, 3)

    # 4E. Evaluate RC+UKF RMSE
    rcukf_rmse = calculate_rmse(test_outputs, rcukf_predicted)
    print("==== RCUKF RMSE (Rössler) ====")
    print(f"RCUKF    RMSE for X: {rcukf_rmse['RMSE_X']:.4f}")
    print(f"RCUKF    RMSE for Y: {rcukf_rmse['RMSE_Y']:.4f}")
    print(f"RCUKF    RMSE for Z: {rcukf_rmse['RMSE_Z']:.4f}")
    print(f"Mean RMSE: {((rcukf_rmse['RMSE_X'] + rcukf_rmse['RMSE_Y'] + rcukf_rmse['RMSE_Z']) / 3):.4f}")

    # ----------------------------------------------------------------
    # C) Plot Comparison: RC-Only vs RC+UKF vs Ground Truth
    # ----------------------------------------------------------------
    time = np.arange(N_test)
    labels = ['x', 'y', 'z']
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 7), sharex=True)

    for dim in range(3):
        axes[dim].plot(time, test_outputs[:, dim], label=f"True {labels[dim]}", color='blue')
        axes[dim].plot(time, rc_predicted[:, dim], label=f"RC-Only {labels[dim]}", linestyle='--', color='green')
        axes[dim].plot(time, rcukf_predicted[:, dim], label=f"RCUKF {labels[dim]}", linestyle=':', color='red')
        axes[dim].set_ylabel(labels[dim])
        axes[dim].legend(loc='best')

    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Rössler System: RC vs RCUKF vs Ground Truth")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
