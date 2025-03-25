"""
RC_vsRCplusUKF_MackeyGlass.py

Compare RC-only vs. RC+UKF on the Mackey-Glass time series.
Plots ground truth vs both RC-only and RC+UKF predictions,
and prints RMSE for each approach.

Dependencies:
    - RCplusUKF.py
    - RC.py
    - UKF.py
    - noisy_mackey_glass.py
    - (optional) compute_error.py if you want a consistent RMSE function
"""

import numpy as np
import matplotlib.pyplot as plt

from RCplusUKF import RC_UKF
from RC import ReservoirComputer
from noisy_mackey_glass import generate_noisy_mackey_glass_data

def main():
    # -------------------------------------------------------
    # 1) Generate Mackey-Glass Data
    # -------------------------------------------------------
    num_steps = 10000
    data = generate_noisy_mackey_glass_data(num_steps=num_steps)  # shape: (num_steps, 1)

    # Prepare input -> output pairs for next-step prediction
    inputs = data[:-1]   # (num_steps-1, 1)
    outputs = data[1:]   # (num_steps-1, 1)

    # -------------------------------------------------------
    # 2) Split into Train / Test
    # -------------------------------------------------------
    split_ratio = 0.7
    split_index = int(len(inputs) * split_ratio)

    train_inputs = inputs[:split_index]      # shape (split_index, 1)
    train_outputs = outputs[:split_index]    # shape (split_index, 1)

    test_inputs = inputs[split_index:]       # shape (N_test, 1)
    test_outputs = outputs[split_index:]     # shape (N_test, 1)
    N_test = len(test_outputs)

    # -------------------------------------------------------
    # A) RC-Only Approach
    # -------------------------------------------------------
    # 3A. Create bare ReservoirComputer
    rc_params = {
        "n_inputs": 1,
        "n_reservoir": 100,
        "spectral_radius": 0.9,
        "reg": 1e-5,
        "leak_rate": 1.0,
        "washout": 100,
    }
    rc_model = ReservoirComputer(**rc_params)

    # 3B. Train the RC on the training set
    #    run_reservoir yields states (T - washout, n_reservoir)
    train_states = rc_model.run_reservoir(train_inputs)
    valid_outputs = train_outputs[rc_model.washout:]  # match shape
    rc_model.train_readout(train_states, valid_outputs)

    # 3C. RC-Only Free-run on the test set
    rc_model_state = np.zeros((rc_model.n_reservoir, 1))
    # Optionally warm up the reservoir with last few steps of training
    # warm_len = 30
    # for i in range(warm_len):
    #     inp_col = train_inputs[-warm_len + i].reshape(-1,1)  # shape (1,1)
    #     rc_model_state = rc_model.update_reservoir_state(rc_model_state, inp_col)

    rc_predicted = []
    for i in range(N_test):
        # Update reservoir state with the current input
        inp_col = test_inputs[i].reshape(-1,1)  # shape (1,1)
        rc_model_state = rc_model.update_reservoir_state(rc_model_state, inp_col)
        # Predict next step (1D)
        rc_input = rc_model_state.T  # shape (1, n_reservoir)
        rc_next = rc_model.predict(rc_input)  # shape (1,1)
        rc_predicted.append(rc_next.item())   # store as scalar

    rc_predicted = np.array(rc_predicted).reshape(-1, 1)  # shape (N_test, 1)

    # 3D. Evaluate RC-Only RMSE
    rc_rmse = np.sqrt(np.mean((test_outputs - rc_predicted)**2))
    print("==== RC-Only RMSE (Mackey-Glass) ====")
    print(f"RC-Only: {rc_rmse:.4f}")

    # -------------------------------------------------------
    # B) RC+UKF Approach
    # -------------------------------------------------------
    # 4A. Build RC+UKF
    measurement_noise_std = 0.01
    framework = RC_UKF(
        n_inputs=1,
        n_reservoir=100,
        process_noise=np.array([[1e-3]]),  # 1x1 process noise
        measurement_noise=np.array([[measurement_noise_std**2]]),
        ukf_params={"alpha": 0.1}
    )
    # 4B. Train the reservoir inside RC+UKF
    framework.train_reservoir(train_inputs, train_outputs)

    # 4C. Init UKF state
    framework.ukf.x = train_outputs[-1].copy()  # shape (1,)
    framework.reset_reservoir_state()

    # 4D. Filter on the test data
    rcukf_predicted = []
    for i in range(N_test):
        # Create a measurement from the ground truth
        # or add noise if desired
        measurement = test_outputs[i]  # shape (1,)
        # measurement += np.random.normal(0, measurement_noise_std, size=1)
        
        x_est = framework.filter_step(measurement)
        rcukf_predicted.append(x_est.item())  # store as scalar

    rcukf_predicted = np.array(rcukf_predicted).reshape(-1, 1)

    # 4E. Evaluate RC+UKF RMSE
    rcukf_rmse = np.sqrt(np.mean((test_outputs - rcukf_predicted)**2))
    print("==== RCUKF RMSE (Mackey-Glass) ====")
    print(f"RCUKF: {rcukf_rmse:.4f}")

    # -------------------------------------------------------
    # C) Plot Comparison: RC-Only vs RC+UKF vs Ground Truth
    # -------------------------------------------------------
    time = np.arange(N_test)
    plt.figure(figsize=(9, 5))
    plt.plot(time, test_outputs[:, 0], label='True x', color='blue')
    plt.plot(time, rc_predicted[:, 0], label='RC-Only x', linestyle='--', color='green')
    plt.plot(time, rcukf_predicted[:, 0], label='RCUKF x', linestyle=':', color='red')
    plt.xlabel("Time Step")
    plt.ylabel("x(t)")
    plt.title("Mackey-Glass: RC vs. RCUKF vs. True")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
