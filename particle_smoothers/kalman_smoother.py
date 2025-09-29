#############################
#%% Import necessary libraries
#############################
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from particle_filters.kalman_filter import KalmanFilter



class KalmanSmoother:
    """
    Rauch–Tung–Striebel (RTS) smoother using stored Kalman filter output.
    """
    def __init__(self, kalman_filter):
        """
        Parameters:
        :kalman_filter: an instance of KalmanFilter that has been run on data
        """
        self.kf = kalman_filter

        # Check required fields exist
        required = [
            "filtered_states", "filtered_covs",
            "predicted_states", "predicted_covs",
            "A", "Q"
        ]
        for attr in required:
            if getattr(kalman_filter, attr, None) is None:
                raise ValueError(f"KalmanFilter object missing attribute: {attr}")

    def smooth(self):
        """
        Runs the RTS smoother.

        Returns:
        - smoothed_states: (T x r) smoothed means
        - smoothed_covs: (T x r x r) smoothed covariances
        """
        T, r = self.kf.filtered_states.shape
        A = self.kf.A
        Q = self.kf.Q

        smoothed_states = np.zeros_like(self.kf.filtered_states)
        smoothed_covs = np.zeros_like(self.kf.filtered_covs)

        # Initialization at T
        smoothed_states[-1] = self.kf.filtered_states[-1]
        smoothed_covs[-1] = self.kf.filtered_covs[-1]

        # RTS backward recursion
        for t in reversed(range(T - 1)):
            m_t = self.kf.filtered_states[t]
            P_t = self.kf.filtered_covs[t]
            m_pred = self.kf.predicted_states[t + 1]
            P_pred = self.kf.predicted_covs[t + 1]

            # Smoothing gain
            C_t = P_t @ A.T @ np.linalg.inv(P_pred)

            smoothed_states[t] = m_t + C_t @ (smoothed_states[t + 1] - m_pred)
            smoothed_covs[t] = P_t + C_t @ (smoothed_covs[t + 1] - P_pred) @ C_t.T

        return smoothed_states, smoothed_covs
#############################
#%% Example usage and testing of the KalmanSmoother class
#############################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simulator.mean_time_series import Mean_Time_Series
    from datetime import datetime

    #############################
    # Simulate time series
    #############################
    ts = Mean_Time_Series(
        sigma=10.0,
        include_level=True,
        include_trend=True,
        include_seasonality=True,
        m_0=[0.0, 0.0] + [0.0] * 11,
        P_0=[0.1] * 13,
        process_noise_vector=[0.1] * 13,
        period=12,
        start_date=datetime(2000, 1, 1)
    )

    for _ in range(100):
        ts.move()
        ts.measure()

    y = np.array(ts.all_measurements)

    #############################
    # Run Kalman filter
    #############################
    kf = KalmanFilter(
        sigma=10.0,
        process_noises=[0.1] * 13,
        include_level=True,
        include_trend=True,
        include_seasonality=True,
        period=12
    )
    kf.filter(y)

    #############################
    # Run Kalman smoother
    #############################
    smoother = KalmanSmoother(kf)
    smoothed_states, smoothed_covs = smoother.smooth()

    #############################
    # Extract means
    #############################
    filtered_mu = kf.filtered_states[:, 0] + kf.filtered_states[:, 2]
    smoothed_mu = smoothed_states[:, 0] + smoothed_states[:, 2]
    
    #############################
    #%% Plot filtered mean with confidence intervals
    #############################
    filtered_std = np.sqrt(kf.filtered_covs[:, 0, 0] + kf.filtered_covs[:, 2, 2])
    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, y, label=r"$y_t$ (observed)", color="tab:blue", alpha=0.5)
    plt.plot(ts.index, ts.mu[1:], label=r"True $\mu_t$", linestyle="--", color="tab:green")
    plt.plot(ts.index, filtered_mu, label="Filtered mean", linestyle="-.", color="tab:orange")
    plt.fill_between(
        ts.index,
        filtered_mu - 1.96 * filtered_std,
        filtered_mu + 1.96 * filtered_std,
        color="tab:orange",
        alpha=0.2,
        label="Filtered 95% CI"
    )
    plt.xlabel("Time")
    plt.ylabel(r"$y_t$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #############################
    #%% Plot smoothed mean with confidence intervals
    #############################
    smoothed_std = np.sqrt(smoothed_covs[:, 0, 0] + smoothed_covs[:, 2, 2])
    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, y, label=r"$y_t$ (observed)", color="tab:blue", alpha=0.5)
    plt.plot(ts.index, ts.mu[1:], label=r"True $\mu_t$", linestyle="--", color="tab:green")
    plt.plot(ts.index, smoothed_mu, label="Smoothed mean", linestyle="-", color="tab:red")
    plt.fill_between(
        ts.index,
        smoothed_mu - 1.96 * smoothed_std,
        smoothed_mu + 1.96 * smoothed_std,
        color="tab:red",
        alpha=0.2,
        label="Smoothed 95% CI"
    )
    plt.xlabel("Time")
    plt.ylabel(r"$y_t$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
