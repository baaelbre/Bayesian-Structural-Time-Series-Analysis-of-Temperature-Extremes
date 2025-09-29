#############################
#%% Import necessary libraries
#############################
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.stats import norm
from simulator.mean_time_series import Mean_Time_Series
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm as normal_dist


class KalmanFilter:
    """
    Kalman Filter using structural notation from RJ-PMCMC manuscript.
    Estimates state means and covariances and stores all quantities for downstream smoothing.
    """
    def __init__(self, sigma, process_noises, include_level=True, include_trend=True, include_seasonality=True, period=12):
        self.include_level = include_level
        self.include_trend = include_trend
        self.include_seasonality = include_seasonality
        self.period = period
        self.sigma = sigma

        self.r = int(include_level) + int(include_trend) + (period - 1 if include_seasonality else 0)

        # Transition matrix A
        self.A = np.zeros((self.r, self.r))
        idx = 0

        if include_level:
            self.A[idx, idx] = 1
            if include_trend:
                self.A[idx, idx + 1] = 1
            idx += 1

        if include_trend:
            self.A[idx, idx] = 1
            idx += 1

        if include_seasonality:
            self.A[idx, idx:idx + period - 1] = -1
            for i in range(1, period - 1):
                self.A[idx + i, idx + i - 1] = 1

        # Observation matrix F
        self.F = np.zeros((1, self.r))
        obs_idx = 0
        if include_level:
            self.F[0, obs_idx] = 1
            obs_idx += 1
        elif include_trend:
            obs_idx += 1

        if include_trend:
            obs_idx += 1

        if include_seasonality:
            self.F[0, obs_idx] = 1  # only current seasonal effect

        # Process noise matrix Q
        self.Q = np.diag(process_noises)

        # Storage attributes
        self.filtered_states = None
        self.filtered_covs = None
        self.predicted_states = None
        self.predicted_covs = None
        self.gain_matrices = None
        self.innovations = None
        self.loglik = None

    def initialize(self, m0=None, P0=None):
        if m0 is None:
            m0 = np.zeros(self.r)
        if P0 is None:
            P0 = np.eye(self.r)
        return m0, P0

    def filter(self, y, m0=None, P0=None):
        T = len(y)
        m, P = self.initialize(m0, P0)

        # Allocate arrays
        self.filtered_states = np.zeros((T, self.r))
        self.filtered_covs = np.zeros((T, self.r, self.r))
        self.predicted_states = np.zeros((T, self.r))
        self.predicted_covs = np.zeros((T, self.r, self.r))
        self.gain_matrices = np.zeros((T, self.r, 1))
        self.innovations = np.zeros((T, 1))
        self.loglik = 0.0

        for t in range(T):
            # Predict step
            m_pred = self.A @ m
            P_pred = self.A @ P @ self.A.T + self.Q

            # Save prediction
            self.predicted_states[t] = m_pred
            self.predicted_covs[t] = P_pred

            # Observation update
            y_pred = self.F @ m_pred
            S = self.F @ P_pred @ self.F.T + self.sigma**2
            K = P_pred @ self.F.T / S
            v = y[t] - y_pred

            m = m_pred + (K.flatten() * v)
            P = P_pred - K @ self.F @ P_pred

            # Save results
            self.filtered_states[t] = m
            self.filtered_covs[t] = P
            self.gain_matrices[t] = K
            self.innovations[t] = v
            self.loglik += -0.5 * (np.log(2 * np.pi) + np.log(S) + (v ** 2) / S)

        return self.filtered_states, self.filtered_covs, self.loglik


#############################
#%% Run Kalman Filter Example
##############################

if __name__ == '__main__':
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

    kf = KalmanFilter(
        sigma=10.0,
        process_noises=[0.1] * 13,
        include_level=True,
        include_trend=True,
        include_seasonality=True,
        period=12
    )
    states, covs, loglik = kf.filter(y)
    
    mu = states[:, 0] + states[:, 2]
    std = np.sqrt(np.clip(covs[:, 0, 0] + covs[:, 2, 2] + 2 * covs[:, 0, 2], 0, np.inf))

    lower, upper = mu - 1.64 * std, mu + 1.64 * std # 90% confidence interval for mu_t
    
    # Predictive standard deviation for y_t
    predictive_std = np.sqrt(std**2 + kf.sigma**2)

    # 90% predictive interval for y_t
    pred_lower = mu - 1.64 * predictive_std
    pred_upper = mu + 1.64 * predictive_std


    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, y, label=r"$y_t$ (observed)", linestyle='-', color='tab:blue')
    plt.plot(ts.index, ts.mu[1:], label=r"$\mu_t$ (latent mean)", linestyle='--', color='tab:green')
    plt.plot(ts.index, mu, label=r"$\mu_t$ (filtered mean)", linestyle='--', color='tab:orange')
    plt.fill_between(ts.index, lower, upper, color='tab:orange', alpha=0.3, label=r"90% interval for $\mu_t$")
    plt.fill_between(ts.index, pred_lower, pred_upper, color='tab:blue', alpha=0.2, label=r"90% predictive interval for $y_t$")
    plt.xlabel("Time")
    plt.ylabel(r"Observations $y_t$")
    plt.title("Kalman Filter testing")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
