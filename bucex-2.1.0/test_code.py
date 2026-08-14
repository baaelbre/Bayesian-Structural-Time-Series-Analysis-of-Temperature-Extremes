import numpy as np
import matplotlib.pyplot as plt

def simulate(
    T=100,
    sigma_alpha2=0.1,
    sigma_epsilon2=1.0,
    seed=None
):
    """
    Local-level model:
        alpha[t+1] = alpha[t] + eta[t]
        y[t]       = alpha[t+1] + epsilon[t]

    where:
        eta[t]     ~ N(0, sigma_alpha2)
        epsilon[t] ~ N(0, sigma_epsilon2)
    """
    rng = np.random.default_rng(seed)

    alpha = np.empty(T + 1)
    y = np.empty(T)

    alpha[0] = 0.0

    for t in range(T):
        alpha[t + 1] = (
            alpha[t]
            + rng.normal(0.0, np.sqrt(sigma_alpha2))
        )

        y[t] = (
            alpha[t + 1]
            + rng.normal(0.0, np.sqrt(sigma_epsilon2))
        )

    return alpha, y


def forward_filter(
    y,
    sigma_alpha2,
    sigma_epsilon2,
    m0=0.0,
    C0=1.0
):
    """
    Kalman filter for the local-level model.

    m[t], C[t] describe:
        alpha[t] | y[0:t-1]
    after assimilating the first t observations.
    """
    T = len(y)

    m = np.empty(T + 1)
    C = np.empty(T + 1)

    # Optional diagnostic quantities
    a = np.empty(T)
    R = np.empty(T)
    v = np.empty(T)
    F = np.empty(T)
    K = np.empty(T)

    m[0] = m0
    C[0] = C0

    for t in range(T):
        # Prediction:
        # alpha[t+1] | y[0:t-1]
        a[t] = m[t]
        R[t] = C[t] + sigma_alpha2

        # Observation prediction error
        v[t] = y[t] - a[t]
        F[t] = R[t] + sigma_epsilon2

        # Kalman gain
        K[t] = R[t] / F[t]

        # Filtering update:
        # alpha[t+1] | y[0:t]
        m[t + 1] = a[t] + K[t] * v[t]
        C[t + 1] = (1.0 - K[t]) * R[t]

    diagnostics = {
        "a": a,
        "R": R,
        "v": v,
        "F": F,
        "K": K,
    }

    return m, C, diagnostics


def backward_sample(
    m,
    C,
    sigma_alpha2,
    seed=None
):
    """
    Draw alpha[0:T] jointly from the smoothing distribution.

    m[t], C[t] are the filtered moments of alpha[t].
    """
    rng = np.random.default_rng(seed)

    T = len(m) - 1
    alpha = np.empty(T + 1)

    # Draw final state from its filtered distribution
    alpha[T] = rng.normal(
        m[T],
        np.sqrt(C[T])
    )

    for t in range(T - 1, -1, -1):
        # Prediction variance for alpha[t+1] given y[0:t-1]
        R_next = C[t] + sigma_alpha2

        # Backward smoothing gain
        J = C[t] / R_next

        # Conditional moments of alpha[t] given alpha[t+1]
        h = m[t] + J * (alpha[t + 1] - m[t])
        H = C[t] - J**2 * R_next

        alpha[t] = rng.normal(
            h,
            np.sqrt(max(H, 0.0))
        )

    return alpha


if __name__ == "__main__":
    sigma_alpha2 = 0.1
    sigma_epsilon2 = 1.0

    true_alpha, y = simulate(
        T=100,
        sigma_alpha2=sigma_alpha2,
        sigma_epsilon2=sigma_epsilon2,
        seed=42
    )

    m, C, diagnostics = forward_filter(
        y,
        sigma_alpha2=sigma_alpha2,
        sigma_epsilon2=sigma_epsilon2
    )

    # do this a 1000 times to get a posterior distribution of the states
    alpha_draws = []
    for _ in range(1000):
        alpha_draw = backward_sample(
            m,
            C,
            sigma_alpha2=sigma_alpha2,
            seed=43
        )
        alpha_draws.append(alpha_draw)

    plt.figure(figsize=(10, 6))
    plt.plot(true_alpha[1:], label="True state", color="black", linewidth=2)
    plt.plot(y, label="Observations", color="gray", alpha=0.5)
    plt.plot(m[1:], label="Filtered mean", color="blue", linewidth=2)
    for alpha_draw in alpha_draws:
        plt.plot(alpha_draw, color="red", alpha=0.1)

    print("Shapes:")
    print("true states:", true_alpha.shape)
    print("observations:", y.shape)
    print("filtered means:", m.shape)
    print("smoothed draw:", alpha_draw.shape)

    print("\nFinal state:")
    print("true:", true_alpha[-1])
    print("filtered mean:", m[-1])
