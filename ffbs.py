import numpy as np


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

def sample_innovation_variance(
    alpha,
    a0=2.0,
    b0=0.01,
    rng=None
):
    rng = np.random.default_rng(rng)

    innovations = np.diff(alpha)
    T = len(innovations)

    a_post = a0 + T / 2
    b_post = b0 + 0.5 * np.sum(innovations**2)

    precision = rng.gamma(
        shape=a_post,
        scale=1.0 / b_post
    )

    q = 1.0 / precision

    return q



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ============================================================
    # 1. Simulate data
    # ============================================================

    T = 100

    sigma_alpha2_true = 0.1
    sigma_epsilon2 = 1.0

    true_alpha, y = simulate(
        T=T,
        sigma_alpha2=sigma_alpha2_true,
        sigma_epsilon2=sigma_epsilon2,
        seed=42
    )

    # ============================================================
    # 2. One filtering and smoothing run
    # ============================================================

    m, C, diagnostics = forward_filter(
        y,
        sigma_alpha2=sigma_alpha2_true,
        sigma_epsilon2=sigma_epsilon2
    )

    alpha_draw = backward_sample(
        m,
        C,
        sigma_alpha2=sigma_alpha2_true,
        seed=43
    )

    print("Shapes:")
    print("true states:   ", true_alpha.shape)
    print("observations:  ", y.shape)
    print("filtered means:", m.shape)
    print("smoothed draw: ", alpha_draw.shape)

    print("\nFinal state:")
    print("true:          ", true_alpha[-1])
    print("filtered mean: ", m[-1])
    print("posterior draw:", alpha_draw[-1])

    # ============================================================
    # 3. Gibbs sampler
    # ============================================================

    n_iter = 5000
    burn = 1000

    q_draws = np.empty(n_iter)

    # One complete state trajectory per MCMC iteration
    alpha_draws = np.empty((n_iter, T + 1))

    # Initial value
    q = 0.5

    # Observation variance is treated as known
    r = sigma_epsilon2

    for iteration in range(n_iter):

        # Step 1: sample the complete state trajectory
        m, C, _ = forward_filter(
            y,
            sigma_alpha2=q,
            sigma_epsilon2=r
        )

        alpha = backward_sample(
            m,
            C,
            sigma_alpha2=q
        )

        # Step 2: sample the state innovation variance
        q = sample_innovation_variance(
            alpha,
            a0=2.0,
            b0=0.01
        )

        # Store both draws
        q_draws[iteration] = q
        alpha_draws[iteration, :] = alpha

    # ============================================================
    # 4. Remove burn-in and calculate posterior summaries
    # ============================================================

    q_post = q_draws[burn:]
    alpha_post = alpha_draws[burn:, :]

    alpha_mean = np.mean(alpha_post, axis=0)
    alpha_median = np.median(alpha_post, axis=0)

    alpha_lower = np.quantile(alpha_post, 0.025, axis=0)
    alpha_upper = np.quantile(alpha_post, 0.975, axis=0)

    print("\nPosterior innovation variance:")
    print("true q:          ", sigma_alpha2_true)
    print("posterior mean:  ", np.mean(q_post))
    print("posterior median:", np.median(q_post))
    print(
        "95% interval:    ",
        np.quantile(q_post, [0.025, 0.975])
    )

    # ============================================================
    # 5. Plot results
    # ============================================================

    time_states = np.arange(T + 1)
    time_observations = np.arange(1, T + 1)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 8)
    )

    # ------------------------------------------------------------
    # Panel 1: posterior state estimate
    # ------------------------------------------------------------

    ax = axes[0, 0]

    ax.scatter(
        time_observations,
        y,
        color="gray",
        alpha=0.45,
        s=16,
        label="Observations"
    )

    ax.plot(
        time_states,
        true_alpha,
        color="black",
        linewidth=2,
        label="True state"
    )

    ax.plot(
        time_states,
        alpha_mean,
        color="tab:blue",
        linewidth=2,
        label="Posterior mean"
    )

    ax.fill_between(
        time_states,
        alpha_lower,
        alpha_upper,
        color="tab:blue",
        alpha=0.2,
        label="95% credible interval"
    )

    ax.set_title("Latent-state posterior")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\alpha_t$")
    ax.legend()

    # ------------------------------------------------------------
    # Panel 2: some complete posterior trajectories
    # ------------------------------------------------------------

    ax = axes[0, 1]

    # Plot every 100th retained trajectory
    selected_draws = alpha_post[::100, :]

    for trajectory in selected_draws:
        ax.plot(
            time_states,
            trajectory,
            color="tab:blue",
            alpha=0.12,
            linewidth=0.8
        )

    ax.plot(
        time_states,
        true_alpha,
        color="black",
        linewidth=2,
        label="True state"
    )

    ax.set_title("Posterior trajectory draws")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\alpha_t$")
    ax.legend()

    # ------------------------------------------------------------
    # Panel 3: trace plot for q
    # ------------------------------------------------------------

    ax = axes[1, 0]

    ax.plot(
        q_draws,
        color="tab:blue",
        linewidth=0.7
    )

    ax.axvline(
        burn,
        color="gray",
        linestyle="--",
        label="End of burn-in"
    )

    ax.axhline(
        sigma_alpha2_true,
        color="black",
        linestyle="--",
        label="True value"
    )

    ax.set_title("Trace plot of innovation variance")
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel(r"$q=\sigma_\alpha^2$")
    ax.legend()

    # ------------------------------------------------------------
    # Panel 4: posterior distribution of q
    # ------------------------------------------------------------

    ax = axes[1, 1]

    ax.hist(
        q_post,
        bins=40,
        density=True,
        color="tab:blue",
        alpha=0.7,
        edgecolor="white"
    )

    ax.axvline(
        sigma_alpha2_true,
        color="black",
        linestyle="--",
        linewidth=2,
        label="True value"
    )

    ax.axvline(
        np.median(q_post),
        color="tab:red",
        linewidth=2,
        label="Posterior median"
    )

    ax.set_title("Posterior innovation variance")
    ax.set_xlabel(r"$q=\sigma_\alpha^2$")
    ax.set_ylabel("Posterior density")
    ax.legend()

    plt.tight_layout()
    plt.show()