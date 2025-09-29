from SLLT.particle_filters.particle_filter_sir_vector import ParticleFilterSIR, open_pf
import numpy as np
class ParticleFilterNEPR(ParticleFilterSIR):
    """
    Notes:
        * Apply approximate number of effective particles core (NEPR)
    """

    def __init__(self,
                 number_of_particles,
                 process_noise_vector,
                 n_time_steps,
                 parameters,
                 period,
                 resampling_algorithm,
                 number_of_effective_particles_threshold,
                 confidence_level):
        """
        Initialize a particle filter that performs resampling whenever the approximated number of effective particles
        falls below a user specified threshold value.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param resampling_algorithm: Algorithm that must be used for core.
        :param number_of_effective_particles_threshold: Resample whenever approximate number of effective particles
        falls below this value.
        """
        # Initialize sir particle filter class
        ParticleFilterSIR.__init__(self, number_of_particles, process_noise_vector, n_time_steps,
                                   parameters, period, resampling_algorithm, confidence_level)

        # Set NEPR specific properties
        self.resampling_threshold = number_of_effective_particles_threshold

    def needs_resampling(self):
        """
        Override method that determines whether or not a step is needed for the current particle filter state
        estimate. Resampling only occurs if the approximated number of effective particles falls below the
        user-specified threshold. Approximate number of effective particles: 1 / (sum_i^N wi^2), P_N^2 in [1].

        [1] Martino, Luca, Victor Elvira, and Francisco Louzada. "Effective sample size for importance sampling based on
        discrepancy measures." Signal Processing 131 (2017): 386-401.

        :return: Boolean indicating whether or not core is needed.
        """
        #
        weights = self.particles[:, 0]
        sum_weights_squared = np.sum(weights ** 2)
        print('number of effective particles', 1.0 / sum_weights_squared)
        return 1.0 / sum_weights_squared < self.resampling_threshold
