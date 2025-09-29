from SLLT.particle_filters.particle_filter_base import ParticleFilter, open_pf
from SLLT.particle_filters.resampling.resampler import naive_search, cumulative_sum, Resampler


import numpy as np


class AuxiliaryParticleFilter(ParticleFilter):
    """
    Notes:
        * State is (m_t, d_t, gamma_t, ..., gamma_t-s+1)
    """

    def __init__(self,
                 number_of_particles,
                 evolution_matrix,
                 process_noise_vector,
                 parameters,
                 period,
                 confidence_level,
                 number_of_effective_particles_threshold=1,
                 resampling_algorithm='multinomial'):
        """
        Initialize the Auxiliary Particle Filter using the particle_filter_base module.

        :param number_of_particles: Number of particles
        :param evolution_matrix: 2D evolution matrix (A)
        :param process_noise_vector: 2D process noise vector (Q)
        :param parameters: GEV parameters: [sigma, xi]
        :param resampling_algorithm: Algorithm that must be used for core
        """
        
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, evolution_matrix, process_noise_vector, 
                                parameters, period, confidence_level)
        self.resampling_threshold = number_of_effective_particles_threshold
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()

    @staticmethod
    def sample_multinomial_indices(samples):
        """
        Particles indices are sampled with replacement proportional to their weight an in arbitrary order. This leads
        to a maximum variance on the number of times a particle will be resampled, since any particle will be resampled
        between 0 and N times.
        Computational complexity: O(N log(M)

        :param samples: Samples that must be resampled.
        :return: Resampled indices.
        """

        # Number of samples
        N = len(samples)

        # Get list with only weights
        weights = [weighted_sample[0] for weighted_sample in samples]

        # Compute cumulative sum
        Q = cumulative_sum(weights)

        # As long as the number of new samples is insufficient
        n = 0
        new_indices = []
        while n < N:
            # Draw a random sample u
            u = np.random.uniform(1e-6, 1, 1)[0]

            # Get first sample for which cumulative sum is above u using naive search
            m = naive_search(Q, u)

            # Store index
            new_indices.append(m)

            # Added another sample
            n += 1

        return new_indices
    
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
        sum_weights_squared = 0
        for par in self.particles:
            sum_weights_squared += par[0] * par[0]

        #return 1.0 / sum_weights_squared < self.resampling_threshold
        return False
    def update(self,  measurement):
        """
        Apply the auxiliary particle filter to the current state estimate.
        """

        # First loop: propagate characterizations and compute weights
        tmp_particles = []
        tmp_likelihoods = []

        for par in self.particles:

            # Compute characterization
            propagated_state = self.propagate_sample(par[1])

            # Compute and store current particle's weight
            likelihood = self.compute_likelihood(propagated_state, measurement) 
            weight = likelihood * par[0]
            tmp_likelihoods.append(likelihood)

            # Store (notice mu will not be used later)
            tmp_particles.append([weight, propagated_state])

        # Normalize particle weights
        tmp_particles = self.normalize_weights(tmp_particles)
############### TOT HIER PERFECTE OVEREENKOMST ############################
        # Resample indices from propagated particles

        new_indices = self.sample_multinomial_indices(tmp_particles)

        # Second loop: now propagate the state of all particles indices that survived
        new_samples = []

        for idx in new_indices:

            # Get particle state associated with current index from original set of particles
            par = self.particles[idx]

            # Propagate the particle state

            propagated_state = self.propagate_sample(par[1])

            # Compute current particle's weight using the measurement likelihood of the characterization
            wi_tmp = tmp_likelihoods[idx]
            if wi_tmp < 1e-10:
                wi_tmp = 1e-10  # avoid division by zero
            weight = self.compute_likelihood(propagated_state, measurement) / wi_tmp

            # Store
            new_samples.append([weight, propagated_state])

        # Update particles
        self.particles = self.normalize_weights(new_samples)
        
        # Resample if needed
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
            
        # Store particles
        self.all_particles.append(self.particles)
