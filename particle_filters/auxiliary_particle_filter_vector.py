from SLLT.particle_filters.particle_filter_base_vector import ParticleFilter, open_pf
from SLLT.particle_filters.resampling.resampler import naive_search, cumulative_sum, Resampler
from scipy.stats import genextreme as gev

import numpy as np


class AuxiliaryParticleFilter(ParticleFilter):
    """
    Notes:
        * State is (m_t, d_t, gamma_t, ..., gamma_t-s+1)
    """

    def __init__(self,
                 number_of_particles,
                 process_noise_vector,
                 n_time_steps,
                 parameters,
                 period,
                 confidence_level,
                 number_of_effective_particles_threshold=1):
        """
        Initialize the Auxiliary Particle Filter using the particle_filter_base module.

        :param number_of_particles: Number of particles
        :param evolution_matrix: 2D evolution matrix (A)
        :param process_noise_vector: 2D process noise vector (Q)
        :param parameters: GEV parameters: [sigma, xi]
        :param resampling_algorithm: Algorithm that must be used for core
        """
        
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, process_noise_vector, n_time_steps,
                                parameters, period, confidence_level)
        self.resampling_threshold = number_of_effective_particles_threshold
        self.resampling_algorithm = 'multinomial'
        self.resampler = Resampler()

    @staticmethod
    def sample_multinomial_indices(particles):
        """
        Particles indices are sampled with replacement proportional to their weight an in arbitrary order. This leads
        to a maximum variance on the number of times a particle will be resampled, since any particle will be resampled
        between 0 and N times.

        :param particles: (nd array) N+(d+1) array where first element is weight and the rest is the state.
        :return: Resampled indices.
        """
        weights = particles[:, 0]
        # Number of samples
        N = len(weights)

        # Compute cumulative sum
        Q = np.cumsum(weights)

        # Draw N random samples u
        u = np.random.uniform(1e-6, 1, N)

        # Get indices for which cumulative sum is above u using naive search
        new_indices = np.searchsorted(Q, u)

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
        weights = self.particles[:, 0]
        sum_weights_squared = np.sum(weights ** 2)

        #return 1.0 / sum_weights_squared < self.resampling_threshold
        return False
    
    def update(self,  measurement):
        """
        Apply the auxiliary particle filter to the current state estimate.
        """

        # First loop: propagate characterizations and compute weights

        tmp_particles = self.propagate_particles(self.particles)
        tmp_likelihoods = self.compute_likelihood(tmp_particles, measurement)
        tmp_particles[:,0] *= tmp_likelihoods

        tmp_particles = self.normalize_weights(tmp_particles)

        # Resample indices from propagated particles

        new_indices = self.sample_multinomial_indices(tmp_particles)

        # Second loop: now propagate the state of all particles indices that survived

        new_samples = self.propagate_particles(self.particles[new_indices])
        tmp_likelihoods[tmp_likelihoods < 1e-10] = 1e-10
        new_samples[:,0] = self.compute_likelihood(new_samples, measurement) / tmp_likelihoods[new_indices]
        #for idx in new_indices:

            # Get particle state associated with current index from original set of particles
        #    par = self.particles[idx]

        #    # Propagate the particle state
        #    propagated_state = self.propagate_sample(par[1])

        #    # Compute current particle's weight using the measurement likelihood of the characterization
        #    wi_tmp = tmp_likelihoods[idx]
        #    if wi_tmp < 1e-10:
        #        wi_tmp = 1e-10  # avoid division by zero
        #    weight = self.compute_likelihood(propagated_state, measurement) / wi_tmp
        #
        #    # Store
        #    new_samples.append([weight, propagated_state])

        # Update particles
        self.particles = self.normalize_weights(new_samples)
        
        # Resample if needed
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
            
        # Store particles
        self.all_particles[self.time] = self.particles
        
        mu = self.get_mean(self.time)
        self.get_confidence_interval(self.time)
        
        # add this to PF BASE **TODO**
        # Obtain RMSE
        yhat = mu + self.sigma*(np.log(2)**(-self.xi)-1)/self.xi
        self.rmse += (measurement-yhat)**2
        
        # Obtain (approximate) LLH and AIC
        # TODO: weigh with the MC weights
        z = (measurement - mu) / self.sigma
        u = 1 + self.xi * z
        if self.xi != 0 and u > 0:
            self.llh += gev.logpdf(measurement, -self.xi, loc=mu, scale=self.sigma)
        elif self.xi == 0:
            self.llh += gev.logpdf(measurement, 0, loc=mu, scale=self.sigma)
        
        self.aic = 2*(2*self.dim+5) - 2*self.llh
        
        self.time += 1
