from SLLT.particle_filters.particle_filter_base_vector import ParticleFilter, open_pf
from SLLT.particle_filters.resampling.resampler_vector import Resampler
import copy
import numpy as np
from tqdm import tqdm


class ParticleFilterSIR(ParticleFilter):
    """
    Notes:
        * State is (alpha_t, beta_t, gamma_t, ..., gamma_t-s+1)
    """

    def __init__(self,
                 number_of_particles,
                 process_noise_vector,
                 n_time_steps,
                 parameters,
                 period,
                 resampling_algorithm,
                 confidence_level):
        """
        Initialize the SIR particle filter using the particle_filter_base module.

        :param number_of_particles: Number of particles
        :param evolution_matrix: 2D evolution matrix (A)
        :param process_noise_vector: 2D process noise vector (Q)
        :param parameters: GEV parameters: [sigma, xi]
        :param resampling_algorithm: Algorithm that must be used for core
        """
        
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, process_noise_vector, n_time_steps,
                                parameters, period, confidence_level)
        # Set SIR specific properties
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()

    def needs_resampling(self):
        """
        Method that determines whether not a resampling step is needed for the current particle filter state estimate. 
        The sequential importance sampling (SIR) scheme resamples every time step hence always return true.

        :return: Boolean indicating whether or not core is needed.
        """
        return True
    
    def update(self, measurement):
        """
        Process a measurement given the proposed new state vector and resample if needed.

        :param measurement: Current measurement.
        :param parameters: [sigma, xi].
        """
        # Predict states
        new_particles = self.propagate_particles(self.particles)
        # Update weights
        new_particles[:,0] *= self.compute_likelihood(new_particles, measurement)
        # Normalize weights
        self.particles = self.normalize_weights(new_particles)
            
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles)
        
        # Store particles
        self.all_particles[self.time] = self.particles

        self.get_mean(self.time)
        self.get_confidence_interval(self.time)
        
        self.time += 1
        
