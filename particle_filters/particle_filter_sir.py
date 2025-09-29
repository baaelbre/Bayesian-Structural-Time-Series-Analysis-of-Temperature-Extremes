from SLLT.particle_filters.particle_filter_base import ParticleFilter, open_pf
from SLLT.particle_filters.resampling.resampler import Resampler
import copy
import numpy as np


class ParticleFilterSIR(ParticleFilter):
    """
    Notes:
        * State is (alpha_t, beta_t, gamma_t, ..., gamma_t-s+1)
    """

    def __init__(self,
                 number_of_particles,
                 evolution_matrix,
                 process_noise_vector,
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
        ParticleFilter.__init__(self, number_of_particles, evolution_matrix, process_noise_vector, 
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

        # Loop over all particles
        new_particles = []
        for par in self.particles:

            # Propagate the particle state according to the current particle
            propagated_state = self.propagate_sample(par[1])
            
            # Compute current particle's weight
            weight = par[0] * self.compute_likelihood(propagated_state, measurement)
            # Store
            new_particles.append([weight, propagated_state])

        # Update particles
        self.particles = self.normalize_weights(new_particles)

        # Resample if needed
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
        
        # Store particles
        self.all_particles.append(self.particles)
