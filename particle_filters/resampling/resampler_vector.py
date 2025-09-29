#!/usr/bin/env python

# Numpy
import numpy as np

# Enum
from enum import Enum

# Deep copy samples
import copy

class ResamplingAlgorithms(Enum):
    MULTINOMIAL = 1

class Resampler:
    """
    Resample class that implements different resampling methods.
    """

    def __init__(self):
        """Initialize the Resampler."""
        self.initialized = True
        self.m = None  # Indices of resampled particles

    def resample(self, particles):
        """
        Resample particles using the specified method.

        This method performs resampling of the particles based on their weights. 
        Currently, only multinomial resampling is implemented.

        :param particles: (np.ndarray) Array of shape (N, d+1) where N is the 
                         number of particles, and d is the dimension of the 
                         state vector. The first column contains the particle 
                         weights, and the remaining columns contain the state 
                         vectors.
        :return: (np.ndarray) Resampled particles with uniform weights.
        """
        return self.__multinomial(particles)

    def __multinomial(self, particles):
        """
        Perform multinomial resampling.

        Particles are sampled with replacement proportional to their weight 
        and in arbitrary order. This leads to a maximum variance on the 
        number of times a particle will be resampled, since any particle 
        will be resampled between 0 and N times.

        :param particles: (np.ndarray) Array of particles (as described in `resample`).
        :return: (np.ndarray) Resampled particles with uniform weights.
        """

        weights = particles[:, 0]  # Extract the particle weights
        N = len(weights)  # Number of particles

        # Compute cumulative sum of weights
        Q = np.cumsum(weights)

        # Generate random numbers and find the corresponding particle indices
        u = np.random.uniform(1e-6, 1, N)  # Generate N random numbers between 0 and 1
        self.m = np.searchsorted(Q, u)  # Find the indices of the particles to resample

        # Create new particles with uniform weights and resampled states
        new_particles = np.zeros(particles.shape)  # Initialize an array for the new particles
        new_particles[:, 0] = 1/N  # Assign uniform weights
        new_particles[:, 1:] = particles[self.m, 1:]  # Resample the states based on the indices

        return new_particles  # Return the resampled particles


