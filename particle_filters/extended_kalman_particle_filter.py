# load required packages
from abc import abstractmethod
import copy
import pickle
from datetime import datetime
import numpy as np
from scipy.stats import genextreme, norm, multivariate_normal

from scipy.optimize import brentq
from tqdm import tqdm
from SLLT.particle_filters.extended_kalman_filter import ExtendedKalmanFilter
from SLLT.particle_filters.resampling.resampler_vector import Resampler

class ExtendedKalmanParticleFilter:
    """
    Extended Kalman Particle Filter (EKPF) for GEV-distributed observations.

    This class implements an Extended Kalman Particle Filter (EKPF) specifically 
    designed for analyzing time series data where the observations follow a 
    Generalized Extreme Value (GEV) distribution. 

    The EKPF estimates the time-varying location parameter of the GEV distribution, 
    which is modeled as a random walk in state space with a trend and seasonal 
    component.

    A key feature of this EKPF is the use of a Gaussian approximation, provided by 
    the Extended Kalman Filter (EKF), as the proposal distribution for the 
    particle filter. This enhances the efficiency and accuracy of the filter, 
    especially when dealing with the nonlinearities inherent in the GEV distribution.

    To save memory and computation power, a global proposal distribution (namely
    the parallel runned EKF) is used for 
    all particles instead of individual proposals for each particle.

    This implementation is designed for offline filtering, where the number of time 
    steps is known in advance. It provides methods for filtering the data, computing
    the marginal likelihood, and calculating the Akaike Information Criterion (AIC)
    for model selection.
    """
    def __init__(self, number_of_particles, process_noise_vector, n_time_steps, 
                 parameters, period=4, confidence_level=0.90, resampling_threshold=None):
        """
        Initialize the Extended Kalman Particle Filter (EKPF).

        :param number_of_particles: (int) Number of particles.
        :param process_noise_vector: (list) 3D process noise vector Q for the EKF.
        :param n_time_steps: (int) Number of time steps in the data.
        :param parameters: (list) GEV parameters [sigma, xi].
        :param period: (int) Seasonal period.
        :param confidence_level: (float) Confidence level for state estimates.
        :param resampling_threshold: (int) Threshold for resampling based on the number of effective particles.
        """
        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles
        self.period = period
        self.n_time_steps = n_time_steps

        # Initialize particle states and weights
        self.particles = np.zeros((self.n_particles, 3 + period - 1))  # Array to store particles (weight, state)
        self.all_particles = np.zeros((self.n_time_steps + 1, self.n_particles, 3 + period - 1))  # Store particles at all time steps

        # Initialize arrays to store results (means, confidence intervals)
        self.means = np.zeros((self.n_time_steps + 1, 3))  # Mean of the state variables over time
        self.means_alpha = np.zeros(self.n_time_steps + 1)  # Mean of alpha (trend) over time
        self.means_beta = np.zeros(self.n_time_steps + 1)  # Mean of beta (slope) over time
        self.means_gamma = np.zeros(self.n_time_steps + 1)  # Mean of gamma (seasonal) over time
        self.means_mu = np.zeros(self.n_time_steps + 1)  # Mean of mu (location) over time
        self.lwr_alpha, self.upr_alpha = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)  # Confidence intervals for alpha
        self.lwr_beta, self.upr_beta = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)  # Confidence intervals for beta
        self.lwr_gamma, self.upr_gamma = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)  # Confidence intervals for gamma
        self.lwr_mu, self.upr_mu = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)  # Confidence intervals for mu

        self.process_noise_vector = process_noise_vector  # Store the process noise vector

        # Initialize an EKF
        self.ekfs = [ExtendedKalmanFilter(process_noise_vector=process_noise_vector,
                                          n_time_steps=n_time_steps,
                                          parameters=parameters,
                                          period=period,
                                          confidence_level=confidence_level) for _ in range(number_of_particles)]

        # Initialize evolution matrix and process noise matrix from the EKF
        self.evolution_matrix = self.ekfs[0].evolution_matrix
        self.process_noise_matrix = self.ekfs[0].process_noise_matrix

        # Store GEV parameters
        self.parameters = parameters
        self.sigma, self.xi = self.parameters  # Extract sigma and xi from the parameters list
        self.confidence_level = confidence_level  # Store the confidence level

        # Initialize log-likelihood and AIC to 0
        self.llh, self.aic = 0, 0

        # Set SIR specific properties
        self.resampler = Resampler()
        if resampling_threshold is None:
            self.resampling_threshold = 0.5 * self.n_particles
        else:
            self.resampling_threshold = resampling_threshold

        self.time = 0  # Initialize the current time step to 0
        print('Initializing particle filter with {} particles and {} time steps'.format(self.n_particles, self.n_time_steps))
        print('filtering...')

    def initialize(self, m_0, P_0):
        """
        Initialize the EKPF with a Gaussian distribution for the initial state.

        This method initializes the particles of the EKPF by sampling from a 
        multivariate Gaussian distribution with the specified mean (`m_0`) and 
        covariance matrix (`P_0`). It also initializes each Extended Kalman Filter 
        (EKF) component with the same initial state and covariance.

        The particles array is structured as follows (example with N=4 particles):

        ```
        self.particles = 
        [
          [weight_1, alpha_1, beta_1, gamma_1, gamma_1_lag1, gamma_1_lag2],  # Particle 1
          [weight_2, alpha_2, beta_2, gamma_2, gamma_2_lag1, gamma_2_lag2],  # Particle 2
          [weight_3, alpha_3, beta_3, gamma_3, gamma_3_lag1, gamma_3_lag2],  # Particle 3
          [weight_4, alpha_4, beta_4, gamma_4, gamma_4_lag1, gamma_4_lag2]   # Particle 4
        ]
        ```
        where:
        - `weight_i`: The weight of the i-th particle.
        - `alpha_i`: The trend component of the GEV location parameter for the i-th particle.
        - `beta_i`: The slope of the trend for the i-th particle.
        - `gamma_i`: The current seasonal component for the i-th particle.
        - `gamma_i_lag1`: The seasonal component from the previous time step for the i-th particle.
        - `gamma_i_lag2`: The seasonal component from two time steps ago for the i-th particle.

        :param m_0: (np.array) Mean vector of the initial state distribution.
        :param P_0: (np.array) Covariance matrix of the initial state distribution.
        """

        # Initialize particles with uniform weights
        weight = 1.0 / self.n_particles  # Calculate the initial weight for each particle

        # Create the particles array
        self.particles = np.concatenate(
            (
                [np.full(self.n_particles, weight)],  # Array of uniform weights
                np.random.multivariate_normal(m_0, np.diag(P_0), size=self.n_particles).T  # Sample particle states from Gaussian
            )
        ).T  

        # Store the initial particles in the all_particles array
        self.all_particles[0] = self.particles  
        for i, ekf in enumerate(self.ekfs):
            ekf.initialize(self.particles[i,1:], P_0)  # Initialize each EKF with the same initial state and covariance
        # Calculate and store initial mean and confidence intervals
        self.get_mean(self.time)  
        self.get_confidence_interval(self.time)  
        self.time = 1  # Set the current time step to 1
        
    def filter_bootstrap(self, all_measurements):
        """
        Filter the data using the particle filter with bootstrap proposal.

        :param all_measurements: (list) of all block maxima.
        """
        for measurement in tqdm(all_measurements):
            self.predict_bootstrap()
            self.update_bootstrap(measurement)
        
    def predict_bootstrap(self):
        """
        Perform the prediction step for the Bootstrap Particle Filter.

        :param measurement: Measurement to update the EKFs.
        """
        for i, ekf in enumerate(self.ekfs):
            # EKF predicts the next state
            ekf.predict()
            
            # Time management
            ekf.time = self.time
            
            # Draw a new particle from the EKF's proposal distribution
            new_particle = np.random.multivariate_normal(mean=ekf.state, cov=ekf.covariance)
            
            # Update the particle state
            self.particles[i, 1:] = new_particle
    
    def update_bootstrap(self, measurement):
        """
        Perform the update step for the Bootstrap Particle Filter.

        :param measurement: The measurement at the current time step.
        """
        # Step 1: Compute likelihoods based on the GEV likelihood function
        likelihood_particles = self.compute_likelihood(self.particles, measurement)

        # Step 2: Compute weights
        for i, ekf in enumerate(self.ekfs):
            # EKF state and covariance represent the proposal distribution
            proposal_mean = ekf.state[:3]
            proposal_cov = ekf.covariance[:3,:3]
            
            # Compute the transition density for this particle
            previous_state = self.all_particles[self.time - 1, i, 1:]
            predicted_state = (self.evolution_matrix @ previous_state.T).T
            transition_density = multivariate_normal.pdf(
                self.particles[i, 1:4], mean=predicted_state[:3], cov=np.diag(self.process_noise_vector)
            )
            
            # Compute the proposal density
            proposal_density = multivariate_normal.pdf(
                self.particles[i, 1:4], mean=proposal_mean, cov=proposal_cov
            )
            #print(f"Particle {i}: transition_density={transition_density}, proposal_density={proposal_density},\
            #      likelihood_density={likelihood_particles[i]}, product={likelihood_particles[i] * transition_density / proposal_density}")
            # Update the weight of the particle
            self.particles[i, 0] *= likelihood_particles[i]
        
        # Step 3: Normalize weights
        self.particles[:, 0] /= np.sum(self.particles[:, 0])
        
        # Step 4: Resample if necessary
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles)
            print('Resampling at time step {}'.format(self.time))
            
        # Step 5: Store results and advance time
        self.all_particles[self.time] = self.particles
        self.get_mean(self.time)
        self.get_confidence_interval(self.time)
        self.time += 1
            
            

    def filter(self, all_measurements):
        """
        Filter the data using the particle filter with EKF proposal.

        :param all_measurements: (list) of all block maxima.
        """
        for measurement in tqdm(all_measurements):
            self.predict(measurement)
            self.update(measurement)

    def predict(self, measurement):
        """
        Perform the prediction step for the Extended Kalman Particle Filter (EKPF).

        :param measurement: Measurement to update the EKFs.
        """
        for i, ekf in enumerate(self.ekfs):
            # EKF predicts the next state
            ekf.predict()
            
            # EKF refines its state estimate based on the measurement
            ekf.update(measurement)
            
            # Time management
            ekf.time = self.time
            
            # Draw a new particle from the EKF's proposal distribution
            new_particle = np.random.multivariate_normal(mean=ekf.state, cov=ekf.covariance)
            
            # Update the particle state
            self.particles[i, 1:] = new_particle


    def update(self, measurement):
        """
        Perform the update step for the Extended Kalman Particle Filter (EKPF).

        After each update step, align the EKF mean (state) with the corresponding particle state. 
        This ensures consistency between the EKF and the particle representation, preventing divergence.

        Reasons for alignment:
        - Particle Representation Alignment: The EKF is used as the proposal for the particle. 
          Once the particle is sampled, the EKF mean should be set to the particle state to represent 
          the particle’s belief in state space.
        - Resetting the Proposal: By aligning the EKF mean to the particle state, the EKF is effectively 
          "re-centered" around the particle for the next prediction-update cycle.
        - Covariance Update: The EKF's covariance naturally reflects the process and measurement updates. 
          However, it must also be consistent with the new particle state, ensuring correct propagation 
          during the next time step.

        :param measurement: The measurement at the current time step.
        """
        # Step 1: Compute likelihoods based on the GEV likelihood function
        likelihood_particles = self.compute_likelihood(self.particles, measurement)

        # Step 2: Compute weights
        for i, ekf in enumerate(self.ekfs):
            # EKF state and covariance represent the proposal distribution
            proposal_mean = ekf.state[:3]
            proposal_cov = ekf.covariance[:3,:3]
            
            # Compute the transition density for this particle
            previous_state = self.all_particles[self.time - 1, i, 1:]
            predicted_state = (self.evolution_matrix @ previous_state.T).T
            transition_density = multivariate_normal.pdf(
                self.particles[i, 1:4], mean=predicted_state[:3], cov=np.diag(self.process_noise_vector)
            )
            
            # Compute the proposal density
            proposal_density = multivariate_normal.pdf(
                self.particles[i, 1:4], mean=proposal_mean, cov=proposal_cov
            )
            #print(f"Particle {i}: transition_density={transition_density}, proposal_density={proposal_density},\
            #      likelihood_density={likelihood_particles[i]}, product={likelihood_particles[i] * transition_density / proposal_density}")
            # Update the weight of the particle
            self.particles[i, 0] *= likelihood_particles[i] * transition_density / proposal_density
            
            # Update the EKF state to align with the particle state
            ekf.state = self.particles[i, 1:] 

        # Step 3: Normalize weights
        self.particles[:, 0] /= np.sum(self.particles[:, 0])

        # Step 4: Resample if necessary
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles)
            indices = self.resampler.m
            self.ekfs = [self.ekfs[i] for i in indices]
            print('Resampling at time step {}'.format(self.time))
        
        for ekf in self.ekfs:
            ekf.time = self.time
        # Step 5: Store results and advance time
        self.all_particles[self.time] = self.particles
        self.get_mean(self.time)
        self.get_confidence_interval(self.time)
        self.time += 1

    def needs_resampling(self):
            """
            Determine if resampling is needed based on effective particle count.

            This method checks if the effective number of particles falls below a 
            threshold. Resampling is performed to avoid particle degeneracy, where 
            a few particles dominate the weights, leading to poor approximation 
            of the posterior distribution.

            The effective number of particles is approximated using the formula:
            1 / (sum_i^N wi^2), where wi are the particle weights.

            :return: Boolean indicating whether resampling is needed.
            """
            weights = self.particles[:, 0]  # Get the particle weights
            sum_weights_squared = np.sum(weights ** 2)  # Calculate the sum of squared weights
            print(1/sum_weights_squared)
            return 1.0 / sum_weights_squared < self.resampling_threshold  # Check if the effective particle count is below the threshold

    def get_mean(self, time=1):
        """
        Compute the weighted average state of the particles.

        This method calculates the weighted average of the state variables 
        (alpha, beta, gamma) across all particles at a given time step.

        :param time: (int) Time step for which to calculate the mean.
        :return: The weighted average of mu (location parameter) at the given time step.
        """
        weights = self.particles[:, 0]  # Get the particle weights
        self.means_alpha[time] = np.average(self.particles[:, 1], weights=weights)  # Calculate weighted average of alpha
        self.means_beta[time] = np.average(self.particles[:, 2], weights=weights)  # Calculate weighted average of beta
        self.means_gamma[time] = np.average(self.particles[:, 3], weights=weights)  # Calculate weighted average of gamma
        self.means_mu[time] = self.means_alpha[time] + self.means_gamma[time]  # Calculate weighted average of mu
        return self.means_mu[time]

    def get_confidence_interval(self, time=1):
        """
        Compute the confidence intervals for the state variables.

        This method calculates the confidence intervals for alpha, beta, gamma, 
        and mu (location parameter) at a given time step, based on the 
        weighted particle distribution.

        :param time: (int) Time step for which to calculate the confidence intervals.
        :return: A tuple containing the lower and upper bounds of the confidence interval for mu.
        """

        # Sort particles based on the values of mu, alpha, beta, and gamma
        particles_sorted = self.particles[np.argsort(self.particles[:, 1] + self.particles[:, 3])]  
        particles_sorted_alpha = self.particles[np.argsort(self.particles[:, 1])]
        particles_sorted_beta = self.particles[np.argsort(self.particles[:, 2])]
        particles_sorted_gamma = self.particles[np.argsort(self.particles[:, 3])]

        # Calculate the cumulative sum of weights for each sorted set of particles
        cumulative_sum = np.cumsum(particles_sorted[:, 0])
        cumulative_sum_alpha = np.cumsum(particles_sorted_alpha[:, 0])
        cumulative_sum_beta = np.cumsum(particles_sorted_beta[:, 0])
        cumulative_sum_gamma = np.cumsum(particles_sorted_gamma[:, 0])

        # Create CDFs for mu, alpha, beta, and gamma
        CDF = np.column_stack((cumulative_sum, particles_sorted[:, 1] + particles_sorted[:, 3]))
        CDF_alpha = np.column_stack((cumulative_sum_alpha, particles_sorted_alpha[:, 1]))
        CDF_beta = np.column_stack((cumulative_sum_beta, particles_sorted_beta[:, 2]))
        CDF_gamma = np.column_stack((cumulative_sum_gamma, particles_sorted_gamma[:, 3]))

        # Calculate the probabilities for the lower and upper bounds of the confidence intervals
        prob_lwr, prob_upr = (1 - self.confidence_level) / 2, (1 + self.confidence_level) / 2

        # Extract the lower and upper bounds from the CDFs
        self.lwr_mu[time] = np.min(CDF[:, 1][CDF[:, 0] >= prob_lwr])
        self.upr_mu[time] = np.max(CDF[:, 1][CDF[:, 0] <= prob_upr])
        self.lwr_alpha[time] = np.min(CDF_alpha[:, 1][CDF_alpha[:, 0] >= prob_lwr])
        self.upr_alpha[time] = np.max(CDF_alpha[:, 1][CDF_alpha[:, 0] <= prob_upr])
        self.lwr_beta[time] = np.min(CDF_beta[:, 1][CDF_beta[:, 0] >= prob_lwr])
        self.upr_beta[time] = np.max(CDF_beta[:, 1][CDF_beta[:, 0] <= prob_upr])
        self.lwr_gamma[time] = np.min(CDF_gamma[:, 1][CDF_gamma[:, 0] >= prob_lwr])
        self.upr_gamma[time] = np.max(CDF_gamma[:, 1][CDF_gamma[:, 0] <= prob_upr])

        return self.lwr_mu[time], self.upr_mu[time]

    def get_states(self, time, end_time=None):
        """
        Retrieve the particle states at a given time.

        This method returns the states of all particles at a specific time step. 
        It can also be used to retrieve states at future time steps (for prediction) 
        by specifying an end_time.

        :param time: Time step for which to retrieve the states (int).
        :param end_time: Optional end time for prediction (int). If None, defaults to the last filtering time step.
        :return: NumPy array of shape (n_particles, state_dim) containing the particle states.
        """
        if end_time is None:
            end_time = len(self.all_particles) - 1
        if time >= end_time:
            T = end_time
            return self.all_particles[T, :, 1:]  # Return states from the last filtering time step
        else:
            return self.all_particles[time + 1, :, 1:]  # Return states from the specified time step

    def get_alpha(self, time, end_time=None):
        """
        Retrieve the alpha (trend) component of the particle states at a given time.

        This method returns the alpha values of all particles at a specific time step.
        For future time steps, it extrapolates the trend using the beta (slope) component.

        :param time: Time step for which to retrieve alpha (int).
        :param end_time: Optional end time for prediction (int). If None, defaults to the last filtering time step.
        :return: NumPy array of shape (n_particles,) containing the alpha values.
        """
        if end_time is None:
            end_time = len(self.all_particles) - 1
        if time >= end_time:
            T = end_time
            return self.all_particles[T, :, 1] + (time - end_time) * self.get_beta(T)  # Extrapolate trend
        else:
            return self.all_particles[time + 1, :, 1]  # Return alpha from the specified time step

    def get_gamma(self, time, end_time=None):
        """
        Retrieve the gamma (seasonal) component of the particle states at a given time.

        This method returns the gamma values of all particles at a specific time step.
        For future time steps, it retrieves the gamma value from the corresponding seasonal period.

        :param time: Time step for which to retrieve gamma (int).
        :param end_time: Optional end time for prediction (int). If None, defaults to the last filtering time step.
        :return: NumPy array of shape (n_particles,) containing the gamma values.
        """
        if end_time is None:
            end_time = len(self.all_particles) - 1
        if time >= end_time:
            T = end_time
            if time % self.period == 0:
                T_ = T  # Use the last filtering time step if time is a multiple of the period
            else:
                rem = self.period - (time % self.period)
                T_ = T - rem  # Otherwise, use the corresponding time step from the previous period
            return self.all_particles[T_, :, 3]  # Return gamma from the corresponding seasonal time step
        else:
            return self.all_particles[time + 1, :, 3]  # Return gamma from the specified time step

    def get_beta(self, time, end_time=None):
        """
        Retrieve the beta (slope) component of the particle states at a given time.

        This method returns the beta values of all particles at a specific time step.
        For future time steps, it retrieves the beta value from the last filtering time step.

        :param time: Time step for which to retrieve beta (int).
        :param end_time: Optional end time for prediction (int). If None, defaults to the last filtering time step.
        :return: NumPy array of shape (n_particles,) containing the beta values.
        """
        if end_time is None:
            end_time = len(self.all_particles) - 1
        if time >= end_time:
            T = end_time
            return self.all_particles[T, :, 2]  # Return beta from the last filtering time step
        else:
            return self.all_particles[time + 1, :, 2]  # Return beta from the specified time step

    def get_mu(self, time, end_time=None):
        """
        Calculate the mu (location) parameter at a given time.

        This method calculates mu as the sum of alpha (trend) and gamma (seasonal) components 
        for all particles at a specific time step.

        :param time: Time step for which to calculate mu (int).
        :param end_time: Optional end time for prediction (int). If None, defaults to the last filtering time step.
        :return: NumPy array of shape (n_particles,) containing the mu values.
        """
        alpha = self.get_alpha(time, end_time=end_time)
        gamma = self.get_gamma(time, end_time=end_time)
        return alpha + gamma

    def get_weights(self, time, end_time=None):
        """
        Retrieve the particle weights at a given time.

        This method returns the weights of all particles at a specific time step.

        :param time: Time step for which to retrieve the weights (int).
        :param end_time: Optional end time for prediction (int). If None, defaults to the last filtering time step.
        :return: NumPy array of shape (n_particles,) containing the particle weights.
        """
        if end_time is None:
            end_time = len(self.all_particles) - 1
        if time >= end_time:
            T = end_time
            return self.all_particles[T, :, 0]  # Return weights from the last filtering time step
        else:
            return self.all_particles[time + 1, :, 0]  # Return weights from the specified time step

    def compute_likelihood(self, particles, measurement):
        """
        Compute likelihood p(y|particle, parameters) for a specific measurement 
        given particle states and parameters.

        :param particles: Array of particles (each particle is a weighted state).
        :param measurement: Measurement (int).
        :return: Likelihood for each particle (array).
        """
        likelihood_particles = np.zeros(self.n_particles)
        i = 0
        for sample in particles:
            # Extract location parameter
            alpha, gamma = sample[1], sample[3]
            mu = alpha + gamma
            
            # Compute likelihood using the GEV distribution 
            likelihood_particles[i] = genextreme.pdf(measurement, -self.xi, mu, self.sigma)

            i += 1
        return likelihood_particles


    def get_marginal_likelihood(self, all_measurements):
        """
        Calculate the marginal likelihood of the data given the parameters.

        This method computes the marginal likelihood of the observed data (all_measurements) 
        given the estimated parameters of the model. It uses the particle filter's 
        approximation of the posterior distribution to compute this likelihood.

        :param all_measurements: List of all observed measurements (block maxima).
        :return: The marginal likelihood (float).
        """

        self.llh = 0  # Initialize the log-likelihood
        T = len(all_measurements)  # Number of time steps

        for t in range(T):
            particles = self.all_particles[t + 1]  # Get particles at time t+1
            measurement = all_measurements[t]  # Get the measurement at time t

            # Compute the likelihood of the measurement given each particle
            likelihood_samples = self.compute_likelihood(particles, measurement)  

            # Compute the weighted average likelihood across all particles
            contribution = particles[:, 0] @ likelihood_samples  

            # Add the log of the contribution to the overall log-likelihood (if positive)
            if contribution > 0:
                self.llh += np.log(contribution)  

        return self.llh  # Return the computed marginal log-likelihood

    def get_AIC(self, all_measurements):
        """
        Calculate the Akaike Information Criterion (AIC) for the model.

        This method computes the AIC, which is a measure of the model's goodness of fit 
        that penalizes models with more parameters. It is used to compare different models 
        and select the one that best balances goodness of fit with complexity.

        :param all_measurements: List of all observed measurements (block maxima).
        :return: The AIC value (float).
        """

        if self.llh == 0:  # If the log-likelihood hasn't been computed yet
            self.get_marginal_likelihood(all_measurements)  # Compute the marginal log-likelihood

        # Calculate the number of parameters in the model
        # In this case: sigma, xi, Q1, Q2, Q3, m_0, V_0 = 5 + 2*(s+1) parameters
        num_params = 5 + 2 * (self.period + 1)  

        # Calculate the AIC using the formula: AIC = 2 * (k - log(L))
        # where k is the number of parameters and L is the maximum likelihood
        self.aic = 2 * (num_params - self.llh)  

        return self.aic  # Return the computed AIC value

    def save(self, file_path=None):
        """
        Saves the PF object to a file with a standard title including the date,
        or with a custom file name if provided.
        
        :param file_path: (str) Path to save the PF object (optional)
        :returns: (str) The file path where the PF object is saved
        """
        if file_path is None:
            file_path = f"PF_{datetime.today().strftime('%Y-%m-%d')}.pkl"
        
        with open('SLLT/results/'+file_path, 'wb') as file:
            pickle.dump(self, file)
        
        return file_path

@staticmethod
def normalize_weights(particles):
    """
    Normalize all particle weights.
    """

    # Compute sum weighted samples
    sum_weights = np.sum(particles[:, 0])

    # Check if weights are non-zero
    if sum_weights < 1e-15:
        print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))
        # set uniform weights
        particles[:, 0] = 1.0 / len(particles)
        return particles
    else:
        particles[:,0] /= sum_weights
    # Return normalized weights
    return particles
        
def open_pf(file_path=None):
    """
    Opens a PF object that has been stored as a pkl file.
            
    :param file_path: (str) Path to the pkl file (optional)
    :returns: (ParticleFilter) The opened PF object
    """
    if file_path is None:
        file_path = f"EKF_PF_{datetime.today().strftime('%Y-%m-%d')}.pkl"
            
    with open('SLLT/results/'+file_path, 'rb') as file:
        eva = pickle.load(file)
    return eva

