# load required packages
from abc import abstractmethod
import copy
import pickle
from datetime import datetime
import numpy as np
from scipy.stats import genextreme, norm
from scipy.optimize import brentq
from tqdm import tqdm

class ParticleFilter:
    """
    Notes:
        * State is s+1 dimensional (alpha_t, beta_t, gamma_t, gamma_t-1,..., gamma_t-s+1), alpha being the stochastic trend and gamma the seasonal
        component of the GEV location parameter mu_t = alpha_t + gamma_t. Static parameters are shape (xi) and scale (sigma)
        * Abstract class
    """
    def __init__(self, number_of_particles, process_noise_vector, n_time_steps,
                 parameters, period = 4, confidence_level=0.90, number_of_predictive_samples=None):
        """
        Initialize the abstract particle filter.
        This is a non online implementation of the particle filter as the number of time steps is known in advance.
        
        :param number_of_particles: (int) Number of particles
        :param process_noise_vector: (list) 3D process noise vector Q
        :param n_time_steps: (int) Number of time steps
        :param parameters: (list) GEV parameters [sigma, xi]
        :param period: (int) Seasonal period
        :param confidence_level: (float) confidence level of the states
        :param number_of_predictive_samples: (int) Number of MC samples for the predictive distribution
        """

        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles
        if number_of_predictive_samples is None:
            self.n_predictive_samples = number_of_particles
        else:
            self.n_predictive_samples = number_of_predictive_samples
        self.period = period
        self.dim = self.period + 1
        self.n_time_steps = n_time_steps
        self.particles = np.zeros((self.n_particles, 3+period-1))
        self.all_particles = np.zeros((self.n_time_steps+1, self.n_particles, 3+period-1))
        self.means = np.zeros((self.n_time_steps+1, 3))
        self.means_alpha = np.zeros(self.n_time_steps+1)
        self.means_beta = np.zeros(self.n_time_steps+1)
        self.means_gamma = np.zeros(self.n_time_steps+1)
        self.means_mu = np.zeros(self.n_time_steps+1)
        self.lwr_alpha, self.upr_alpha = np.zeros(self.n_time_steps+1), np.zeros(self.n_time_steps+1)
        self.lwr_beta, self.upr_beta = np.zeros(self.n_time_steps+1), np.zeros(self.n_time_steps+1)
        self.lwr_gamma, self.upr_gamma = np.zeros(self.n_time_steps+1), np.zeros(self.n_time_steps+1)
        self.lwr_mu, self.upr_mu = np.zeros(self.n_time_steps+1), np.zeros(self.n_time_steps+1)
        self.quantiles = {}
        self.predictive_quantiles, self.predictive_probabilities = {}, {}
        self.quantiles_alpha, self.quantiles_beta, self.quantiles_gamma = [], [], []
        self.quantiles_mu = {}
        self.predictive_quantiles_mu = {}
        
        self.process_noise_vector = process_noise_vector
        
        # Initialize evolution matrix
        self.evolution_matrix = np.zeros((self.period+1, self.period+1))
        self.evolution_matrix[0, :2] = [1, 1]
        self.evolution_matrix[1, 1] = 1
        self.evolution_matrix[2, 2:self.period+1] = -1
        np.fill_diagonal(self.evolution_matrix[3:, 2:], 1)
        # Initialize process noise matrix                  
        self.process_noise_matrix = np.diag(np.concatenate((process_noise_vector, np.zeros(self.period-2))))
        
        self.parameters = parameters
        self.sigma, self.xi = self.parameters
        self.confidence_level = confidence_level
        self.llh, self.aic, self.rmse = 0, 0, 0
        
        self.time = 0
        print('Initializing particle filter with {} particles and {} time steps'.format(self.n_particles, self.n_time_steps))
        print('filtering...')

    def initialize_particles_gaussian(self, m_0, P_0):
        """
        Initialize particle filter using a (s+1)D (independent) Gaussian distribution with 

        :param m_0: (list) mean of the initial state
        :param P_0: (list) variances of the initial state
        """
 
        # Initialize particles with uniform weight distribution
        weight = 1.0 / self.n_particles
        self.particles = np.concatenate(([np.full(self.n_particles, weight)], np.random.multivariate_normal(m_0, np.diag(P_0), size=self.n_particles).T)).T
        self.all_particles[0] = self.particles
        
        self.get_mean(self.time)
        self.get_confidence_interval(self.time)
        self.time = 1


    def get_mean(self, time=1):
        """
        Compute average state according to all weighted particles
        :param time: (int) Time step
        :return: Average state (mu_t, beta_t, gamma_t)
        """
        # Compute weighted average
        weights = self.particles[:,0]
        self.means_alpha[time] = np.average(self.particles[:, 1], weights=weights)
        self.means_beta[time] = np.average(self.particles[:, 2], weights=weights)
        self.means_gamma[time] = np.average(self.particles[:, 3], weights=weights)
        self.means_mu[time] = self.means_alpha[time] + self.means_gamma[time]

        return self.means_mu[time]
    
    def get_confidence_interval(self, time=1):
        
        particles_sorted = self.particles[np.argsort(self.particles[:,1]+self.particles[:,3])]    
        particles_sorted_alpha = self.particles[np.argsort(self.particles[:,1])]
        particles_sorted_beta = self.particles[np.argsort(self.particles[:,2])]
        particles_sorted_gamma = self.particles[np.argsort(self.particles[:,3])]
        
        cumulative_sum = np.cumsum(particles_sorted[:,0])
        cumulative_sum_alpha = np.cumsum(particles_sorted_alpha[:,0])
        cumulative_sum_beta = np.cumsum(particles_sorted_beta[:,0])
        cumulative_sum_gamma = np.cumsum(particles_sorted_gamma[:,0])
        
        CDF = np.column_stack((cumulative_sum, particles_sorted[:, 1] + particles_sorted[:, 3]))
        CDF_alpha = np.column_stack((cumulative_sum_alpha, particles_sorted_alpha[:, 1]))
        CDF_beta = np.column_stack((cumulative_sum_beta, particles_sorted_beta[:, 2]))
        CDF_gamma = np.column_stack((cumulative_sum_gamma, particles_sorted_gamma[:, 3]))

        prob_lwr, prob_upr = (1-self.confidence_level)/2, (1+self.confidence_level)/2
    
        self.lwr_mu[time] = np.min(CDF[:, 1][CDF[:, 0] >= prob_lwr])
        self.upr_mu[time] = np.max(CDF[:, 1][CDF[:, 0] <= prob_upr])
        self.lwr_alpha[time]= np.min(CDF_alpha[:, 1][CDF_alpha[:, 0] >= prob_lwr])
        self.upr_alpha[time] = np.max(CDF_alpha[:, 1][CDF_alpha[:, 0] <= prob_upr])
        self.lwr_beta[time] = np.min(CDF_beta[:, 1][CDF_beta[:, 0] >= prob_lwr])
        self.upr_beta[time] = np.max(CDF_beta[:, 1][CDF_beta[:, 0] <= prob_upr])
        self.lwr_gamma[time] = np.min(CDF_gamma[:, 1][CDF_gamma[:, 0] >= prob_lwr])
        self.upr_gamma[time] = np.max(CDF_gamma[:, 1][CDF_gamma[:, 0] <= prob_upr])

        return self.lwr_mu[time], self.upr_mu[time]
    
    def get_quantile(self, p):
        """
        Compute quantiles of the approximated filtering distribution.
        # NOT VECTORISED YET
        """
        if self.quantiles.get(p) is None:
            self.quantiles[p] = []
            self.quantiles_alpha[p] = []
            self.quantiles_beta[p] = []
            self.quantiles_gamma[p] = []
            self.quantiles_mu[p] = []
            
        # quantile associated with posterior distribution
        quantile_alpha = np.quantile([self.particles[i][1][0] for i in range(self.n_particles)],p)
        quantile_beta = np.quantile([self.particles[i][1][1] for i in range(self.n_particles)],p)
        quantile_gamma = np.quantile([self.particles[i][1][2] for i in range(self.n_particles)],p)
        quantile_mu = quantile_alpha + quantile_gamma
        
        self.quantiles_alpha[p].append(quantile_alpha)
        self.quantiles_beta[p].append(quantile_beta)
        self.quantiles_gamma[p].append(quantile_gamma)
        self.quantiles_mu[p].append(quantile_mu)
        self.quantiles[p].append([quantile_alpha, quantile_gamma])
        
        return [quantile_alpha, quantile_beta, quantile_gamma]

    def get_max_weight(self):
        """
        Find maximum weight in particle filter.

        :return: Maximum particle weight
        """
        return np.max(self.particles[:, 0])

    def print_particles(self):
        """
        Print all particles: index, state and weight.
        """

        print("Particles:")
        for i in range(self.n_particles):
            print(" ({}): {} with w: {}".format(i+1, self.particles[i,1:], self.particles[i,0]))
            
    def get_states(self, time, end_time=None):
        if end_time is None:
            end_time = len(self.all_particles)-1
        if time >= end_time:
            T = end_time
            return self.all_particles[T, :, 1:]
        else:
            return self.all_particles[time+1, :, 1:]
     
    def get_alpha(self, time, end_time=None):
        if end_time is None:
            end_time = len(self.all_particles)-1
        if time >= end_time:
            T = end_time
            return self.all_particles[T, :, 1] + (time-end_time)*self.get_beta(T)
        else:
            return self.all_particles[time+1, :, 1]
    
    def get_gamma(self, time, end_time=None):
        if end_time is None:
            end_time = len(self.all_particles)-1
        if time >= end_time:
            T = end_time
            if time % self.period == 0:
                T_ = T
            else:
                rem = self.period - (time % self.period)
                T_ = T - rem
            return self.all_particles[T_, :, 3]
        else:       
            return self.all_particles[time+1, :, 3]
    
    def get_beta(self, time, end_time=None):
        if end_time is None:
            end_time = len(self.all_particles)-1
        if time >= end_time:
            T = end_time
            return self.all_particles[T, :, 2]
        else:
            return self.all_particles[time+1, :, 2]
     
    def get_mu(self, time, end_time=None):
        alpha, gamma = self.get_alpha(time, end_time=end_time), self.get_gamma(time, end_time=end_time)

        return alpha + gamma

    def get_weights(self, time, end_time=None):
        if end_time is None:
            end_time = len(self.all_particles)-1
        if time >= end_time:
            T = end_time
            return self.all_particles[T, :, 0]
        else:
            return self.all_particles[time+1, :, 0]

    @staticmethod
    def normalize_weights(particles):
        """
        Normalize all particle weights.
        TODO
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

    def compute_likelihood(self, particles, measurement):
        """
        Compute likelihood p(y|sample, parameters) for a specific measurement given sample state and parameters.

        :param sample: Sample (unweighted particle) that must be propagated
        :param measurement: Measurement (int)
        :param parameters: Parameters [sigma, xi]
        :return Likelihood (int)
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

    def propagate_particles(self, particles):
        """
        Propagate an individual sample according to a Gaussian transition. 
        Return the propagated sample (leave input unchanged).

        :param sample: Sample (weighted particle) that must be propagated
        :return: propagated sample
        """

        new_particles = np.zeros((self.n_particles,self.period+2))

        new_particles[:, 1:] = np.random.multivariate_normal(np.zeros(self.period+1), self.process_noise_matrix, size=self.n_particles) + particles[:, 1:] @ self.evolution_matrix.T
        new_particles[:, 0] = particles[:, 0]

        return new_particles

    
    def get_marginal_likelihood(self, all_measurements):
        """
        Function to calculate the marginal likelihood of the data given the parameters.
        After the filtering process this follows immediately.
        :param all_measurements: (list) of all block maxima.
        
        :returns: marginal likelihood
        """
        self.llh = 0
        T = len(all_measurements)
        for t in range(T):
            particles = self.all_particles[t+1]
            measurement = all_measurements[t]
            # Compute weighted average
            # if sum equals zero don't add it to the likelihood ...
            contribution = particles[:,0] @ self.compute_likelihood(particles, measurement)

            if contribution > 0:
                self.llh += np.log(contribution)

        return self.llh

    def get_AIC(self, all_measurements):
        
        self.get_marginal_likelihood(all_measurements)
        
        # sigma, xi, Q1, Q2, Q3, m_0, P_0 = 5 + 2*(s+1) parameters
        self.aic = 2*(5+2*(self.period+1)-self.llh)
        
        return self.aic
    
    @abstractmethod
    def update(self, measurements):
        """
        Process a measurement. Abstract method that must be implemented in derived
        class.

        :param measurements: Measurement.
        :param parameters: Parameters [sigma, xi].
        """

        pass
    
    def filter(self, all_measurements):
        """
        Filter the data using the particle filter.
        
        :param all_measurements: (list) of all block maxima.
        """

        for measurement in tqdm(all_measurements):
            self.update(measurement)

        self.rmse = np.sqrt(self.rmse/len(all_measurements))

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
        
def open_pf(file_path=None):
    """
    Opens a PF object that has been stored as a pkl file.
            
    :param file_path: (str) Path to the pkl file (optional)
    :returns: (ParticleFilter) The opened PF object
    """
    if file_path is None:
        file_path = f"PF_{datetime.today().strftime('%Y-%m-%d')}.pkl"
            
    with open('SLLT/results/'+file_path, 'rb') as file:
        eva = pickle.load(file)
    return eva

