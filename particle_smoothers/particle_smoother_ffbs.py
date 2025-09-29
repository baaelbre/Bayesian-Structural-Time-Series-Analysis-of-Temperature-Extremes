"""
Class to calculate the approximate smoothing distribution of the states given the extremal time series. 
The algorithm used is 'Forward Filtering and Backward Smoothing', hence a filtering step is always required when running this.
"""

from scipy.stats import multivariate_normal as norm
from scipy.stats import genextreme as gev, kstest
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
import time as TIME
import os
import multiprocessing as mp
from scipy.optimize import fsolve

class ParticleSmootherFFBS:
    """
    Notes:
        * State is (alpha_t, beta_t, gamma_t, ..., gamma_t-s+1)
    """
    def __init__(self, particle_filter, optimize=True):
        """
        Initialize the backward particle smoother given the set of filtered particles by a previous algorithm.
        
        :param particle_filter: (class) ParticleFilter object that contains data, filtered particles etc.
       
        """

        self.all_filtered_particles = particle_filter.all_particles
        self.n_time_steps = particle_filter.n_time_steps+1 # is this +1 necessary?
        self.period = particle_filter.period
        self.n_particles = particle_filter.n_particles
        self.particles, self.all_particles = [], []
        self.process_noise_vector = np.array(particle_filter.process_noise_vector)
        self.dim = 2 + (self.period-1)
        self.sigma, self.xi = particle_filter.parameters
        self.estimations, self.residuals = 0,0
        self.optimize = optimize
        
        self.current_states = []
        self.current_alpha = []
        self.current_beta = []
        self.current_gamma = []
        self.current_gamma_sum = []
        self.current_weights = []
        self.current_weights_matrix = []
        
        self.future_states = self.all_filtered_particles[-1]
        self.future_alpha = self.all_filtered_particles[-1][:,1]
        self.future_beta = self.all_filtered_particles[-1][:,2]
        self.future_gamma = self.all_filtered_particles[-1][:,3]
        self.future_gamma_sum = -np.sum(self.all_filtered_particles[-1][:,3:], axis=1)
        
        self.means = []
        self.means_alpha, self.means_beta, self.means_gamma = [], [], []
        self.means_mu = []
        self.means_y = []
        self.lwr_alpha, self.upr_alpha = [], []
        self.lwr_beta, self.upr_beta = [], []
        self.lwr_gamma, self.upr_gamma = [], []
        self.lwr_mu, self.upr_mu = [], []
        self.lwr_y, self.upr_y = [], []
        self.quantiles = {}
        self.quantiles_alpha, self.quantiles_beta, self.quantiles_gamma = {}, {}, {}
        self.quantiles_mu = {}
        self.probabilities, self.exceedance_probabilities = {}, {}
        self.confidence_level = particle_filter.confidence_level
        
        self.rmse, self.llh, self.aic = 0, 0, 0
        self.ks_stat, self.ks_p = 0, 0
                
    def initialize_trajectories(self):
        """
        Initialize the smoothing weights at time T, which equal the filtering weights. 
        From there we work backwards until t = 0 adjusting the filtering weights to obtain the smoothing weights,
        while leaving the particle states untouched.
        """
     
        weights = self.all_filtered_particles[-1][:,0]
        weights_matrix = np.zeros((self.n_particles,self.n_particles))
        states = self.all_filtered_particles[-1][:,1:]
        
        self.particles = [weights, weights_matrix, states]
        # Store trajectories at time zero
        self.all_particles = [self.particles]
    
    def get_weights(self, time):
        
        return self.all_particles[time][0]
    
    def get_weights_matrix(self, time):
        
        return self.all_particles[time][1]

    def get_states(self, time, component):	
            
        return self.all_filtered_particles[time][:,component+1]
    
    def get_alpha(self, time):
        
        return self.all_filtered_particles[time][:,1]
        
    def get_beta(self, time):
        
        return self.all_filtered_particles[time][:,2]
    
    def get_gamma(self, time):
        
        return self.all_filtered_particles[time][:,3] 
     
    def get_gamma_sum(self, time):
            
        #return [-np.sum(self.all_filtered_particles[time][i][1][2:]) for i in range(self.n_particles)]
        return -np.sum(self.all_filtered_particles[time][:,3:], axis=1)
    
    def get_mu(self, time):
        alpha, gamma = self.get_alpha(time), self.get_gamma(time)
        
        return alpha + gamma
    
    def get_filtering_weights(self, time):
        
        return self.all_filtered_particles[time][:,0]
    
    def get_means(self):
        """
        Compute average states according to all weighted particles

        :return: (list) List of averages [avg_mu, avg_d]
        """
        self.means_alpha, self.means_beta, self.means_gamma = [], [], []
        self.means, self.means_mu = [], []
        for t in range(self.n_time_steps):
            weights, alpha, beta, gamma = self.get_weights(t), self.get_alpha(t), self.get_beta(t), self.get_gamma(t)
            avg_alpha, avg_beta, avg_gamma = sum(weights*alpha), sum(weights*beta), sum(weights*gamma)
            self.means_alpha.append(avg_alpha)
            self.means_beta.append(avg_beta)
            self.means_gamma.append(avg_gamma)
            self.means_mu.append(avg_alpha + avg_gamma)
            self.means.append([avg_alpha, avg_beta, avg_gamma])
        
        return self.means
    
    def get_confidence_interval(self):
        
        for particles in self.all_particles:
            ## 1) Calculate CDF (order particles + calculate cumulative sum of the particle weights)
            combined = list(zip(particles[2], particles[0]))
            particles_sorted = sorted(combined, key=lambda x: x[0][0] + x[0][2])        
            particles_sorted_alpha = sorted(combined, key=lambda x: x[0][0])
            particles_sorted_beta = sorted(combined, key=lambda x: x[0][1])
            particles_sorted_gamma = sorted(combined, key=lambda x: x[0][2])
            states_sorted, weights_sorted = zip(*particles_sorted)  
            states_sorted_alpha, weights_sorted_alpha = zip(*particles_sorted_alpha)
            states_sorted_beta, weights_sorted_beta = zip(*particles_sorted_beta)
            states_sorted_gamma, weights_sorted_gamma = zip(*particles_sorted_gamma)
            cumulative_sum = np.cumsum(weights_sorted)
            cumulative_sum_alpha = np.cumsum(weights_sorted_alpha)
            cumulative_sum_beta = np.cumsum(weights_sorted_beta)
            cumulative_sum_gamma = np.cumsum(weights_sorted_gamma)
            CDF = list(zip(cumulative_sum, states_sorted))
            CDF_alpha = list(zip(cumulative_sum_alpha, states_sorted_alpha))
            CDF_beta = list(zip(cumulative_sum_beta, states_sorted_beta))
            CDF_gamma = list(zip(cumulative_sum_gamma, states_sorted_gamma))       
            ## 2) Find the closest lower and upper quantiles of that CDF
            prob_lwr, prob_upr = (1-self.confidence_level)/2, (1+self.confidence_level)/2
            lwr_mu, upr_mu = None, None
            for cumulative_weight, state in CDF:
                if cumulative_weight >= prob_lwr:
                    lwr_mu = state[0] + state[2]
                    break
            for cumulative_weight, state in reversed(CDF):
                if cumulative_weight <= prob_upr:
                    upr_mu = state[0] + state[2]
                    break
            self.lwr_mu.append(lwr_mu)
            self.upr_mu.append(upr_mu)
            
            lwr_alpha, upr_alpha = None, None
            for cumulative_weight, state in CDF_alpha:
                if cumulative_weight >= prob_lwr:
                    lwr_alpha = state[0]
                    break
            for cumulative_weight, state in reversed(CDF_alpha):
                if cumulative_weight <= prob_upr:
                    upr_alpha = state[0]
                    break
            self.lwr_alpha.append(lwr_alpha)
            self.upr_alpha.append(upr_alpha)
            
            lwr_beta, upr_beta = None, None
            for cumulative_weight, state in CDF_beta:
                if cumulative_weight >= prob_lwr:
                    lwr_beta = state[1]
                    break
            for cumulative_weight, state in reversed(CDF_beta):
                if cumulative_weight <= prob_upr:
                    upr_beta = state[1]
                    break
            self.lwr_beta.append(lwr_beta)
            self.upr_beta.append(upr_beta)
            
            lwr_gamma, upr_gamma = None, None
            for cumulative_weight, state in CDF_gamma:
                if cumulative_weight >= prob_lwr:
                    lwr_gamma = state[2]
                    break
            for cumulative_weight, state in reversed(CDF_gamma):
                if cumulative_weight <= prob_upr:
                    upr_gamma = state[2]
                    break
            self.lwr_gamma.append(lwr_gamma)
            self.upr_gamma.append(upr_gamma)
                    
        return self.lwr_mu, self.upr_mu
    
    def get_RMSE(self, all_measurements, mode=False, approximation=True):
        """
        Compute the root mean squared error of the posterior mean of mu_t.
        
        :param all_measurements: (list) List of all measurements
        :returns: (float) RMSE
        """
        if not approximation:
            assert self.means_y != [], 'Please run get_quantiles() first.'
            self.estimations = self.means_y
        else:
            if mode: 
                self.estimations = np.array(self.means_mu[1:]) +\
                    self.sigma*((1+self.xi)**(-self.xi)-1)/self.xi
            else:
                self.estimations = np.array(self.means_mu[1:]) +\
                    self.sigma*(np.log(2)**(-self.xi)-1)/self.xi
            
        all_measurements = np.array(all_measurements)
        self.residuals = all_measurements - self.estimations
        self.rmse = np.sqrt(np.mean((self.residuals)**2))
        
        return self.rmse
    
    def get_marginal_likelihood(self, all_measurements, approximation=True):
        """
        Function to calculate the marginal likelihood of the data given the parameters.
        After the filtering process this follows immediately.
        :param all_measurements: (list) of all block maxima.
        :param approximation: (bool) whether to use the approximation of using the means of the particles.
        
        :returns: marginal likelihood
        """
        self.llh = 0
        T = len(all_measurements)
        if approximation:
            measurements = np.array(all_measurements)
            mu = np.array(self.means_mu[1:T+1])
            
            # Compute likelihood using the GEV distribution
            contributions = gev.logpdf(measurements, -self.xi, mu, self.sigma)
            self.llh = np.sum(contributions)
        else:
            for t in range(T):
                weights, mu = self.get_weights(t+1), self.get_mu(t+1)
                measurement = all_measurements[t]
            
                # Compute likelihood using the GEV distribution 
                contribution = np.sum([weights[i]*gev.pdf(measurement, -self.xi, mu[i], self.sigma) for i in range(self.n_particles)])

                if contribution > 0:
                    self.llh += np.log(contribution)

        return self.llh
    
    def get_AIC(self, all_measurements):
        """
        Calculate the Akaike Information Criterion of the model based on
        the approximate likelihood
        """
        self.get_marginal_likelihood(all_measurements, approximation=True)
        self.aic = 2*(2*self.dim+5) - 2*self.llh
        
        return self.aic
    
    def KS_test(self, all_measurements, uniform=False):
        """
        Perform the Kolmogorov-Smirnov test to check the goodness-of-fit of the model.
        """
        empirical_quantiles = []
        t = 0
        for measurement in all_measurements:
            z = (measurement-self.means_mu[t+1])/self.sigma
            if self.xi != 0:
                s = np.log(1+self.xi*z)/self.xi
            else:
                s = z
            empirical_quantiles.append(s)
            t += 1
            
        # Perform KS test on the transformed data

        if uniform:
            theoretical_probabilities = gev.cdf(np.sort(empirical_quantiles), 0, 0, 1)
            if np.isnan(theoretical_probabilities).any():
                print("The input array contains NaN values.")
            # Remove NaN values
            valid_indices = ~np.isnan(theoretical_probabilities)
            theoretical_probabilities = theoretical_probabilities[valid_indices]
            self.ks_stat, self.ks_p = kstest(theoretical_probabilities, 'uniform')
            
        else:
            self.ks_stat, self.ks_p = kstest(empirical_quantiles, 'gumbel_r', args=(0, 1))
        
        return self.ks_stat, self.ks_p
    
    def calculate_quadrant(self, rank, size, future_alpha, future_beta, future_gamma):
        """
        Caculate vertical slices of the transition matrix T
        :param rank: The rank of the current worker (0,1,2, size-1)
        :param size: The total number of workers
        :returns: A vertical slice of the matrix
        """
        # Calculate the start and end indices for the current rank
        N = self.n_particles
        start_index = int(rank * N / size)
        end_index = int((rank + 1) * N / size)
        time0 = TIME.time()
        # Calculate the quadrant, vectorised
        #future = np.array(list(zip(future_alpha, future_beta, future_gamma)))
        future = np.column_stack((future_alpha, future_beta, future_gamma))
        current = np.column_stack((self.current_alpha, self.current_beta, self.current_gamma_sum))
        #current = np.array(list(zip(self.current_alpha, self.current_beta, self.current_gamma_sum)))

        i_indices, j_indices = np.meshgrid(range(start_index, end_index), range(N), indexing='ij')
        i_indices, j_indices = i_indices.flatten(), j_indices.flatten()
        process_noise_vector_replicated = np.tile(self.process_noise_vector, (len(i_indices), 1))
        Test_matrix = vectorized_gaussian_pdf(future[j_indices], current[i_indices], process_noise_vector_replicated)
        Test_matrix = Test_matrix.reshape((end_index - start_index, N))

        return Test_matrix
    
    def get_probability_single(self, y, time, alpha=True):
            """
            Compute the probability of a measurement for a single time step.

            Args:
                y: The value for which to compute the probability.
                time: The time index.
                alpha: Whether to use only the alpha (level) component or the full mu (level + seasonality).

            Returns:
                The probability.
            """
            if alpha:
                cdf = np.dot(self.get_weights(time), gev.cdf(y, -self.xi, loc=self.get_alpha(time), scale=self.sigma))
            else:
                cdf = np.dot(self.get_weights(time), gev.cdf(y, -self.xi, loc=self.get_mu(time), scale=self.sigma))
            return cdf

    def get_probability_yearly(self, y, time):
        """
        Compute the yearly probability of a measurement by considering all seasons.

        Args:
            y: The value for which to compute the probability.
            time: The time index.

        Returns:
            The yearly probability.
        """
        if time > 0:  # We initialize one season before the first year.
            year = (time-1) // self.period
            start_time = year * self.period
            end_time = (year + 1) * self.period - 1
            cdf = 1  # Initialize CDF
            for t in range(start_time, min(end_time + 1, self.n_time_steps)):
                cdf_t = np.dot(self.get_weights(t), gev.cdf(y, -self.xi, loc=self.get_mu(t), scale=self.sigma))
                cdf *= cdf_t  # Multiply CDFs for each season
        else:
            cdf = np.dot(self.get_weights(0), gev.cdf(y, -self.xi, loc=self.get_mu(0), scale=self.sigma))
        return cdf

    def get_probability(self, y, time, alpha=False, yearly=True):
        """
        Compute the probability of a measurement, with option for yearly calculation.

        Args:
            y: The value for which to compute the probability.
            time: The time index.
            alpha: Whether to use only the alpha (level) component or the full mu (level + seasonality).
            yearly: If True, calculates the yearly exceedance probability by considering all seasons.

        Returns:
            The probability.
        """
        if not yearly:
            return self.get_probability_single(y, time, alpha)
        else:
            return self.get_probability_yearly(y, time)

    def get_probabilities(self, quantiles=None, alpha=True, yearly=True):
        """
        Compute the probabilities for all time steps, for a list of quantiles.

        Args:
            quantiles: A list of quantiles for which to compute the probabilities.
            alpha: Whether to use only the alpha (level) component or the full mu (level + seasonality).
            yearly: If True, calculates the yearly probability by considering all seasons.

        Returns:
            A dictionary mapping quantiles to lists of probabilities, where each list 
            contains the probabilities for a specific quantile at all time steps.
        """
        if quantiles is None:
            # take as y the first mu!
            print('Please provide a list of quantiles. Taking the first mu as default.')
            quantiles = [self.means_mu[1]]

        self.probabilities = {}
        for quantile in quantiles:
            self.probabilities[quantile] = [self.get_probability(quantile, t, alpha, yearly) for t in 
                                            range(1,self.n_time_steps)]

        return self.probabilities

    def get_exceedance_probability(self, y, time, alpha=True, yearly=True):
        """
        Compute the exceedance probability for a single time step.

        Args:
            y: The value for which to compute the exceedance probability.
            time: The time index.
            alpha: Whether to use only the alpha (level) component or the full mu (level + seasonality).
            yearly: If True, calculates the yearly exceedance probability by considering all seasons.

        Returns:
            The exceedance probability.
        """
        # Make sure the probability is not negative
        return max(0, 1 - self.get_probability(y, time, alpha, yearly))

    def get_exceedance_probabilities(self, quantiles=None, alpha=True, yearly=True):
        """
        Compute the exceedance probabilities for all time steps, for a list of quantiles.

        Args:
            quantiles: A list of quantiles for which to compute the exceedance probabilities.
            alpha: Whether to use only the alpha (level) component or the full mu (level + seasonality).
            yearly: If True, calculates the yearly exceedance probability by considering all seasons.

        Returns:
            A dictionary mapping quantiles to lists of exceedance probabilities, where each list 
            contains the exceedance probabilities for a specific quantile at all time steps.
        """
        if quantiles is None:
            # take as y the first mu!
            print('Please provide a list of quantiles. Taking the first mu as default.')
            quantiles = [self.means_mu[1]]

        self.exceedance_probabilities = {}
        for quantile in quantiles:
            self.exceedance_probabilities[quantile] = [self.get_exceedance_probability(quantile, t, alpha, yearly) for t 
                                                       in range(1,self.n_time_steps)]

        return self.exceedance_probabilities
    
    def get_quantile(self, probability, time, alpha=True, yearly=False):
        """
        Compute a single quantile for a given probability at a specific time.

        Args:
            time: The time index.
            probability: The probability for which to compute the quantile.
            alpha: Whether to use only the alpha (level) component or the full mu (level + seasonality).
            yearly: If True, calculates the yearly quantile by considering all seasons.

        Returns:
            The quantile value.
        """
        if alpha:
            x0 = self.means_alpha[time]
        elif yearly:
            x0 = max(self.means_mu[(time // self.period) * self.period:
                (time // self.period) * self.period + self.period]) + \
                    self.sigma/self.xi *((-np.log(probability))**(-self.xi)-1)
        else:
            x0 = self.means_mu[time]

        y_t = fsolve(lambda y: self.get_probability(y, time=time, alpha=alpha, yearly=yearly) - probability, x0=x0)
        return float(y_t)

    def get_quantiles(self, probabilities=None, alpha=True, yearly=False):
        """
        Compute quantiles for given probabilities, with option for yearly calculation.

        Args:
            probabilities: List of probabilities for which to compute quantiles.
            alpha: Whether to use only the alpha (level) component or the full mu (level + seasonality).
            yearly: If True, calculates the yearly quantiles by considering all seasons.

        Returns:
            A dictionary mapping probabilities to lists of quantiles over time, 
            or a list of median quantiles with lower and upper bounds if probabilities is None.
        """
        if probabilities is None:
            self.means_y, self.lwr_y, self.upr_y = [], [], []
            for t in tqdm(range(1,self.n_time_steps)):
                if alpha:
                    x0 = self.means_alpha[t]
                else:
                    x0 = self.means_mu[t]
                y_t = self.get_quantile(0.5, t, alpha, yearly)  
                y_lwr = self.get_quantile((1 - self.confidence_level) / 2, t, alpha,
                                          yearly) 
                y_upr = self.get_quantile((1 + self.confidence_level) / 2, t, alpha,
                                          yearly) 
                self.means_y.append(y_t)
                self.lwr_y.append(y_lwr)
                self.upr_y.append(y_upr)

            return self.means_y, self.lwr_y, self.upr_y

        else:
            assert type(probabilities) == list, 'Please provide a list of probabilities.'
            for probability in probabilities:
                self.quantiles[probability] = []

            for t in tqdm(range(1,self.n_time_steps)):
                for probability in probabilities:
                    y_t = self.get_quantile(probability, t, alpha, yearly) 
                    self.quantiles[probability].append(y_t)

            return self.quantiles
                    
    def get_updated_weights(self, time):
        """
        Method to use in self.smooth.
        """
        future_states = self.all_filtered_particles[time+1][:,1:]
        future_alpha = np.array(self.get_alpha(time+1))
        future_beta = np.array(self.get_beta(time+1))
        future_gamma = np.array(self.get_gamma(time+1))
        future_weights = self.particles[0]
        
        self.current_states = self.all_filtered_particles[time][:,1:]
        self.current_alpha = np.array(self.get_alpha(time))
        self.current_beta = np.array(self.get_beta(time))
        self.current_gamma_sum = np.array(self.get_gamma_sum(time))
        current_filtering_weights = np.array(self.get_filtering_weights(time))
        
        # Calculate the transition matrix T multi-core
        #args = [(i, self.processorCount, future_m, future_d, future_gamma) for i in range(self.processorCount)]
        #T =  np.vstack(pool.starmap_async(self.calculate_quadrant, args).get())
        T = self.calculate_quadrant(0, 1, future_alpha, future_beta, future_gamma)
        # calculate normalized T
        T_ = np.divide(T, np.full(T.shape, np.dot(current_filtering_weights,T)))
        #T_ = T / np.dot(current_filtering_weights, T)

        #weights_matrix = np.einsum('i,ij,j->ij', current_filtering_weights, T_, future_weights)
        #weights = np.einsum('i,ij,j->i', current_filtering_weights, T_, future_weights)
        weights = current_filtering_weights * np.dot(T_, future_weights)
        if self.optimize: 
            weights_matrix = (current_filtering_weights[:, np.newaxis] * T_) * future_weights
        else:
            weights_matrix = np.zeros((self.n_particles, self.n_particles))
        return weights, weights_matrix
            
    def smooth(self):
        self.initialize_trajectories()
        print('smoothing...')
        
        # Backward smoothing
        for t in tqdm(range(self.n_time_steps-2,-1,-1)):
            self.current_weights, self.current_weights_matrix = self.get_updated_weights(t)
            self.particles = [self.current_weights, self.current_weights_matrix, self.current_states]
            
            self.all_particles.append(self.particles)
            
        self.all_particles.reverse()
        self.get_means()
        self.get_confidence_interval()
        
    def save(self, file_path=None):
        """
        Saves the PS object to a file with a standard title including the date,
        or with a custom file name if provided.
        
        :param file_path: (str) Path to save the PS object (optional)
        :returns: (str) The file path where the PS object is saved
        """
        if file_path is None:
            file_path = f"PS_{datetime.today().strftime('%Y-%m-%d')}.pkl"
        
        with open('SLLT/results/'+file_path, 'wb') as file:
            pickle.dump(self, file)
        
        return file_path
        
def open_ps(file_path=None):
    """
    Opens a PS object that has been stored as a pkl file.
            
    :param file_path: (str) Path to the pkl file (optional)
    :returns: (ParticleSmootherFFBS) The opened PS object
    """
    if file_path is None:
        file_path = f"PS_{datetime.today().strftime('%Y-%m-%d')}.pkl"
            
    with open('SLLT/results/'+file_path, 'rb') as file:
        ps = pickle.load(file)
    return ps

def vectorized_gaussian_pdf(X, means, variances):
    """
    Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i
    Args:
        X : shape (n, d)
            Data points
        means : shape (n, d)
            Mean vectors
        covariances : shape (n, d)
            Diagonal covariance matrices
    Returns:
        logpdfs : shape (n,)
            Log probabilities
    """
    _, d = X.shape
    determinants = np.prod(variances, axis=1)
    prefactor = 1 / np.sqrt((2 * np.pi) ** d * determinants)
    deviations = X - means
    inverses = 1 / variances
    return prefactor*np.exp(-np.sum(deviations * inverses * deviations, axis=1)/2)

   
            
            
        