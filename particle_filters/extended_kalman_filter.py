"""
This module implements an Extended Kalman Filter (EKF) for estimating the 
states of a structural DGEV model.

The EKF is used to track the time-varying parameters of the Generalized 
Extreme Value (GEV) distribution, which is often used to model extreme 
events. The structural DGEV model assumes that the location parameter of 
the GEV distribution varies over time according to a stochastic process 
that includes a trend and a seasonal component.

The EKF predicts the next state of the system based on the current state 
and the system dynamics, and then updates the state estimate based on new 
measurements. This process is repeated for each time step, providing 
estimates of the hidden states and their uncertainties.
"""

from abc import abstractmethod
import numpy as np
from scipy.stats import norm
from scipy.stats import genextreme as gev
from scipy.linalg import block_diag
from tqdm import tqdm
import pickle
from datetime import datetime
from scipy.special import gamma

class ExtendedKalmanFilter:
    """
    Abstract base class for Extended Kalman Filter (EKF).

    This class defines the basic structure and methods for an EKF, 
    including initialization, prediction, and update steps. It assumes a 
    state vector that includes a stochastic trend (alpha), a slope (beta), 
    and a seasonal component (gamma) for the GEV location parameter.

    Subclasses need to implement specific methods for calculating the 
    Jacobian matrix and other components required for the EKF.
    """

    def __init__(self, process_noise_vector, n_time_steps, parameters, period=4, confidence_level=0.90, llh_method="normal"):
        """
        Initialize the extended Kalman filter.

        :param process_noise_vector: Process noise vector Q (list).
        :param n_time_steps: Number of time steps (int).
        :param parameters: GEV parameters [sigma, xi] (list).
        :param period: Seasonal period (int).
        :param confidence_level: Confidence level for state estimates (float).
        :param llh_method: Method for likelihood calculation ("normal" or "gev").
        """

        self.period = period
        self.n_time_steps = n_time_steps
        self.state_dim = 2 + period - 1
        self.process_noise_vector = process_noise_vector
        self.parameters = parameters
        self.sigma, self.xi = self.parameters
        self.variance = gev.var(-self.xi, scale=self.sigma)
        self.alpha, self.beta, self.gamma, self.mu = None, None, None, None
        self.confidence_level = confidence_level

        # Initialize state and covariance
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim)

        # Initialize evolution matrix
        self._initialize_evolution_matrix()
        
        # Initialize observation matrix
        self.H = np.zeros(self.state_dim)
        self.H[0] = 1
        self.H[2] = 1
        
        # Initialize process noise matrix
        self.process_noise_matrix = np.diag(
            np.concatenate((process_noise_vector, np.zeros(self.period - 2)))
        )

        # Initialize storage for results
        self._initialize_result_arrays()
        
        # Initialize likelihood method
        self.llh_method = llh_method
        if self.llh_method not in ["normal", "gev"]:
            raise ValueError("llh_method must be 'normal' or 'gev'")

        self.llh, self.aic, self.rmse = 0, 0, 0
        self.time = 0

    def _initialize_evolution_matrix(self):
        """
        Initialize the evolution matrix for state prediction.
        """
        self.evolution_matrix = np.zeros((self.state_dim, self.state_dim))
        self.evolution_matrix[0, :2] = [1, 1]
        self.evolution_matrix[1, 1] = 1
        self.evolution_matrix[2, 2:self.period+1] = -1
        np.fill_diagonal(self.evolution_matrix[3:, 2:], 1)

    def _initialize_result_arrays(self):
        """
        Initialize arrays to store EKF results.
        """
        self.states = np.zeros((self.n_time_steps + 1, self.state_dim))
        self.covariances = np.zeros((self.n_time_steps + 1, self.state_dim, self.state_dim))
        self.lwr_alpha, self.upr_alpha = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)
        self.lwr_beta, self.upr_beta = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)
        self.lwr_gamma, self.upr_gamma = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)
        self.lwr_mu, self.upr_mu = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)
        self.means_alpha, self.means_beta = np.zeros(self.n_time_steps + 1), np.zeros(self.n_time_steps + 1)
        self.means_gamma = np.zeros(self.n_time_steps + 1)
        self.means_mu = np.zeros(self.n_time_steps + 1)
        
        
    def initialize(self, initial_state, initial_covariance):
        """
        Initialize the filter with the initial state and covariance.
        
        :param initial_state: Initial state vector (array).
        :param initial_covariance: Initial covariance matrix (array).
        """
        self.state = initial_state
        self.alpha, self.beta, self.gamma = self.state[0], self.state[1], self.state[2]
        self.mu = self.alpha + self.gamma
        self.predicted_mean = self.H @ self.state
        self.means_alpha[0], self.means_beta[0], self.means_gamma[0] = self.alpha, self.beta, self.gamma
        self.means_mu[0] = self.mu
        self.states[0] = self.state
        self.covariance = np.diag(initial_covariance)
        self.covariances[0] = self.covariance
        self.predicted_covariance = self.H @ self.covariance @ self.H.T + self.variance
        self.get_confidence_interval()
        self.time = 1

    def predict(self):
        """
        Predict the next state and covariance.
        """
        self.state = self.evolution_matrix @ self.state
        self.alpha, self.beta, self.gamma = self.state[0], self.state[1], self.state[2]
        self.mu = self.alpha + self.gamma
        self.covariance = self.evolution_matrix @ self.covariance @ self.evolution_matrix.T + self.process_noise_matrix
        
        # Calculate and return predicted mean and covariance for likelihood calculation
        self.predicted_mean = self.H @ self.state
        self.predicted_covariance = self.H @ self.covariance @ self.H.T + self.variance

    def update(self, measurement):
        """
        Update the state and covariance with a new measurement.
        
        :param measurement: New measurement (float).
        """
        
        mu = self.predicted_mean
        z = (measurement - mu) / self.sigma
        u = 1 + self.xi * z

        # Calculate log-likelihood contribution based on selected method
        if self.llh_method == "normal":
            self.llh += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(self.predicted_covariance) \
                        - 0.5 * (measurement - mu) ** 2 / self.predicted_covariance
        elif self.llh_method == "gev":
            if self.xi != 0 and u > 0:
                self.llh += gev.logpdf(measurement, -self.xi, loc=mu, scale=self.sigma)
            elif self.xi == 0:
                self.llh += gev.logpdf(measurement, 0, loc=mu, scale=self.sigma)
                
        self.aic = 2*(2*self.state_dim+5) - 2*self.llh # initial states, sigma, xi, process noise
        
        mu = self.state[0] + self.state[2]
        z = (measurement - mu) / self.sigma
        u = 1 + self.xi * z

        # Exclude observations outside the domain of the GEV distribution
        if u < 0:
            self.get_confidence_interval()
            self._store_results()
            return
        
        # Measurement prediction
        S = self.H @ self.covariance @ self.H.T + self.variance

        # Kalman gain
        K = self.covariance @ self.H.T / S

        # Update state and covariance
        self.state = self.state + K * (measurement - mu)
        self.covariance = self.covariance - np.outer(K, K) * S
        
        # Obtain confidence intervals
        self.get_confidence_interval()

        # Store results
        self._store_results()
        
        # Calculate RMSE
        self.rmse += (measurement - mu) ** 2
                
    def _store_results(self):
        """
        Store the current state and covariance estimates.
        """
        self.states[self.time] = self.state
        self.alpha, self.beta, self.gamma = self.state[0], self.state[1], self.state[2]
        self.mu = self.alpha + self.gamma
        self.covariances[self.time] = self.covariance
        self.means_mu[self.time] = self.mu
        self.means_alpha[self.time] = self.alpha
        self.means_beta[self.time] = self.beta
        self.means_gamma[self.time] = self.gamma
        self.time += 1

    def filter(self, all_measurements):
        """
        Filter the data using the extended Kalman filter.
        
        :param all_measurements: List of all block maxima (list).
        """
        for measurement in tqdm(all_measurements):
            self.predict()
            self.update(measurement)
        
        self.rmse = np.sqrt(self.rmse / len(all_measurements))
            
    def get_confidence_interval(self):
        """
        Compute the confidence interval for the state at a given self.time.

        :return: Lower and upper bounds of the confidence interval (tuple).
        """
        prob_lwr, prob_upr = (1-self.confidence_level)/2, (1+self.confidence_level)/2
        lwr_alpha, upr_alpha = norm.ppf([prob_lwr, prob_upr], loc=self.alpha, scale=np.sqrt(self.covariance[0,0]))
        lwr_beta, upr_beta = norm.ppf([prob_lwr, prob_upr], loc=self.beta, scale=np.sqrt(self.covariance[1,1]))
        lwr_gamma, upr_gamma = norm.ppf([prob_lwr, prob_upr], loc=self.gamma, scale=np.sqrt(self.covariance[2,2]))
        lwr_mu, upr_mu = norm.ppf([prob_lwr, prob_upr], loc=self.mu, scale=np.sqrt(self.covariance[0,0] + self.covariance[2,2]))
        
        self.lwr_alpha[self.time], self.upr_alpha[self.time] = lwr_alpha, upr_alpha
        self.lwr_beta[self.time], self.upr_beta[self.time] = lwr_beta, upr_beta
        self.lwr_gamma[self.time], self.upr_gamma[self.time] = lwr_gamma, upr_gamma
        self.lwr_mu[self.time], self.upr_mu[self.time] = lwr_mu, upr_mu


        return lwr_mu, upr_mu

    def save(self, file_path=None):
        """
        Saves the EKF object to a file.

        :param file_path: Path to save the EKF object (optional) (str).
        :returns: The file path where the EKF object is saved (str).
        """
        if file_path is None:
            file_path = f"EKF_{datetime.today().strftime('%Y-%m-%d')}.pkl"
        
        with open('SLLT/results/' + file_path, 'wb') as file:
            pickle.dump(self, file)
        
        return file_path

def open_ekf(file_path=None):
    """
    Opens an EKF object that has been stored as a pkl file.

    :param file_path: Path to the pkl file (optional) (str).
    :returns: The opened EKF object (ExtendedKalmanFilter).
    """
    if file_path is None:
        file_path = f"EKF_{datetime.today().strftime('%Y-%m-%d')}.pkl"
        
    with open('SLLT/results/' + file_path, 'rb') as file:
        ekf = pickle.load(file)
    return ekf