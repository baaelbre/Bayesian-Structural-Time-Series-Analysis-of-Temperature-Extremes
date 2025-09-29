import copy
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from scipy.special import gamma

class ExtendedKalmanSmoother:
    """
    Notes:
        * The Extended Kalman Smoother (EKS) is used for non-linear state space models.
        * State is s+1 dimensional (m_t, d_t, gamma_t, gamma_t-1, ..., gamma_t-s+1) m being the stochastic trend and gamma the seasonal
        component of the location parameter of the Gaussian distribution.
    """
    def __init__(self, extended_kalman_filter):
        # observation matrix
        self.H = extended_kalman_filter.H
        self.process_noise_vector = extended_kalman_filter.process_noise_vector
        self.parameters = extended_kalman_filter.parameters
        self.sigma, self.xi = self.parameters
        self.period = extended_kalman_filter.period
        self.dim = 2 + (self.period-1)
        self.confidence_level = extended_kalman_filter.confidence_level
        self.evolution_matrix = extended_kalman_filter.evolution_matrix
        self.process_noise_matrix = extended_kalman_filter.process_noise_matrix
        self.m, self.P = extended_kalman_filter.state, extended_kalman_filter.covariance
        self.states = extended_kalman_filter.states
        self.all_states = [0]*len(self.states)
        self.covariances = extended_kalman_filter.covariances
        self.all_Gs = self.states
        self.n_time_steps = len(self.states)-1
        
        self.current_states = []
        self.current_alpha = []
        self.current_beta = []
        self.current_gamma = []
        self.current_gamma_sum = []
        
        self.m, self.P, self.P_lag = [], [], []
        self.G = []
        self.alpha, self.beta, self.gamma = [], [], []
        self.means_alpha, self.means_beta, self.means_gamma = [], [], []
        self.means_mu = []
        self.lwr_alpha, self.upr_alpha = [], []
        self.lwr_beta, self.upr_beta = [], []
        self.lwr_gamma, self.upr_gamma = [], []
        self.lwr_mu, self.upr_mu = [], []

        self.rmse = 0
        self.llh = extended_kalman_filter.llh
        self.aic = extended_kalman_filter.aic
        print('smoothing...')
                
    def initialize(self):
        """
        Initialize Extended Kalman Smoother 
        """
        self.all_states = [0]*len(self.states)
        self.all_Gs = [0]*len(self.states)
        self.means_alpha = [0]*len(self.states)
        self.means_beta = [0]*len(self.states)
        self.means_gamma = [0]*len(self.states)
        self.means_mu = [0]*len(self.states)
        
        self.lwr_alpha = [0]*len(self.states)
        self.upr_alpha = [0]*len(self.states)
        self.lwr_beta = [0]*len(self.states)
        self.upr_beta = [0]*len(self.states)
        self.lwr_gamma = [0]*len(self.states)
        self.upr_gamma = [0]*len(self.states)
        self.lwr_mu = [0]*len(self.states)
        self.upr_mu = [0]*len(self.states)
        
        self.m, self.P = self.states[-1], self.covariances[-1]
        P_T_1 = self.covariances[-2]
        P_T_1_ = self.evolution_matrix@P_T_1@self.evolution_matrix.T + self.process_noise_matrix
        variance = self.sigma**2 * (gamma(1-2/self.xi) - gamma(1-1/self.xi)**2)
        K_T = P_T_1_ @ self.evolution_matrix.T / (self.H@P_T_1_@self.H.T + variance)
        self.P_lag = (np.identity(self.dim) - K_T @ self.H.T) @ self.evolution_matrix@P_T_1
        
        self.all_states[-1] = [self.m, self.P, self.P_lag]
        self.alpha, self.beta, self.gamma = self.states[-1,:3]
        self.means_alpha[-1] = self.alpha
        self.means_beta[-1] = self.beta
        self.means_gamma[-1] = self.gamma
        self.means_mu[-1] = self.alpha + self.gamma

        
        self.lwr_alpha[-1] = self.alpha - norm.ppf(1-self.confidence_level/2)*np.sqrt(self.covariances[-1][0,0])
        self.upr_alpha[-1] = self.alpha + norm.ppf(1-self.confidence_level/2)*np.sqrt(self.covariances[-1][0,0])
        self.lwr_beta[-1] = self.beta - norm.ppf(1-self.confidence_level/2)*np.sqrt(self.covariances[-1][1,1])
        self.upr_beta[-1] = self.beta + norm.ppf(1-self.confidence_level/2)*np.sqrt(self.covariances[-1][1,1])
        self.lwr_gamma[-1] = self.gamma - norm.ppf(1-self.confidence_level/2)*np.sqrt(self.covariances[-1][2,2])
        self.upr_gamma[-1] = self.gamma + norm.ppf(1-self.confidence_level/2)*np.sqrt(self.covariances[-1][2,2])
        self.lwr_mu[-1] = self.alpha + self.gamma - norm.ppf(1-self.confidence_level/2)*np.sqrt(self.covariances[-1][0,0] + self.covariances[-1][2,2])
        self.upr_mu[-1] = self.alpha + self.gamma + norm.ppf(1-self.confidence_level/2)*np.sqrt(self.covariances[-1][0,0] + self.covariances[-1][2,2])

    def get_RMSE(self, all_measurements):
        """
        Calculates the root-mean-square error of the Kalman Smoother
        
        :param all_measurements: (np.array) All measurements in the time series
        
        :returns: (float) The root-mean-square error of the Kalman Smoother
        """
        all_measurements, means_mu = np.array(all_measurements), np.array(self.means_mu)[1:]
        for i in range(len(all_measurements)):
            self.rmse += (all_measurements[i] - means_mu[i])**2
            
        self.rmse = np.sqrt(self.rmse/len(all_measurements))
        
        return self.rmse
    
    def smooth(self):
        """
        Update Extended Kalman Smoother
        """
        self.initialize()
        
        for t in range(len(self.states)-2, 0, -1):
            m_t, P_t = self.states[t], self.covariances[t]
            m_t_ = self.evolution_matrix@m_t
            P_t_= self.evolution_matrix@P_t@self.evolution_matrix.T + self.process_noise_matrix
            G_t = P_t @ self.evolution_matrix.T @ np.linalg.inv(P_t_)
            
            P_t_1 = self.covariances[t-1]
            P_t_1_ = self.evolution_matrix@P_t_1@self.evolution_matrix.T + self.process_noise_matrix
            G_t_1 = P_t_1 @ self.evolution_matrix.T @ np.linalg.inv(P_t_1_)
            
            self.m, self.P, self.P_lag = self.all_states[t+1]
            
            self.m = m_t + G_t @ (self.m - m_t_)
            self.P = P_t + G_t @ (self.P - P_t_) @ G_t.T
            self.P_lag = P_t @ G_t_1.T + G_t @ (self.P_lag - self.evolution_matrix@P_t) @ G_t_1.T
            self.P_lag = G_t_1 @ self.P
            # Compute Kalman gain
            self.G =  G_t
            
            self.all_states[t] = [self.m, self.P, self.P_lag]
            self.all_Gs[t] = self.G
            self.means_alpha[t] = self.m[0]
            self.means_beta[t] = self.m[1]
            self.means_gamma[t] = self.m[2]
            self.means_mu[t] = self.m[0] + self.m[2]
            
            prob_lwr, prob_upr = (1-self.confidence_level)/2, (1+self.confidence_level)/2
            self.lwr_alpha[t], self.upr_alpha[t] = norm.ppf([prob_lwr, prob_upr], loc=self.m[0], scale=np.sqrt(self.P[0,0]))
            self.lwr_beta[t], self.upr_beta[t] = norm.ppf([prob_lwr, prob_upr], loc=self.m[1], scale=np.sqrt(self.P[1,1]))
            self.lwr_gamma[t], self.upr_gamma[t] = norm.ppf([prob_lwr, prob_upr], loc=self.m[2], scale=np.sqrt(self.P[2,2]))
            self.lwr_mu[t], self.upr_mu[t] = norm.ppf([prob_lwr, prob_upr], loc=self.m[0] + self.m[2], scale=np.sqrt(self.P[0,0] + self.P[2,2]))
        
        m_t, P_t = self.states[0], self.covariances[0]
        m_t_ = self.evolution_matrix@m_t
        P_t_= self.evolution_matrix@P_t@self.evolution_matrix.T + self.process_noise_matrix
        G_t = P_t @ self.evolution_matrix.T @ np.linalg.inv(P_t_)
            
        self.m, self.P, self.P_lag = self.all_states[1]
            
        self.m = m_t + G_t @ (self.m - m_t_)
        self.P = P_t + G_t @ (self.P - P_t_) @ G_t.T
        # Compute Kalman gain
        self.G =  G_t
            
        self.all_states[0] = [self.m, self.P, np.zeros((self.dim, self.dim))]
        self.all_Gs[0] = self.G
        self.means_alpha[0] = self.m[0]
        self.means_beta[0] = self.m[1]
        self.means_gamma[0] = self.m[2]
        self.means_mu[0] = self.m[0] + self.m[2]
            
        prob_lwr, prob_upr = (1-self.confidence_level)/2, (1+self.confidence_level)/2
        self.lwr_alpha[0], self.upr_alpha[0] = norm.ppf([prob_lwr, prob_upr], loc=self.m[0], scale=np.sqrt(self.P[0,0]))
        self.lwr_beta[0], self.upr_beta[0] = norm.ppf([prob_lwr, prob_upr], loc=self.m[1], scale=np.sqrt(self.P[1,1]))
        self.lwr_gamma[0], self.upr_gamma[0] = norm.ppf([prob_lwr, prob_upr], loc=self.m[2], scale=np.sqrt(self.P[2,2]))
        self.lwr_mu[0], self.upr_mu[0] = norm.ppf([prob_lwr, prob_upr], loc=self.m[0] + self.m[2], scale=np.sqrt(self.P[0,0] + self.P[2,2]))
        
    def save(self, file_path=None):
        """
        Saves the EKS object to a file with a standard title including the date,
        or with a custom file name if provided.
        
        :param file_path: (str) Path to save the EKS object (optional)
        :returns: (str) The file path where the EKS object is saved
        """
        if file_path is None:
            file_path = f"EKS_{datetime.today().strftime('%Y-%m-%d')}.pkl"
        
        with open('SLLT/results/'+file_path, 'wb') as file:
            pickle.dump(self, file)
        
        return file_path
        
def open_ps(file_path=None):
    """
    Opens an EKS object that has been stored as a pkl file.
    :param file_path: (str) Path to the pkl file (optional)
    :returns: (ExtendedKalmanSmoother) The opened EKS object
    """
    
    if file_path is None:
        file_path = f"EKS_{datetime.today().strftime('%Y-%m-%d')}.pkl"
            
    with open('SLLT/results/'+file_path, 'rb') as file:
        eks = pickle.load(file)
    return eks
