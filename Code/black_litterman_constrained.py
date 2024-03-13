import numpy as np
import pandas as pd
from scipy.optimize import minimize
import toolkit as kit

class BlackLittermanModel():
    def __init__(self, frequency, market_return, industy_return, inds_names, industry_market_cap, 
                 absolute_view, view_confidence, risk_free_rate, adjustment, omega_method='idzorek', max_obj='mv', weights_constraint=1):
        self.T = frequency
        self.market_ret = market_return
        self.ind_ret = industy_return
        self.inds_names = inds_names
        self.mcap = industry_market_cap
        self.cov_matrix = self.ind_ret.cov() * self.T
        self.absolute_view = absolute_view
        self.view_confidence = view_confidence
        self.view_confidence = np.array(view_confidence[absolute_view != 0]).reshape(-1,1)
        self.rf_rate = risk_free_rate
        self.adjustment = adjustment
        # For value of $\tau$, the Black and Litterman suggest using a small number. A common approach is to set $\tau = \frac{1}{T}$, where $T$ is number of periods 
        self.tau = 1/self.T
        self.omega_method = omega_method
        self.max_obj = max_obj
        self.weights_constraint = weights_constraint
        
        
        self.delta = None
        self.pi = None
        self.P = None
        self.Q = None
        self.omega = None
        self.BL_return = None
        self.BL_covariance = None
        self.BL_weights = None
        
        self.update_all_params()
    
    def update_all_params(self):
        self.delta = self.market_implied_risk_aversion(self.market_ret, self.rf_rate, frequency=self.T, adjustment=self.adjustment)
        self.pi = self.market_implied_prior_returns()
        self.P, self.Q = self.update_P_Q(self.absolute_view)
        
        if self.omega_method == 'idzorek':
            self.omega = self.idzorek_method(self.view_confidence, self.cov_matrix, self.Q, self.P, self.tau, self.delta)
        else:
            self.omega = self.default_omega(self.cov_matrix, self.P, self.tau)
        
        
        self.BL_return = self.bl_returns()
        self.BL_covariance = self.bl_cov()
        self.BL_weights = self.bl_weights(self.max_obj)
    
    def update_P_Q(self, absolute_view):
        """
        Given a collection of absolute views, construct
        the appropriate views vector and picking matrix. 
        """
        K = np.sum(absolute_view != 0)
        N = 30
        Q = np.zeros((K, 1))
        P = np.zeros((K, N))
        for i, ind in enumerate(absolute_view.keys()):
            Q[i] = absolute_view[ind]
            P[i, list(absolute_view.keys()).index(ind)] = 1
        return P, Q
    
    def idzorek_method(self, view_confidences, cov_matrix, Q, P, tau, risk_aversion):
        """
        Use Idzorek's method to create the uncertainty matrix given user-specified
        percentage confidences. We use the closed-form solution described by
        Jay Walters in The Black-Litterman Model in Detail (2014).

        :param view_confidences: Kx1 vector of percentage view confidences (between 0 and 1),
                                required to compute omega via Idzorek's method.
        :type view_confidences: np.ndarray, pd.Series, list,, optional
        :return: KxK diagonal uncertainty matrix
        :rtype: np.ndarray
        """
        view_omegas = []
        for ind in range(len(Q)):
            conf = view_confidences[ind]

            if conf <= 1e-60:
                view_omegas.append(1e6)
                continue

            P_view = P[ind]
            alpha = (1 - conf) / conf  # formula (44)
            omega = tau * alpha * P_view @ cov_matrix @ P_view.T  # formula (41)
            view_omegas.append(omega.item())

        return np.diag(view_omegas)
    
    
    def default_omega(self, cov_matrix, P, tau):
        """
        If the uncertainty matrix omega is not provided, we calculate using the method of
        He and Litterman (1999), such that the ratio omega/tau is proportional to the
        variance of the view portfolio.

        :return: KxK diagonal uncertainty matrix
        :rtype: np.ndarray
        """
        return np.diag(np.diag(tau * P @ cov_matrix @ P.T))
    
    
    def bl_returns(self):
        """
        Calculate the posterior estimate of the returns vector,
        given views on some assets.
        """

        pi = np.array(self.pi)
        A = np.linalg.inv(self.tau * self.cov_matrix) + self.P.T@self.omega@self.P
        b = np.linalg.inv(self.tau * self.cov_matrix)@pi + self.P.T@self.omega@self.Q    
        post_rets = np.linalg.solve(A, b)
        return pd.Series(post_rets.flatten(), index=self.inds_names)
    
    def bl_cov(self):
        """
        Calculate the posterior estimate of the covariance matrix,
        given views on some assets. 
        
        Use Tikhonov regularization if omega is singular.
        """
        if np.linalg.det(self.omega) != 0:
            omega_inv = np.linalg.inv(self.omega)
        else:
            epsilon = 1e-5  
            omega_reg = self.omega + epsilon * np.eye(self.omega.shape[0])
            omega_inv = np.linalg.inv(omega_reg)
        
        M = np.linalg.inv(np.linalg.inv(self.tau * self.cov_matrix) + self.P.T@omega_inv@self.P)
        posterior_cov = np.array(self.cov_matrix) + M
    
        return pd.DataFrame(posterior_cov, index=self.inds_names, columns=self.inds_names)
    
    def mean_variance_weights(self, method='SLSQP'):
        """
        The optimal portfolio can be constructed using the standard 
        mean-variance optimization method with constraints with no short selling 
        and sum of weights adds to 1.
        """
        n = len(self.BL_return)
    
        def objective(w):
            return - (w @ self.BL_return - self.delta / 2 * w @ self.BL_covariance @ w)
    
        if self.weights_constraint == 1:
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        else:
            constraints = [
                {'type': 'ineq', 'fun': lambda w: 1 - np.sum(w)},  # Sum less than or equal to 1
                {'type': 'ineq', 'fun': lambda w: np.sum(w)}        # Sum greater than or equal to 0
            ]
        bounds = ((0.0, 1.0),) * n 
        options = {'maxiter': 1000}
        init_guess = np.array([1/n] * n)
        result = minimize(objective, init_guess, method=method, bounds=bounds, constraints=constraints, options=options)
    
        if not result.success and method=='SLSQP':
            ## regularize to make cov matrix PD
            regularization_factor = 1e-6
            self.BL_covariance = self.BL_covariance + regularization_factor * np.eye(self.BL_covariance.shape[0])
            return self.mean_variance_weights('trust-constr')

        return pd.Series(result.x, index=self.inds_names)
    
    
    def msr_weights(self, method='SLSQP'):
        """
        Returns the weights of the portfolio that gives you the maximum sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        """
        if np.any(np.isnan(self.BL_covariance)) or np.any(np.isinf(self.BL_covariance)):
            print("COV contains NaNs or Infs.")
        
        n = len(self.BL_return)
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n 
        options = {'maxiter': 1000}
        if self.weights_constraint == 1:
            constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        else:
            constraints = {'type': 'ineq', 'fun': lambda weights: 1 - np.sum(weights),  # Sum less than or equal to 1
                'type': 'ineq', 'fun': lambda weights: np.sum(weights)}        # Sum greater than or equal to 0

       
        def neg_sharpe(weights):
            """
            Returns the negative of the sharpe ratio
            of the given portfolio
            """
            r = kit.portfolio_return(weights, self.BL_return)
            vol = kit.portfolio_vol(weights, self.BL_covariance)
            return -(r - self.rf_rate)/vol
    
        result = minimize(neg_sharpe, init_guess,
                           method=method,
                           options=options,
                           constraints=(constraints),
                           bounds=bounds)
        
        return pd.Series(result.x, index=self.inds_names)
    
    
    def bl_weights(self, max_obj='mv'):
        """
        Compute the weights implied by the posterior returns, given the
        market price of risk either by maximizing mean-variance objective or the Sharpe ratio.
        """
        if max_obj == 'msr':
            return self.msr_weights()
        return self.mean_variance_weights()
    
    def get_BL_Returns(self):
        return self.BL_return
    
    def get_BL_Covariance_Matrix(self):
        return self.BL_covariance
    
    def get_BL_Weights(self):
        return self.BL_weights
    
    
    def market_implied_risk_aversion(self, market_ret, risk_free_rate, frequency, adjustment=None):
        """
        Calculate the market-implied risk-aversion parameter (i.e market price of risk)
        based on market monthly returns.
        """
        annualized_var = market_ret.var() * frequency
        annualized_return = market_ret.mean() * frequency
        delta = (annualized_return - risk_free_rate * frequency) / annualized_var
        if adjustment is None:
            return delta
        elif delta > adjustment[1]: 
            delta = adjustment[1]
        elif delta < adjustment[0]:
            delta = adjustment[0]
        return delta
    
    def market_implied_prior_returns(self):
        """
        The neutral prior distribution is obtained by reverse engineering assuming 
        market or benchmark is the optimal portfolio. 
    
        Weights are taking from the begining of the rolling period.
        """
        # Pi is excess returns so must add risk_free_rate to get return.
        return (self.delta * self.cov_matrix@self.mcap + self.rf_rate * self.T)
    
    
    def BL_Portfolio_Return(self, actual_return):
        return self.BL_weights.T @ actual_return.values