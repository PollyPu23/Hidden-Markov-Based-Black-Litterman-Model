import numpy as np
import pandas as pd
class BlackLittermanModel():
    def __init__(self, frequency, market_return, industy_return, inds_names, industry_market_cap, 
                 absolute_view, view_confidence, risk_free_rate, adjustment, omega_method='idzorek'):
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
        self.pi = self.market_implied_prior_returns(self.mcap, self.delta, self.cov_matrix, self.rf_rate)
        self.P, self.Q = self.update_P_Q(self.absolute_view)
        
        if self.omega_method == 'idzorek':
            self.omega = self.idzorek_method(self.view_confidence, self.cov_matrix, self.Q, self.P, self.tau, self.delta)
        else:
            self.omega = self.default_omega(self.cov_matrix, self.P, self.tau)
        
        
        self.BL_return = self.bl_returns()
        self.BL_covariance = self.bl_cov()
        self.BL_weights = self.bl_weights()
    
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

        tau_sigma_P = self.tau * self.cov_matrix @ self.P.T
        pi = self.pi.values.reshape(-1, 1)

        # Solve the linear system Ax = b to avoid inversion
        A = (self.P @ tau_sigma_P) + self.omega
        b = self.Q - self.P @ pi
        post_rets = pi + tau_sigma_P @ np.linalg.solve(A, b)
        return pd.Series(post_rets.values.flatten(), index=self.inds_names)
    
    def bl_cov(self):
        """
        Calculate the posterior estimate of the covariance matrix,
        given views on some assets. Based on He and Litterman (2002).
        It is assumed that omega is diagonal. If this is not the case,
        please manually set omega_inv.
        """
        tau_sigma_P = self.tau * self.cov_matrix @ self.P.T

        # Solve the linear system Ax = b to avoid inversion
        A = (self.P @ tau_sigma_P) + self.omega
        b = tau_sigma_P.T

        M = np.array(self.tau * self.cov_matrix) - np.array(tau_sigma_P @ np.linalg.solve(A, b))
        posterior_cov = np.array(self.cov_matrix) + M
        return pd.DataFrame(posterior_cov, index=self.inds_names, columns=self.inds_names)
    
    def bl_weights(self):
        """
        Compute the weights implied by the posterior returns, given the
        market price of risk. Technically this can be applied to any
        estimate of the expected returns, and is in fact a special case
        of mean-variance optimization. In short, these are the MSR portfolio weights.
        """
        posterior_rets = self.BL_return
        A = self.delta * self.BL_covariance
        b = posterior_rets
        raw_weights = np.linalg.solve(A, b)
        weights = raw_weights / raw_weights.sum()
        return pd.Series(weights, index=self.inds_names)
    
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
        delta = (annualized_return - risk_free_rate) / annualized_var
        if adjustment is None:
            return delta
        elif delta > adjustment[1]: 
            delta = adjustment[1]
        elif delta < adjustment[0]:
            delta = adjustment[0]
        return delta
    
    def market_implied_prior_returns(self, market_caps, risk_aversion, cov_matrix, risk_free_rate):
        """
        The neutral prior distribution is obtained by reverse engineering assuming 
        market or benchmark is the optimal portfolio. 
    
        Weights are taking from the begining of the rolling period.
        """
        # Pi is excess returns so must add risk_free_rate to get return.
        return (risk_aversion * cov_matrix@market_caps + risk_free_rate)
    
    def BL_Portfolio_Return(self, actual_return):
        return self.BL_weights.T @ actual_return.values