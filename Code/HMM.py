from sklearn.mixture import GaussianMixture 
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toolkit as kit
import warnings
from hmmlearn import hmm


sns.set(style="dark")
plt.style.use("ggplot")
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [12, 8]

class IndustryHMM:
    def __init__(self, industry, name):
        ALL_DATA = kit.get_ind_30_ret()
        self.name = name
        self.data = ALL_DATA[industry]
        self.data_arr = np.array(self.data)
        self.num_states = 3
        self.initial_matrix = []
        self.transition_matrix = []
        self.means = []
        self.covariances = []
        self.hidden_states = []
        self.strategy1ret = []
        self.start_time = 0
        self.strategy1decisions = []
        self.confidence = []
        self.absolute_view = []

    def model_selection(self, print_table=False, message=False):
        aics = []
        bics = []
        ll = []
        for i in range(3, 6):
            model = hmm.GaussianHMM(n_components=i, covariance_type='full', n_iter=1000, random_state=2023)
            model.fit(self.data_arr.reshape(-1, 1))
            aics.append(model.aic(self.data_arr.reshape(-1, 1)))
            bics.append(model.bic(self.data_arr.reshape(-1, 1)))
            ll.append(model.score(self.data_arr.reshape(-1, 1)))
        num_states_scores = pd.DataFrame({'Number of Latent States': np.arange(3,6), 'Log Likelihood': ll, 'AIC': aics, 'BIC': bics})
        num_states_scores = num_states_scores.set_index('Number of Latent States')
        if print_table:
            display(num_states_scores)
        self.num_states = num_states_scores['BIC'].idxmin()
        if message:
            print(f'Best number of States selected by BIC is {self.num_states}')
    
    def industry_HMM_with_all_data(self, display_states=False):
        model = hmm.GaussianHMM(n_components=self.num_states, covariance_type='full', n_iter=1000, random_state=2023)
        model.fit(self.data_arr.reshape(-1, 1))
        Z = model.predict(self.data_arr.reshape(-1, 1))
        states = pd.unique(Z)
        
        if display_states:
            for i in states:
                index = (Z == i)
                x = self.data.iloc[index]
                x.plot(style='.')
            plt.legend(states, fontsize=16)
            plt.grid(True)
            plt.ylabel(f"{self.name} Monthly Return", fontsize=16)
            plt.show()
        
        self.initial_matrix = model.get_stationary_distribution()
        self.transition_matrix = np.around(model.transmat_, 2)
        self.means = model.means_
        self.covariances = model.covars_
        self.hidden_states = model.predict(self.data_arr.reshape(-1,1))
        
    
    def display_hidden_states_mean_return(self):
        self.data.plot()
        plt.plot(pd.Series(self.means[self.hidden_states].reshape(-1), index=self.data.index), alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Monthly Return')
        plt.title('Expected Return on Hidden States');
    
    def display_countplot_positive_negative_hidden_states(self):
        df = pd.DataFrame({'Ret':self.data, 'State': self.hidden_states})
        df['Positive?'] = (df['Ret'] > 0).replace({True: '> 0', False: '<= 0'})
        sns.countplot(df, x="State", hue='Positive?', alpha=0.8)
        plt.title('Positive and Negative Returns in Hidden States');
        
        
    def trailing_CV(self, train_size=0.6, trailing_window=120, omega=0.6, mu=0.6, plot=False, decision_metrics='simple'):
        """
        hyperparameters to be tuned: train_size, trailing_window, omega, mu
        omega, mu can represent investor's risk averseness, higher values imply more risk averse.
        """
        n = len(self.data_arr)
        t0 = int(n*train_size)
        self.start_time = t0
        
        strategy1decisions = []
        confidence = []
        absolute_view = []
        
        for t in range(t0+1, n):
            # Fit HMM Model
            model = hmm.GaussianHMM(n_components=self.num_states, covariance_type='diag', n_iter=1000, random_state=2023)
            start = t-trailing_window-1
            train = self.data_arr[start:t]
            model.fit(train.reshape(-1, 1))
            
            #  Extract Parameter
            transition_matrix = model.transmat_
            hidden_states = model.predict(train.reshape(-1,1))
            Z_curr = hidden_states[-1]
            prob_Z_curr = model.predict_proba(train[-1].reshape(-1,1))[0][Z_curr]
            confidence.append(prob_Z_curr)
            
            # Find observations with the same state as O_t-1 in [t-W-1: t-2]
            obs_same_state = train[hidden_states==Z_curr][:-1]
            long_win_rate = np.sum(obs_same_state > 0) / len(obs_same_state)
            short_selling_rate = np.sum(obs_same_state < 0) / len(obs_same_state)
            
            true_indices = np.where(hidden_states==Z_curr)[0]
            next_growth_indices = true_indices + 1
            # Filter out indices that are out of bounds
            next_growth_indices = next_growth_indices[next_growth_indices < len(hidden_states)]
            next_growth = train[next_growth_indices]
            
            # similar to simple voting
            average_growth = np.mean(next_growth)
            # TODO:Try weighted voting by transition matrix 
            if decision_metrics == 'weighted':
                next_hidden_states = hidden_states[next_growth_indices]
                weight = transition_matrix[Z_curr][next_hidden_states]
                average_growth = np.average(next_growth, weights=weight)
            
            
            absolute_view.append(np.exp(np.mean(next_growth))-1)
                
            if average_growth > 0 and long_win_rate > omega:
                strategy1decisions.append(1)
            elif average_growth < 0 and short_selling_rate > mu:
                strategy1decisions.append(-1)
            else:
                strategy1decisions.append(0)
            
        strategy1decisions = pd.Series(strategy1decisions, index = self.data.index[t0+1:])
        self.strategy1decisions = strategy1decisions
        self.confidence = confidence
        self.absolute_view = absolute_view
        if plot:
            self.plot_decision_on_cum_return(strategy1decisions.index, strategy1decisions)
    
    def trading_strategy_ret(self, plot=False):
        """
        If the trading signal for y_t+1 = 1, buy portfolio.
        If the trading signal for y_t+1 = -1, sell portfolio.
        Else, stay still
        """
        adjusted_returns = []
        previous_signal = 0

        for signal, monthly_return in zip(self.strategy1decisions, self.data_arr[self.start_time+1:]):
            if signal == 1:
                adjusted_returns.append(monthly_return)
                previous_signal = 1
            elif signal == -1:
                adjusted_returns.append(0)
                previous_signal = -1
            elif signal == 0:
                if previous_signal == 1:
                    adjusted_returns.append(monthly_return)
                else:
                    adjusted_returns.append(0)
                
                ## Assuming lack of signal means volatile state -> we buy at the time
        
        
        buy_hold = (self.data[self.start_time+1:]).cumsum()
        strategy_hmm = pd.Series(adjusted_returns, index = self.data.index[self.start_time+1:])
        strategy_hmm_cumprod = strategy_hmm.cumsum()
        print(len(strategy_hmm_cumprod))
        print(len(buy_hold))
        if plot:
            buy_hold.plot(label='Buy and Hold')
            strategy_hmm_cumprod.plot(label='Strategy HMM Return')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.title(f'Buy and Hold Compared with Trading Strategy');
            
        print(f'Buy and Hold: {buy_hold[-1]}\n Strategy HMM: {strategy_hmm_cumprod[-1]}')
        
        
    def all_past_CV(self, train_size=0.6, omega=0.6, mu=0.6, plot=False, decision_metrics='simple'):
        """
        hyperparameters to be tuned: train_size, omega, mu
        omega, mu can represent investor's risk averseness, higher values imply more risk averse.
        """
        n = len(self.data_arr)
        t0 = int(n*train_size)
        self.start_time = t0
        
        strategy1decisions = []
        confidence = []
        absolute_view = []
        
        for t in range(t0+1, n):
            # Fit HMM Model
            model = hmm.GaussianHMM(n_components=self.num_states, covariance_type='diag', n_iter=1000, random_state=2023)
            train = self.data_arr[:t]
            model.fit(train.reshape(-1, 1))
            
            #  Extract Parameter
            transition_matrix = model.transmat_
            hidden_states = model.predict(train.reshape(-1,1))
            Z_curr = hidden_states[-1]
            prob_Z_curr = model.predict_proba(train[-1].reshape(-1,1))[0][Z_curr]
            confidence.append(prob_Z_curr)
            
            # Find observations with the same state as O_t-1 in [0: t-2]
            obs_same_state = train[hidden_states==Z_curr][:-1]
            long_win_rate = np.sum(obs_same_state > 0) / len(obs_same_state)
            short_selling_rate = np.sum(obs_same_state < 0) / len(obs_same_state)
            
            true_indices = np.where(hidden_states==Z_curr)[0]
            next_growth_indices = true_indices + 1
            # Filter out indices that are out of bounds
            next_growth_indices = next_growth_indices[next_growth_indices < len(hidden_states)]
            next_growth = train[next_growth_indices]
            
            # similar to simple voting
            average_growth = np.mean(next_growth)
            # TODO:Try weighted voting by transition matrix 
            if decision_metrics == 'weighted':
                next_hidden_states = hidden_states[next_growth_indices]
                weight = transition_matrix[Z_curr][next_hidden_states]
                average_growth = np.average(next_growth, weights=weight)
            
            
            absolute_view.append(np.exp(np.mean(next_growth))-1)
                
            if average_growth > 0 and long_win_rate > omega:
                strategy1decisions.append(1)
            elif average_growth < 0 and short_selling_rate > mu:
                strategy1decisions.append(-1)
            else:
                strategy1decisions.append(0)
            
        strategy1decisions = pd.Series(strategy1decisions, index = self.data.index[t0+1:])
        self.strategy1decisions = strategy1decisions
        self.confidence = confidence
        self.absolute_view = absolute_view
        if plot:
            self.plot_decision_on_cum_return(strategy1decisions.index, strategy1decisions)
        
        
        
        
    def get_decison_array(self):
        return self.strategy1decisions
    
    def get_initial_matrix(self):
        return self.initial_matrix
    
    def get_transition_matrix(self):
        return self.transition_matrix
    
    def get_means(self):
        return self.means
    
    def get_covariances(self):
        return self.covariances
    
    def get_hidden_states(self):
        return self.hidden_states
    
    def get_confidence_array(self):
        return self.confidence
    
    def get_absolute_view(self):
        return self.absolute_view
    
    def get_CV_train_data(self):
        return self.data[self.start_time+1:]
    
    def get_CV_index(self):
        return self.data[self.start_time+1:].index
    
    def plot_decision_on_cum_return(self, time, arr):

        # Create patches for the legend
        patches = [
            plt.Rectangle((0,0),1,1,fc="green", alpha=0.3),
            plt.Rectangle((0,0),1,1,fc="red", alpha=0.3),
            plt.Rectangle((0,0),1,1,fc="white", alpha=0.3)
        ]
        self.data[self.start_time+1:].plot()
        for i in range(len(arr)-1):
            if arr[i] == 1:
                plt.axvspan(time[i].to_timestamp(), time[i+1].to_timestamp(), color='green', alpha=0.3)
            elif arr[i] == -1:
                plt.axvspan(time[i].to_timestamp(), time[i+1].to_timestamp(), color='red', alpha=0.3)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Highlight Regions Based on Decision')
        plt.legend(patches, ['Buy', 'Sell', 'Stay Still'], loc='best')