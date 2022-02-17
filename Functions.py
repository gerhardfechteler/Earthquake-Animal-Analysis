import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import bisect
import seaborn as sns
import datetime as dt

def accumulate_pga(Data_Animal, Data_EQ, method = 'max'):
    """
    accumulate_pga(Data_Animal, Data_EQ, method = 'max')
    
    accumulates the PGA values and returns a dataframe with the respective
    animal and EQ data together.
    
    Parameters
    ----------
    Data_Animal: pandas dataframe
        animal data, 15min frequency, including datetime column
    Data_EQ: pandas dataframe
        EQ data, all events, including datetime column
    method: str
        accumulation method, default 'max'
    
    Returns
    -------
    Data_accumulated: pandas dataframe
        animal and PGA events at same frequency
    """
    
    n_obs = len(Data_Animal.datetime)
    # create variable for PGA accumulated to 15 minute scale and one for the 
    # corresponding hypocentral distance
    pga_accumulated = np.zeros(n_obs)
    hyp_dis = np.zeros(n_obs)
    for i,date in enumerate(Data_EQ.datetime):
        # infer corresponding time index for EQ event
        index = (date - Data_Animal.datetime[0]).total_seconds() / 60 / 15
        if (index > 0) & (int(index) < n_obs):
            if Data_EQ.PGA[i] > pga_accumulated[int(index)]:
                pga_accumulated[int(index)] = Data_EQ.PGA[i]
                hyp_dis[int(index)] = Data_EQ['HYP-DIS'][i]
    # create dataframe with animal and EQ data
    Data_accumulated = Data_Animal.copy()
    Data_accumulated['eq'] = pga_accumulated
    Data_accumulated['HYP-DIS'] = hyp_dis
    
    return Data_accumulated


def fourier_filter(Data, names, period, max_freq = 30, fixed_freq = False):
    """
    fourier_filter(Data, names, period, max_freq = 30, fixed_freq = False)
    
    Fourier filter 
    
    Parameters
    ----------
    Data: pandas dataframe
        contains (among others) the variables that shall be filtered
    names: list of strings
        list with column names of variables that shall be filtered
    period: float or int
        period of series
    max_freq: int
        maximum number of frequencies to consider, 0 is intercept only
    fixed_freq: boot or int
        if False, compute information criteria
        if int, compute residual from filtering with fixed_freq frequencies,
        max_freq is then ignored
    
    Returns
    -------
    if fixed_freq == False:
        dict
            information criteria for considered number of frequencies, keys:
                AIC_values - Values of AIC for all series
                BIC_values - 
                AIC_recommendation - recommended number of frequencies for all 
                                     series by AIC
                BIC_recommendation - recommended number of frequencies for all 
                                     series by BIC
    else:
        Data_filtered: pandas dataframe
            dataframe Data where variables in names are replaced by fourier
            residuals
    """
    
    # create dataframe with variables to filter
    Y = Data[names]
    # number of observations, i.e. length of series
    n_obs = Y.shape[0]
    # index of series
    ind = np.arange(n_obs)
    
    if fixed_freq == False:
        # create regressor dataframe
        X = pd.DataFrame()
        X['0'] = np.ones(n_obs)
        coef = pd.DataFrame(inv(X.transpose() @ X), X.columns, X.columns) @ X.transpose() @ Y
        # residuals
        resid = Y - X @ coef
        # variance of error term
        sig2 = np.sum(resid**2, axis=0) / (n_obs+1)
        # log-likelihood
        loglik = -n_obs / 2 * (1 + np.log(2*np.pi*sig2))
        # information criteria
        AIC = np.zeros((max_freq+1, len(names)))
        BIC = np.zeros((max_freq+1, len(names)))
        AIC[0,:] = 2 * (2*0 + 1) - 2 * loglik
        BIC[0,:] = np.log(n_obs) * (2*0 + 1) - 2 * loglik
        for k in np.arange(1, max_freq+1):
            # add frequency k to X
            X[str(k)+'sin'] = np.sin(2*np.pi / period * k * ind)
            X[str(k)+'cos'] = np.cos(2*np.pi / period * k * ind)
            # coefficient estimates for frequencies
            coef = pd.DataFrame(inv(X.transpose() @ X), X.columns, X.columns) @ X.transpose() @ Y
            # residuals
            resid = Y - X @ coef
            # variance of error term
            sig2 = np.sum(resid**2, axis=0) / (n_obs+1)
            # log-likelihood
            loglik = -n_obs / 2 * (1 + np.log(2*np.pi*sig2))
            # information criteria
            AIC[k,:] = 2 * (2*k + 1) - 2 * loglik
            BIC[k,:] = np.log(n_obs) * (2*k + 1) - 2 * loglik
        number_AIC = pd.DataFrame(np.argmin(AIC, axis=0), index = names)
        number_BIC = pd.DataFrame(np.argmin(BIC, axis=0), index = names)
        AIC = pd.DataFrame(AIC, columns = names)
        BIC = pd.DataFrame(BIC, columns = names)
        
        return {'AIC_values':AIC, 'BIC_values':BIC, 
                'AIC_recommendation':number_AIC, 'BIC_recommendation':number_BIC}
    
    else:
        # create regressor dataframe
        X = pd.DataFrame()
        X['0'] = np.ones(n_obs)
        for k in np.arange(1, fixed_freq+1):
            # add frequency k to X
            X[str(k)+'sin'] = np.sin(2*np.pi / period * k * ind)
            X[str(k)+'cos'] = np.cos(2*np.pi / period * k * ind)
        # coefficients for frequencies
        coef = pd.DataFrame(inv(X.transpose() @ X), X.columns, X.columns) @ X.transpose() @ Y
        residual = Y - np.array(X @ coef)
        Data_filtered = Data.copy()
        Data_filtered[names] = residual
        # periodicity
        periodicity = (X @ coef).iloc[:round(period),:]
        
        return Data_filtered, {'coef':coef, 'periodicity':periodicity}


class threshold_choice():
    def __init__(self, t_grids, trigger, select, horizon):
        self.t_grids = t_grids
        self.trigger = trigger
        self.select = select
        self.horizon = horizon
    
    def plot(self):
        fig = plt.figure(figsize = (6,5))
        # indices for subplots
        indices = ['A) ', 'B) ', 'C) ', 'D) ']
        for i,key in enumerate(self.select):
            plt.subplot(2,2,i+1)
            heatmap = sns.heatmap(self.t_grids[key], 
                                  vmin=-4, vmax=4,
                                  cmap = 'coolwarm',
                                  )
            
            plt.title(indices[i] + key)
            plt.tight_layout()
            heatmap.invert_yaxis()
            ax = plt.gca()
            plt.yticks(rotation=45)
            ax.set_xticks([0, 9, 19, 29, 39])
            ax.set_yticks([0, 9, 19, 29, 39])
            ax.set_xticklabels([1, 1.5, 2, 2.5, 3])
            ax.set_yticklabels([1, 1.5, 2, 2.5, 3])
            if i in [0,1]:
                ax.axes.xaxis.set_ticklabels([])
            if i in [1,3]:
                ax.axes.yaxis.set_ticklabels([])
        fig.text(0.5, 0.005, 'Animal threshold [in multiples of std. dev.]', ha='center', fontsize = 12)
        fig.text(0.005, 0.5, 'EQ threshold [in multiples of std. dev.]', 
                     va='center', rotation='vertical', fontsize = 12)
        fig.text(0.95, 0.5, 't-statistic for slope parameter', 
                     va='center', rotation='vertical', fontsize = 12)
        plt.tight_layout(rect = (0.02, 0.02, 0.95, 0.95))
        return fig
            
            


class threshold_select():
    def __init__(self, Data_selected, regression_results, horizon, thresholds):
        self.Data_selected = Data_selected
        self.regression_results = regression_results
        self.horizon = horizon
        self.thresholds = thresholds
    
    def plot(self, bounds = 'c'):
        """
        Parameters
        ----------
        bounds: str
            kind of bounds that shall be added to plot, shall be one of
            'c' for confidence bands
        """
        fig = plt.figure(figsize = (6,5.5))
        # indices for subplots
        indices = ['A) ', 'B) ', 'C) ', 'D) ']
        for i,key in enumerate(self.Data_selected):
            # print regression table
            print(self.regression_results[key].summary())
            plt.subplot(2,2,i+1)
            # obtain x and y values and relevant parameters
            x = self.Data_selected[key]['distance']
            y = self.Data_selected[key]['warning_time']
            xlim = [0, np.max(x)*1.05]
            intercept, slope = self.regression_results[key].params
            cov_params = self.regression_results[key].normalized_cov_params
            
            # x resolution for confidence bounds, number of points
            x_resol = 20
            x_val = np.linspace(0, xlim[1], x_resol)
            X_val = np.array([np.ones(x_resol), x_val]).transpose()
            # confidence interval
            CI = np.zeros((x_resol, 2))
            for j,xj in enumerate(x_val):
                var_y = X_val[j,:].reshape((1,-1)) @ np.array(cov_params) @ X_val[j,:].reshape((-1,1))
                CI[j,0] = intercept + xj * slope - 1.96 * np.sqrt(var_y)
                CI[j,1] = intercept + xj * slope + 1.96 * np.sqrt(var_y)
            # plot confidence bounds
            plt.fill_between(x_val, CI[:,0], CI[:,1],
                             alpha = 0.5,
                             color = 'grey',
                             linewidth = 0,
                             label = '95% confidence bounds')
            
            # add datapoints as scatterplot
            plt.scatter(x, y,
                        alpha = 0.3,
                        label = 'Selected Data')
            
            # plot regression line
            plt.plot([0, xlim[1]], [intercept, intercept + xlim[1]*slope],
                     color = 'black',
                     label = 'Median regression')
            
            if i in [0,1]:
                plt.gca().set_xticklabels('')
#            if i in [1,3]:
#                plt.gca().set_yticklabels('')
            plt.xlim(tuple(xlim))
            plt.ylim((0,20))
            plt.title(indices[i] + key)
        
        fig.text(0.5, 0.09, 'Hypocentral Distance [km]', ha='center', fontsize = 12)
        fig.text(0.005, 0.5, 'Anticipation Time [h]', 
                     va='center', rotation='vertical', fontsize = 12)
        hand, lab = plt.gca().get_legend_handles_labels()
        fig.legend(hand, lab,
                   loc = 'center',
                   bbox_to_anchor = (0,0,1,0.06),
                   ncol = 3)
        plt.tight_layout(rect = (0.02,0.1,1,0.95))
        plt.subplots_adjust()


class threshold_model():
    def __init__(self, Data_trigger, Data_select, trigger, select):
        """
        Parameters
        ----------
        Data_trigger: pandas dataframe
            data, containing trigger variable and datetime variable
        Data_select: pandas dataframe
            data, containing select variable(s) and datetime variable
        trigger: str
            variable to trigger selection
        select: list of str
            variables to select from when triggered
        """
        self.Data_trigger = Data_trigger
        self.Data_select = Data_select
        self.trigger = trigger
        self.select = select
    
    def choose_threshold(self, horizon, th_trigger, th_select):
        """
        Parameters
        ----------
        horizon: datetime.timedelta
            time span before trigger event to consider for selection
        th_trigger: 1D numpy array
            threshold values to consider for trigger variable (multiples of
            standard deviation)
        th_select: 1D numpy array
            threshold values to consider for select variables (multiples of
            standard deviation)
        """
        # standardize trigger and select variables
        trigger = self.Data_trigger[self.trigger]
        trigger = (trigger - np.mean(trigger)) / np.std(trigger)
        select = self.Data_select[self.select]
        select = (select - np.mean(select, axis = 0)) / np.std(select, axis = 0)
        select['datetime'] = self.Data_select.datetime
        # in October, the mowing time on the fourth day has to be excluded
        mow_start = dt.datetime(year=2017, month = 11, day=2, hour = 10)
        mow_end = dt.datetime(year=2017, month = 11, day=2, hour = 15)
        # create dict with lists of selected events
        selected_events = {key:list() for key in self.select}
        for i, trig_i in enumerate(trigger):
            if trig_i > th_trigger[0]:
                th_trigger_index = bisect.bisect_left(th_trigger, trig_i) - 1
                # time of trigger event
                trigger_time = self.Data_trigger['datetime'][i]
                # vector to indicate which select events are in horizon before the trigger event
                relevant_bool = (select.datetime <= trigger_time) & \
                                (select.datetime >= trigger_time - horizon)
                # select rows of select that are in horizon before trigger event
                relevant_select = select.loc[relevant_bool,:].reset_index(drop = True)
                for j, group in enumerate(self.select):
                    for k, sel_k in enumerate(relevant_select[group]):
                        # take negative events for cows and sheep
                        sel_k = sel_k * (1 - 2*(group in ['cow']))
                        if sel_k > th_select[0]:
                            th_select_index = bisect.bisect_left(th_select, sel_k) - 1
                            # if select event in horizon before trigger event
                            # exceeds threshold, add line to selected_events,
                            # containing warning time and distance
                            select_time =  relevant_select.datetime[k]
                            if mow_start <= select_time <= mow_end:
                                continue
                            warning_time = (trigger_time - select_time).total_seconds() / 60 / 60
                            hyp_dis = self.Data_trigger['HYP-DIS'][i]
                            selected_events[group].append([warning_time, hyp_dis, th_trigger_index, th_select_index])
        
        t_grids = {key:np.full((len(th_trigger), len(th_select)), np.nan) for key in self.select}
        for key in selected_events:
            # replace event lists by dataframes
            selected_events[key] = pd.DataFrame(selected_events[key], 
                           columns = ['warning_time', 'distance', 'max_trigger_index', 'max_select_index'])
            for i, th_trigger_i in enumerate(th_trigger):
                events_trigger_filtered = selected_events[key].loc[selected_events[key].max_trigger_index >= i]
                for j, th_select_i in enumerate(th_select):
                    events_filtered = events_trigger_filtered.loc[events_trigger_filtered.max_select_index >= j]
                    # perform regression for filtered events
                    if len(events_filtered) > 3:
                        # regression_result = smf.quantreg('warning_time ~ distance', events_filtered).fit(q = 0.5)
                        regression_result = smf.ols('warning_time ~ distance', events_filtered).fit()
                        t_grids[key][i,j] = regression_result.tvalues[1]
            t_grids[key] = pd.DataFrame(t_grids[key], index = np.round(th_trigger,1), columns = np.round(th_select,1))
                    
        
        return threshold_choice(t_grids, self.trigger, self.select, horizon)
            
    
    def select_for_threshold(self, horizon, thresholds = (1,1)):
        """
        Parameters
        ----------
        horizon: datetime.timedelta
            time span before trigger event to consider for selection
        thresholds: tuple of two integers
            threshold values for trigger and select variables, (multiples of
            standard deviation), defaults to (1,1)
        
        Returns
        -------
        threshold_selec: threshold_select object
        """
        # standardize trigger and select variables
        trigger = self.Data_trigger[self.trigger]
        trigger = (trigger - np.mean(trigger)) / np.std(trigger)
        select = self.Data_select[self.select]
        select = (select - np.mean(select, axis = 0)) / np.std(select, axis = 0)
        select['datetime'] = self.Data_select.datetime
        select['HYP-DIS'] = self.Data_select['HYP-DIS']
        # in October, the mowing time on the fourth day has to be excluded
        mow_start = dt.datetime(year=2017, month = 11, day=2, hour = 10)
        mow_end = dt.datetime(year=2017, month = 11, day=2, hour = 15)
        # create dict with lists of selected events
        selected_events = {key:list() for key in self.select}
        for i, trig_i in enumerate(trigger):
            if trig_i > thresholds[0]:
                # time of trigger event
                trigger_time = self.Data_trigger['datetime'][i]
                # vector to indicate which select events are in horizon before the trigger event
                relevant_bool = (select.datetime <= trigger_time) & \
                                (select.datetime >= trigger_time - horizon)
                # select rows of select that are in horizon before trigger event
                relevant_select = select.loc[relevant_bool,:].reset_index(drop = True)
                for j, group in enumerate(self.select):
                    for k, sel_k in enumerate(relevant_select[group]):
                        if sel_k * (1 - 2*(group in ['cow'])) > thresholds[1]:
                            # if select event in horizon before trigger event
                            # exceeds threshold, add line to selected_events,
                            # containing warning time and distance
                            select_time =  relevant_select.datetime[k]
                            if mow_start <= select_time <= mow_end:
                                continue
                            warning_time = (trigger_time - select_time).total_seconds() / 60 / 60
                            # hyp_dis = relevant_select['HYP-DIS'][k]
                            hyp_dis = self.Data_trigger['HYP-DIS'][i]
                            selected_events[group].append([warning_time, hyp_dis, trigger_time])
        
        regression_results = {key:None for key in self.select}
        for key in selected_events:
            # replace event lists by dataframes
            selected_events[key] = pd.DataFrame(selected_events[key], columns = ['warning_time', 
                                                                                 'distance', 
                                                                                 'trigger_time'])
            # perform quantile regression for selected events
            regression_results[key] = smf.quantreg('warning_time ~ distance', selected_events[key]).fit(q = 0.5)
        
        return threshold_select(selected_events, regression_results, horizon, thresholds)











