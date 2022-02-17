# software packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from statsmodels.tsa.api import VAR
# load functions and classes from separate file
import Functions as fcts

# Raw Data Import ############################################################
##############################################################################

# import animal data for Oct-Nov period
Oct_Nov_Animals = pd.read_excel('Data_Anim_Oct_Nov.xlsx',
                                names = ['datetime','cow','dog','sheep','animal'])
# drop information of first day on which no EQ data available
Oct_Nov_Animals = Oct_Nov_Animals.iloc[48:,:].reset_index(drop = True)

# import animal data for Jan-Apr period
Jan_Apr_Animals = pd.read_excel('Data_Anim_Jan_Apr.xlsx',
                                names = ['datetime','animal','cow','dog','sheep'])
# drop incomplete information of first day and last day, respectively
Jan_Apr_Animals = Jan_Apr_Animals.iloc[47:-10,:].reset_index(drop = True)

# import EQ data for Oct-Nov period
Oct_Nov_EQ = pd.read_excel('Data_EQ_Oct_Nov.xlsx',
                           skiprows = [1])

# import EQ data for Jan-Apr period
Jan_Apr_EQ = pd.read_excel('Data_EQ_Jan_Apr.xlsx',
                           skiprows = [0,1,2,3,4,6,7])

# split animal and EQ data in Jan-Mar and Mar-Apr period
split_date = dt.datetime(2017, 3, 11)
Jan_Mar_Animals = Jan_Apr_Animals.loc[Jan_Apr_Animals.datetime < split_date,:]
Mar_Apr_Animals = Jan_Apr_Animals.loc[Jan_Apr_Animals.datetime >= split_date,:].reset_index(drop = True)
Jan_Mar_EQ = Jan_Apr_EQ.loc[Jan_Apr_EQ.datetime < split_date,:]
Mar_Apr_EQ = Jan_Apr_EQ.loc[Jan_Apr_EQ.datetime >= split_date,:].reset_index(drop = True)


# Accumulate PGA #############################################################
##############################################################################

# accumulate PGA events on animal time scale for Oct-Nov period
Data_accumulated_Oct_Nov = fcts.accumulate_pga(Oct_Nov_Animals, Oct_Nov_EQ)

# accumulate PGA events on animal time scale for Jan-Apr period
Data_accumulated = fcts.accumulate_pga(Jan_Apr_Animals, Jan_Apr_EQ)
# spit dataframe for Jan-Mar and Mar-Apr, i.e. stable and pasture period
Data_accumulated_Jan_Mar = Data_accumulated.loc[Data_accumulated.datetime < split_date,:]
Data_accumulated_Mar_Apr = Data_accumulated.loc[Data_accumulated.datetime >= split_date,:].reset_index(drop = True)

# list of periods
periods = ['A) Oct-Nov Period (Stable)', 'B) Jan-Mar Period (Stable)', 'C) Mar-Apr Period (Pasture)']

# remove superfluous variables
del(Data_accumulated, Jan_Apr_Animals, Jan_Apr_EQ)

# Remove Daily Patterns from Animal Series ###################################
##############################################################################

# animal names
groups = ['animal','cow','dog','sheep']

# Obtain information criteria for selection of number of frequencies
Fourier_IC_Oct_Nov = fcts.fourier_filter(Data_accumulated_Oct_Nov, groups, 4*24)
Fourier_IC_Jan_Mar = fcts.fourier_filter(Data_accumulated_Jan_Mar, groups, 4*24)
Fourier_IC_Mar_Apr = fcts.fourier_filter(Data_accumulated_Mar_Apr, groups, 4*24)
# print table with suggested frequencies by BIC
BIC_suggest = pd.DataFrame()
BIC_suggest['Oct-Nov'] = Fourier_IC_Oct_Nov['BIC_recommendation'].iloc[:,0]
BIC_suggest['Jan-Mar'] = Fourier_IC_Jan_Mar['BIC_recommendation'].iloc[:,0]
BIC_suggest['Mar-Apr'] = Fourier_IC_Mar_Apr['BIC_recommendation'].iloc[:,0]
print(BIC_suggest)

# chosen number of frequencies (max suggestion by BIC)
number_of_frequencies = 16

# Obtain fourier filtered data
Data_fourier_Oct_Nov, Fourier_info_Oct_Nov = fcts.fourier_filter(Data_accumulated_Oct_Nov, 
                                                                 groups, 4*24, fixed_freq=number_of_frequencies)
Data_fourier_Jan_Mar, Fourier_info_Jan_Mar = fcts.fourier_filter(Data_accumulated_Jan_Mar, 
                                                                 groups, 4*24, fixed_freq=number_of_frequencies)
Data_fourier_Mar_Apr, Fourier_info_Mar_Apr = fcts.fourier_filter(Data_accumulated_Mar_Apr, 
                                                                 groups, 4*24, fixed_freq=number_of_frequencies)

# Plot periodicity
fig_fourier = plt.figure(figsize = (6,5))
for i,Fourier_info in enumerate([Fourier_info_Oct_Nov, Fourier_info_Jan_Mar, Fourier_info_Mar_Apr]):
    plt.subplot(2,2,i+1)
    for j,name in enumerate(groups):
        # create labels for legend in first subplot
        if i==0:
            plt.plot(np.arange(0,24,0.25), Fourier_info['periodicity'][name], label = name)
        else:
            plt.plot(np.arange(0,24,0.25), Fourier_info['periodicity'][name])
    plt.xlabel('Time of day [h]')
    plt.ylabel('Activity (ODBA)')
    yliminfo = plt.ylim()
    plt.ylim(0, yliminfo[1])
    ytickinfo = plt.yticks()
    plt.yticks(ytickinfo[0], ytickinfo[0]/100000)
    plt.title(periods[i])
fig_fourier.legend(loc='center', 
                   bbox_to_anchor=(0.5, 0., 0.5, 0.5),
                   title = 'Daily activity patterns of')
plt.tight_layout()
plt.savefig('S4_Daily_Activity_Patterns.pdf', dpi = 600, format = 'pdf')

# remove superfluous variables
del(i, Fourier_info, j, name, yliminfo, ytickinfo)


# Remove Reactive Pattern from Animal Series #################################
##############################################################################

# ordering for orthogonalized IRFs
ordering = ['eq','cow','dog','sheep']

# compute information criteria for lag lengths between 0 and 9
info_crit_VAR = {'BIC':np.zeros((10,3)),
                 'AIC':np.zeros((10,3))}
periods_data = {'Oct-Nov':Data_fourier_Oct_Nov[ordering],
                'Jan-Mar':Data_fourier_Jan_Mar[ordering],
                'Mar-Apr':Data_fourier_Mar_Apr[ordering]}
for j,period in enumerate(periods_data):
    for i in range(10):
        info_i = VAR(periods_data[period]).fit(i).info_criteria
        info_crit_VAR['BIC'][i,j] = info_i['bic']
        info_crit_VAR['AIC'][i,j] = info_i['aic']
for crit in info_crit_VAR:
    info_crit_VAR[crit] = pd.DataFrame(info_crit_VAR[crit],
                 columns = list(periods_data))
    info_crit_VAR[crit]['lags'] = np.arange(0,10)
    info_crit_VAR[crit] = info_crit_VAR[crit].set_index('lags')
# print BIC values
print(info_crit_VAR['BIC'])
    

# number of lags to use
var_lags = 6
# perform var analysis for separate animal behavior
var_Oct_Nov = VAR(Data_fourier_Oct_Nov[ordering]).fit(var_lags)
print(var_Oct_Nov.summary())
var_Jan_Mar = VAR(Data_fourier_Jan_Mar[ordering]).fit(var_lags)
print(var_Jan_Mar.summary())
var_Mar_Apr = VAR(Data_fourier_Mar_Apr[ordering]).fit(var_lags)
print(var_Mar_Apr.summary())
# Impulse response functions for Oct-Nov period
fig_var_Oct_Nov = var_Oct_Nov.irf(5*4).plot(orth=True)
fig_var_Oct_Nov.set_size_inches(10,10)
stds = np.std(Data_fourier_Oct_Nov)
axes = fig_var_Oct_Nov.get_axes()
for i,ax in enumerate(axes):
    if i in [0,5,10,15]:
        ax.set_ylim((-0.15 * stds[ordering[i//5]], 1 * stds[ordering[i//5]]))
        ax.set_yticks([x * stds[ordering[i//5]] for x in [-0.1,0,1]])
        ax.set_yticklabels([-0.1,0,1])
    else:
        ax.set_ylim((-0.15 * stds[ordering[i//4]], 0.15 * stds[ordering[i//4]]))
        ax.set_yticks([x * stds[ordering[i//4]] for x in [-0.1, 0, 0.1]])
        ax.set_yticklabels([-0.1, 0, 0.1])
    ax.set_xticks([0,4,8,12,16,20])
    ax.set_xticklabels([0,1,2,3,4,5])
plt.suptitle('Impulse Responses (orthogonalized) for the Oct-Nov Period (Stable)')
fig_var_Oct_Nov.text(0.5, 0.005, 'Time after shock [h]', ha='center', fontsize = 12)
fig_var_Oct_Nov.text(0.005, 0.5, 'Excess activity after shock [in multiples of std. dev.]', 
                     va='center', rotation='vertical', fontsize = 12)
plt.tight_layout(rect = (0.01, 0.01, 1, 0.95))
plt.savefig('S7_IR_Oct_Nov.pdf', dpi = 600, format = 'pdf')
# Impulse response functions for Jan-Mar period
fig_var_Jan_Mar = var_Jan_Mar.irf(5*4).plot(orth=True)
fig_var_Jan_Mar.set_size_inches(10,10)
stds = np.std(Data_fourier_Jan_Mar)
axes = fig_var_Jan_Mar.get_axes()
for i,ax in enumerate(axes):
    if i in [0,5,10,15]:
        ax.set_ylim((-0.15 * stds[ordering[i//5]], 1 * stds[ordering[i//5]]))
        ax.set_yticks([x * stds[ordering[i//5]] for x in [-0.1,0,1]])
        ax.set_yticklabels([-0.1,0,1])
    else:
        ax.set_ylim((-0.15 * stds[ordering[i//4]], 0.15 * stds[ordering[i//4]]))
        ax.set_yticks([x * stds[ordering[i//4]] for x in [-0.1, 0, 0.1]])
        ax.set_yticklabels([-0.1, 0, 0.1])
    ax.set_xticks([0,4,8,12,16,20])
    ax.set_xticklabels([0,1,2,3,4,5])
plt.suptitle('Impulse Responses (orthogonalized) for the Jan-Mar Period (Stable)')
fig_var_Jan_Mar.text(0.5, 0.005, 'Time after shock [h]', ha='center', fontsize = 12)
fig_var_Jan_Mar.text(0.005, 0.5, 'Excess activity after shock [in multiples of std. dev.]', 
                     va='center', rotation='vertical', fontsize = 12)
plt.tight_layout(rect = (0.01, 0.01, 1, 0.95))
plt.savefig('S8_IR_Jan_Mar.pdf', dpi = 600, format = 'pdf')
# Impulse response functions for Mar_Apr period
fig_var_Mar_Apr = var_Mar_Apr.irf(5*4).plot(orth=True)
fig_var_Mar_Apr.set_size_inches(10,10)
stds = np.std(Data_fourier_Mar_Apr)
axes = fig_var_Mar_Apr.get_axes()
for i,ax in enumerate(axes):
    if i in [0,5,10,15]:
        ax.set_ylim((-0.15 * stds[ordering[i//5]], 1 * stds[ordering[i//5]]))
        ax.set_yticks([x * stds[ordering[i//5]] for x in [-0.1,0,1]])
        ax.set_yticklabels([-0.1,0,1])
    else:
        ax.set_ylim((-0.15 * stds[ordering[i//4]], 0.15 * stds[ordering[i//4]]))
        ax.set_yticks([x * stds[ordering[i//4]] for x in [-0.1, 0, 0.1]])
        ax.set_yticklabels([-0.1, 0, 0.1])
    ax.set_xticks([0,4,8,12,16,20])
    ax.set_xticklabels([0,1,2,3,4,5])
plt.suptitle('Impulse Responses (orthogonalized) for the Mar-Apr Period (Pasture)')
fig_var_Mar_Apr.text(0.5, 0.005, 'Time after shock [h]', ha='center', fontsize = 12)
fig_var_Mar_Apr.text(0.005, 0.5, 'Excess activity after shock [in multiples of std. dev.]', 
                     va='center', rotation='vertical', fontsize = 12)
plt.tight_layout(rect = (0.01, 0.01, 1, 0.95))
plt.savefig('S9_IR_Mar_Apr.pdf', dpi = 600, format = 'pdf')

# obtain var filtered data, replace separate animal by separate var residuals
# and accumulate animal by accumulate var residuals
Data_var_Oct_Nov = Data_fourier_Oct_Nov.copy().iloc[var_lags:,:]
Data_var_Oct_Nov[['dog','cow','sheep']] = var_Oct_Nov.resid[['dog','cow','sheep']]
Data_var_Oct_Nov['animal'] = var_Oct_Nov.resid['dog'] - var_Oct_Nov.resid['cow'] + var_Oct_Nov.resid['sheep']
Data_var_Jan_Mar = Data_fourier_Jan_Mar.copy().iloc[var_lags:,:]
Data_var_Jan_Mar[['dog','cow','sheep']] = var_Jan_Mar.resid[['dog','cow','sheep']]
Data_var_Jan_Mar['animal'] = var_Jan_Mar.resid['dog'] - var_Jan_Mar.resid['cow'] + var_Jan_Mar.resid['sheep']
Data_var_Mar_Apr = Data_fourier_Mar_Apr.copy().iloc[var_lags:,:]
Data_var_Mar_Apr[['dog','cow','sheep']] = var_Mar_Apr.resid[['dog','cow','sheep']]
Data_var_Mar_Apr['animal'] = var_Mar_Apr.resid['dog'] - var_Mar_Apr.resid['cow'] + var_Mar_Apr.resid['sheep']
# reset index to zero
Data_var_Oct_Nov = Data_var_Oct_Nov.reset_index(drop = True)
Data_var_Jan_Mar = Data_var_Jan_Mar.reset_index(drop = True)
Data_var_Mar_Apr = Data_var_Mar_Apr.reset_index(drop = True)

# delete unnecessary variables
del(i, ax, axes, stds, periods_data, period, info_i, crit)


# Threshold Analysis #########################################################
##############################################################################

# create theshold model objects for each period
Threshold_Oct_Nov = fcts.threshold_model(Oct_Nov_EQ, Data_var_Oct_Nov, 'PGA', groups)
Threshold_Jan_Mar = fcts.threshold_model(Jan_Mar_EQ, Data_var_Jan_Mar, 'PGA', groups)
Threshold_Mar_Apr = fcts.threshold_model(Mar_Apr_EQ, Data_var_Mar_Apr, 'PGA', groups)
# lookback horizon
horizon = dt.timedelta(hours = 20)

# threshold ranges to consider
th_eq = np.linspace(1, 3, 40)
th_animals = np.linspace(1, 3, 40)
# compute t-statistics for threshold ranges
Threshold_choice_Oct_Nov = Threshold_Oct_Nov.choose_threshold(horizon, th_eq, th_animals)
Threshold_choice_Jan_Mar = Threshold_Jan_Mar.choose_threshold(horizon, th_eq, th_animals)
Threshold_choice_Mar_Apr = Threshold_Mar_Apr.choose_threshold(horizon, th_eq, th_animals)
# plot heatmaps
Threshold_choice_Oct_Nov.plot()
plt.suptitle('Significance Patterns Oct-Nov period')
plt.savefig('S13_TH_Oct_Nov.pdf', dpi = 600, format = 'pdf')
Threshold_choice_Jan_Mar.plot()
plt.suptitle('Significance Patterns Jan-Mar period')
plt.savefig('S14_TH_Jan_Mar.pdf', dpi = 600, format = 'pdf')
Threshold_choice_Mar_Apr.plot()
plt.suptitle('Significance Patterns Mar-Apr period')
plt.savefig('S15_TH_Mar_Apr.pdf', dpi = 600, format = 'pdf')


# threshold values to use for distance - warning time plots
thresholds = (2, 2)
# perform threshold regression for given thresholds
Threshold_select_Oct_Nov = Threshold_Oct_Nov.select_for_threshold(horizon, thresholds)
Threshold_select_Jan_Mar = Threshold_Jan_Mar.select_for_threshold(horizon, thresholds)
Threshold_select_Mar_Apr = Threshold_Mar_Apr.select_for_threshold(horizon, thresholds)
# plot distance against warning time 
Threshold_select_Oct_Nov.plot()
plt.suptitle('Oct-Nov 2016 (Stable)')
plt.savefig('S10_dist_warn_Oct_Nov.pdf', format = 'pdf', dpi = 600)
Threshold_select_Jan_Mar.plot()
plt.suptitle('Jan-Mar 2017 (Stable)')
plt.savefig('S11_dist_warn_Jan_Mar.pdf', format = 'pdf', dpi = 600)
Threshold_select_Mar_Apr.plot()
plt.suptitle('Mar-Apr 2017 (Pasture)')
plt.savefig('S12_dist_warn_Mar_Apr.pdf', format = 'pdf', dpi = 600)

# Figure 5 - Animal Anticipatory Behavior
fig_th_animal = plt.figure(figsize = (6,5))
Data_selected = {'A) Oct-Nov 2016 (Stable)': Threshold_select_Oct_Nov.Data_selected['animal'],
                      'B) Jan-Mar 2017 (Stable)': Threshold_select_Jan_Mar.Data_selected['animal'],
                      'C) Mar-Apr 2017 (Pasture)': Threshold_select_Mar_Apr.Data_selected['animal'],}
regression_results = {'A) Oct-Nov 2016 (Stable)': Threshold_select_Oct_Nov.regression_results['animal'],
                      'B) Jan-Mar 2017 (Stable)': Threshold_select_Jan_Mar.regression_results['animal'],
                      'C) Mar-Apr 2017 (Pasture)': Threshold_select_Mar_Apr.regression_results['animal'],}
for i,key in enumerate(Data_selected):
    plt.subplot(2,2,i+1)
    # obtain x and y values and relevant parameters
    x = Data_selected[key]['distance']
    y = Data_selected[key]['warning_time']
    xlim = [0, np.max(x)*1.05]
    intercept, slope = regression_results[key].params
    cov_params = regression_results[key].normalized_cov_params
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
    
    plt.xlabel('Hypocentral Distance [km]')
    plt.ylabel('Anticipation Time [h]')
    plt.xlim(tuple(xlim))
    plt.ylim((0,20))
    plt.title(key)
l,h = plt.gca().get_legend_handles_labels()
fig_th_animal.legend(l, h, 
                     loc='center', 
                     bbox_to_anchor=(0.5, 0., 0.5, 0.5))
plt.tight_layout()
plt.subplots_adjust()
plt.savefig('5_anticipation_time_animals.pdf', dpi = 600, format = 'pdf')

del(l, h, i, key, j, xj, x, y, xlim, intercept, slope, cov_params, x_resol, 
    x_val, X_val, CI, var_y, Data_selected, regression_results)

