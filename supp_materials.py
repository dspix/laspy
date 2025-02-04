
import glob
import os
from functools import reduce

import matplotlib.pyplot as plt

from scipy import stats
import numpy as np
import pandas as pd

from sweep_landscapes.helper_functions import import_cosmos_daily
from sweep_landscapes.sweep_landscape_functions import sweep_locate, sweep_extract
from sweep_landscapes.dlm_functions import forwardFilteringM, forwardFilteringM2, Model, computeAnormaly
from scipy.stats import t as tdstr
import numpy as np
import scipy.linalg
from scipy.stats import t as tdstr
import itertools
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.optimize import minimize
import random
import math

def dlm(N, anCLM, vid, fs, deltas, prior = None):
    Y = N[1:]-np.nanmean(N) 
    X = np.column_stack((N[:-1]-np.nanmean(N),anCLM.values[:-1])) 

    Y[0] = 0
    X[0:5, :] = 0

    M = Model(Y,X,rseas,deltas)

    if prior is not None:
        M.prior = prior

    FF = forwardFilteringM2(M, fs)

    sm = FF.get('sm')[vid,:] # mean of autocorrelation
    sC = FF.get('sC')[vid,vid,:] # variance of autocorrelation
    snu = FF.get('snu') # degree of freedom

    return sm, sC, snu, FF, M


def calculate_bounds(sm, sC, snu, quantile):

    lower_bounds = np.array(list(map(lambda m, C, nu: m - np.sqrt(C) * tdstr.ppf(quantile, nu), sm, sC, snu)))
    upper_bounds = np.array(list(map(lambda m, C, nu: m + np.sqrt(C) * tdstr.ppf(quantile, nu), sm, sC, snu)))
    
    return lower_bounds, upper_bounds


def generate_synthetic_data(pre_ar1, post_ar1, change_point, n_points, fs):
    #np.random.seed(42)  # For reproducibility

    t = np.arange(n_points)
    freq1 = 1/365.25 #frequency component
    freq2 = 2/365.25 #frequency component
    
    signal = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)
    
    noise = np.random.normal(size=n_points)
    ar1 = np.zeros(n_points)
    ar1[0] = noise[0]
    ar1_coefficient = pre_ar1
    change_timestep = math.floor(n_points * change_point)

    for i in range(1, n_points):
        if i == change_timestep:
            ar1_coefficient = post_ar1

        ar1[i] = ar1_coefficient * ar1[i-1] + noise[i]

    # Generate the regressive series X
    X = np.random.normal(size=n_points)
    beta = 0.7  # Coefficient for the regressive component
    regressive_noise = beta * X

    synthetic_data = signal + ar1 + regressive_noise
    
    sample = range(0, len(synthetic_data), fs)
    synthetic_data = synthetic_data.copy()[sample]
    ar1 = ar1.copy()[sample]
    X = X.copy()[sample]
    signal = signal.copy()[sample]
    regressive_noise = regressive_noise.copy()[sample]

    return synthetic_data, ar1, X, signal, regressive_noise

def calculate_rolling_ar1(series, window_length):
    series = pd.Series(series)
    def calc_ar1(window):
        if window.isnull().any():
            return np.nan
        lag_acf = acf(window, nlags=1)
        return lag_acf[1]
    rolling_ar1 = series.rolling(window = window_length, min_periods = window_length).apply(calc_ar1, raw=False)
    return rolling_ar1


def sample_delta_fit(n_deltas, n_sims, l_delta, u_delta, pre_ar1, post_ar1, change_point, window_length, n_years, fs, warmup, percent_drop, name):

    results = []
    vid = 2

    window_length_fs = math.floor(window_length / fs)
    warmup = math.floor(warmup / fs)
    
    for _ in tqdm(range(n_deltas)):
        delta = random.uniform(l_delta, u_delta)

        for _ in range(n_sims):

            synthetic_data, ar1, X, signal, regressive_noise = generate_synthetic_data(pre_ar1, post_ar1, change_point, n_years * 365, fs)
            
            # Randomly set a percentage of data points to NaN
            n_points = len(synthetic_data)
            n_nan = math.floor(percent_drop * n_points / 100)
            nan_indices = random.sample(range(n_points), n_nan)
            
            synthetic_data_dec = synthetic_data.copy()
            synthetic_data_dec[nan_indices] = np.nan

            win_ar1 = calculate_rolling_ar1(synthetic_data - signal - regressive_noise, window_length_fs)

            deltas = np.ones(4) * delta
            dlm_ar1, sC, snu, FF, M = dlm(synthetic_data_dec, pd.Series(X), vid, fs, deltas)
        
            win_ar1 = win_ar1[warmup:]
            dlm_ar1 = dlm_ar1[warmup:]

            valid_mask = ~np.isnan(win_ar1) & ~np.isnan(dlm_ar1)
            
            valid_observed = win_ar1[valid_mask]
            valid_predicted = dlm_ar1[valid_mask]
            mse = np.mean((valid_observed - valid_predicted) ** 2)
            rmse = np.sqrt(mse)
            r2 = np.round(pearsonr(valid_observed, valid_predicted)[0] ** 2, 3)
            
            y_mean = np.mean(valid_observed)
            ss_res = np.sum((valid_observed - valid_predicted) ** 2)
            ss_tot = np.sum((valid_observed - y_mean) ** 2)
            r2_ss = 1 - (ss_res / ss_tot)

            # Append the results to the lis
            results.append({"delta": delta, 
                            "rmse": rmse, 
                            "r2": r2, 
                            "r2_ss": r2_ss,
                            "ar1_window_length": window_length, 
                            "pre_ar1": pre_ar1,
                            "post_ar1": post_ar1, 
                            "series_length": n_years, 
                            "change_point": change_point,
                            "fs": fs,
                            "pc_drop": percent_drop,
                            "name": name})
    
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def sample_delta_fit2(delta, pre_ar1, post_ar1, change_point, window_length, n_years, fs, warmup, percent_drop):

    results = []
    vid = 2

    window_length_fs = math.floor(window_length / fs)
    warmup = math.floor(warmup / fs)
    
    synthetic_data, ar1, X, signal, regressive_noise = generate_synthetic_data(pre_ar1, post_ar1, change_point, n_years * 365, fs)
    
    # Randomly set a percentage of data points to NaN
    n_points = len(synthetic_data)
    n_nan = math.floor(percent_drop * n_points / 100)
    nan_indices = random.sample(range(n_points), n_nan)
    
    synthetic_data_dec = synthetic_data.copy()
    synthetic_data_dec[nan_indices] = np.nan

    win_ar1 = calculate_rolling_ar1(synthetic_data - signal - regressive_noise, window_length_fs)

    deltas = np.ones(4) * delta
    dlm_ar1, sC, snu, FF, M = dlm(synthetic_data_dec, pd.Series(X), vid, fs, deltas)

    win_ar1 = win_ar1[warmup:]
    dlm_ar1 = dlm_ar1[warmup:]

    valid_mask = ~np.isnan(win_ar1) & ~np.isnan(dlm_ar1)
    
    valid_observed = win_ar1[valid_mask]
    valid_predicted = dlm_ar1[valid_mask]
    mse = np.mean((valid_observed - valid_predicted) ** 2)
    rmse = np.sqrt(mse)
    r2 = np.round(pearsonr(valid_observed, valid_predicted)[0] ** 2, 3)
    
    y_mean = np.mean(valid_observed)
    ss_res = np.sum((valid_observed - valid_predicted) ** 2)
    ss_tot = np.sum((valid_observed - y_mean) ** 2)
    r2_ss = 1 - (ss_res / ss_tot)

    # Append the results to the lis
    results = ({"rmse": rmse, 
                    "r2": r2})
    
    return results



#FINAL SUPP MATT CODE#######

import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Define your variables and parameters
l_delta = 0.9 #0.9
u_delta = 0.999
n_deltas = 1000
n_sims = 1
pre_ar1 = 0.3
post_ar1 = 0.7
change_point = 0.6
n_years = 16
warmup_days = 365*2
pc_drop = [0, 40, 80]
fs_n = [1, 16]
windows = [50, 100, 200]
ar1_list = [0.5, 0.7, 0.9]
rseas = [1, 2]

#pc drop
#window
#ar1_list
#delta

# Initialize results DataFrame
#results = pd.DataFrame()

from scipy.stats import pearsonr

# Function to wrap sample_delta_fit
def run_sample_delta_fit(pc=None, fs=None, window_days=None, ar1=None):
    if ar1 is not None:
        result_df = sample_delta_fit(n_deltas, n_sims, l_delta, u_delta, pre_ar1, ar1, change_point, 200, n_years, fs, warmup_days, 0, "ar1")
    elif window_days is not None:
        result_df = sample_delta_fit(n_deltas, n_sims, l_delta, u_delta, pre_ar1, post_ar1, change_point, window_days, n_years, fs, warmup_days, 0, "window")
    else:
        result_df = sample_delta_fit(n_deltas, n_sims, l_delta, u_delta, pre_ar1, post_ar1, change_point, 200, n_years, fs, warmup_days, pc, "dropout")
    return result_df

# Parallel execution for pc_drop, fs_n
results_pc = Parallel(n_jobs=-1)(delayed(run_sample_delta_fit)(pc=pc, fs=fs) for pc in pc_drop for fs in fs_n)
results = pd.concat(results_pc, axis=0)

# Parallel execution for windows, fs_n
results_windows = Parallel(n_jobs=-1)(delayed(run_sample_delta_fit)(fs=fs, window_days=window_days) for window_days in windows for fs in fs_n)
results = pd.concat([results, pd.concat(results_windows, axis=0)], axis=0)

# Parallel execution for ar1_list, fs_n
results_ar1 = Parallel(n_jobs=-1)(delayed(run_sample_delta_fit)(fs=fs, ar1=ar1) for ar1 in ar1_list for fs in fs_n)
results = pd.concat([results, pd.concat(results_ar1, axis=0)], axis=0)

#results_old = results.copy()

#NEW PLOTS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure results DataFrame has necessary columns
required_columns = ['delta', 'rmse', 'pc_drop', 'ar1_window_length', 'post_ar1', 'name', 'fs']
if not all(col in results.columns for col in required_columns):
    raise ValueError("Missing required columns in the results DataFrame.")

# Create a grid of scatter plots
unique_names = results['name'].unique()
unique_fs = results['fs'].unique()

# Set up grid dimensions
num_rows = len(unique_names)
num_cols = len(unique_fs)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

for i, (name, fs) in enumerate([(n, f) for n in unique_names for f in unique_fs]):
    ax = axes[i]
    
    # Filter data for the specific 'name' and 'fs'
    subset = results[(results['name'] == name) & (results['fs'] == fs)]
    if subset.empty:
        ax.axis('off')  # Turn off axes for empty plots
        continue

    # Identify the parameter that changes
    changing_params = []
    for param in ['pc_drop', 'ar1_window_length', 'post_ar1']:
        if subset[param].nunique() > 1:
            changing_params.append(param)
    
    if len(changing_params) == 0:
        # No parameter varies, skip this plot
        ax.axis('off')
        print(f"Skipping plot for name={name}, fs={fs} as no parameter varies.")
        continue

    if len(changing_params) > 1:
        raise ValueError(f"Expected exactly one parameter to vary for name={name}, fs={fs}, but found: {changing_params}")

    changing_param = changing_params[0]
    subset['label'] = subset[changing_param].astype(str)  # Use the varying parameter as the label

    # Assign colors to labels
    unique_labels = subset['label'].unique()
    palette = ['black', 'red', 'blue']

    # Scatter plot and rolling min
    for j, label in enumerate(unique_labels):
        label_data = subset[subset['label'] == label].sort_values('delta')  # Ensure sorted by delta
        color = palette[j]  # Get the color for this label
        ax.scatter(label_data['delta'], label_data['rmse'], 
                   label=label, color=color, alpha=0.7, s = 0.1)
        
        # Compute and plot smoothed rolling minima
        rolling_minima = pd.Series(label_data['rmse']).rolling(window=10).mean()
        #ax.plot(label_data['delta'].unique(), rolling_minima, color=color, linewidth=1)

    ax.set_title(f'{name}, {fs}', fontsize=10)
    ax.set_xlabel('Delta')
    ax.set_ylabel('RMSE')
    ax.set_ylim(-1, 10)
    ax.legend(loc='upper right', fontsize=8)

# Adjust layout and display
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

for i, (name, fs) in enumerate([(n, f) for n in unique_names for f in unique_fs]):
    ax = axes[i]
    ax.set_ylim(-1, 10)

    # Filter data for the specific 'name' and 'fs'
    subset = results[(results['name'] == name) & (results['fs'] == fs)]
    if subset.empty:
        ax.axis('off')  # Turn off axes for empty plots
        continue

    # Identify the parameter that changes
    changing_params = []
    for param in ['pc_drop', 'ar1_window_length', 'post_ar1']:
        if subset[param].nunique() > 1:
            changing_params.append(param)
    
    if len(changing_params) == 0:
        # No parameter varies, skip this plot
        ax.axis('off')
        print(f"Skipping plot for name={name}, fs={fs} as no parameter varies.")
        continue

    if len(changing_params) > 1:
        raise ValueError(f"Expected exactly one parameter to vary for name={name}, fs={fs}, but found: {changing_params}")

    changing_param = changing_params[0]
    subset['label'] = subset[changing_param].astype(str)  # Use the varying parameter as the label

    # Assign colors to labels
    unique_labels = subset['label'].unique()
    palette = sns.color_palette("hsv", len(unique_labels))
    palette = ['black', 'red', 'blue']

    for j, label in enumerate(unique_labels):
        label_data = subset[subset['label'] == label].sort_values('delta').reset_index(drop=True)  # Ensure sorted and reset index
        color = palette[j]  # Get the color for this label
        ax.scatter(label_data['delta'], label_data['rmse'], 
                label=label, color=color, alpha=0.9, s=0.1)
        
        # Compute and plot smoothed rolling minima
        rolling_minima = label_data['rmse'].rolling(window=10).mean()
        #ax.plot(label_data['delta'], rolling_minima, color=color, linewidth=1)

        # Find the delta corresponding to the minimum rolling mean
        min_index = rolling_minima.idxmin()
        min_delta = label_data.iloc[min_index]['delta'] if not pd.isna(min_index) else None

        if min_delta is not None:
            ax.axvline(x=min_delta, color=color, linestyle='--', linewidth=0.8, alpha=0.8)
            
            # Dynamic Y spreading to avoid label overlap
            ylim = ax.get_ylim()
            spread_factor = 0.05  # Adjust this for more/less spread
            y_position = ylim[0] + (ylim[1] - ylim[0]) * (0.8 - spread_factor * j)

            # Add a left-pointing arrow stopping at x=0.94
            ax.annotate(f'{min_delta:.3f}',
                        xy=(min_delta, y_position),  # Arrow points to vertical line
                        xytext=(0.92, y_position),  # Fixed x-position for text
                        color=color,
                        fontsize=8,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                        horizontalalignment='left',
                        verticalalignment='center')

    ax.set_title(f'{name}, {fs}', fontsize=8)
    ax.set_xlabel('Delta')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper left', fontsize=6)

# Adjust layout and display
plt.tight_layout()
plt.show()


param_titles = {
    'pc_drop': '% decimation',
    'ar1_window_length': 'Window length',
    'post_ar1': 'Post-shift AR1'
}

fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 5), sharex=True, sharey=True)
axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

for i, (name, fs) in enumerate([(n, f) for n in unique_names for f in unique_fs]):
    ax = axes[i]
    ax.set_ylim(-1, 10)
    ax.tick_params(axis='x', labelsize=6) 
    ax.tick_params(axis='y', labelsize=6) 

    # Filter data for the specific 'name' and 'fs'
    subset = results[(results['name'] == name) & (results['fs'] == fs)]
    if subset.empty:
        ax.axis('off')  # Turn off axes for empty plots
        continue

    # Identify the parameter that changes
    changing_params = []
    for param in ['pc_drop', 'ar1_window_length', 'post_ar1']:
        if subset[param].nunique() > 1:
            changing_params.append(param)
    
    if len(changing_params) == 0:
        # No parameter varies, skip this plot
        ax.axis('off')
        continue

    if len(changing_params) > 1:
        raise ValueError(f"Expected exactly one parameter to vary for name={name}, fs={fs}, but found: {changing_params}")

    changing_param = changing_params[0]
    subset['label'] = subset[changing_param].astype(str)  # Use the varying parameter as the label

    # Assign colors to labels
    unique_labels = subset['label'].unique()
    palette = sns.color_palette("hsv", len(unique_labels))
    palette = ['black', 'red', 'blue']

    # Scatter plot and rolling min
    for j, label in enumerate(unique_labels):
        label_data = subset[subset['label'] == label].sort_values('delta').reset_index(drop=True)  # Ensure sorted and reset index
        color = palette[j]  # Get the color for this label
        ax.scatter(label_data['delta'], label_data['rmse'], 
                   label=label, color=color, alpha=0.6, s=0.1)
        
        # Compute and plot smoothed rolling minima
        rolling_minima = label_data['rmse'].rolling(window=10).mean()
        #ax.plot(label_data['delta'], rolling_minima, color=color, linewidth=1)

        # Find the delta corresponding to the minimum rolling mean
        min_index = rolling_minima.idxmin()
        min_delta = label_data.iloc[min_index]['delta'] if not pd.isna(min_index) else None

        if min_delta is not None:
            ax.axvline(x=min_delta, color=color, linestyle='--', linewidth=0.8, alpha=0.8)
            
            # Dynamic Y spreading to avoid label overlap
            ylim = ax.get_ylim()
            spread_factor = 0.1  # Adjust this for more/less spread
            y_position = ylim[0] + (ylim[1] - ylim[0]) * (0.4 - spread_factor * j)

            # Add a left-pointing arrow stopping at x=0.94
            ax.annotate(f'{min_delta:.3f}',
                        xy=(min_delta, y_position),  # Arrow points to vertical line
                        xytext=(0.92, y_position),  # Fixed x-position for text
                        color=color,
                        fontsize=4,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                        horizontalalignment='left',
                        verticalalignment='center',
                        bbox=dict(facecolor="white", alpha=0.5, edgecolor='none'))

    # Set legend title instead of panel title
    legend_title = param_titles.get(changing_param, changing_param) 
    legend_title = f'Sample int: {fs}\n{legend_title}:'
    ax.legend(title=legend_title, loc='upper left', fontsize=4, title_fontsize=5)

    ax.set_xlabel("Delta", fontsize = 6)
    ax.set_ylabel("RMSE", fontsize = 6)

    # Suppress X-axis labels for top rows
    if i // num_cols < num_rows - 1:  # Rows other than the last
        ax.set_xlabel("")
    
    # Suppress Y-axis labels for the second column
    if i % num_cols == 1:  # Second column
        ax.set_ylabel("")


os.chdir("C:/Users/Will.Rust/OneDrive - Cranfield University/postdoc/Environment/Projects/RESTRECO/sweep_paper")

path_name = f"data_out/figs/SM_sensitivity_RMSE.png"
plt.savefig(path_name, dpi=300, bbox_inches='tight', pad_inches=0)



fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 5), sharex=True, sharey=True)
axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

for i, (name, fs) in enumerate([(n, f) for n in unique_names for f in unique_fs]):
    ax = axes[i]
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelsize=6) 
    ax.tick_params(axis='y', labelsize=6) 

    # Filter data for the specific 'name' and 'fs'
    subset = results[(results['name'] == name) & (results['fs'] == fs)]
    if subset.empty:
        ax.axis('off')  # Turn off axes for empty plots
        continue

    # Identify the parameter that changes
    changing_params = []
    for param in ['pc_drop', 'ar1_window_length', 'post_ar1']:
        if subset[param].nunique() > 1:
            changing_params.append(param)
    
    if len(changing_params) == 0:
        # No parameter varies, skip this plot
        ax.axis('off')
        continue

    if len(changing_params) > 1:
        raise ValueError(f"Expected exactly one parameter to vary for name={name}, fs={fs}, but found: {changing_params}")

    changing_param = changing_params[0]
    subset['label'] = subset[changing_param].astype(str)  # Use the varying parameter as the label

    # Assign colors to labels
    unique_labels = subset['label'].unique()
    palette = sns.color_palette("hsv", len(unique_labels))
    palette = ['black', 'red', 'blue']

    # Scatter plot and rolling min
    for j, label in enumerate(unique_labels):
        label_data = subset[subset['label'] == label].sort_values('delta').reset_index(drop=True)  # Ensure sorted and reset index
        color = palette[j]  # Get the color for this label
        ax.scatter(label_data['delta'], label_data['r2'], 
                   label=label, color=color, alpha=0.6, s=0.1)
        
        # Compute and plot smoothed rolling minima
        rolling_minima = label_data['r2'].rolling(window=10).mean()
        #ax.plot(label_data['delta'], rolling_minima, color=color, linewidth=1)

        # Find the delta corresponding to the minimum rolling mean
        min_index = rolling_minima.idxmax()
        min_delta = label_data.iloc[min_index]['delta'] if not pd.isna(min_index) else None

        if min_delta is not None:
            ax.axvline(x=min_delta, color=color, linestyle='--', linewidth=0.8, alpha=0.8)
            
            # Dynamic Y spreading to avoid label overlap
            ylim = ax.get_ylim()
            spread_factor = 0.1  # Adjust this for more/less spread
            y_position = ylim[0] + (ylim[1] - ylim[0]) * (0.4 - spread_factor * j)

            # Add a left-pointing arrow stopping at x=0.94
            ax.annotate(f'{min_delta:.3f}',
                        xy=(min_delta, y_position),  # Arrow points to vertical line
                        xytext=(0.92, y_position),  # Fixed x-position for text
                        color=color,
                        fontsize=4,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                        horizontalalignment='left',
                        verticalalignment='center',
                        bbox=dict(facecolor="white", alpha=1, edgecolor='none'))

    # Set legend title instead of panel title
    legend_title = param_titles.get(changing_param, changing_param) 
    legend_title = f'Sample int: {fs}\n{legend_title}:'
    ax.legend(title=legend_title, loc='upper left', fontsize=4, title_fontsize=5)

    ax.set_xlabel("Delta", fontsize = 6)
    ax.set_ylabel("R2", fontsize = 6)

    # Suppress X-axis labels for top rows
    if i // num_cols < num_rows - 1:  # Rows other than the last
        ax.set_xlabel("")
    
    # Suppress Y-axis labels for the second column
    if i % num_cols == 1:  # Second column
        ax.set_ylabel("")


os.chdir("C:/Users/Will.Rust/OneDrive - Cranfield University/postdoc/Environment/Projects/RESTRECO/sweep_paper")

path_name = f"data_out/figs/SM_sensitivity_R2.png"
plt.savefig(path_name, dpi=300, bbox_inches='tight', pad_inches=0)















