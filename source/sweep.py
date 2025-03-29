"""
GNU GENERAL PUBLIC LICENSE
Copyright (C) 2025 Will D. Rust, Marko Stojanovic, Daniel M. Simms

https://github.com/dspix/laspy/blob/main/LICENSE
"""

import numpy as np
import pandas as pd
import ssqueezepy as ssq
from scipy import stats
from ssqueezepy.experimental import scale_to_freq
from joblib import Parallel, delayed
import random
import math
from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import gaussian_kde
import os
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import ast
import imageio

def calculate_landscape(state_estimates, state_variances, state_1_ind, state_2_ind, warmup):
    """Calculates a potential energy surface (stability landscape) from 
    two selected state variables.

    Args:
        state_estimates (np.ndarray): State estimate matrix of shape
            (n_states, n_timesteps).
        state_variances (np.ndarray): State variance tensor of shape
            (n_states, n_states, n_timesteps).
        state_1_ind (int): Index for the first state variable.
        state_2_ind (int): Index for the second state variable.
        warmup (int): Number of timesteps to exclude from the beginning.

    Returns:
        potential (np.ndarray): 2D potential energy surface.
        X (np.ndarray): Meshgrid X-coordinates.
        Y (np.ndarray): Meshgrid Y-coordinates.
        pdf (np.ndarray): Probability density function over the state space.
        local_mean (np.ndarray): Time series of the first state variable.
        ar1_resilience (np.ndarray): Time series of the second state variable.
    """
    local_mean = state_estimates[state_1_ind, :][warmup:]
    ar1_resilience = state_estimates[state_2_ind, :][warmup:]

    local_mean_variance = state_variances[state_1_ind, state_1_ind, :][warmup:]
    ar1_resilience_variance = state_variances[state_2_ind, state_2_ind, :][warmup:]

    state_estimates_2d = np.vstack((local_mean, ar1_resilience))

    # Inverse of standard deviation as weights
    weights = 1 / np.sqrt(local_mean_variance * ar1_resilience_variance)
    valid_indices = np.isfinite(local_mean) & np.isfinite(ar1_resilience) & np.isfinite(weights)
    filtered_state_estimates_2d = state_estimates_2d[:, valid_indices]
    filtered_weights = weights[valid_indices]

    kde = gaussian_kde(filtered_state_estimates_2d, weights=filtered_weights)
    xmin, ymin = filtered_state_estimates_2d.min(axis=1)
    xmax, ymax = filtered_state_estimates_2d.max(axis=1)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pdf = np.reshape(kde(positions).T, X.shape)

    # Potential function: V(x) = -ln(P(x))
    potential = -np.log(pdf + 1e-10)  # Add a small value to avoid log(0)

    return potential, X, Y, pdf, local_mean, ar1_resilience

def synth_max_wp(win_mean, win_range, wavelet, fs, n, l_period, u_period):
    """
    Calculate maximum wavelet power values from synthetic data.

    Args:
        win_mean (np.ndarray): Array of window mean values.
        win_range (np.ndarray): Array of window range values.
        wavelet (str): Name of the wavelet used.
        fs (float): Sampling frequency.
        n (int): Number of synthetic trials.
        l_period (float): Lower bound of target period.
        u_period (float): Upper bound of target period.

    Returns:
        np.ndarray: Sorted array of max wavelet powers from synthetic trials.
    """
    def process_sample(sample_ind, max_space, wavelet, fs):
        winrange_wps = np.zeros(max_space)

        for j in range(max_space):
            synthetic_series = np.random.normal(size = math.ceil(u_period))

            n_nan = math.floor(0 * len(synthetic_series))
            nan_indices = random.sample(range(len(synthetic_series)), n_nan)
            synthetic_series[nan_indices] = 0
            wx_i, scales, *_ = ssq.cwt(synthetic_series, wavelet, fs=fs)
            freqs = scale_to_freq(scales, wavelet, len(synthetic_series), fs=fs)
            periods = 1 / freqs

            wx_slice = wx_i[:, sample_ind]

            p_ind = np.where((periods <= u_period) & (periods >= l_period))[0]
            wx_p = wx_slice[p_ind]
            winrange_wps[j] = np.nanmean(np.abs(wx_p))

        return np.nanmax(winrange_wps)

    sample_ind = int(u_period / 2)
    max_space = len(win_mean) * len(win_range)

    results = Parallel(n_jobs=-1)(
        delayed(process_sample)(
            sample_ind,
            max_space,
            wavelet,
            fs
        ) for i in tqdm(range(n)))

    wx_sorted = np.sort(results)

    return wx_sorted

def sweep_locate(
        x,
        win_mean,
        win_range,
        wavelet,
        fs,
        l_period,
        u_period,
        min_thresh,
        sig_wp,
        cost,
        buff
    ):
    """Locate significant wavelet power regions within 
    unfiltered data over specified center and range values.

    Args:
        x (np.ndarray): unfiltered time series signal.
        win_mean (np.ndarray): Array of window center values.
        win_range (np.ndarray): Array of window half-widths.
        wavelet (str): Name of wavelet.
        fs (float): Sampling frequency.
        l_period (float): Lower period threshold.
        u_period (float): Upper period threshold.
        min_thresh (float): Minimum signal value for masking.
        sig_wp (float): Power threshold for significance.
        cost (float): Weighting cost factor.
        buff (float): Buffer added to extracted region.

    Returns: 
        dict: Dictionary containing:
        - signal_index (dict): Indices of best-fitting window centers and ranges.
        - signal_location (dict): Locations (min, max, center) in signal space.
        - best_power (np.ndarray): Maximum power for each timestep.
        - sig_vector (np.ndarray): Boolean array indicating significant times.
        - av_power_mat (np.ndarray): Mean power across all windows.
        - power_mat (np.ndarray): Raw power matrix (mean, range, time).
        - center_mat (np.ndarray): Center axis power matrix.
        - range_mat (np.ndarray): Range axis power matrix.
        - center_mat_w (np.ndarray): Weighted center matrix.
        - range_mat_w (np.ndarray): Weighted range matrix.
    """
    x_trim = x.copy()
    x_trim[x_trim < min_thresh] = np.nan

    power_mat = np.full((len(win_mean), len(win_range), len(x)), np.nan)

    def process_combination(i, j):
        """
        Computes the mean wavelet power over a window defined by the i-th mean
        and j-th range combination in the sweep grid. Function inside parralelisation.

        Args:
            i (int): Index of the current mean in the window mean array.
            j (int): Index of the current range in the window range array.

        Returns:
            tuple or None: (i, j, power_mat_comb) if computation is successful,
            otherwise None.
            - i (int): Index of the tested mean value.
            - j (int): Index of the tested range value.
            - power_mat_comb (np.ndarray): 1D array of mean wavelet power over time.
        """
        mean_i = win_mean[i]
        range_j = win_range[j]
        
        min_i = mean_i - range_j
        max_i = mean_i + range_j
        
        if max_i > np.nanmax(x_trim) or min_i < np.nanmin(x_trim):
            return None

        x_subset = x_trim.copy()
        
        x_subset[np.where(x_subset > max_i)] = np.nan
        x_subset[np.where(x_subset < min_i)] = np.nan
            
        if np.all(np.isnan(x_subset)):
            return None
        
        actual_mean = np.nanmean(x_subset)
        
        na_ind = np.where(np.isnan(x_subset))[0]

        if (len(x_subset) - len(na_ind)) / len(x_subset) < 0.05:
            return None

        x_subset[na_ind] = actual_mean
        std_x = np.std(x_subset)

        if np.std(x_subset) == 0 or np.isnan(std_x) or len(set(x_subset)) < 2:
            return None

        x_subset = stats.zscore(x_subset)

        x_smooth = lowess(x_subset, np.arange(len(x_subset)), frac=0.75)
        x_subset = x_subset.copy() - x_smooth[:, 1]

        wx, scales, *_ = ssq.cwt(x_subset, wavelet, fs=fs)
        freqs = scale_to_freq(scales, wavelet, len(x_subset), fs=fs)
        periods = 1/freqs

        wx_rec = wx.copy()
        mask = (periods <= u_period) & (periods >= l_period)
        wx_rec[~mask, :] = np.nan
        
        power_mat_comb = np.nanmean(np.abs(wx_rec), axis=0)
        
        return i, j, power_mat_comb

    results = Parallel(n_jobs=-1)(
        delayed(process_combination)(i, j) for i in range(power_mat.shape[0]) for j in range(power_mat.shape[1])
        )

    for res in results:
        if res is not None:
            i, j, power_mat_comb = res
            power_mat[i, j, :] = power_mat_comb

    #find average power mat
    av_power_mat = np.nanmean(power_mat, axis = 2)
    slice_min = np.nanmin(av_power_mat)
    slice_max = np.nanmax(av_power_mat)
    normalized_matrix = (av_power_mat - slice_min) / (slice_max - slice_min)
    adjusted_slice = normalized_matrix * (1 - cost) + cost

    #extract annual signal
    center_mat = np.nanmax(power_mat, axis = 1)
    range_mat = np.nanmax(power_mat, axis = 0)

    best_center_ind = np.nanargmax(center_mat, axis = 0)
    best_range_ind = np.nanargmax(range_mat, axis = 0)

    weighted_power_mat = power_mat.copy()


    for t in range(power_mat.shape[2]):
        weighted_power_mat[:, :, t] *= adjusted_slice

    #extract annual signal
    center_mat_w = np.nanmax(weighted_power_mat, axis = 1)
    range_mat_w = np.nanmax(weighted_power_mat, axis = 0)

    best_center_ind = np.nanargmax(center_mat_w, axis = 0)
    best_range_ind = np.nanargmax(range_mat_w, axis = 0)

    loc_mean = win_mean[best_center_ind] 
    loc_min = loc_mean - win_range[best_range_ind] - buff
    loc_max = loc_mean + win_range[best_range_ind] + buff

    best_power = center_mat[best_center_ind, np.arange(center_mat.shape[1])]

    #define signal significance
    sig_vector = best_power >= sig_wp
    
    signal_index = {
        'ind_center': best_center_ind,
        'ind_range': best_range_ind,
    }
    
    signal_location = {
        'loc_center': loc_mean,
        'loc_min': loc_min,
        'loc_max': loc_max
    }

    return {
        'signal_index': signal_index,
        'signal_location': signal_location,
        'best_power': best_power,
        'sig_vector': sig_vector,
        'av_power_mat': av_power_mat,
        'power_mat': power_mat,
        'center_mat': center_mat,
        'range_mat': range_mat,
        'center_mat_w': center_mat_w,
        'range_mat_w': range_mat_w 
    }

def sweep_infil(
    signal_index, 
    signal_location, 
    best_power, 
    sig_vector, 
    av_power_mat, 
    win_mean, 
    win_range, 
    buff
    ):

    """Fill gaps in detected signals for plotting, also returns significant 
    indicies and values for plotting.

    Args:
        signal_index (dict): Dictionary of best window indices.
        signal_location (dict): Dictionary of spatial signal boundaries.
        best_power (np.ndarray): Power values of best signals.
        sig_vector (np.ndarray): Boolean array indicating significant points.
        av_power_mat (np.ndarray): Averaged power matrix.
        win_mean (np.ndarray): Window center values.
        win_range (np.ndarray): Window half-widths.
        buff (float): Buffer added to extracted region.

    Returns:
        dict: Dictionary containing:
        - signal_index_infil (dict): Interpolated signal index values.
        - signal_index_sigonly (dict): Significant-only signal index values.
        - signal_location_infil (dict): Interpolated spatial signal bounds.
        - signal_location_sigonly (dict): Significant-only spatial bounds.
    """

    best_center_ind = signal_index['ind_center']
    best_range_ind = signal_index['ind_range']
    loc_mean = signal_location['loc_center']
    loc_min = signal_location['loc_min']
    loc_max = signal_location['loc_max']

    best_power_sig = best_power.copy()
    best_power_sig[~sig_vector] = np.nan

    best_center_ind_sig = best_center_ind.copy()
    best_center_ind_sig = best_center_ind_sig.astype('float')
    best_center_ind_sig[~sig_vector] = np.nan

    best_range_ind_sig = best_range_ind.copy()
    best_range_ind_sig = best_range_ind_sig.astype('float')
    best_range_ind_sig[~sig_vector] = np.nan

    loc_mean_sig = loc_mean.copy()
    loc_min_sig = loc_min.copy()
    loc_max_sig = loc_max.copy()

    loc_mean_sig[~sig_vector] = np.nan
    loc_min_sig[~sig_vector] = np.nan
    loc_max_sig[~sig_vector] = np.nan

    loc_mean_if = pd.Series(loc_mean_sig).ffill()
    loc_min_if = pd.Series(loc_min_sig).ffill()
    loc_max_if = pd.Series(loc_max_sig).ffill()

    av_center_mat = np.nanmax(av_power_mat, axis = 1)
    av_range_mat = np.nanmax(av_power_mat, axis = 0)

    av_center_ind = np.nanargmax(av_center_mat)
    av_range_ind = np.nanargmax(av_range_mat)

    loc_mean_if[np.isnan(loc_mean_if)] = win_mean[av_center_ind] 
    loc_min_if[np.isnan(loc_min_if)] = (loc_mean_if[np.isnan(loc_min_if)]
                                        - win_range[av_range_ind] - buff)
    loc_max_if[np.isnan(loc_max_if)] = (loc_mean_if[np.isnan(loc_max_if)]
                                        + win_range[av_range_ind] + buff)

    best_center_ind_if = best_center_ind.copy() 
    best_center_ind_if = best_center_ind_if.astype('float')
    best_center_ind_if[~sig_vector] = np.nan
    best_center_ind_if = pd.Series(best_center_ind_if).ffill()
    best_center_ind_if[np.isnan(best_center_ind_if)] = av_center_ind

    best_range_ind_if = best_range_ind.copy() 
    best_range_ind_if = best_range_ind_if.astype('float')
    best_range_ind_if[~sig_vector] = np.nan
    best_range_ind_if = pd.Series(best_range_ind_if).ffill()
    best_range_ind_if[np.isnan(best_range_ind_if)] = av_range_ind

    signal_index_infil = {
        'ind_center': best_center_ind_if,
        'ind_range': best_range_ind_if
    }
    
    signal_location_infil = {
        'loc_center': loc_mean_if,
        'loc_min': loc_min_if,
        'loc_max': loc_max_if
    }

    signal_index_sigonly = {
        'ind_center': best_center_ind_sig,
        'ind_range': best_range_ind_sig
    }

    signal_location_sigonly = {
        'loc_center': loc_mean_sig,
        'loc_min': loc_min_sig,
        'loc_max': loc_max_sig
    }

    return {
        'signal_index_infil': signal_index_infil,
        'signal_index_sigonly': signal_index_sigonly,
        'signal_location_infil': signal_location_infil,
        'signal_location_sigonly': signal_location_sigonly
        }

def sweep_extract(x, signal_location_infil, wavelet, fs, l_period, exp):
    """ Extract normalized and smoothed signal from defined sweep region.

    Args:
        x (np.ndarray): Time series input.
        signal_location_infil (dict): Dictionary with signal bounds.
        wavelet (str): Wavelet name.
        fs (float): Sampling frequency.
        l_period (float): Lower cutoff period.
        exp (float): Exponent for frequency tapering.

    Returns:
        x_sweep (np.ndarray): Final reconstructed and normalized signal.
        wx_og (np.ndarray): Original wavelet coefficients.
        wx_filt (np.ndarray): Frequency-filtered wavelet coefficients.
        factor_mask (np.ndarray): Frequency-based filtering mask.
    """

    loc_mean_if = signal_location_infil['loc_center']
    loc_max_if = signal_location_infil['loc_max']
    loc_min_if = signal_location_infil['loc_min']
    loc_env_if = loc_max_if - loc_min_if

    x_subset = x.copy()
    na_ind = (x_subset > loc_max_if) | (x_subset < loc_min_if)
    x_subset[na_ind] = np.nan

    x_subset = pd.Series(x_subset).interpolate(method='linear', limit_direction='both').values
    x_subset[np.where(np.isnan(x_subset))] = np.mean(x_subset[~np.isnan(x_subset)])

    x_subset_mean = np.nanmean(x_subset)

    x_subset -= loc_mean_if
    x_subset /= loc_env_if

    X = stats.zscore(x_subset)
    wx_og, scales, *_ = ssq.cwt(X, wavelet, fs=fs)
    freqs = scale_to_freq(scales, wavelet, len(x_subset), fs=fs)
    periods = 1/freqs

    lower_ind = np.argmin(np.abs(periods - l_period))
    factor_mask = np.ones((wx_og.shape[0], wx_og.shape[1]), dtype='float')

    def normalize_vector(v):
        min_v = np.min(v)
        max_v = np.max(v)
        if max_v == min_v:
            return np.zeros_like(v)
        return (v - min_v) / (max_v - min_v)

    for j in range(factor_mask.shape[1]):
        indices = np.arange(0, lower_ind)
        factor_mask[indices, j] = normalize_vector(indices) ** exp

    wx_filt = wx_og * factor_mask
    x_sweep = ssq.icwt(wx_filt, wavelet, scales)

    x_sweep = stats.zscore(x_sweep)
    x_sweep *= np.std(x_subset)

    x_sweep *= loc_env_if 
    x_sweep += loc_mean_if

    corr_factor = np.mean(loc_mean_if) - ((x_subset_mean + np.mean(loc_mean_if)) / 2)

    x_sweep -= corr_factor

    x_sweep[na_ind] = np.nan

    return x_sweep, wx_og, wx_filt, factor_mask

def plot_landscape(FF_sweep, warmup, dates, plot_params, output_path=None):
    """
    Generate a 3D landscape plot showing the system trajectory over a
    potential energy surface.

    Args:
        FF_sweep (dict): Dictionary with keys 'sm' and 'sC' containing state
            means and covariances.
        warmup (int): Number of timesteps to exclude from the beginning.
        dates (pd.Series): Timestamps corresponding to the state estimates.
        plot_params (list): Plotting parameters, site_name, label_offset_csv, [x1,y1][x2,y2] minima locations.
        file_name (str): Output filename for the saved image.

    Returns:
        None. Saves a plotly figure as a png file.
    """

    camera_distance = 1.75
    rotation_degree = 145
    rotation_rad = np.radians(rotation_degree)

    # Camera position (rotating around Z-axis)
    eye_x = np.sin(rotation_rad) * camera_distance  # Rotates around Z-axis
    eye_y = np.cos(rotation_rad) * camera_distance
    eye_z = 1.28  # Keeping z fixed

    # Light position (rotating around Z-axis)
    n = 5

    potential, X, Y, pdf, local_mean, ar1_resilience = calculate_landscape(
        FF_sweep['sm'], 
        FF_sweep['sC'], 
        0, 
        2, 
        warmup
        )
    
    dates_clip = dates[warmup: ]

    potential = np.sqrt(potential - np.nanmin(potential) + 1) - 1
    potential = gaussian_filter(potential, sigma=3)

    # Subsample every nth timestep
    subsampled_local_mean = local_mean[::n]
    subsampled_ar1_resilience = ar1_resilience[::n]
    dates_clip = dates_clip[::n]

    def moving_average(data, window_size):
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd number to ensure central averaging.")

        half_window = window_size // 2
        smoothed = np.array(data, dtype=float)  # Ensure floating point calculations
        result = np.copy(smoothed)

        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            result[i] = np.mean(smoothed[start:end])

        return result

    subsampled_local_mean = moving_average(
        subsampled_local_mean,
        window_size=31
        )
    subsampled_ar1_resilience = moving_average(
        subsampled_ar1_resilience,
        window_size=31
        )

    # Create a meshgrid for X, Y, and flatten it for interpolation
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = potential.ravel()

    z_offset = 0.2

    # Prepare static surface with proper shading, lighting, and aspect ratio
    attr_surface = go.Surface(
        z=potential,
        x=X,
        y=Y,
        surfacecolor=potential,
        colorscale="Spectral",
        cmin=0.5,
        cmax=3.8,
        opacity=1,
        lighting=dict(
            ambient=0.4,    # surface
            diffuse=0.6,    # surface
            specular=0.3,   # surface
            roughness=0.5,  # surface
            fresnel=0.1,    # surface
        ),
        lightposition=dict(
            x=3,
            y=3,
            z=1,
        ),
        contours=dict(
            z=dict(
                show=True,
                start=0,
                end=4,
                size=0.2,
                color="black",
                width=2,
            )
        ),
        showscale=False,  # No color bar
        name="Potential Energy Surface",
    )


    # Define fixed axis ranges with margins
    pad = 0.05
    x_range_full = [X.min(), X.max()]
    y_range_full = [Y.min(), Y.max()]
    z_range_full = [potential.min(), potential.max() + z_offset]

    x_margin = pad * (x_range_full[1] - x_range_full[0])
    y_margin = pad * (y_range_full[1] - y_range_full[0])
    z_margin = pad * (z_range_full[1] - z_range_full[0])

    x_range_fixed = [x_range_full[0] - x_margin, x_range_full[1] + x_margin]
    y_range_fixed = [y_range_full[0] - y_margin, y_range_full[1] + y_margin]
    z_range_fixed = [z_range_full[0] - z_margin, z_range_full[1] + z_margin]

    time_traj = {"x": [], "y": [], "z": []}

    for j in range(len(subsampled_local_mean)):
        x, y = subsampled_local_mean[j], subsampled_ar1_resilience[j]
        z = griddata((x_flat, y_flat), z_flat, (x, y), method="linear")
        
        if z is None:
            z = potential.min()

        time_traj["x"].append(x)
        time_traj["y"].append(y)
        time_traj["z"].append(z)  


    path_trace = go.Scatter3d(
        x=time_traj["x"],  
        y=time_traj["y"],
        z=[z + 0.2 for z in time_traj["z"]],
        mode="lines",
        line=dict(color="red", width=7), 
        name="Smoothed trajectory",
    )

    start_x = time_traj["x"][0]
    start_y = time_traj["y"][0]
    start_z = time_traj["z"][0] + 0.2


    start_marker = go.Scatter3d(
        x=[start_x],
        y=[start_y],
        z=[start_z],
        mode="markers",
        marker=dict(
            symbol="square",  
            size=6,  
            color="red",  
            opacity=1
        ),
        name="Start Point"
    )

    frame_fig = go.Figure(data=[attr_surface, path_trace, start_marker])

   
    frame_fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="NDVI Anomaly", 
                range=x_range_fixed,
                tickvals = np.round(np.arange(-1, 1, 0.05), 2),
                ticktext = np.round(np.arange(-1, 1, 0.05), 2),
                showspikes=False,
                showbackground=False,  
                zeroline=True,
                zerolinecolor="black",
                showline=True,
                linecolor="black",
                linewidth=2,
                tickcolor="black",
                tickwidth=2,
                gridcolor="lightgray"

            ),
            yaxis=dict(
                title="System speed (-AC1)", 
                tickmode = "array",
                tickvals = np.round(np.arange(-1, 1.1, 0.1), 2),
                ticktext = -np.round(np.arange(-1, 1.1, 0.1), 2),
                range=y_range_fixed,
                showspikes=False,
                showbackground=False,
                zeroline=True,
                zerolinecolor="black",
                showline=True,
                linecolor="black",
                linewidth=2,
                tickcolor="black",
                tickwidth=2,
                gridcolor="lightgray"

            ),
            zaxis=dict(
                title="Potential", 
                range=z_range_fixed,
                showspikes=False,
                showbackground=False,
                zeroline=True,
                zerolinecolor="black",
                showline=True,
                linecolor="black",
                linewidth=2,
                tickcolor="black",
                tickwidth=2,
                gridcolor="lightgray"

            ),
            aspectmode="manual", 
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                eye=dict(x=-eye_x, y=eye_y, z=eye_z),  
                up=dict(x=1, y=0, z=1),  
                center=dict(x=0, y=0, z=0),  
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0), 
        paper_bgcolor="white",  
        plot_bgcolor="white", 
    )

    if isinstance(plot_params.iloc[2], str) and plot_params.iloc[2].strip():  
            time_traj_points = np.column_stack(
                (time_traj["x"], time_traj["y"], time_traj["z"])
                )
            coord_string = plot_params.iloc[2]  
            minima_points = np.array(
                [ast.literal_eval(f"[{x}]") for x in coord_string.strip("][").split("][")]
                )
            grid_coords = np.column_stack((X.ravel(), Y.ravel()))
            grid_distances = cdist(minima_points, grid_coords)
            closest_grid_indices = np.argmin(grid_distances, axis=1)
            closest_grid_z = potential.ravel()[closest_grid_indices]
            matched_minima = np.column_stack((minima_points, closest_grid_z))
            traj_points_xy = time_traj_points[:, :2] 
            distances = cdist(minima_points, traj_points_xy)
            closest_traj_indices = np.argmin(distances, axis=1)
            closest_traj_x = np.array(time_traj["x"])[closest_traj_indices]
            closest_traj_y = np.array(time_traj["y"])[closest_traj_indices]
            closest_traj_z = np.array(time_traj["z"])[closest_traj_indices]
            closest_dates = [dates_clip.iloc[j] for j in closest_traj_indices]

            results = [
                {
                    "minima": {"x": minima[0], "y": minima[1], "z": minima[2]},
                    "closest_traj": {
                        "x": closest_traj_x[i], 
                        "y": closest_traj_y[i],
                        "z": closest_traj_z[i]},
                    "date": closest_dates[i].strftime('%d-%m-%Y'),
                }
                for i, minima in enumerate(matched_minima)
            ]


            z_max_offset = potential.max() + 0.1  

            dogleg_traces = []
            annotations = []

            offset_list = list(map(float, plot_params.iloc[1].split(",")))

            for j, result in enumerate(results):
                # Minima coordinates
                minima_x = result["closest_traj"]["x"]
                minima_y = result["closest_traj"]["y"]
                minima_z = result["closest_traj"]["z"] + 0.2

                # Vertical line endpoint
                vert_x = minima_x
                vert_y = minima_y
                vert_z = z_max_offset

                # Horizontal line endpoint
                horiz_y = vert_y
                horiz_z = vert_z

                # Vertical line trace
                vertical_trace = go.Scatter3d(
                    x=[minima_x, vert_x],
                    y=[minima_y, vert_y],
                    z=[minima_z, vert_z],
                    mode="lines",
                    line=dict(color="black", width=3),
                    showlegend=False,
                    name=f"Vertical Line {j + 1}",
                )
                dogleg_traces.append(vertical_trace)

                # Horizontal line trace
                horizontal_trace = go.Scatter3d(
                    x=[vert_x, vert_x + offset_list[j]],
                    y=[vert_y, horiz_y],
                    z=[vert_z, horiz_z],
                    mode="lines",
                    line=dict(color="black", width=3),
                    showlegend=False,
                    name=f"Horizontal Line {j + 1}",
                )
                dogleg_traces.append(horizontal_trace)

                if offset_list[j] > 0:
                    xanchor_j = "left"
                else:
                    xanchor_j = "right"

                annotations.append(
                    dict(
                        x=vert_x + offset_list[j],
                        y=horiz_y,
                        z=horiz_z,
                        text=result["date"],
                        showarrow=False,  
                        font=dict(size=17, color="black", family="Arial"),  
                        xanchor=xanchor_j,  
                        yanchor="middle",
                        bgcolor="white",  
                        opacity=0.8, 
                    )
                )

            for trace in dogleg_traces:
                frame_fig.add_trace(trace)

            frame_fig.update_layout(
                scene=dict(
                    annotations=annotations,  
                ),
                margin=dict(l=0, r=0, t=0, b=0), 
            )

    frame_fig.update_layout(showlegend=False)
    frame_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)  
    )

    if output_path:
        frame_fig.write_image(output_path, engine="kaleido", width=1080/1.5, height=960/1.5)
    else:
        return frame_fig

    

def plot_landscape_gif(FF_sweep, warmup, dates, plot_params):
    """Generate a GIF showing system evolution on a 3D potential surface.

    Args:
        FF_sweep (dict): Dictionary with keys 'sm' and 'sC' containing state
            means and covariances.
        warmup (int): Number of timesteps to exclude from the beginning.
        dates (pd.Series): Timestamps corresponding to the state estimates.
        plot_params (list): Plotting parameters including animation labels.

    Returns:
        None. Saves a GIF animation to file.
    """

    camera_distance = 1.75
    rotation_degree = 145
    rotation_rad = np.radians(rotation_degree)

    eye_x = np.sin(rotation_rad) * camera_distance  
    eye_y = np.cos(rotation_rad) * camera_distance
    eye_z = 1.28 
    
    n = 5

    potential, X, Y, pdf, local_mean, ar1_resilience = calculate_landscape(
        FF_sweep['sm'], 
        FF_sweep['sC'], 
        0, 
        2, 
        warmup
        )

    potential = np.sqrt(potential - np.nanmin(potential) + 1) - 1
    potential = gaussian_filter(potential, sigma=3)
    dates_clip = dates[warmup: ]

    subsampled_local_mean = local_mean[::n]
    subsampled_ar1_resilience = ar1_resilience[::n]
    dates_clip = dates_clip[::n]
    dates_clip.reset_index(drop=True, inplace=True)

    def moving_average(data, window_size):
        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be an odd number to ensure central averaging."
                )

        half_window = window_size // 2
        smoothed = np.array(data, dtype=float)  
        result = np.copy(smoothed)

        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            result[i] = np.mean(smoothed[start:end])

        return result

    subsampled_local_mean = moving_average(
        subsampled_local_mean,
        window_size=31
        )
    subsampled_ar1_resilience = moving_average(
        subsampled_ar1_resilience,
        window_size=31
        )

    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = potential.ravel()

    z_offset = 0.2

    attr_surface = go.Surface(
        z=potential,
        x=X,
        y=Y,
        surfacecolor=potential,
        colorscale="Spectral",
        cmin=0.8,
        cmax=3.8,
        opacity=1,
        lighting=dict(
            ambient=0.4,    
            diffuse=0.6,    
            specular=0.3,   
            roughness=0.5,  
            fresnel=0.1,    
        ),
        lightposition=dict(
            x=3,
            y=3,
            z=1,
        ),
        contours=dict(
            z=dict(
                show=True,
                start=0,
                end=4,
                size=0.2,
                color="black",
                width=2,
            )
        ),
        showscale=False,  
        name="Potential Energy Surface",
    )


    pad = 0.05
    x_range_full = [X.min(), X.max()]
    y_range_full = [Y.min(), Y.max()]
    z_range_full = [potential.min(), potential.max() + z_offset]

    x_margin = pad * (x_range_full[1] - x_range_full[0])
    y_margin = pad * (y_range_full[1] - y_range_full[0])
    z_margin = pad * (z_range_full[1] - z_range_full[0])

    x_range_fixed = [x_range_full[0] - x_margin, x_range_full[1] + x_margin]
    y_range_fixed = [y_range_full[0] - y_margin, y_range_full[1] + y_margin]
    z_range_fixed = [z_range_full[0] - z_margin, z_range_full[1] + z_margin]

    axis_limits = (x_range_fixed, y_range_fixed, z_range_fixed)
    aspect_ratio = (1, 1, 0.5)  


    time_traj = {"x": [], "y": [], "z": []}


    for j in range(len(subsampled_local_mean)):
        x, y = subsampled_local_mean[j], subsampled_ar1_resilience[j]
        z = griddata((x_flat, y_flat), z_flat, (x, y), method="linear")
        

        if z is None:
            z = potential.min()

        time_traj["x"].append(x)
        time_traj["y"].append(y)
        time_traj["z"].append(z + 0.3)  


    dx = np.diff(time_traj["x"])
    dy = np.diff(time_traj["y"])
    dz = np.diff(time_traj["z"])
    norms = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norms  # Normalize
    dy /= norms
    dz /= norms

    # Sample at regular intervals (e.g., every 5 points)
    indices = np.arange(0, len(dx), 182)

    # Generate Sphere
    phi, theta = np.linspace(0, np.pi, 20), np.linspace(0, 2 * np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    sphere_x_base = np.sin(phi) * np.cos(theta)
    sphere_y_base = np.sin(phi) * np.sin(theta)
    sphere_z_base = np.cos(phi)
    radius_percentage = 0.05  # Ball diameter as 5% of axis limits

    def update_sphere_scaled(center, radius_percentage, axis_limits, aspect_ratio):
        """Generate a sphere scaled according to axis limits and aspect ratio."""
        x_range, y_range, z_range = axis_limits
        x_aspect, y_aspect, z_aspect = aspect_ratio

        # Scale the radius based on axis limits and aspect ratio
        x_scale = (x_range[1] - x_range[0]) * radius_percentage / x_aspect
        y_scale = (y_range[1] - y_range[0]) * radius_percentage / y_aspect
        z_scale = (z_range[1] - z_range[0]) * radius_percentage / z_aspect

        x = center[0] + x_scale * sphere_x_base
        y = center[1] + y_scale * sphere_y_base
        z = center[2] + z_scale * sphere_z_base
        return x, y, z

    # Parallelize frame rendering
    def render_frame(i):
        x, y = subsampled_local_mean[i], subsampled_ar1_resilience[i]
        z = griddata((x_flat, y_flat), z_flat, (x, y), method="linear")
        
        # Handle interpolation failures
        if z is None:
            z = potential.min()

        # Path as a dotted red line slightly above the surface
        path_trace = go.Scatter3d(
            x=time_traj["x"][:i + 1],  # Use only the path up to the current frame
            y=time_traj["y"][:i + 1],
            z=time_traj["z"][:i + 1],
            mode="lines",
            line=dict(color="red", width=15),  # Dotted red line
            name="Path",
        )

        sphere_x, sphere_y, sphere_z = update_sphere_scaled([x, y, z + z_offset], radius_percentage, axis_limits, aspect_ratio)
        ball_trace = go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            surfacecolor=np.zeros_like(sphere_x),  
            colorscale=[[0, "red"], [1, "red"]], 
            showscale=False, 
            lighting=dict(
                ambient=0.7,
                diffuse=0.2,
                specular=0.8,
                roughness=0.5,
                fresnel=0.2,
            ),
            lightposition=dict(
                x=3,
                y=3,
                z=1,
            ),
            opacity=1,
            name="Ball",
        )

        # Create figure for each frame
        frame_fig = go.Figure(data=[attr_surface, ball_trace, path_trace])

        date_str = dates_clip[i].strftime("%d/%m/%Y")

        # Update layout with consistent camera, aspect ratio, and formatting
        frame_fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="NDVI Anomaly",  # No title
                    range=x_range_fixed,
                    showspikes=False,
                    showbackground=False,  # Remove background
                    zeroline=True,
                    zerolinecolor="black",
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    tickcolor="black",
                    tickwidth=2,
                    gridcolor="lightgray"
                ),
                yaxis=dict(
                    title="System speed (-AC1)",  # No title
                    tickmode = "array",
                    tickvals = np.round(np.arange(-1, 1.1, 0.1), 2),
                    ticktext = -np.round(np.arange(-1, 1.1, 0.1), 2),
                    range=y_range_fixed,
                    showspikes=False,
                    showbackground=False,
                    zeroline=True,
                    zerolinecolor="black",
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    tickcolor="black",
                    tickwidth=2,
                    gridcolor="lightgray"
                ),
                zaxis=dict(
                    title="Potential",  # No title
                    range=z_range_fixed,
                    showspikes=False,
                    showbackground=False,
                    zeroline=True,
                    zerolinecolor="black",
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    tickcolor="black",
                    tickwidth=2,
                    gridcolor="lightgray",
                ),
                aspectmode="manual",  # Maintain the aspect ratio
                aspectratio=dict(x=1, y=1, z=0.5),
                camera=dict(
                    eye=dict(x=-eye_x, y=eye_y, z=eye_z),  # Camera position
                    up=dict(x=1, y=0, z=1),  # "Up" direction
                    center=dict(x=0, y=0, z=0),  # Center of the view
                ),
            ),
            margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
            paper_bgcolor="white",  # White background
            plot_bgcolor="white",  # White grid
            annotations=[
            go.layout.Annotation(
                text=date_str,
                x=0.02,       # a small offset from the left
                y=0.02,       # near the top
                xref="paper", 
                yref="paper",
                showarrow=False,
                font=dict(size=25, color="black")
            )
            ],
        )   

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame_fig.write_image(frame_path, engine="kaleido", width=1080, height=960)

    
    frames_dir = "C:/data/gif_frames"
    os.makedirs(frames_dir, exist_ok=True)

    Parallel(n_jobs=-1)(delayed(render_frame)(i) for i in range(len(subsampled_local_mean)))

    output_gif = f"data_out/figs/gifs/surface_{plot_params.iloc[0]}.gif"
    with imageio.get_writer(output_gif, mode="I", fps=12, loop=0, codec="png", quantizer="nq") as writer:
        for frame_file in sorted(os.listdir(frames_dir)):
            if frame_file.endswith(".png"):
                writer.append_data(imageio.imread(os.path.join(frames_dir, frame_file)))

    for frame_file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame_file))

    print(f"GIF saved at: {output_gif}")

