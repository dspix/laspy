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

def PES(state_estimates, state_variances, state_1_ind, state_2_ind, warmup):

    local_mean = state_estimates[state_1_ind, :][warmup: ]
    ar1_resilience = state_estimates[state_2_ind, :][warmup: ]

    local_mean_variance = state_variances[state_1_ind, state_1_ind, :][warmup: ]
    ar1_resilience_variance = state_variances[state_2_ind, state_2_ind, :][warmup: ]

    state_estimates_2d = np.vstack((local_mean, ar1_resilience))

    weights = 1 / np.sqrt(local_mean_variance * ar1_resilience_variance)  # Inverse of standard deviation as weights

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

    results = Parallel(n_jobs=-1)(delayed(process_sample)(sample_ind, max_space, wavelet, fs) for i in tqdm(range(n)))

    wx_sorted = np.sort(results)

    return wx_sorted

def sweep_locate(x, win_mean, win_range, wavelet, fs, l_period, u_period, min_thresh, sig_wp, cost, buff):
    x_trim = x.copy()
    x_trim[x_trim < min_thresh] = np.nan

    power_mat = np.full((len(win_mean), len(win_range), len(x)), np.nan)

    def process_combination(i, j):
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

    results = Parallel(n_jobs=-1)(delayed(process_combination)(i, j) for i in range(power_mat.shape[0]) for j in range(power_mat.shape[1]))

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

    #best_power = np.nanmax(center_mat, axis = 0)
    best_center_ind = np.nanargmax(center_mat, axis = 0)
    best_range_ind = np.nanargmax(range_mat, axis = 0)

    weighted_power_mat = power_mat.copy()

    #mean_weight = cost
    #slice_weight = 1 - cost

    for t in range(power_mat.shape[2]):
        #power_mat_t = weighted_power_mat[:, :, t]
        #slice_min = np.nanmin(power_mat_t)
        #slice_max = np.nanmax(power_mat_t)
        #power_mat_t_norm = (power_mat_t - slice_min) / (slice_max - slice_min)
        #weighted_power_mat[:, :, t] = (normalized_matrix * mean_weight) + (power_mat_t_norm * slice_weight)
        weighted_power_mat[:, :, t] *= adjusted_slice

    #extract annual signal
    center_mat_w = np.nanmax(weighted_power_mat, axis = 1)
    range_mat_w = np.nanmax(weighted_power_mat, axis = 0)

    best_center_ind = np.nanargmax(center_mat_w, axis = 0)
    best_range_ind = np.nanargmax(range_mat_w, axis = 0)

    loc_mean = win_mean[best_center_ind] 
    loc_min = loc_mean - win_range[best_range_ind] - buff
    loc_max = loc_mean + win_range[best_range_ind] + buff

    #for t in range(center_mat.shape[1]):
    #    best_power[t] = center_mat[best_center_ind[t], t]

    best_power = center_mat[best_center_ind, np.arange(center_mat.shape[1])]
    #best_power = power_mat[best_center_ind, best_range_ind, np.arange(power_mat.shape[2])]

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

    return signal_index, signal_location, best_power, sig_vector, av_power_mat, power_mat, center_mat, range_mat, center_mat_w, range_mat_w

def sweep_infil(signal_index, signal_location, best_power, sig_vector, av_power_mat, win_mean, win_range, buff):

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
    loc_min_if[np.isnan(loc_min_if)] = loc_mean_if[np.isnan(loc_min_if)] - win_range[av_range_ind] - buff
    loc_max_if[np.isnan(loc_max_if)] = loc_mean_if[np.isnan(loc_max_if)] + win_range[av_range_ind] + buff

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

    return signal_index_infil, signal_index_sigonly, signal_location_infil, signal_location_sigonly

def sweep_extract(x, signal_location_infil, wavelet, fs, l_period, exp):

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
    #x_sweep += ((loc_mean_if - np.mean(loc_mean_if)) + np.mean(x_subset)) + np.mean(loc_mean_if)

    x_sweep[na_ind] = np.nan

    return x_sweep, wx_og, wx_filt, factor_mask
