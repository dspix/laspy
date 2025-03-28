import numpy as np
from scipy.stats import t as tdstr
from pycwt import wavelet, wct, cwt
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
from scipy.stats import t as tdstr
import ssqueezepy as ssq   
from ssqueezepy.experimental import scale_to_freq
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker

def expand_to_df(dict_of_feats):
  df = pd.DataFrame(dict_of_feats)
  cols = [i for i in df.columns if isinstance(df[i][0], dict)]
  for col in cols:
    df = pd.concat([df.drop([col], axis=1), df[col].apply(pd.Series)], axis=1)
  return df

def to_df(response):
  data = expand_to_df(response['features'])
  data['date'] = pd.to_datetime(data['date'])
  data.set_index('date', inplace=True)

  return data

def get_api_response(url):
  response = requests.get(url)
  return response.json()

def calculate_bounds(sm, sC, snu, quantile):
  """
  Calculate confidence interval bounds from dlm.

  Args:
      sm (np.ndarray): Mean estimates.
      sC (np.ndarray): Variance estimates.
      snu (np.ndarray): Degrees of freedom for t-distribution.
      quantile (float): Quantile value.

  Returns:
      lower_bounds (np.ndarray): Lower confidence interval bounds.
      upper_bounds (np.ndarray): Upper confidence interval bounds.
  """

  lower_bounds = np.array(list(map(lambda m, C, nu: m - np.sqrt(C) * tdstr.ppf(quantile, nu), sm, sC, snu)))
  upper_bounds = np.array(list(map(lambda m, C, nu: m + np.sqrt(C) * tdstr.ppf(quantile, nu), sm, sC, snu)))
  
  return lower_bounds, upper_bounds

def wave_variance(x, y, fs, u_period, l_period):
  """
  Calcualte coherence-power ratio (CPR) between two time series.

  Args:
      x (np.ndarray): First signal (e.g., environmental driver).
      y (np.ndarray): Second signal (e.g., ecosystem response).
      fs (float): Sampling frequency.
      u_period (float): Upper period threshold (e.g., 365 for annual).
      l_period (float): Lower period threshold.

  Returns:
      WCT (np.ndarray): Wavelet coherence values.
      wav_coh (float): Average coherence over target periods.
      wav_r2 (float): Power-weighted coherence (R^2-like).
      mean_phase (float): Mean phase offset.
      mean_phase_day (float): Phase offset in units of days.
  """
  x_nan = np.isnan(x)

  x = pd.Series(x).interpolate(method='linear', limit_direction='both').values
  x[np.where(np.isnan(x))] = np.mean(x[~np.isnan(x)])
  
  y = pd.Series(y).interpolate(method='linear', limit_direction='both').values
  y[np.where(np.isnan(y))] = np.mean(y[~np.isnan(y)])

  wavelet_mother=wavelet.Morlet(6)
  WCT, phase, coi, freq, *_ = wct(
      x, y, 1, 1/10, s0=-1, J=-1, 
      sig=False, significance_level=0.95,
      wavelet=wavelet_mother, normalize=True
      )
  
  periods = 1 / freq
  period_indices = np.where((periods >= l_period/fs) & (periods <= u_period/fs))[0]
  Wx, _, _, _, _, _ = cwt(x, 1, dj=1/10, s0=-1, J=-1, wavelet=wavelet_mother)
  power_x = np.abs(Wx) ** 2
  
  WCT[:, ~x_nan] = np.nan
  power_x[:, ~x_nan] = np.nan
  phase[:, ~x_nan] = np.nan

  # Coherence measures correlation irrespective of power - so can be misleading as all powers may not contribute
  coherence_power = np.nansum(WCT[period_indices, :] * power_x[period_indices, :], axis=0)
  total_power = np.nansum(power_x[period_indices, :], axis=0)

  wav_r2 = np.nansum(coherence_power) / np.sum(total_power)
  wav_coh = np.nanmean(WCT[period_indices, :])

  annual = 365.25
  half_annual = 365.25 / 2

  ann_ind = np.argmin(np.abs(periods - (annual/fs)))
  sub_ind = np.argmin(np.abs(periods - (half_annual/fs)))

  ann_angles = np.nanmean(phase[ann_ind, :])
  sub_angles = np.nanmean(phase[sub_ind, :])

  mean_phase = (ann_angles + sub_angles) / 2

  def phase_to_day(phase, period):
      return (np.abs(phase) / np.pi) * (period / 2)

  mean_phase_day = ((phase_to_day(ann_angles, annual)
                     + phase_to_day(sub_angles, half_annual))
                     / 2)

  return WCT, wav_coh, wav_r2, mean_phase, mean_phase_day

def ews(quant, sm, sC, snu, warmup, fs, n):
    """
    Detect early warning signals (Critical slowing down) in 
    a DLM state series by comparing recent values to a rolling upper 
    confidence boundary.

    Args:
        quant (float): Quantile for the upper confidence boundary.
        sm (np.ndarray): Smoothed mean state estimates.
        sC (np.ndarray): State covariance estimates.
        snu (np.ndarray): Degrees of freedom for Student's t-distribution.
        warmup (int): Number of initial indices to skip.
        fs (int): Sampling frequency (1 for daily, 16 for MODIS, etc.).
        n (int): Length of historical window in real days.

    Returns:
        np.ndarray: Array of indices where sm exceeds median confidence bound.
    """
    bd2 = list(map(lambda m, C, nu: m + np.sqrt(C) * tdstr.ppf(quant, nu),
                   sm, sC, snu))
    ews_indices = []
    n = int(n / fs)

    for i in range(warmup, len(sm)):
        start_idx = max(i - n, 0)
        median_bd2 = np.median(bd2[start_idx:i])
        if sm[i] > median_bd2:
            ews_indices.append(i)

    return np.array(ews_indices)

def plot_dlm_results(daily_dates, x_sweep, sig_vector, comp_dates,
                     ndvi_comp, cosmos_vwc, FF_sweep, FF_modcomp,
                     quantile, warmup, output_path=None):
    """
    Plot DLM-fitted NDVI data, confidence bounds, and early warning signals
    (CSD) from both SWEEP and composite MODIS sources.

    Args:
        daily_dates (pd.Series): Dates for daily SWEEP NDVI.
        x_sweep (np.ndarray): SWEEP NDVI values.
        sig_vector (np.ndarray): Boolean mask for significant SWEEP values.
        comp_dates (pd.Series): Dates for MODIS composite NDVI.
        ndvi_comp (np.ndarray): MODIS composite NDVI values.
        cosmos_vwc (np.ndarray): Volumetric water content time series.
        FF_sweep (dict): DLM output for SWEEP (keys: 'sm', 'sC', 'snu').
        FF_modcomp (dict): DLM output for MODIS composite NDVI.
        quantile (float): Quantile used for confidence bounds.
        warmup (int): Number of initial time steps to exclude.
        output_path (str, ): File path to save the plot.

    Returns:
        None
    """

    point_size = 0.7

    sm_sweep = FF_sweep['sm'][2,:]
    sC_sweep = FF_sweep['sC'][2,2,:]
    snu_sweep = FF_sweep['snu']

    sm_modcomp = FF_modcomp['sm'][2,:]
    sC_modcomp = FF_modcomp['sC'][2,2,:]
    snu_modcomp = FF_modcomp['snu']

    sweep_lbounds, sweep_ubounds = calculate_bounds(
        sm_sweep, sC_sweep, snu_sweep, quantile)
    comp_lbounds, comp_ubounds = calculate_bounds(
        sm_modcomp, sC_modcomp, snu_modcomp, quantile)

    sweep_ews_ind = ews(quantile, sm_sweep, sC_sweep, snu_sweep,
                        warmup, 1, 365)
    comp_ews_ind = ews(quantile, sm_modcomp, sC_modcomp, snu_modcomp,
                       int(warmup / 16), 16, 365)

    sweep_ews = np.full(len(x_sweep), np.nan)
    if len(sweep_ews_ind) != 0:
        sweep_ews[sweep_ews_ind] = sm_sweep[sweep_ews_ind]

    comp_ews = np.full(len(ndvi_comp), np.nan)
    if len(comp_ews_ind) != 0:
        comp_ews[comp_ews_ind] = sm_modcomp[comp_ews_ind]

    fig, ax = plt.subplots(3, 1, figsize=(4, 4),
                           gridspec_kw={'width_ratios': [1],
                                        'height_ratios': [1, 1, 2]})

    ax[0].scatter(daily_dates[sig_vector], x_sweep[sig_vector],
                  color='black', label='SWEEP (sig.)', s=point_size)
    ax[0].scatter(daily_dates[~sig_vector], x_sweep[~sig_vector],
                  color='black', label='SWEEP (non-sig.)', marker='+',
                  s=point_size + 5, linewidths=0.3)
    ax[0].scatter(comp_dates, ndvi_comp, color='red', label='Comp.',
                  s=point_size)
    ax[0].legend(loc='lower right', fontsize=6, ncol=3)
    ax[0].set_ylabel('NDVI', fontsize=6)
    ax[0].tick_params(axis='x', labelsize=6)
    ax[0].tick_params(axis='y', labelsize=6)

    ax[1].plot(daily_dates, cosmos_vwc, color='black',
               label='Volumetric water content (%)',
               linewidth=0.6, zorder=1)
    ax[1].set_ylabel('Soil moisture (%)', fontsize=6)
    ax[1].tick_params(axis='x', labelsize=6)
    ax[1].tick_params(axis='y', labelsize=6)

    ax[2].fill_between(comp_dates, -comp_lbounds, -comp_ubounds,
                       facecolor='red', alpha=0.2, zorder=1)
    ax[2].plot(comp_dates, -sm_modcomp, color='red', linestyle='dashed',
               label='Comp. Speed', linewidth=0.6, zorder=2)
    ax[2].plot(comp_dates, -comp_ews, color='red', label='Comp. CSD',
               linewidth=2, zorder=3)

    ax[2].fill_between(daily_dates, -sweep_lbounds, -sweep_ubounds,
                       facecolor='lightgrey', alpha=0.8, zorder=4)
    ax[2].plot(daily_dates, -sm_sweep, color='black', linestyle='dashed',
               label='SWEEP Speed', linewidth=0.6, zorder=5)
    ax[2].plot(daily_dates, -sweep_ews, color='black', label='SWEEP CSD',
               linewidth=2, zorder=6)

    ax[2].set_ylim(-1, 1)
    ax[2].legend(loc='upper right', fontsize=6, ncol=2)
    ax[2].set_ylabel('System Speed (-AC1)', fontsize=6)
    ax[2].set_xlabel('Date', fontsize=6)
    ax[2].tick_params(axis='x', labelsize=6)
    ax[2].tick_params(axis='y', labelsize=6)

    for ax_i in [ax[0], ax[1]]:
        plt.setp(ax_i.get_xticklabels(), visible=False)
        ax_i.tick_params(axis='x', which='both', bottom=False, top=False)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        return fig

def plot_gcc_comparison(daily_dates, x_sweep, sig_vector, comp_dates,
                        ndvi_comp, gcc, output_path=None):
    """
    Plot NDVI time series from SWEEP and MODIS composite, with GCC on a
    secondary y-axis.

    Args:
        daily_dates (pd.Series): Dates corresponding to SWEEP and GCC data.
        x_sweep (np.ndarray): NDVI values from SWEEP.
        sig_vector (np.ndarray): Boolean vector for significant SWEEP values.
        comp_dates (pd.Series): Dates corresponding to composite NDVI.
        ndvi_comp (np.ndarray): Composite NDVI time series.
        gcc (np.ndarray): Green Chromatic Coordinate (GCC) values.
        output_path (str, optional): Path to save the figure. If None, the
            figure is returned.

    Returns:
        matplotlib.figure.Figure or None: Returns the figure if output_path
        is not provided.
    """

    point_size = 0.7

    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    ax.scatter(daily_dates[sig_vector], x_sweep[sig_vector], color='black',
               label='Sweep (sig.)', s=point_size)
    ax.scatter(daily_dates[~sig_vector], x_sweep[~sig_vector], color='black',
               label='Sweep (non-sig.)', marker='+', s=point_size + 5,
               linewidths=0.3)
    ax.scatter(comp_dates, ndvi_comp, color='red', label='Mod13Q1', s=2)

    ax.set_ylim(-0.5, 1.2)
    ax.set_ylabel('NDVI', fontsize=6)
    ax.tick_params(axis='both', labelsize=6)

    ax2 = ax.twinx()
    ax2.scatter(daily_dates, gcc, color="green", s=1, label='GCC')
    ax2.set_ylim(0.3, 0.5)
    ax2.set_ylabel('GCC', fontsize=6)
    ax2.tick_params(axis='y', labelsize=6)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left',
              fontsize=6)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        return fig

def plot_sweep_location(daily_dates, x, x_sweep, sig_vector,
                        signal_location_infil, 
                        signal_index_infil, 
                        center_mat_w,
                        output_path=None):
    """
    Plot SWEEP NDVI signal and significance alongside phenological signal
    location from wavelet analysis.

    Args:
        daily_dates (pd.Series): Time index for x_sweep and phenology signals.
        x (np.ndarray): unfiltered NDVI time series.
        x_sweep (np.ndarray): Sweep NDVI series.
        sig_vector (np.ndarray): Boolean array for significant values in x_sweep.
        signal_location_infil (dict): Output from Sweep_infil
             Contains 'loc_center', 'loc_max', 'loc_min', and 'ind_center'.
        signal_index_infil (dict): Output from Sweep_infil
             Contains index equivolent of signal_location_infil.
        center_mat_w (np.ndarray): Wavelet power surface.
        output_path (str, optional): Path to save the plot. If None,
            returns the figure.

    Returns:
        matplotlib.figure.Figure or None: Returns the figure if output_path
        is not provided.
    """

    point_size = 0.7

    x_sig = x_sweep.copy()
    x_non = x_sweep.copy()
    x_sig[~sig_vector] = np.nan
    x_non[sig_vector] = np.nan

    loc_c = signal_location_infil['loc_center']
    loc_max = signal_location_infil['loc_max']
    loc_min = signal_location_infil['loc_min']

    best_ind = signal_index_infil['ind_center']

    year_starts = daily_dates[daily_dates.dt.dayofyear == 1].index
    year_labels = daily_dates.dt.year[daily_dates.dt.dayofyear == 1]
    ytick_positions = np.linspace(-1, 20, num=11)  # -1 to 20 with 6 points
    ytick_labels = np.linspace(0, 1, num=11) 
    #ytick_labels = np.insert(ytick_labels, 0, 0)
    xlim = [daily_dates.min(), daily_dates.max()]

    grey_cmap = mcolors.LinearSegmentedColormap.from_list(
        "", ["#F0F0F0", "darkgrey"])

    fig, ax = plt.subplots(2, 1, figsize=(6, 4),
                           gridspec_kw={'height_ratios': [1, 1.5]})
    fig.subplots_adjust(wspace=0.08, hspace=0.08)

    ax[0].scatter(daily_dates, x, c="lightgrey",  label="MOD09GQ", s = point_size)
    ax[0].scatter(daily_dates, x_sig, c="black", label="Sweep (sig.)",
                  s=point_size)
    ax[0].scatter(daily_dates, x_non, color='black', label="Sweep (non-sig.)",
                  marker='+', s=point_size + 5, linewidths=0.3)
    ax[0].plot(daily_dates, loc_c, color="black", label="Phenology center",
               linewidth=0.9)
    ax[0].plot(daily_dates, loc_max, color="black", linestyle='dashed',
               linewidth=0.6)
    ax[0].plot(daily_dates, loc_min, color="black", linestyle='dashed',
               linewidth=0.6)
    ax[0].legend(loc='lower right', fontsize=6, ncol=2)
    ax[0].set_ylabel('NDVI', fontsize=6)
    ax[0].set_xlim(xlim)

    interval = 0.015
    round_lvl = 0.05
    l_min = math.floor(np.nanmin(center_mat_w) / round_lvl) * round_lvl
    l_max = math.floor(np.nanmax(center_mat_w) / round_lvl) * round_lvl
    levels = np.arange(l_min, l_max, interval)

    ax[1].contour(center_mat_w, origin='lower', levels=levels,
                  cmap=grey_cmap, linewidths=0.8)
    ax[1].plot(np.arange(len(best_ind)), best_ind + 0.5, color="black",
               linewidth=0.9, label="Selected phenology center")

    norm = mcolors.Normalize(vmin=l_min, vmax=l_max)
    sm = cm.ScalarMappable(cmap=grey_cmap, norm=norm)
    sm.set_array([])

    pos = ax[1].get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])  
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_ylabel('Wavelet Power Contours', rotation=270, labelpad=-5, fontsize=6, verticalalignment='center')

    cbar.set_ticks([l_min, l_max])
    cbar.set_ticklabels([l_min, l_max], fontsize = 5)

    ax[1].set_xlim(0, len(x_sweep)-1)
    ax[1].set_xticks(year_starts)
    ax[1].set_xticklabels(year_labels, rotation=0, ha='center', fontsize = 6)
    ax[1].set_yticks(ytick_positions)
    ax[1].set_yticklabels(np.round(ytick_labels, 1), rotation=0, ha='right')
    valid_rows = ~np.isnan(center_mat_w).any(axis=1)
    min_index = np.where(valid_rows)[0][0]  # First index with no NaNs
    max_index = np.where(valid_rows)[0][-1]  # Last index with no NaN
    ax[1].set_ylim(min_index + 1, max_index)

    ax[0].set_ylabel('NDVI', fontsize=6)
    ax[1].set_ylabel('Potential centers', fontsize=6)
    ax[0].tick_params(axis='both', which='major', labelsize=5)
    ax[1].tick_params(axis='both', which='major', labelsize=5)

    for ax_i in ax[:-1]:
        plt.setp(ax_i.get_xticklabels(), visible=False)
        ax_i.tick_params(axis='x', which='both', bottom=False, top=False)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        return fig

def plot_sweep_extract(daily_dates, x, x_sweep, signal_location_infil,
                       wx_og, wx_filt, l_period, output_path=None):
    """
    Plot the original and filtered wavelet transform of a sweep NDVI signal
    and highlight its power distribution in relation to phenology bounds.

    Args:
        daily_dates (pd.Series): Time index.
        x (np.ndarray): unfiltered NDVI time series.
        signal_location_infil (dict): Output from Sweep_infil, 
            Contains 'loc_max' and 'loc_min' bounds.
        wx_og (np.ndarray): Original wavelet transform (2D time-frequency).
        wx_filt (np.ndarray): Filtered wavelet transform.
        output_path (str, optional): Path to save figure. If None, returns fig.

    Returns:
        matplotlib.figure.Figure or None: Returns the figure if not saved.
    """
    year_labels = daily_dates.dt.year[daily_dates.dt.dayofyear == 1]

    point_size = 0.7

    loc_max = signal_location_infil['loc_max']
    loc_min = signal_location_infil['loc_min']

    x_masked = x.copy()
    x_masked[(x > loc_max) | (x < loc_min)] = np.nan

    wav = ('morlet', {'mu': 6})

    wx, scales, *_ = ssq.cwt(x, wav, fs=1)
    freqs = scale_to_freq(scales, wav, len(x), fs=1)
    periods = 1/freqs

    og_av = np.mean(np.abs(wx_og), axis=1)
    filt_av = np.mean(np.abs(wx_filt), axis=1)

    labels = (2, 4, 8, 16, 32, 64, 128, 365)

    indices = []
    for label in labels:
        index = np.argmin(np.abs(label - periods))
        indices.append(index)

    grey_cmap = mcolors.LinearSegmentedColormap.from_list("", ["#F0F0F0", "black"])

    fig, ax = plt.subplots(2, 3, figsize=(6, 3),
                           gridspec_kw={'height_ratios': [1, 1],
                                        'width_ratios': [1, 0.4, 0.15]})
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    for ax_i in [ax[0, 2], ax[1, 2]]:
        pos = ax_i.get_position()
        pos.x0 -= 0.00 
        pos.x1 -= 0.02  
        ax_i.set_position(pos)

    levels = np.linspace(np.min(np.abs(wx_og)), np.max(np.abs(wx_og)), 10)

    #plot ndvi series
    ax[0, 0].scatter(daily_dates, x, c="lightgrey",  label="MOD09GQ", s=point_size)
    ax[0, 0].scatter(daily_dates, x_masked, c="black",  label="Phenology envelope", s=point_size)
    ax[0, 0].set_ylabel('Enveloped NDVI', fontsize = 6)

    ax[1, 0].scatter(daily_dates, x, c="lightgrey",  label="MOD09GQ", s=point_size)
    ax[1, 0].scatter(daily_dates, x_sweep, c="black",  label="Sweep phenology", s=point_size)
    ax[1, 0].set_xlabel('Year', fontsize=6)
    ax[1, 0].set_ylabel('Noise-supressed NDVI', fontsize = 6)

    ax[0, 1].contourf(np.abs(wx_og), levels=levels, cmap=grey_cmap)
    ax[0, 1].axhline(y=np.argmin(np.abs(l_period - periods)), color='black', linestyle='--', linewidth=0.6)
    ax[0, 1].set_yticks(indices, labels)
    
    contour = ax[1, 1].contourf(np.abs(wx_filt), levels = levels, cmap = grey_cmap)
    ax[1, 1].axhline(y=np.argmin(np.abs(l_period - periods)), color='black', linestyle='--', linewidth=0.6)
    ax[1, 1].set_yticks(indices, labels)
    ax[1, 1].set_xticks(year_labels.iloc[[0, -1]].index)
    ax[1, 1].set_xticklabels(year_labels.iloc[[0, -1]], rotation=0, ha='center', fontsize = 6)
    ax[1, 1].set_xlabel('Year', fontsize = 6)
    # Create an inset axis for the colorbar

    norm = mcolors.Normalize(vmin = np.min(levels), vmax = np.max(levels))
    sm = cm.ScalarMappable(cmap=grey_cmap, norm=norm)
    sm.set_array([])
    
    pos = ax[1, 1].get_position()
    cbar_ax = fig.add_axes([pos.x1 - 0.185, pos.y0 + 0.02, 0.006, 0.1])  # Adjust the 0.01 and 0.02 values to position and size the color bar
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([np.min(levels), np.max(levels)])
    cbar.set_ticklabels([np.round(np.min(levels), 3), np.round(np.max(levels), 2)], fontsize = 5)
    cbar.ax.text(-0.1, 1.45, 'Wavelet\nPower', rotation=0, va='center', ha='left', fontsize = 6, transform=cbar.ax.transAxes)
    cbar.ax.xaxis.label.set_position((0, 1))
    cbar.ax.xaxis.label.set_ha('left') 

    ax[0, 2].plot(og_av, np.arange(len(og_av)), color = "black", linewidth=0.6)
    ax[0, 2].axhline(y=np.argmin(np.abs(l_period - periods)), color='black', linestyle='--', linewidth=0.6)
    ax[0, 2].set_yticks(indices, labels)
    ax[0, 2].set_ylim(0, wx_filt.shape[0])
    ax[0, 2].set_ylabel('Period (years)', fontsize = 6)
    ax[0, 2].yaxis.set_label_position("right")
    ax[0, 2].yaxis.tick_right()

    ax[1, 2].plot(filt_av, np.arange(len(filt_av)), label = "Average", color = "black", linewidth=0.6)
    ax[1, 2].axhline(y=np.argmin(np.abs(l_period - periods)), color='black', linestyle='--', linewidth=0.6)
    ax[1, 2].set_yticks(indices, labels)
    ax[1, 2].set_ylim(0, wx_filt.shape[0])
    ax[1, 2].set_xlabel('Av. Power', fontsize = 6)
    ticks = ax[1, 2].get_xticks()
    ax[1, 2].xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax[1, 2].set_xticklabels([f'{tick:.1f}' for tick in ticks])
    ax[1, 2].set_ylabel('Period (years)', fontsize = 6)
    ax[1, 2].yaxis.set_label_position("right")
    ax[1, 2].yaxis.tick_right()

    for ax_i in ax.flat:
        ax_i.tick_params(axis='both', which='major', labelsize=6)

    for ax_i in [ax[0, 1], ax[1, 1]]:
        plt.setp(ax_i.get_yticklabels(), visible=False)
        ax_i.tick_params(axis='y', which='both', left=False, top=False)

    for ax_i in [ax[0, 0], ax[0, 1], ax[0, 2]]:
        plt.setp(ax_i.get_xticklabels(), visible=False)
        ax_i.tick_params(axis='x', which='both', bottom=False, top=False)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        return fig
