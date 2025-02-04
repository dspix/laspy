import numpy as np
from scipy.stats import t as tdstr
from pycwt import wavelet, wct, cwt
import requests
import numpy as np
import pandas as pd

def calculate_bounds(sm, sC, snu, quantile):

    lower_bounds = np.array(list(map(lambda m, C, nu: m - np.sqrt(C) * tdstr.ppf(quantile, nu), sm, sC, snu)))
    upper_bounds = np.array(list(map(lambda m, C, nu: m + np.sqrt(C) * tdstr.ppf(quantile, nu), sm, sC, snu)))
    
    return lower_bounds, upper_bounds


def expand_to_df(dict_of_feats):
  '''Expand key-value pairs to columns'''
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

def wave_variance(x, y, fs, u_period, l_period):

    x_nan = np.isnan(x)

    x = pd.Series(x).interpolate(method='linear', limit_direction='both').values
    x[np.where(np.isnan(x))] = np.mean(x[~np.isnan(x)])
    
    y = pd.Series(y).interpolate(method='linear', limit_direction='both').values
    y[np.where(np.isnan(y))] = np.mean(y[~np.isnan(y)])

    wavelet_mother=wavelet.Morlet(6)
    WCT, phase, coi, freq, *_ = wct(x, y, 1, 1/10, s0=-1, J=-1, sig=False, significance_level=0.95, wavelet=wavelet_mother, normalize=True)
    
    periods = 1 / freq
    period_indices = np.where((periods >= l_period/fs) & (periods <= u_period/fs))[0]
    Wx, _, _, _, _, _ = cwt(x, 1, dj=1/10, s0=-1, J=-1, wavelet=wavelet_mother)
    power_x = np.abs(Wx) ** 2
    
    WCT[:, ~x_nan] = np.nan
    power_x[:, ~x_nan] = np.nan
    phase[:, ~x_nan] = np.nan

    #coherence measures correlation irrespective of power - so can be misleading as all powers may not contribute
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

    #phase_av = np.nanmean(phase[period_indices, :], axis=1)#, weights=WCT[period_indices, :] * power_x[period_indices, :])
    mean_phase_day = (phase_to_day(ann_angles, annual) + phase_to_day(sub_angles, half_annual)) / 2

    return WCT, wav_coh, wav_r2, mean_phase, mean_phase_day




