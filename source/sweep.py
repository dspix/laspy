from datetime import datetime
import scipy.linalg
from scipy.stats import t as tdstr
from scipy.stats import norm
from scipy.interpolate import interp1d
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.rc('font', size=14)
import matplotlib.pyplot as plt


from scipy import stats
from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess
import ssqueezepy as ssq
from ssqueezepy.experimental import scale_to_freq
from wavelet_functions import ssq_cwt_sig

import pandas as pd
import numpy as np
from scipy import signal


def normalize_vector(v):
    min_v = np.min(v)
    max_v = np.max(v)
    if max_v == min_v:
        return np.zeros_like(v)
    return (v - min_v) / (max_v - min_v)


#WAVELT FUNCTIONS

def ssq_cwt_sig(x_in, wavelet, fs, n, sig_lvl):
    X = pd.Series(x_in)
    X = stats.zscore(X)
    ar1 = X.autocorr(lag=1)
    
    wx_og, scales_og, *_ = ssq.cwt(X.values.ravel(), wavelet, fs=fs)
    wx_sur = np.zeros((wx_og.shape[0], wx_og.shape[1], n))
    freqs = scale_to_freq(scales_og, wavelet, len(x_in), fs=fs)
    periods = 1 / freqs
    
    length = len(X)
    
    for i in tqdm(range(n)):
        # Generate synthetic series using AR(1) process
        synthetic_series = np.random.normal(size=length)
        for j in range(1, length):
            synthetic_series[j] += ar1 * synthetic_series[j - 1]
        
        synthetic_series = stats.zscore(synthetic_series)
        wx_i, *_ = ssq.cwt(synthetic_series, wavelet, fs=fs)
        wx_sur[:, :, i] = np.abs(wx_i)
    
    p_values = np.sum(wx_sur > np.abs(wx_og[:, :, np.newaxis]), axis=2) / n
    sig_mat = p_values < sig_lvl
    
    return wx_og, periods, scales_og, p_values, sig_mat


""" def ssq_cwt_sig(x_in, wavelet, fs, n, sig_lvl):
     
    import pandas as pd
    import numpy as np
    from ssqueezepy.experimental import scale_to_freq
    from scipy import stats
    import ssqueezepy as ssq
    from scipy import signal
    from tqdm import tqdm

    X = pd.Series(x_in)
    X = stats.zscore(X)
    ar1 = X.autocorr(lag=1)
    
    wx_og, scales_og, *_ = ssq.cwt(X.values.ravel(), wavelet, fs=fs)
    #wx_og = np.abs(wx_og)#**2
    wx_sur = np.zeros((wx_og.shape[0], wx_og.shape[1], n))
    freqs = scale_to_freq(scales_og, wavelet, len(x_in), fs=fs)
    periods = 1/freqs
    
    for i in tqdm(range(n)):
        length = len(X)
        synthetic_series = np.zeros(length)
        synthetic_series[0] = np.random.normal()  # Initial value
        
        for j in range(1, length):
            synthetic_series[j] = ar1 * synthetic_series[j-1] + np.random.normal()
        
        synthetic_series = stats.zscore(synthetic_series)
        wx_i, *_ = ssq.cwt(synthetic_series, wavelet, fs=fs)
        wx_sur[:,:,i] = np.abs(wx_i)#**2
    
    p_values = np.sum(wx_sur > np.abs(wx_og[:, :, np.newaxis]), axis = 2) / n
    sig_mat = p_values < sig_lvl
    
    return wx_og, periods, scales_og, p_values, sig_mat """

#WAVELETCOMP R PACKAGE TRANSFORM CODE##############
def wc_cwt_morlet(x, dt=1, dj=1/20, lowerPeriod=None, upperPeriod=None):
    
    import numpy as np
    from scipy.fftpack import fft, ifft
    from scipy.signal import detrend

    if lowerPeriod is None:
        lowerPeriod = 2 * dt
    if upperPeriod is None:
        upperPeriod = int(np.floor(len(x) * dt / 3))

    series_length = len(x)
    pot2 = int(np.log2(series_length) + 0.5)
    pad_length = 2**(pot2 + 1) - series_length
    omega0 = 6
    fourier_factor = (2 * np.pi) / omega0
    min_scale = lowerPeriod / fourier_factor
    max_scale = upperPeriod / fourier_factor
    J = int(np.log2(max_scale / min_scale) / dj)
    scales = min_scale * 2**(np.arange(0, J + 1) * dj)
    scales_length = len(scales)
    periods = fourier_factor * scales
    N = series_length + pad_length
    omega_k = np.arange(1, N // 2 + 1)
    omega_k = omega_k * (2 * np.pi) / (N * dt)
    omega_k = np.concatenate(([0], omega_k, -omega_k[int(np.floor((N - 1) / 2)) - 1::-1]))

    def morlet_wavelet_transform(x):
        st = np.std(x, ddof=1)
        x = (x - np.mean(x)) / st # Remove the mean and trend
        xpad = np.concatenate((x, np.zeros(pad_length)))
        fft_xpad = fft(xpad, axis=0)
        wave = np.zeros((scales_length, N), dtype=np.complex_)
        
        for ind_scale in range(scales_length):
            my_scale = scales[ind_scale]
            norm_factor = np.pi**(1/4) * np.sqrt(2 * my_scale / dt)
            expnt = -((my_scale * omega_k - omega0)**2 / 2) * (omega_k > 0)
            daughter = norm_factor * np.exp(expnt)
            daughter = daughter * (omega_k > 0)
            wave[ind_scale, :] = ifft(fft_xpad * daughter, axis=0) / N

        wave = wave[:, :series_length]
        return wave

    Wave = morlet_wavelet_transform(x)
    Power = np.abs(Wave)**2 / np.tile(scales, (series_length, 1)).T
    Phase = np.angle(Wave)
    Ampl = np.abs(Wave) / np.tile(np.sqrt(scales), (series_length, 1)).T

    return Wave, Phase, Ampl, periods, scales, Power


def wc_rec(wx, scales, dt, dj, nr, nc):

    import numpy as np

    rec_waves = np.zeros((nr, nc))
    
    for s_ind in range(nr):
        rec_waves[s_ind, :] = (np.real(wx[s_ind, :]) / np.sqrt(scales[s_ind])) * dj * np.sqrt(dt) / (np.pi**(-1/4) * 0.776)

    x_r = np.nansum(rec_waves, axis=0)
    return(x_r)


#SWEEP FUNCTIONS##############
def extract_ind(results_mat):

    row_out = []
    col_out = []
    for i in range(results_mat.shape[2]):

        mat_i = results_mat[:,:,i]
        mat_i[np.isnan(mat_i)] = -np.inf
        #mat_i = np.where((mat_i < min_val) | (mat_i > max_val), mat_i, -np.inf)
        max_val = np.max(mat_i, axis=None)
        max_indices = np.transpose(np.where(mat_i == max_val))
        max_indices_sorted = sorted(max_indices, key=lambda x: x[1])
        ind_i = max_indices_sorted[0]

        row_out.append(ind_i[0])
        col_out.append(ind_i[1])
        
    df = pd.DataFrame({'mean_i': row_out, 'range_i': col_out})   
    return df

def sweep_locate(x, win_mean, win_range, win_time, u_bound, l_bound, wavelet, min_thresh, win_length, fs):

    i = 0
    j = 0
    t = 0
    
    results_mat = np.full((len(win_mean), len(win_range), len(win_time)), np.nan)
    mean_err_mat = np.full((len(win_mean), len(win_range), len(win_time)), np.nan)
    
    #x = x.values.ravel()
    x_trim = x.copy()
    x_trim[x_trim < min_thresh] = np.nan
    
    for t in tqdm(range(results_mat.shape[2])):
        t0 = win_time[t] - np.floor(win_length/2)
        t1 = win_time[t] + np.floor(win_length/2)

        if t0 < 0:
            t0 = 0
        if t1 > len(x)-1:
            t1 = len(x)-1

        for i in range(results_mat.shape[0]):
            for j in range(results_mat.shape[1]):
                #print(f"{t}, {i}, {j}")

                #grab mean and range
                mean_i = win_mean[i]
                range_j = win_range[j]
                
                min_i = mean_i - range_j
                max_i = mean_i + range_j
                
                if max_i > np.nanmax(x_trim) or min_i < np.nanmin(x_trim):
                    continue
        
                x_subset = x_trim.copy()
                x_subset = x_subset[int(t0):int(t1)]
                
                x_subset[np.where(x_subset > max_i)] = np.nan
                x_subset[np.where(x_subset < min_i)] = np.nan
                    
                if np.all(np.isnan(x_subset)):
                    continue
                
                actual_mean = np.nanmean(x_subset)
                mean_error = mean_i - actual_mean
    
                mean_err_mat[i, j, t] = mean_error
                
                na_ind = np.where(np.isnan(x_subset))
                x_subset[na_ind] = actual_mean
                std_x = np.std(x_subset)

                if np.std(x_subset) == 0 or np.isnan(std_x) or len(set(x_subset)) < 2:
                   continue

                x_subset = stats.zscore(x_subset)

                x_smooth = lowess(x_subset, np.arange(len(x_subset)), frac=0.75)
                x_subset = x_subset.copy() - x_smooth[:, 1]

                #####squeezepy cwt#####
                wx, scales, *_ = ssq.cwt(x_subset, wavelet, fs=fs)
                freqs = scale_to_freq(scales, wavelet, len(x_subset), fs=fs)
                periods = 1/freqs

                mask = (periods <= u_bound) & (periods >= l_bound)
                wx[~mask, :] = 0
                rec = ssq.icwt(wx, wavelet, scales)

                #####waveletcomp cwt#####

                #wc_wx, Phase, Ampl, wc_periods, wc_scales, Power = wc_cwt_morlet(x_subset, dt=1, dj=1/20, lowerPeriod=l_bound, upperPeriod=u_bound)
                #rec = wc_rec(wc_wx, wc_scales, dt = 1, dj = 1/20, nr = len(wc_scales), nc = len(x_subset))
                
                results_mat[i, j, t] = np.std(rec, ddof =1)


    return results_mat, mean_err_mat

def sweep_extract(x, wavelet, fs, signal_location_matrix, signal_mean_error, win_mean, win_range, win_time, u_period, l_period, buff, sig_lvl, exp):

    sig_loc = extract_ind(signal_location_matrix)

    sig_loc['time'] = win_time
    sig_loc['mean'] = win_mean[sig_loc['mean_i']]
    sig_loc['range'] = win_range[sig_loc['range_i']]

    sig_loc['mean_err'] = signal_mean_error[sig_loc.iloc[:, 0], sig_loc.iloc[:, 1], np.arange(len(sig_loc))]
    sig_loc['mean_adj'] = sig_loc['mean'] - sig_loc['mean_err']

    mean_infil = pd.Series([np.nan] * len(x))
    range_infil = pd.Series([np.nan] * len(x))
    error_infil = pd.Series([np.nan] * len(x))

    mean_infil[win_time] = sig_loc['mean_adj']
    range_infil[win_time] = sig_loc['range']
    error_infil[win_time] = sig_loc['mean_err']

    mean_infil = mean_infil.interpolate(method='linear')
    range_infil = range_infil.interpolate(method='linear')
    error_infil = error_infil.interpolate(method='linear')

    min_infil = mean_infil - range_infil - buff
    max_infil = mean_infil + range_infil + buff

    x_subset = x.copy()
    ind_rm = (x_subset > max_infil) | (x_subset < min_infil)
    x_subset[ind_rm] = np.nan
    x_subset = pd.Series(x_subset).interpolate(method='linear', limit_direction='both').values
    x_subset[np.where(np.isnan(x_subset))] = np.mean(x_subset[~np.isnan(x_subset)])

    wx_og, periods, scales_og, p_values, sig_mat = ssq_cwt_sig(x_subset, wavelet, fs, n = 1000, sig_lvl = sig_lvl)

    lower_ind = np.argmin(np.abs(periods - l_period))
    upper_ind = np.argmin(np.abs(periods - u_period))

    sig_vector = np.any(sig_mat[lower_ind:upper_ind], axis=0)
    #p_inds = np.argmin(np.abs(periods - l_period)) + 1, np.argmin(np.abs(periods - u_period)) + 1
    #suppression_mask = np.full((wx_og.shape[0], wx_og.shape[1]), np.nan)

    factor_mask = np.ones((wx_og.shape[0], wx_og.shape[1]), dtype='float')

    mask = (p_values < sig_lvl) & (np.arange(p_values.shape[0])[:, None] >= lower_ind) & (np.arange(p_values.shape[0])[:, None] <= upper_ind)
    max_indices = np.argmax(mask[::-1], axis=0)
    max_indices = mask.shape[0] - max_indices - 1  # Convert from reverse index

    mask[:, :] = False
    for j in range(p_values.shape[1]):
        if max_indices[j] > 0:
            mask[max_indices[j]:, j] = True
        else:
            mask[upper_ind:, j] = True

    for j in range(mask.shape[1]):
        indices = np.where(~mask[:, j])[0]
        factor_mask[indices, j] = normalize_vector(indices) ** exp

    wx_filt = wx_og * factor_mask
    rec = ssq.icwt(wx_filt, wavelet, scales_og)
    rec_rescale = (rec / np.std(rec)) * np.std(x_subset) + np.mean(x_subset)
    rec_rescale[ind_rm] = np.nan

    return(rec_rescale, sig_vector)


#LIU DLM FUNCTIONS

class Prior:
    def __init__(self, m, C, S, nu): 
        self.m = m # mean of t-distribution 
        self.C = C # scale matrix of t-distribution
        self.S = S # precision ~ IG(nu/2,S*nu/2)
        self.nu = nu # degree of freedom

class Model:
    def __init__(self,Y,X,rseas,deltas):
        self.Y = Y
        self.X = X
        self.rseas = rseas
        dd = deltas
        self.deltas = dd
        ntrend = 2;nregn = X.shape[1]; pseas = len(rseas);nseas = pseas*2;
        m = np.zeros([ntrend+nregn+nseas,1])
        C = scipy.linalg.block_diag(1*np.eye(ntrend),1*np.eye(nregn),1*np.eye(nseas))
        S = np.power(0.2,2); nu = ntrend+nregn+pseas;
        pr = Prior(m,C,S,nu)
        self.prior = pr
        
        
def forwardFilteringM(Model, fs):
    # All the parameters estimated here correspond Eqs. 13-16 and the related ones in the Supplementary Information of Liu et al. (2019)
    # notation in the code -> notation in Liu et al., 2019: 
    # m -> m_t; C -> C_t^{**}; nu -> n_t; 
    # a -> a_t; R -> R_t^{**}; F -> F_t; e -> e_t; y -> y_t; Q -> q_t^{**}; f -> f_t; S -> s_t = d_t/n_t
    
    Y = Model.Y
    X = Model.X
    rseas = Model.rseas
    delta = Model.deltas
    Prior = Model.prior
    period = 365.25/fs
    deltrend = delta[0];delregn = delta[1];delseas = delta[2];delvar = delta[3]
    Ftrend = np.array([[1],[0]]);ntrend = len(Ftrend); Gtrend = np.array([[1,1],[0,1]]);itrend = np.arange(0,ntrend)
    nregn = X.shape[1];Fregn = np.zeros([nregn,1]);Gregn=np.eye(nregn);iregn = np.arange(ntrend,ntrend+nregn)
    pseas = len(rseas);nseas = pseas*2;iseas = np.arange(ntrend+nregn,ntrend+nregn+nseas)
    Fseas = np.tile([[1],[0]],[pseas,1]);Gseas = np.zeros([nseas,nseas]);
    for j in range(pseas):
        c = np.cos(2*np.pi*rseas[j]/period);
        s = np.sin(2*np.pi*rseas[j]/period);
        i = np.arange(2*j,2*(j+1))
        Gseas[np.reshape(i,[2,1]),i] = [[c,s],[-s,c]]
    F = np.concatenate((Ftrend,Fregn,Fseas),axis=0)
    G = scipy.linalg.block_diag(Gtrend,Gregn,Gseas) 
    m = Prior.m; C = Prior.C; S = Prior.S; nu = Prior.nu

    T = len(Y)
    sm = np.zeros(m.shape)
    sC = np.zeros([C.shape[0],C.shape[1],1])
    sS = np.zeros(1)
    snu = np.zeros(1)
    slik = np.zeros(1)
    for t in range(T):
        a = np.dot(G,m)
        R = np.dot(np.dot(G,C),np.transpose(G))
        R[np.reshape(itrend,[-1,1]),itrend] = R[np.reshape(itrend,[-1,1]),itrend]/deltrend
        R[np.reshape(iregn,[-1,1]),iregn] = R[np.reshape(iregn,[-1,1]),iregn]/delregn
        R[np.reshape(iseas,[-1,1]),iseas] = R[np.reshape(iseas,[-1,1]),iseas]/delseas
        nu = delvar*nu
        F[iregn,0] = X[t,]

        A = np.dot(R,F);Q = np.squeeze(np.dot(np.transpose(F),A)+S); A = A/Q; f = np.squeeze(np.dot(np.transpose(F),a))
        y = Y[t]
        
        if ~np.isnan(y):
            e = y-f; ac = (nu+np.power(e,2)/Q)/(nu+1)
            rQ = np.sqrt(Q)
            mlik = tdstr.pdf(e/rQ,nu)/rQ
            m = a+A*e; C = ac*(R-np.dot(A,np.transpose(A))*Q); nu = nu+1; S = ac*S; 
            # About "S = ac*S" (using the notations in Liu et al. (2019)): 
            # s_t = d_t/n_t = (d_{t-1}+e_t^2/(q_t^{**}/s_t))/n_t = s_{t-1} * (n_{t-1}+e_t^2/(q_t^{**})/n_t = ac * s_{t-1}
        else:
            m = a; C = R;
            if t<T-1:
                X[t+1,0] = f
            mlik = np.nan
        sm = np.concatenate((sm,m),axis=1)
        sC = np.concatenate((sC,np.reshape(C,[C.shape[0],C.shape[1],1])),axis=2)
        snu = np.concatenate((snu,[nu]),axis=0)
        sS = np.concatenate((sS,[S]),axis=0)
        slik = np.concatenate((slik,[mlik]),axis=0)  
    return {'sm':sm, 'sC':sC ,'snu':snu,'slik':slik} 

def computeAnormaly(CLM,AvgCLM,date0):
    deltaT = timedelta(days=16)
    anCLM = np.zeros([1,CLM.shape[1]])
    for i in range(CLM.shape[0]):
        st = date0+deltaT*(i); st = st.timetuple().tm_yday
        et = date0+deltaT*(i+1); et = et.timetuple().tm_yday               
        if et<st:
            window = np.concatenate((np.arange(st,365),np.arange(0,et)))
        else:
            window = np.arange(st,et)
        window[window==365] = 0  # leap year
        anCLM = np.concatenate((anCLM,np.reshape(CLM[i,:]- np.mean(AvgCLM[window,:],axis = 0),[1,CLM.shape[1]])),axis=0)
    return anCLM[1:,:]

def Index_low(nn,date0,percentile):
    intervel = 16
    date0_num = date0.toordinal()
    dd = np.arange(date0,date0+timedelta(days=intervel)*len(nn),timedelta(days=intervel))
    dd_num = np.arange(date0_num,date0_num+intervel*(len(nn)),intervel)
    idq = [i for i in range(len(nn)) if np.isfinite(nn[i])] 
    tt1_num = np.arange(dd_num[idq[0]],dd_num[idq[-1]],1)
    f_itp = interp1d(dd_num[idq], nn[idq],kind = 'linear')
    nn_itp = f_itp(tt1_num)
    
    yday = np.array([date.fromordinal(tt1_num[i]).timetuple().tm_yday for i in range(len(tt1_num))])
    
    ndvi_mean = np.array([np.mean(nn_itp[yday==i]) for i in range(1,366)])
    ndvi_std = np.array([np.std(nn_itp[yday==i]) for i in range(1,366)])
    if len(ndvi_mean)==365:
        ndvi_mean = np.concatenate((ndvi_mean,[ndvi_mean[-1]]),axis = 0)
        ndvi_std = np.concatenate((ndvi_std,[ndvi_std[-1]]),axis = 0)
    
    tt2 = np.arange(dd[0],dd[-1],timedelta(days=1))
    tt2_num = np.arange(dd_num[0],dd_num[-1],1)
    yday2 = np.array([date.fromordinal(tt2_num[i]).timetuple().tm_yday for i in range(len(tt2_num))])
    nv = norm.ppf(1-(1-percentile)/2)
    lowboundary = np.array([ndvi_mean[yday2[i]-1]-nv*ndvi_std[yday2[i]-1] for i in range(len(tt2))])
    
    index_low = [i for i in range(len(dd)) if (~np.isnan(nn[i])) and (nn[i]<lowboundary[tt2_num==dd_num[i]])] 
    return index_low

def PlotEWS(N,date0,sm,sC,snu):
    # thresholds for identification of abnormally high autocorrelation (EWS)  
    quantile1 = 0.90
    quantile2 = 0.70
    
    steps = [date0+relativedelta(days=16*i) for i in range(len(N))]
    lown = Index_low(N,date0,0.8)
    lown_continuous = []
    for i in range(len(lown)):
        tile = [j for j in lown if (j<=lown[i] and j>=lown[i]-5)] 
        if len(tile)>2: 
            #NDVI being abnormally low fro more than half of the time within 3 mon
            lown_continuous = np.concatenate([lown_continuous,[lown[i]]])
            lown_continuous = np.array(lown_continuous).astype(int)
    tmp = np.array([steps[i] for i in lown_continuous])
    diebackdate = tmp[0]
    
    steps = np.array(steps)
    
    xpos = datetime(1996,1,1)
    xtick = [datetime(2000,1,1)+relativedelta(years=2*i) for i in range(0,9)]
    
    plt.figure(figsize=(7, 8))
    ax1 = plt.subplot(211)
    
    xlim = [datetime(1999,1,1),datetime(2016,7,1)]
    ylim = [0.3,0.85]
    ax1.plot(steps,N,'o-k',markersize=5,label='NDVI')
    ax1.plot(steps[lown_continuous],N[lown_continuous],'or',label='ALN')
    ax1.axvspan(datetime(2007,1,1), datetime(2010,1,1), color='brown', alpha=0.1, lw=0)
    ax1.axvspan(datetime(2011,1,1), datetime(2016,1,1), color='brown', alpha=0.1, lw=0)
    
    xtick = [datetime(2000,1,1)+relativedelta(years=2*i) for i in range(0,9)]
    ax1.set_xticks(xtick,('00','02','04','06','08','10','12','14','16'))
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax1.set_ylabel('NDVI')
    ax1.legend(loc = 'lower left',ncol=2)
    ax1.text(xpos, 0.82, '(a)',fontsize=20)
    ax1.set_xticks(xtick)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    warmup = 47
    bd = list(map(lambda m,C,nu: m+np.sqrt(C)*tdstr.ppf(quantile1,nu),sm,sC,snu))
    bd2 = list(map(lambda m,C,nu: m+np.sqrt(C)*tdstr.ppf(quantile2,nu),sm,sC,snu))
    
    mbd = np.median(bd2[warmup:])
    ews = np.array([i for i,im in enumerate(sm) if im >mbd])
    ews = ews[ews>warmup]
    ews_continuous = []
    window = int(90/16) # three months
    
    for i in range(len(ews)):
        tile = [j for j in ews if (j<=ews[i] and j>=ews[i]-window)]
        if len(tile)>window-1:
            ews_continuous = np.concatenate([ews_continuous,[ews[i]]])
    ews_continuous = np.array(ews_continuous).astype(int)
    tmp = steps[ews_continuous]
    ewsdate = tmp[tmp>datetime(2012,7,15)][0]
    mortdate = datetime(2015,7,15)
    arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth=10)
    ax2 = plt.subplot(212)
    ylim = [-0.5,0.7]
    ax2.plot(steps[1:], sm[1:], lw=2, label='mean')
    ax2.fill_between(steps, 2*sm-bd, bd, facecolor='0.7',label=str(int(quantile1*100))+'% range')
    ax2.fill_between(steps, 2*sm-bd2, bd2, facecolor='0.5',label=str(int(quantile2*100))+'% range')
    ax2.plot([steps[warmup],steps[-1]],[mbd,mbd],'--',color='0.4')
    ax2.plot(steps[ews_continuous],sm[ews_continuous],'^r',markersize=3,label='EWS')
    ax2.axvspan(datetime(2007,1,1), datetime(2010,1,1), color='brown', alpha=0.1, lw=0)
    ax2.axvspan(datetime(2011,1,1), datetime(2016,1,1), color='brown', alpha=0.1, lw=0)
    ax2.set_xlim(xlim)
    ax2.set_xticks(xtick)
    ax2.set_xticklabels(('00','02','04','06','08','10','12','14','16'))
    ax2.set_ylim(ylim)
    ax2.set_ylabel('Autocorrelation')
    ax2.set_xlabel('Year')
    yend = -0.28
    ft = 14
    hshift = relativedelta(months=3)
    ax2.text(mortdate-hshift, 0.05, 'mortality',rotation='vertical',fontsize=ft)
    ax2.text(diebackdate-hshift, 0.00, 'dieback',rotation='vertical',fontsize=ft)
    ax2.text(ewsdate-hshift, -0.12, 'EWS',rotation='vertical',fontsize=ft)
    
    ax2.annotate('', xy=(mortdate, ylim[0]), 
                xytext=(mortdate, yend),arrowprops=arrowprops)
    ax2.annotate('', xy=(diebackdate, ylim[0]), 
                xytext=(diebackdate, yend),arrowprops=arrowprops)
    ax2.annotate('', xy=(ewsdate, ylim[0]), 
                xytext=(ewsdate, yend),arrowprops=arrowprops)
    ax2.legend(loc=9,ncol=2)
    ax2.text(xpos, 0.63, '(b)',fontsize=20)

def calDrift(ts,window):
    qv = (np.nanmax(ts)-np.nanmin(ts))/20
    delta = (np.nanmax(ts)-np.nanmin(ts))/10
    trd_diffusion = (np.nanmax(ts)-np.nanmin(ts))/20
    drift = np.zeros([len(ts)-window-1,1])
    
    for i in range(drift.shape[0]):
        subn = ts[i:i+window]
        DX = subn[1:]-subn[:-1]
        X1 = subn[:-1]
        xq = np.arange(np.nanmin(ts)-0.1*abs(np.nanmin(ts)),np.nanmax(ts)+0.1*abs(np.nanmax(ts)),qv)
        mx = np.zeros(xq.shape)+np.nan
        mx2 = np.zeros(xq.shape)+np.nan
        for j in range(len(xq)):
            tmp = DX[np.abs(X1-xq[j])<delta]
            if len(tmp)>0:
                mx[j] = np.nanmedian(tmp)
                mx2[j] = np.nanmedian(tmp**2)
        idx = [j for j,x in enumerate(mx2) if x<trd_diffusion]
        idx = [j for j in idx if np.isfinite(mx[j])] # remove nan
        if np.size(idx):            
            drift[i] = 1+np.polyfit(xq[idx],mx[idx],1)[0] # slope of <D(x)_1> v.s. x
        else:
            drift[i] = 0
    return drift

def calEmprAC(nn,window):
    empr_ac = np.zeros([len(nn)-window-1,1])
    for i in range(len(empr_ac)):
        t1 = nn[i:i+window]
        t2 = nn[i+1:i+window+1]
        idx = [j for j in range(len(t1)) if np.isfinite(t1[j]+t2[j])]
        if np.size(idx):
            empr_ac[i] = np.corrcoef(t1[idx],t2[idx])[0,1]
        else:
            empr_ac[i] = 0
    return empr_ac

def calEmprVar(nn,window):
    return np.asarray([np.nanstd(nn[i:i+window]) for i in range(len(nn)-window)])


def plotThFig(FF,X,meanX):
    warmup = int(24*1.5)
    quantile = 0.90
    vid = 2

    sm = FF.get('sm')[vid,:]
    sC = np.sqrt(FF.get('sC')[vid,vid,:]) # std
    snu = FF.get('snu')
    bd = list(map(lambda m,C,nu: m+C*tdstr.ppf(quantile,nu),
                      sm,sC,snu))
    mbd = np.median(bd[warmup:])
    ews = np.array([i for i,im in enumerate(sm) if im >mbd])
    ews = ews[ews>warmup]
    ews_continuous = []
    for i in range(len(ews)): # if ews lasts > 3 mon
        tile = [j for j in ews if (j<=ews[i] and j>=ews[i]-5)]
        if len(tile)>4:
            ews_continuous = np.concatenate([ews_continuous,[ews[i]]])
    ews_continuous = np.array(ews_continuous).astype(int)

    xpos = -220
    xlim = [-0.05*len(X),len(X)*1.05]
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(311)
    ax1.plot(X+meanX,'-k')
    ax1.set_ylim([0.05,1.05])
    ax1.set_xlim(xlim)
    ax1.set_ylabel('yt')
    ax1.text(xpos, 0.95, '(a)',fontsize=20)
    
    steps = np.arange(len(sm))
    ax2 = plt.subplot(3,1,2)
    ax2.plot(steps,sm,lw=2, label='mean')
    ax2.fill_between(steps,2*sm-bd, bd, facecolor='0.7',label=str(int(quantile*100))+'% range')
    ax2.plot([warmup,len(sm)],[mbd,mbd],'--',color='0.4')
    ax2.plot(steps[ews_continuous],sm[ews_continuous],'^r',markersize=3,label='EWS')
    
    ax2.set_ylim([-0.2,1.2])
    ax2.set_xlim(xlim)
    ax2.set_ylabel('DLM autocorr')
    ax2.legend(loc=9,ncol=3)
    ax2.text(xpos, 1.1, '(b)',fontsize=20)
    
    window = 50
    emprAc = calEmprAC(X,window)
    emprVar = calEmprVar(X,window)
    Drift = calDrift(X,window)
    
    ax3 = plt.subplot(3,1,3)
    ax4 = ax3.twinx()
    nv = norm.ppf(0.8)
    l1 = ax3.plot(steps[window+1:],emprAc,label='autocorr')
    ax3.plot([steps[window],steps[-1]],np.nanmean(emprAc)+nv*np.nanstd(emprAc)*np.array([1,1]),'--',color=l1[0].get_color())
    l2 = ax3.plot(steps[window+1:],Drift,label='drift')
    ax3.plot([steps[window],steps[-1]],np.nanmean(Drift)+nv*np.nanstd(Drift)*np.array([1,1]),'--',color=l2[0].get_color())
    
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('autocorr/drift')
    ax3.set_xlim(xlim)
    ax3.set_ylim([-0.3,1.05])
    l3 = ax4.plot(steps[window:],emprVar,'-g',label='std')
    ax4.plot([steps[window],steps[-1]],np.nanmean(emprVar)+nv*np.nanstd(emprVar)*np.array([1,1]),'--',color=l3[0].get_color())
    ax4.set_ylabel('std')
    ax4.set_ylim([0.03,0.22])
    
    leg3 = ax3.legend(bbox_to_anchor=(0.48, -0.28),ncol=2,)
    leg4 = ax4.legend(bbox_to_anchor=(0.7, -0.28))
    leg3.get_frame().set_linewidth(0.0)
    leg4.get_frame().set_linewidth(0.0)
    ax3.text(xpos, 0.95, '(c)',fontsize=20)
    
def plotThFig2(FF,X,meanX):
    warmup = int(24*1.5)
    quantile = 0.90

    xpos = -220
    xlim = [-0.05*len(X),len(X)*1.05]
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(311)
    ax1.plot(X+meanX,'-k')
    ax1.set_ylim([0.05,1.05])
    ax1.set_xlim(xlim)
    ax1.set_ylabel('yt')
    ax1.text(xpos, 0.95, '(a)',fontsize=20)
    
    vid = 0
    sm = FF.get('sm')[vid,:]
    sC = np.sqrt(FF.get('sC')[vid,vid,:]) # std
    snu = FF.get('snu')
    bd = list(map(lambda m,C,nu: m+C*tdstr.ppf(quantile,nu),
                      sm,sC,snu))
    steps = np.arange(len(sm))
    ax2 = plt.subplot(3,1,2)
    ax2.plot(steps,sm,lw=2, label='mean')
    ax2.fill_between(steps,2*sm-bd, bd, facecolor='0.7',label=str(int(quantile*100))+'% range')
    ax2.set_ylim([-0.22,0.22])
    ax2.set_xlim(xlim)
    ax2.set_ylabel('DLM local mean')
    ax2.legend(loc=1,ncol=1)
    ax2.text(xpos, 0.21, '(b)',fontsize=20)
    
    
    vid = 2
    sm = FF.get('sm')[vid,:]
    sC = np.sqrt(FF.get('sC')[vid,vid,:]) # std
    snu = FF.get('snu')
    bd = list(map(lambda m,C,nu: m+C*tdstr.ppf(quantile,nu),
                      sm,sC,snu))
    mbd = np.median(bd[warmup:])
    ews = np.array([i for i,im in enumerate(sm) if im >mbd])
    ews = ews[ews>warmup]
    ews_continuous = []
    for i in range(len(ews)): # if ews lasts > 3 mon
        tile = [j for j in ews if (j<=ews[i] and j>=ews[i]-5)]
        if len(tile)>4:
            ews_continuous = np.concatenate([ews_continuous,[ews[i]]])
    ews_continuous = np.array(ews_continuous).astype(int)
    
    ax3 = plt.subplot(3,1,3)
    ax3.plot(steps,sm,lw=2, label='mean')
    ax3.fill_between(steps,2*sm-bd, bd, facecolor='0.7',label=str(int(quantile*100))+'% range')
    ax3.plot([warmup,len(sm)],[mbd,mbd],'--',color='0.4')
    ax3.plot(steps[ews_continuous],sm[ews_continuous],'^r',markersize=3,label='EWS')
    ax3.set_ylim([-0.2,1.2])
    ax3.set_xlim(xlim)
    ax3.set_ylabel('DLM autocorr')
    ax3.legend(loc=9,ncol=3)
    ax3.text(xpos, 1.1, '(c)',fontsize=20)
    ax3.set_xlabel('Time step')
    

