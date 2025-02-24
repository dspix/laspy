from datetime import datetime
import numpy as np
import scipy.linalg
from scipy.stats import t as tdstr
from scipy.stats import norm
from scipy.interpolate import interp1d
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.rc('font', size=14)
import matplotlib.pyplot as plt


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
    return {'sm':sm, 'sC':sC ,'snu':snu,'slik':slik, 'sS': sS} 

def forwardFilteringM2(Model, fs):
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
    sac = np.zeros(1)
    se = np.zeros(1)
    snu = np.zeros(1)
    slik = np.zeros(1)
    predictions = np.zeros(T)
    explained_trend = np.zeros(T)
    explained_seasonal = np.zeros(T)
    explained_regressors = np.zeros((T, nregn))
    explained_lag1 = np.zeros(T)

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

        predictions[t] = f  # Store the one-step ahead prediction
        #ar1_contrib[t] = X[t, 0] * m[ntrend]

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
        sac = np.concatenate((sac,[ac]),axis=0)
        se = np.concatenate((se,[e]),axis=0)

        explained_trend[t] = np.dot(Ftrend.T, m[itrend])
        explained_seasonal[t] = np.dot(Fseas.T, m[iseas])
        explained_regressors[t, 0] = X[t, 0] * m[iregn[0]]
        explained_regressors[t, 1] = X[t, 1] * m[iregn[1]]
        
    explained_lag1 = explained_regressors[:, 0]
    residuals = Y - (explained_trend + explained_seasonal + explained_regressors.sum(axis=1))

    return {
        'sm': sm,
        'sC': sC,
        'snu': snu,
        'slik': slik,
        'explained_trend': explained_trend,
        'explained_seasonal': explained_seasonal, 
        'explained_regressors': explained_regressors,
        'predictions':predictions, 
        'residuals': residuals,
        'explained_lag1':explained_lag1,
        'sS': sS,
        'sac': sac,
        'se':se}

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

def run_dlm(N, anCLM, vid, fs, rseas, deltas, prior = None):
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
