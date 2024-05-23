import pandas as pd
import numpy as np

def get_greenup(x_time, period=[0, 366], norm=False):
  """Returns index of greenup phenological date from 
  multiyear time-series profile.

  The greenup (or *start of season*)
  defined as the start of the *land surface phenology*
  profile. The function returns the index of the maximum
  peak in curvature for a specific period for each year 
  (e.g. spring), using the day-of-the-year to set the 
  thresholds.

  Arguments:
    x_time - a pandas series with datetime index
    period - day-of-year period for search
  """
  x = pd.DataFrame(x_time.copy(), columns=['value'])
  x.sort_index(inplace=True)
  x['year'] = x.index.year
  
  x['grad'] = np.gradient(x['value'].values)
  x['grad2'] = np.gradient(x['grad'])
  x['curve'] =  x.eval('grad2 * (1 + grad**2)**-1.5')

  t, f = period
  if t<f:
    sub = x[(x.index.dayofyear>t) & (x.index.dayofyear<f)]
  else:
    sub = x[(x.index.dayofyear>t) | (x.index.dayofyear<f)]

  return sub.groupby('year')['curve'].idxmax().values