import pandas as pd
import ee

class Profiler():
  """Super/Parent class for time-series profiling using Google
  Earth Engine.
  
  Override `get_profile()` for dataset specific profiling.
  """
  def __init__(self, dataset, scale):
    self.dataset = dataset
    self.scale = scale

    self.collection = ee.ImageCollection(self.dataset)

  def get_profile(self, pnt):
    """Override me"""
    def profile_func(image):
      pass

    return profile_func

  def profile(self, point, from_date, to_date, filter=None):
    collection = (
        self.collection
        .filterDate(from_date, to_date)
        .filterBounds(point)
        # More...
    )
    map_func = self.get_profile(point)
    profile_data = collection.map(map_func)

    if filter:
      profile_data = profile_data.filter(filter)

    return profile_data.getInfo()

  def get_name(self):
    return self.dataset

  def expand_to_df(self, dict_of_feats):
    """Expand key-value pairs to columns"""
    df = pd.DataFrame(dict_of_feats)
    cols = [i for i in df.columns if isinstance(df[i][0], dict)]
    for col in cols:
      df = pd.concat([df.drop([col], axis=1), df[col].apply(pd.Series)], axis=1)
    return df

  def to_df(self, response):
    data = self.expand_to_df(response['features'])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    return data
  
class Mod09gq_profiler(Profiler):
  """
  MOD09GQ.061 Terra Surface Reflectance Daily Global 250m
  https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GQ
  """

  def __init__(self, scale=250):
    super().__init__('MODIS/061/MOD09GQ', scale)

  def get_profile(self, pnt):
    def profile_func(image):
      ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
      quality = image.select('QC_250m')

      # Reduce for single pixel
      q_reduced = quality.reduceRegion(ee.Reducer.max(), pnt, self.scale)
      bits = q_reduced.get('QC_250m')
      
      reduced = ndvi.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      feat = ee.Feature(pnt, {
          'ndvi': reduced.get('nd'),
          'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
          'qc_250m': bits
          })
      return feat

    return profile_func
  
  def to_df(self, response):
    """Overridden"""
    data = self.expand_to_df(response['features'])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    data['qc_250m'] = data['qc_250m'].apply(
        lambda x: format(int(x), '016b') if pd.notnull(x) else x
        )

    flags = {
        'band_1_q': [
            -8, -4,
            {},
            ],
        'band_2_q': [
            -12, -8,
            {},
            ],
        }

    for param in flags.keys():
      f, t, lu = flags[param]
      data[param] = data['qc_250m'].str[f:t]#.map(lu)
    
    return data


class Mod13Q1_profiler(Profiler):
  """
  MOD13Q1.061 Terra Vegetation Indices 16-Day Global 250m
  https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1
  """

  def __init__(self, scale=250):
    super().__init__('MODIS/061/MOD13Q1', scale)

  def get_profile(self, pnt):
    def profile_func(image):
      ndvi = image.select('NDVI').multiply(0.0001)
      reduced = ndvi.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      
      sysdate = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
      doy = image.select('DayOfYear')
      doy_m = doy.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      
      feat = ee.Feature(pnt, {
          'ndvi': reduced.get('NDVI'),
          'date': sysdate,
          'cdoy' : doy_m.get('DayOfYear')
          })
      return feat

    return profile_func

  def to_df(self, response):
    """Overridden"""
    data = self.expand_to_df(response['features'])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    data['cdate'] = pd.to_datetime(
        data.index.year*1000+ data['cdoy'],
        format='%Y%j'
        )

    return data

class Mod09ga_profiler(Profiler):
  """
  MOD09GA.061 Terra Surface Reflectance Daily Global 1km and 500m
  https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA
  """

  def __init__(self, scale=250):
    super().__init__('MODIS/061/MOD09GA', scale)

  def get_profile(self, pnt):
    def profile_func(image):
      ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
      quality = image.select('state_1km')

      # Reduce for single pixel
      q_reduced = quality.reduceRegion(ee.Reducer.max(), pnt, self.scale)
      bits = q_reduced.get('state_1km')
      
      reduced = ndvi.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      feat = ee.Feature(pnt, {
          'ndvi': reduced.get('nd'),
          'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
          'state_1km': bits
          })
      return feat

    return profile_func
  
  def to_df(self, response):
    '''Overridden'''
    data = self.expand_to_df(response['features'])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    data['state_1km'] = data['state_1km'].apply(
        lambda x: format(int(x), '016b') if pd.notnull(x) else x
        )
    
    flags = {
        'cloud_state': [
            -2, None,
            {'00': 'clear', '01': 'cloudy', '10': 'mixed', '11': 'unknown'},
            ],
        'cloud_shadow': [
            -3, -2,
            {'0': 'no', '1': 'yes'},
            ],
        'internal_cloud_flag': [
            -11, -10,
            {'0': 'no cloud', '1': 'cloud'},
            ],
        }

    for param in flags.keys():
      f, t, lu = flags[param]
      data[param] = data['state_1km'].str[f:t].map(lu)
    
    return data

class S2_sr_profiler(Profiler):

  def __init__(self, scale=10):
    super().__init__('COPERNICUS/S2_SR_HARMONIZED', scale)

  def get_profile(self, pnt):
    def profile_func(image):
      ndvi = image.normalizedDifference(['B8', 'B4'])
      reduced = ndvi.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      feat = ee.Feature(pnt, {
          'ndvi': reduced.get('nd'),
          'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
          })
      return feat

    return profile_func

class S1_grd_profiler(Profiler):

  def __init__(self, scale=10):
    super().__init__('COPERNICUS/S1_GRD', scale)

    collection = (
        self.collection
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        #.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    )
    self.collection = collection

  def get_profile(self, pnt):
    def profile_func(image):
      bs = image.select(['VV', 'VH'])
      reduced = bs.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      feat = ee.Feature(pnt, {
          'vv': reduced.get('VV'),
          'vh': reduced.get('VH'),
          'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
          })
      return feat

    return profile_func
