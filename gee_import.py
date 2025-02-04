import ee


class Profiler():
  #marko code
  def __init__(self, dataset, scale):
    self.dataset = dataset
    self.scale = scale

    self.collection = ee.ImageCollection(self.dataset)

  def get_profile(self, pnt):
    '''Overide this function'''
    def profile_func(image):
      pass

    return profile_func

  def profile(self, point, from_date, to_date):
    collection = (
        self.collection
        .filterDate(from_date, to_date)
        .filterBounds(point)
        # More...
    )
    map_func = self.get_profile(point)
    profile_data = collection.map(map_func)

    return profile_data.getInfo()

  def get_name(self):
    return self.dataset

class Mod09gq_profiler(Profiler): # daily reflectance
#Markos Code
  def __init__(self, scale=250):
    super().__init__('MODIS/006/MOD09GQ', scale)

  def get_profile(self, pnt):
    def profile_func(image):
      ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']) # this is your NDVI calculation
      quality = image.select('QC_250m')
      q_reduced = quality.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      reduced = ndvi.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      feat = ee.Feature(pnt, {
          'ndvi': reduced.get('nd'),
          'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
          'qa': q_reduced.get('QC_250m')
          })
      return feat

    return profile_func

class Mod13Q1_profiler(Profiler): #this is the 2 week NDVI modis data
#Markos code
  def __init__(self, scale=250):
    super().__init__('MODIS/006/MOD13Q1', scale)

  def get_profile(self, pnt):
    def profile_func(image):
      ndvi = image.select('NDVI').multiply(0.0001)
      reduced = ndvi.reduceRegion(ee.Reducer.mean(), pnt, self.scale)
      feat = ee.Feature(pnt, {
          'ndvi': reduced.get('NDVI'),
          'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
          })
      return feat

    return profile_func
