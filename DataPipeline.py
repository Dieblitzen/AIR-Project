## This is the superclass for the dataset generation pipeline
# 


class DataPipeline:
  """ 
  DataPipeline fetches image and bounding box data from the source APIs in pixel format.
  Minimal processing is done.
  """

  
  def __init__(self, coords, source, download_path='./downloads'):
    """
    Constructor takes [coords] as an array of 4 latitude and longitude coordinates in 
    the following format [[LAT_MIN, LON_MIN, LAT_MAX, LON_MAX]]
    Constructor also takes the [source] of the data (eg. IBM, Google, etc.)
    Constructor takes the location
    """
    self.coordinates = coords
    self.source = source
    self.download_path = download_path
  

  def query_image(self):
    """  """
    pass
  
  def query_OSM(self):
    pass

  def image_to_array(self):
    pass

  def remove_indices(self, indices_to_remove):
    pass

  def coords_to_pixels(self):
    pass
  
  def visualize_data(self):
    pass
  
  def save_data(self):
    pass
  
  def create_bbox(self):
    pass
  
  def boxes_in_tile(self):
    pass
  
  def tile_image(self):
    pass
  


