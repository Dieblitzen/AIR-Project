import sys
sys.path.append('.')
import os
from Dataset import Dataset

class Drone_Dataset(Dataset):
  def __init__(self, data_path, classes_path='classes.json'):
    super().__init__(data_path, classes_path=classes_path)

    self.meta_path = os.path.join(self.data_path, 'metadata')
    Dataset._create_dirs(self.meta_path)

  pass