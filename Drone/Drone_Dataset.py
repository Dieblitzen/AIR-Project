import sys
sys.path.append('.')
import os
import argparse
from Dataset import Dataset
from ImSeg.ImSeg_Dataset import ImSeg_Dataset

class Drone_Dataset(ImSeg_Dataset):
  def __init__(self, data_path, classes_path='classes.json'):
    super().__init__(data_path, classes_path=classes_path)

    self.meta_path = os.path.join(self.data_path, 'metadata')
    Dataset._create_dirs(self.meta_path)

  
  def create_inference_set(self, set_type):
    """
    Helper:
    Creates quick inference set by copying images in images/ dir to
    dir in im_seg specified by set_type.
    """
    set_path = self.train_path
    if set_type.find("val") > -1:
      set_path = self.val_path
    elif set_type.find("test") > -1:
      set_path = self.test_path
    
    dest_dir = os.path.join(set_path, 'images')
    for i, im_path in enumerate(self.img_list):
      self.format_image(os.path.join(self.images_path, im_path), 
                        os.path.join(dest_dir, f'{i}.jpg'))


def passed_arguments():
  parser = argparse.ArgumentParser(description=\
    "Script to create inference set from already created Drone Dataset.")
  parser.add_argument("-d", "--data_path",
                      type=str,
                      required=True,
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument("-s", "--set_type",
                      type=str,
                      default="val",
                      help="Type of train/val/test set you want to initialise.")
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()
  ds = Drone_Dataset(args.data_path)
  ds.create_inference_set(args.set_type)