import gdal
import argparse
import numpy as np
from PIL import Image

## Tile and save image
def tile_tiff_image(path_to_tiff):
  im = gdal.Open(path_to_tiff)
  im_arr = im.ReadAsArray()
  print(im_arr.shape)
  im_arr = im_arr.transpose(1, 2, 0)
  print(im_arr.shape)

  # Try and remove NaNs
  im_arr[np.isnan(im_arr)] = -128
  row_mask = (im_arr[..., 0] > -128).any(axis=1)
  clean_arr = im_arr[row_mask, :, :]
  col_mask = (clean_arr[..., 0] > -128).any(axis=0)
  clean_arr = clean_arr[:, col_mask, :]
  print(clean_arr.shape)

  clean_im = Image.fromarray(clean_arr).convert('RGB')
  clean_im.save(path_to_tiff.replace('.tiff', '.jpg'))


def passed_arguments():
  parser = argparse.ArgumentParser(description="Temp script to create tiled drone data.")
  parser.add_argument("-p", "--im_path",
                      type=str,
                      required=True,
                      help="Path to image .tiff file.")
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = passed_arguments()
  tile_tiff_image(args.im_path)