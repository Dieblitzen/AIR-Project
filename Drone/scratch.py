import gdal
import argparse
import numpy as np
from PIL import Image

TILE_SIZE = 224

def read_tiff_image(path_to_tiff):
  im = gdal.Open(path_to_tiff)
  im_arr = im.ReadAsArray()
  im_arr = im_arr.transpose(1, 2, 0)
  # print(im_arr)

  # Try and remove NaNs
  im_arr[np.isnan(im_arr)] = -128
  row_mask = (im_arr[..., 0] > -128).any(axis=1)
  clean_arr = im_arr[row_mask, :, :]
  col_mask = (clean_arr[..., 0] > -128).any(axis=0)
  clean_arr = clean_arr[:, col_mask, :]

  return clean_arr
  # clean_im = Image.fromarray(clean_arr).convert('RGB')
  # clean_im.save(path_to_tiff.replace('.tif', '.jpg'))

## Tile and save image
# Per-pixel resolution goal: 1 pix ~ 1 m
def tile_image(im_arr, out_path, resolution=3):
  print(im_arr.shape)

  # Tile according to expanded tile size
  res = int(100/resolution)
  size = res * TILE_SIZE
  print(size)

  tile_arr = im_arr[:size*100, :size*100, :]
  tile_im = Image.fromarray(tile_arr).convert('RGB')
  tile_im = tile_im.resize((TILE_SIZE, TILE_SIZE), resample=Image.BICUBIC)
  print(tile_im.size)
  tile_im.save(out_path)



def passed_arguments():
  parser = argparse.ArgumentParser(description="Temp script to create tiled drone data.")
  parser.add_argument("-p", "--im_path",
                      type=str,
                      required=True,
                      help="Path to image .tif file.")
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()
  im_arr = read_tiff_image(args.im_path)
  tile_image(im_arr, args.im_path.replace(".tif", ".jpg"), resolution=3)