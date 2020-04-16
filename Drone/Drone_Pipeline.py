import os
import gdal
import json
import requests
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import concurrent.futures
from Drone_Dataset import Drone_Dataset


def read_image(path_to_im):
  """
  Reads an image file and converts it to a numpy array.
  If the file is a .tif file, then deletes the .tiff file, saves
  it as a .jpg instead and returns the numpy array.
  Requires:
    path_to_im: Path to .tif or .jpg image file.
  """
  if path_to_im.endswith(".tif") or path_to_im.endswith(".tiff"):
    im = gdal.Open(path_to_im)
    im_arr = im.ReadAsArray()
    im_arr = im_arr.transpose(1, 2, 0)

    # Try and remove NaNs
    im_arr[np.isnan(im_arr)] = -128
    row_mask = (im_arr[..., 0] > -128).any(axis=1)
    clean_arr = im_arr[row_mask, :, :]
    col_mask = (clean_arr[..., 0] > -128).any(axis=0)
    clean_arr = clean_arr[:, col_mask, :]

    file_end = ".tif" if path_to_im.endswith(".tif") else ".tiff"
    clean_im = Image.fromarray(clean_arr).convert('RGB')
    clean_im.save(path_to_im.replace(file_end, '.jpg'))

    os.remove(path_to_im)
  elif path_to_im.endswith(".jpg") or path_to_im.endswith(".jpeg"):
    im = Image.open(path_to_im).convert('RGB')
    clean_arr = np.array(im)

  return clean_arr


def tile_image(im_arr, im_meta, dataset, out_res=1, tile_size=(224, 224)):
  """
  Tiles and saves an image. The tiles that are saved are resized to the intended
  resolution determined by out_res. The input resolution of the file (in metres) 
  is accessed using the `gsd` attribute of the metadata dictionary.
  Requires:
    im_arr: numpy array of entire image to be tiled.
    im_meta: dictionary containing image's metadata (including resolution)
             as specified by the `gsd` attribute (in metres)
    dataset: the dataset object this image is tied to
    out_res: target per-pixel resolution, where 1 pixel ~ `out_res` meters
    tile_size: (h, w) in pixels defining target size of tiles
  """
  print(im_arr.shape)

  # Tile according to expanded tile size
  in_res = im_meta["gsd"]
  res = int(out_res/in_res)
  in_size = (res * tile_size[0], res * tile_size[1])
  
  # Tile up input high res image
  h, w, _ = im_arr.shape
  total_rows, total_cols = h//in_size[0], w//in_size[1]

  image_ind = len(dataset)
  for r in range(total_rows):
    for c in range(total_cols):
      # row_start, row_end, col_start, col_end in pixels relative to entire img
      row_start, row_end = r*in_size[0], (r+1)*in_size[0]
      col_start, col_end = c*in_size[1], (c+1)*in_size[1]
      tile_arr = im_arr[row_start:row_end, col_start:col_end, :]

      tile_im = Image.fromarray(tile_arr).convert('RGB')
      tile_im = tile_im.resize(tile_size, resample=Image.BICUBIC)
      tile_im.save(os.path.join(dataset.images_path, f"img_{image_ind}.jpg"))
      
      image_ind += 1


def tile_and_save(dataset, path_to_im, path_to_meta, out_res=1, tile_size=(224, 224)):
  """
  Tiles and saves an image and the metadata associated with the tiles.
  See `tile_and_save` for argument description.
  This method loads the image from its path as a numpy array, and loads the 
  metadata .json file as a dictionary.
  """
  im_arr = read_image(path_to_im)
  with open(path_to_meta, 'r') as f:
    im_meta = json.load(f)
  
  start = len(dataset)
  tile_image(im_arr, im_meta, dataset, out_res=out_res, tile_size=tile_size) 
  end = len(dataset)

  # Save the metadata associated with the range of tiles
  im_meta["image_indices"] = list(range(start, end))
  with open(os.path.join(dataset.meta_path, f"meta_{start}-{end-1}.json"), 'w') as f:
    json.dump(im_meta, f, indent=2)

  
def parse_image_url(url):
  """
  Parses a url continaing the image id to return the image id.
  """
  # Check if it was a url or actual image id
  # eg: https://map.openaerialmap.org/#/.../59e62b9a3d6412ef7220a51f?_k=vy2p83
  image_id = url.split("/")[-1].split("?")[0]
  return image_id


def query_image_metadata(url):
  """
  Given a url (either image id or URL containing image id), queries
  and downloads the image metadata as a .json object
  Raises an exception if the query fails or if the metadata can't be parsed
  as a json dictionary.
  Requires:
    url: a string that is either an image id or is a URL containing the image id
  Returns:
    A dictionary containing the metadata of the image.
  """
  image_id = parse_image_url(url)
  q_url = f"https://api.openaerialmap.org/meta?_id={image_id}"
  response = requests.get(q_url)

  if not response.ok: 
    raise RuntimeError(f"Query failed with status: {response.status_code}")

  try:
    meta = response.json()
  except ValueError:
    raise ValueError(f"Image metadata cannot be converted to .json dict")

  return meta


def download_tiff(url, out_path):
  """
  Given a url (either an image id or a url to a webpage describing the OAM image),
  downloads the image metadata information and saves the image as a .tiff file.
  Requires:
    url: a string that is either an image id or is a URL containing the image id
    out_path: path to dir where image + metadata will be stored
  """
  im_meta = query_image_metadata(url)

  # Pick first result
  im_meta = im_meta['results'][0]
  im_id = im_meta['_id']
  with open(os.path.join(out_path, f'{im_id}.json'), 'w') as f:
    json.dump(im_meta, f, indent=2)

  # Write tiff image
  tiff_url = im_meta['uuid']
  response = requests.get(tiff_url, stream=True)
  with open(os.path.join(out_path, f"{im_id}.tif"), 'wb') as f:
    for data in tqdm(response.iter_content(chunk_size=8192)):
      f.write(data)


def create_dataset(data_path, query_url_path=None):
  """
  Creates a dataset of drone imagery (no annotations) given the directory path
  to store the data. If specified, will download OpenAerialMap imagery from 
  given a text file containing the image ids with which to create a dataset.
  Requires:
    data_path: Path to directory where extracted dataset is stored.
    query_url_path: Path to .txt file containing URLs or image ids of OpenAerialMap
                    drone images.
  """
  ds = Drone_Dataset(data_path)
  im_ext1, im_ext2 = ".jpg", ".jpeg"

  # Download .tiff files if specified.
  if query_url_path:
    with open(query_url_path, 'r') as f:
      download_urls = f.read().splitlines()
    
    for url in download_urls:
      if url.find("https") == -1: 
        continue
      print(f"\nDownloading image: {url}\n")
      download_tiff(url, ds.raw_data_path)
      print("Done downloading image.\n")

    im_ext1, im_ext2 = ".tif", ".tiff"

  # Construct list of image paths and metadata paths.
  raw_im_paths, raw_meta_paths = [], []
  for f in os.listdir(ds.raw_data_path):
    if f.endswith(im_ext1) or f.endswith(im_ext2):
      raw_im_paths.append(os.path.join(ds.raw_data_path, f))
    elif f.endswith(".json"):
      raw_meta_paths.append(os.path.join(ds.raw_data_path, f))

  assert len(raw_im_paths) == len(raw_meta_paths),\
    "Number of images != number of image metadata files."
  
  # Tile image and save them.
  for im_path, meta_path in zip(raw_im_paths, raw_meta_paths):
    print(f"\nTiling image: {im_path}\n")
    tile_and_save(ds, im_path, meta_path)
    print(f"Done tiling image.\n")


def passed_arguments():
  parser = argparse.ArgumentParser(description="Script to create a drone image dataset.")
  parser.add_argument("-d", "--data_path",
                      type=str,
                      required=True,
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument("-q", "--query_path",
                      type=str,
                      default=None,
                      help="Path to .txt file containing urls of OpenAerialMap queries.")
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = passed_arguments()
  create_dataset(args.data_path, query_url_path=args.query_path)