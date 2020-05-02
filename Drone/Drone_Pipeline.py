import sys
sys.path.append('.')
import os
import json
import requests
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm
import concurrent.futures
from Drone.Drone_Dataset import Drone_Dataset
from DataPipeline import query_OSM, coords_to_pixels, boxes_in_tile


def save_tile_and_labels(tile_arr, tile_labels, out_index, dataset, resize=None):
  """
  Saves a single tile and the labels associated with that tile. Resizes the
  tile if specified, assumes that `tile_labels` coords won't need to be resized.\n
  Requires:\n
    tile_arr: numpy array denoting the specific tile.\n
    tile_labels: dictionary of label coords (in pixel value) associated with tile.\n
    out_index: named index of the tile/bbox to be saved\n
    resize: (h, w) in pixels defining target size of tiles
  """
  tile_im = Image.fromarray(tile_arr).convert('RGB')
  if resize:
    tile_im = tile_im.resize(resize, resample=Image.BILINEAR)
  tile_name = os.path.join(dataset.images_path, f"{out_index}.jpg")
  tile_im.save(tile_name)

  bbox_name = os.path.join(dataset.annotations_path, f"{out_index}.json")
  with open(bbox_name, 'w') as f:
    json.dump(tile_labels, f)


def read_tile(im_path, tile_range):
  """
  Reads a single tile from the `.tiff` file of the entire area corresponding
  to the specified `tile_range`. Cleans the tile and returns a numpy array. \n
  Requires:
    `path_to_im`: Path to `.tif` file of the entire area.
    `tile_range`: A list in the format `[col_start, col_end, row_start, row_end]`\n
  """
  col_start, col_end, row_start, row_end = tile_range
  window = Window.from_slices((row_start, row_end), (col_start, col_end))
  with rasterio.open(im_path) as im:
    im_arr = im.read(window=window)
  
  im_arr = im_arr.transpose(1, 2, 0)
  # Try and remove NaNs
  im_arr[np.isnan(im_arr)] = -128
  row_mask = (im_arr[..., 0] > -128).any(axis=1)
  clean_arr = im_arr[row_mask, :, :]
  col_mask = (clean_arr[..., 0] > -128).any(axis=0)
  clean_arr = clean_arr[:, col_mask, :]

  return clean_arr


def tile_and_annotate(dataset, path_to_im, path_to_meta, 
                      out_res=1, tile_size=(224, 224), overlap=0):
  """
  Tiles and saves an image. The tiles that are saved are resized to the intended
  resolution determined by `out_res`. The input resolution of the file (in metres) 
  is accessed using the `gsd` attribute of the metadata dictionary. \n
  Also queries OpenStreetMap (OSM) for labels in specified region based on the
  `bbox` attribute of the metadata dictionary. \n
  Requires:
    dataset: the dataset object this image is tied to \n
    path_to_im: path to file of entire image that is to be tiled \n
    path_to_meta: path to .json dictionary containing image's metadata 
      (including resolution as specified by the `gsd` attribute (in metres)) \n
    out_res: target per-pixel resolution, where 1 pixel ~ `out_res` meters \n
    tile_size: (h, w) in pixels defining target size of tiles \n
    overlap: Amount of overlapping pixels between adjacent tiles (after resizing)
  """
  with rasterio.open(path_to_im) as im:
    h, w = im.shape

  with open(path_to_meta, 'r') as f:
    im_meta = json.load(f)
  im_id = im_meta['_id']

  # Tile according to expanded tile size
  in_res = im_meta["gsd"]
  ratio = int(out_res/in_res)
  in_size = (ratio * tile_size[0], ratio * tile_size[1])
  overlap = ratio * overlap

  # Get step sizes for tiles based on overlap
  step_h, step_w = in_size[0] - overlap, in_size[1] - overlap

  # Get bounding box region of image as (lat_min, lon_min, lat_max, lon_max)
  bbox = im_meta['geojson']['bbox']
  coords = [bbox[1], bbox[0], bbox[3], bbox[2]]

  # Get OSM data, and convert from lat-lon to pixel coords (in terms of resized image)
  print(f"Querying OpenStreetMap data for labels...")
  raw_osm = query_OSM(coords, dataset.classes)
  label_coords = coords_to_pixels(raw_osm, coords, (h/ratio, w/ratio), 
                                  dataset.raw_data_path, out_file=f"{im_id}")
  
  # Tile up input high res image
  start = len(dataset)
  image_ind = start
  for row_start in range(0, h - step_h, step_h):
    for col_start in range(0, w - step_w, step_w):
      # row_start, row_end, col_start, col_end in pixels relative to entire img
      row_end, col_end = row_start + in_size[0], col_start + in_size[1]
      
      # Get the tile array and the labels in the (resized) tile
      tile_range = np.array([col_start, col_end, row_start, row_end])
      tile_arr = read_tile(path_to_im, tile_range)
      tile_labels = boxes_in_tile(label_coords, tile_range/ratio)

      # Save the tile and labels, resizing the tile to the `tile_size`
      save_tile_and_labels(tile_arr, tile_labels, image_ind, dataset, resize=tile_size)
      
      image_ind += 1

  end = len(dataset)

  # Save the metadata associated with the range of tiles
  if end - start > 0:
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
  and downloads the image metadata as a .json object \n
  Raises an exception if the query fails or if the metadata can't be parsed
  as a json dictionary. \n
  Requires: \n
    url: a string that is either an image id or is a URL containing the image id \n
  Returns: \n
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


def create_dataset(data_path, classes_path, query_url_path=None, overlap=0):
  """
  Creates a dataset of drone imagery (no annotations) given the directory path
  to store the data. If specified, will download OpenAerialMap imagery from 
  given a text file containing the image ids with which to create a dataset.
  Requires:
    data_path: Path to directory where extracted dataset is stored.
    query_url_path: Path to .txt file containing URLs or image ids of OpenAerialMap
                    drone images.
  """
  ds = Drone_Dataset(data_path, classes_path=classes_path)
  im_ext1, im_ext2 = ".tif", ".tiff"

  # Download .tiff files if specified.
  if query_url_path:
    with open(query_url_path, 'r') as f:
      download_urls = f.read().splitlines()
    
    for url in download_urls:
      if url.find("https") == -1: 
        continue
      print(f"\nDownloading image: {url}")
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
  
  # Query OSM, tile image and save them.
  for im_path, meta_path in zip(raw_im_paths, raw_meta_paths):
    print(f"\nTiling image: {im_path}")
    tile_and_annotate(ds, im_path, meta_path, overlap=overlap)
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
  parser.add_argument("-o", "--overlap", 
                      type=int, 
                      default=0,
                      help="Amount of overlapping pixels between adjacent tiles.")
  parser.add_argument("-c", "--classes", 
                      type=str, 
                      default=os.path.join(".", "classes.json"),
                      help="Path to json file determining OSM classes.")
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = passed_arguments()
  create_dataset(
    args.data_path, 
    classes_path=args.classes, 
    query_url_path=args.query_path,
    overlap=args.overlap
  )