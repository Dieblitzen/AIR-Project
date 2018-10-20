#Before importing api_wrapper, had to install some dependencies:
  #pip install geopandas, geojson, coloredlogs
  #conda install gdal

import os
import numpy as np
import geojson
from pytz import UTC
import requests
from requests.auth import HTTPBasicAuth
import zipfile, io
from osgeo import gdal

# Config
PAIRS_USER = "ams792@cornell.edu"
PAIRS_PASS = "myibmpairspassword"
server = "https://pairs.res.ibm.com"
auth = (PAIRS_USER, PAIRS_PASS)
path = './downloads' #Download path

lat1 = 41.0100756423
lat2 = 41.0338409682
lon1 = -73.7792749922
lon2 = -73.7582464736

response = requests.post(
    json = {
    "layers" : [
        {
            "type": "raster",
            "id": 61
        },
        {
            "type": "raster",
            "id": 48695
        },
#         {
#             "type": "raster",
#             "id": 36432
#         },
#         {
#             "type": "raster",
#             "id": 49104
#         }
    ],
    "spatial": {
        "type": "square",
        "coordinates": ["41.0100746", "-73.779274", "41.0338402", "-73.7582474"]
    },
    "temporal": {
        "intervals": [{"start": "2014-12-31T00:00:00Z","end": "2015-01-01T00:00:00Z"}]
    }
},
    url=f'{server}/v2/query',
    auth=auth,
)

res = response.json()
id = res['id']
print("ID: ",id)

print(response)

# Check status of query, make sure it's finished before downloading
response = requests.get(
    url=f'{server}/v2/queryjobs/{id}',
    auth=auth,
)
response.json()

#Download and extract to files
download = requests.get(
    f'{server}/download/{id}', auth=auth, stream=True,
)

z = zipfile.ZipFile(io.BytesIO(download.content))

## Extract to ./downloads
z.extractall(path)

#Convert to np array
## Takes file name and converts to np array
def tiff2array(fn):
    ## Special gdal dataset
    dataset = gdal.Open(fn)
    array = np.array(dataset.GetRasterBand(1).ReadAsArray())
    return array

## Will go through filenames and put the image arrays in here
images_dict = {}

## Loop through files in downloads directory (if multiple)
for i, filename in enumerate(os.listdir(path)):
    if filename.endswith(".tiff"): 
        path_to_file = path + '/' + filename
        images_dict[i] = tiff2array(path_to_file)

## Note: keys start at 1 for image_dict

print(images_dict)

res = requests.get(f'{server}/v2/datalayers/',auth=auth)

print(res.json())