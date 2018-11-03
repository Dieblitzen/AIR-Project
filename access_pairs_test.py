#Had to install some dependencies:
  #pip install geopandas, geojson, coloredlogs
  #conda install gdal

import os
import pandas as pd
import numpy as np
import geojson
from pytz import UTC
import requests
from requests.auth import HTTPBasicAuth
import zipfile, io
from osgeo import gdal
from time import sleep

# Config
PAIRS_USER = "ams792@cornell.edu"
PAIRS_PASS = "myibmpairspassword"
server = "https://pairs.res.ibm.com"
auth = (PAIRS_USER, PAIRS_PASS)
path = './downloads' #Download path

# Old: Cutoff region in Scarsdale
# lat1 = 41.0100756423
# lat2 = 41.0338409682
# lon1 = -73.7792749922
# lon2 = -73.7582464736

# New. No cutoff that we could see
lat_min, lon_min, lat_max, lon_max = 41.0155, -73.7792749922, 41.0338409682, -73.7582464736

response = requests.post(
    json = {
    "layers" : [
        {
            "type": "raster",
            "id": 36431
        },
        {
            "type": "raster",
            "id": 35445
        },
        {
            "type": "raster",
            "id": 36432
        },
    ],
    "spatial": {
        "type": "square",
        "coordinates": ["41.0155", "-73.7792749922", "41.022", "-73.7582464736"]
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

print("Request status code: ", response)

# Check status of query, make sure it's finished before downloading
response = requests.get(
    url=f'{server}/v2/queryjobs/{id}',
    auth=auth,
)

# do not download until the query is succeeded
while response.json()["status"] != "Succeeded" and response.json()["status"] != "Failed":
  sleep(5)
  # recheck status after waiting 5 seconds
  response = requests.get(
      url=f'{server}/v2/queryjobs/{id}',
      auth=auth,
  )
  print("status every 5 seconds: ")
  print(response.json()["status"])
  

print("json eventual response: ")
print(response.json()["status"])

#Download and extract to files
download = requests.get(
    f'{server}/download/{id}', auth=auth, stream=True,
)

z = zipfile.ZipFile(io.BytesIO(download.content))

## Extract to ./downloads
z.extractall(path)

def get_datalayers():
    res = requests.get(f'{server}/v2/datalayers/',auth=auth)
    print(res.json)

