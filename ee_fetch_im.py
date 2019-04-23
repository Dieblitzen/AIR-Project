import ee
import numpy as np


"""Fetches from earth engine by bounding coordinates"""


def fetch_by_coords(coords):
    # coords: [lat_min, lon_min, lat_max, lon_max]

    # Create bounding coords
    lat_min = coords[0]
    lon_min = coords[1]
    lat_max = coords[2]
    lon_max = coords[3]
    bounds = [[lat_min, lon_min], [lat_min, lon_max],
              [lat_max, lon_max], [lat_max, lon_min]]

    # image data filtered by date and bounds, RGB bands
    img = ee.Image(
        "USDA/NAIP/DOQQ").select(['R', 'G', 'B'])

    print(img)

    # Config for exporting the image
    task_config = {
        'description': 'exportedNAIPImage',
        'scale': 30,
        'region': bounds
    }

    # Create a task to export the image
    task = ee.batch.Export.image(img, 'exportedNAIPImage', task_config)

    # Start the task
    task.start()


if __name__ == "__main__":
    ee.Initialize()
    fetch_by_coords([41.0100756423, -73.7792749922,
                     41.0338409682, -73.7582464736])
