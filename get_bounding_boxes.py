
from minimum_bounding_box import MinimumBoundingBox
import overpy

api = overpy.Overpass()

def get_rect (nodes, horizontal=True):
    ## Gets either horizontally aligned (horizontal=True) or minimum bounding box for a set of nodes (horizontal=False)
    points = [(float(str(n.lat)), float(str(n.lon))) for n in nodes]
    if horizontal: 
        lats = [node[0] for node in points] 
        lons = [node[1] for node in points]
        lat_min = min(lats)
        lon_min = min(lons)
        lat_max = max(lats)
        lon_max = max(lons)
        bounding_box = [(lat_min,lon_min),(lat_max,lon_min),(lat_max,lon_max),(lat_min,lon_max)]
    else:
        bounding_boxes = MinimumBoundingBox(points).corner_points

    return bounding_box

#Bounding box of decided area in White Plains, NY
lat_min = 41.0100756423
lat_max = 41.0338409682
lon_min = -73.7792749922
lon_max = -73.7582464736

# Returns: a list of the bounding box location (4 lat/long coordinates) 
#    for each building within the area defined by coordinate parameters.
#    lat_min, lat_max, long_min, long_max: represents the edges of the area
#    that we're interested in. Horizontal=True gets horizontal bounding boxes while
#    horizontal=False gets minimum bounding boxes
def get_bounding_boxes(lat_min, lon_min, lat_max, lon_max, horizontal=True):

    #query overpass for all buildings
    result = api.query(("""
        way
            ({}, {}, {}, {}) ["building"];
        (._;>;);
        out body;
        """).format(lat_min, lon_min, lat_max, lon_max))
    
    buildings = result.ways
    building_coordinates = []
    #use the imported package to find minimum bounding box
    for building in buildings:
        building_coordinates.append(list(get_rect(building.nodes,horizontal)))
    return building_coordinates



