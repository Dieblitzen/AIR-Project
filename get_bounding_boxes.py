
from minimum_bounding_box import MinimumBoundingBox
import overpy

api = overpy.Overpass()

def get_rect (nodes):
    
    points = [(float(str(n.lat)), float(str(n.lon))) for n in nodes]
    bounding_box = MinimumBoundingBox(points)
#     lats = [n.lat for n in nodes]
#     lons = [n.lon for n in nodes]
    
#     min_lat = float(str(min(lats)))
#     min_lon = float(str(min(lons)))
#     max_lat = float(str(max(lats)))
#     max_lon = float(str(max(lons)))
    
    return bounding_box.corner_points

#Bounding box of decided area in White Plains, NY
lat_min = 41.0100756423
lat_max = 41.0338409682
lon_min = -73.7792749922
lon_max = -73.7582464736

# Returns: a list of the bounding box location (4 lat/long coordinates) 
#    for each building within the area defined by coordinate parameters.
# lat_min, lat_max, long_min, long_max: represents the edges of the area
#    that we're interested in
def get_bounding_boxes(lat_min, lon_min, lat_max, lon_max):

    #query overpass for all buildings
    result = api.query(("""
        way
            ({}, {}, {}, {}) ["building"];
        (._;>;);
        out body;
        """).format(lat_min, long_min, lat_max, long_max))
    
    buildings = result.ways
    building_coordinates = []
    #use the imported package to find minimum bounding box
    for building in buildings:
        building_coordinates.append(get_rect(building.nodes))
    return building_coordinates

white_plain_buildings = get_bounding_boxes(41.014456, -73.769573, 41.018465,-73.765043)

