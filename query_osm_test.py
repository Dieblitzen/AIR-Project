import overpy

def query_OSM(coords, classes):
  """
  Sends a request to OSM server and returns a list of all the building nodes
  and classes in the area specified by [coords]. Those buildings not in the
  specified classes are of class "building"

  Returns: 
  [{class: building1_class, coords: building1_node}, {class: building2_class, 
  coords: building2_node}, ...] where each node is in (lat,lon) format.
  """
  api = overpy.Overpass()

  # The list of each building's coordinates and classes.
  # Each item in this list is a list of points in (lat,lon) for each building's nodes.
  building_coords = []

  # the query to request buildings not in the classes, initialized with a header
  building_query = """way({0}, {1}, {2}, {3}) ["building"];""".format(coords[0], coords[1], coords[2], coords[3])

  for building_class in classes:
    # update the bulding query to exclude certain classes
    class_query = """way({0}, {1}, {2}, {3})["amenity"!={4}];""".format(coords[0], coords[1], coords[2], coords[3], building_class)
    building_query += class_query

    class_result = api.query(("""
        way({}, {}, {}, {})["amenity"={}];
        (._;>;);
        out body;
        """).format(coords[0], coords[1], coords[2], coords[3], building_class))

    # Unprocessed class data from the query
    for way in class_result.ways:
      points = [(float(str(n.lat)), float(str(n.lon))) for n in way.nodes]
      building_coords.append({'class': building_class, 'coords': points})

  building_result = api.query(("""
      {}
      (._;>;);
      out body;
      """).format(building_query))
  
  # Unprocessed building data from the query
  buildings = building_result.ways
  for building in buildings:
    points = [(float(str(n.lat)), float(str(n.lon))) for n in building.nodes]
    building_coords.append({'class': 'building', 'coords': points})
  
  return building_coords

if __name__ == "__main__":
    # testing with hospital and parking and output the first 10 instances
    coords = [41.009, -73.779, 41.03, -73.758]
    result = query_OSM(coords, ['hospital', 'parking'])
    print(result[:10])