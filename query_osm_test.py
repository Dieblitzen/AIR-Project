import overpy

def query_OSM(coords, **kwargs):
  """
  Sends a request to OSM server and returns a dictionary of all the buildings
  and roads nodes along with their classes in the area specified by [coords].
  Those buildings and roads not in specified classes are of class "n/a"

  Returns: 
  {building: building_class1: [building_class1_node1...] ..., road: road_class_1: 
  [road_class1_node1...]...} where each node is in (lat,lon) format.
  """
  api = overpy.Overpass()

  # The dictionary of different buildings and roads coordinates to return
  query_data = {}
  query_data["building"] = {}
  query_data["road"] = {}

  # the query to request buildings and roads not in the specified building classes, 
  # initialized with a header
  building_query = """way({0}, {1}, {2}, {3}) ["building"]""".format(coords[0], coords[1], coords[2], coords[3])
  road_query = """way({0}, {1}, {2}, {3}) ["highway"]""".format(coords[0], coords[1], coords[2], coords[3])

  building_classes = kwargs.get('building_classes')
  road_classes = kwargs.get('road_classes')

  if building_classes:
    for building_class in building_classes:
      # Update the bulding query to exclude specified classes for getting the rest buildings later
      class_query = """ ["amenity"!={}]""".format(building_class)
      building_query += class_query

      query_data["building"][building_class] = []

      class_result = api.query(("""
          way({}, {}, {}, {})["amenity"={}];
          (._;>;);
          out body;
          """).format(coords[0], coords[1], coords[2], coords[3], building_class))

      # Unprocesse class data from the query
      for way in class_result.ways:
        points = [(float(str(n.lat)), float(str(n.lon))) for n in way.nodes]
        query_data["building"][building_class].append(points)

  # Get the rest buildings not in building_classes
  query_data["building"]["n/a"] = []

  building_result = api.query(("""
      {};
      (._;>;);
      out body;
      """).format(building_query))
  
  # Unprocess building data from the query
  buildings = building_result.ways
  for building in buildings:
    points = [(float(str(n.lat)), float(str(n.lon))) for n in building.nodes]
    query_data["building"]["n/a"].append(points)

  if road_classes:
    for road_class in road_classes:
      # Update the road query to exclude specified classes for getting the rest roads later
      class_query = """ ["highway"!={}]""".format(road_class)
      road_query += class_query

      query_data["road"][road_class] = []

      class_result = api.query(("""
          way({}, {}, {}, {})["highway"={}];
          (._;>;);
          out body;
          """).format(coords[0], coords[1], coords[2], coords[3], road_class))

      # Unprocesse class data from the query
      for way in class_result.ways:
        points = [(float(str(n.lat)), float(str(n.lon))) for n in way.nodes]
        query_data["road"][road_class].append(points)

  # Get the rest roads not in road_classes
  query_data["road"]["n/a"] = []

  road_result = api.query(("""
      {};
      (._;>;);
      out body;
      """).format(road_query))
  
  # Unprocess road data from the query
  roads = road_result.ways
  for road in roads:
    points = [(float(str(n.lat)), float(str(n.lon))) for n in road.nodes]
    query_data["road"]["n/a"].append(points)
  
  return query_data

if __name__ == "__main__":
    # Test with hospital and parking in buildings and general roads and output the first 3 instances
    coords = [41.009, -73.779, 41.03, -73.758]
    result = query_OSM(coords, building_classes=['hospital', 'parking'])
    print('hospital: {}'.format(result["building"]["hospital"][:3]))
    print('parking: {}'.format(result["building"]["parking"][:3]))
    print('general buildings: {}'.format(result["building"]["n/a"][:3]))
    print('general roads: {}'.format(result["road"]["n/a"][:3]))