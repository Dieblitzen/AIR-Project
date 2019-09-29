import overpy

def query_OSM(coords, classes):
  """
  Sends a request to OSM server and returns a dictionary of all the building nodes
  and all the nodes for certain classes in the area specified by [coords]

  Returns: 
  {building: [[building1_node, ...], [building2_node, ...], ...], 
  1st_class: [[1st_class1_node, ...], [1st_class2_node, ...], ...],
  ...} where each node is in (lat,lon) format.
  """
  api = overpy.Overpass()
  query_result = api.query(("""
      way
          ({}, {}, {}, {}) ["building"];
      (._;>;);
      out body;
      """).format(coords[0], coords[1], coords[2], coords[3]))
  
  # Unprocessed building data from the query
  buildings = query_result.ways

  # The list of each building's coordinates.
  # Each item in this list is a list of points in (lat,lon) for each building's nodes.
  building_coords = {}
  building_coords['building'] = []

  for building in buildings:
    points = [(float(str(n.lat)), float(str(n.lon))) for n in building.nodes]
    building_coords['building'].append(points)

  for building_class in classes:
    building_coords[building_class] = []
    result = api.query(("""
        way({}, {}, {}, {})["amenity"={}];
        (._;>;);
        out body;
        """).format(coords[0], coords[1], coords[2], coords[3], building_class))
    for way in result.ways:
      points = [(float(str(n.lat)), float(str(n.lon))) for n in way.nodes]
      building_coords[building_class].append(points)
  
  return building_coords

if __name__ == "__main__":
    # testing
    coords = [40.7999, -73.9657, 40.8210, -73.9258]
    result = query_OSM(coords, ['hospital'])

    # Output the coordinates of the first instance of a certain class for testing
    for r in result:
      print('first {}: '.format(r), result[r][0])