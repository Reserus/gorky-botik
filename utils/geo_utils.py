from geopy.distance import geodesic

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance between two points in kilometers."""
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers
