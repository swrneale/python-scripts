import numpy as np
from geopy.distance import geodesic

def latlon_to_cartesian(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])

def cartesian_to_latlon(xyz):
    x, y, z = xyz
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)
    return np.degrees(lat), np.degrees(lon)

def normalize(v):
    return v / np.linalg.norm(v)

def spherical_bezier(p0, p1, p2, num_points=20):
    p0 = normalize(latlon_to_cartesian(*p0))
    p1 = normalize(latlon_to_cartesian(*p1))
    p2 = normalize(latlon_to_cartesian(*p2))
    arc_points = []

    for t in np.linspace(0, 1, num_points):
        a = normalize((1 - t) * p0 + t * p1)
        b = normalize((1 - t) * p1 + t * p2)
        point = normalize((1 - t) * a + t * b)
        latlon = cartesian_to_latlon(point)
        arc_points.append(latlon)

    return arc_points