import numpy as np

def convert_sphere_to_cartesian(coordinates):
    """For a list of spherical coordinates of the form (elevation, azimuth), convert to (x, y, z) cartesian
    coordinates for plotting purposes """
    x, y, z = [], [], []
    mask = []

    for elevation, azimuth in coordinates:
        if elevation is not None and azimuth is not None:
            mask.append(True)
            # convert to cartesian coordinates
            x_i = np.cos(elevation) * np.cos(azimuth)
            y_i = np.cos(elevation) * np.sin(azimuth)
            z_i = np.sin(elevation)

            x.append(x_i)
            y.append(y_i)
            z.append(z_i)
        else:
            mask.append(False)

    return np.asarray(x), np.asarray(y), np.asarray(z), mask