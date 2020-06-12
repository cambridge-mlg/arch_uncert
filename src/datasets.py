
"""Useful datasets"""
import math
import numpy as np


def make_spirals(n_samples=100, shuffle=True, noise=None, random_state=None,\
                 n_arms=2, start_angle=0, stop_angle=360):
    """Adapted from: https://github.com/DatCorno/N-Arm-Spiral-Dataset
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
    n_arms : int, optional (default=2)
        The number of arms in the spiral.
    start_angle: int, optional (default=0)
        The starting angle for each spiral arm (in degrees).
    stop_engle: int, optional (default=360)
        The stopping angle for each spiral arm (in degrees).
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """
    n_samples = math.floor(n_samples / n_arms)
    data = np.empty((0, 3))

    # Create a list of the angles at which to rotate the arms.
    angles = [(360 / n_arms) * i for i in range(n_arms)]

    for i, angle in enumerate(angles):
        points = _generate_spiral(n_samples, start_angle, stop_angle, angle, noise)
        classified_points = np.hstack((points, np.full((n_samples, 1), i)))
        data = np.concatenate((data, classified_points))

    if shuffle:
        np.random.shuffle(data)

    return data[:, 0:2], data[:, 2]


def _generate_spiral(n_samples, start_angle, stop_angle, angle, noise):
    """Generate a spiral of points.
    Given a starting end, an end angle and a noise factor, generate a spiral of points along
    an arc.
    Parameters
    ----------
    n_samples: int
        Number of points to generate.
    start_angle: float
        The starting angle of the spiral in degrees.
    stop_angle: float
        The the stopping angle at which to rotate the points, in degrees.
    angle: float
        Angle of rotation in degrees.
    noise: float
        The noisyness of the points inside the spirals. Needs to be less than 1.
    Returns
    -------
    2d numpy array
        Stack of points inside a n_samples x 2 matrix
    """
    # Generate points from the square root of random data inside an uniform distribution on [0, 1).
    points = math.radians(start_angle) + np.sqrt(np.random.rand(n_samples, 1)) * math.radians(stop_angle)

    # Apply a rotation to the points.
    rotated_x_axis = np.cos(points) * points + np.random.rand(n_samples, 1) * noise
    rotated_y_axis = np.sin(points) * points + np.random.rand(n_samples, 1) * noise

    # Stack the vectors inside a samples x 2 matrix.
    rotated_points = np.column_stack((rotated_x_axis, rotated_y_axis))
    return np.apply_along_axis(_rotate_point, 1, rotated_points, math.radians(angle))


def _rotate_point(point, angle):
    """Rotate two point by an angle.
    Parameters
    ----------
    point: 2d numpy array
        The coordinate to rotate.
    angle: float
        The angle of rotation of the point, in degrees.
    Returns
    -------
    2d numpy array
        Rotated point.
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_point = rotation_matrix.dot(point)
    return rotated_point

