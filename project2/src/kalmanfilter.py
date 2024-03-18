from filterpy.kalman import KalmanFilter
import numpy as np

def create_kalman_filter(dt, sigma_p, sigma_o):
    # Create Kalman filter
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Define state transition matrix
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])

    # Define measurement function
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])

    # Define initial state
    kf.x = np.array([0, 0, 0, 0])

    # Define state covariance
    kf.P *= 1000

    # Define process noise covariance
    kf.Q = np.array([[0.25 * dt**4 * sigma_p**2, 0.5 * dt**3 * sigma_p**2, 0, 0],
                     [0.5 * dt**3 * sigma_p**2, dt**2 * sigma_p**2, 0, 0],
                     [0, 0, 0.25 * dt**4 * sigma_p**2, 0.5 * dt**3 * sigma_p**2],
                     [0, 0, 0.5 * dt**3 * sigma_p**2, dt**2 * sigma_p**2]])

    # Define measurement noise covariance
    kf.R = np.array([[sigma_o**2, 0],
                     [0, sigma_o**2]])

    return kf