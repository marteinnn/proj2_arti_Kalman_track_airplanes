from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
import numpy as np
from pykalman import KalmanFilter as PyKalmanFilter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_kalman_filter(dt, sigma_p, sigma_o):
    # Create Kalman filter using filterpy
    kf = FilterPyKalmanFilter(dim_x=4, dim_z=2)

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

def run_kalman_filter_on_flight(flight):
    # Time step
    dt = 10  # seconds

    # Transition matrix F (assuming a constant velocity model)
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])

    # Observation matrix H (we only observe positions, not velocities)
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

    # Initial state (x0, vx0, y0, vy0)
    initial_state = [flight.data.x.iloc[0], 0, flight.data.y.iloc[0], 0]

    # Measurements
    measurements = np.column_stack([flight.data.x, flight.data.y])

    # Initial state covariance P
    P = np.eye(4) * 1000  # Large initial uncertainty

    # Process noise covariance Q
    Q = np.eye(4)

    # Observation noise covariance R
    R = np.eye(2) * 100  # Adjust based on measurement noise

    # Create and initialize the Kalman Filter
    kf = PyKalmanFilter(transition_matrices=F,
                        observation_matrices=H,
                        initial_state_mean=initial_state,
                        transition_covariance=Q,
                        observation_covariance=R,
                        initial_state_covariance=P)

    # Run the Kalman Filter and compute the filtered state estimates
    filtered_state_means, filtered_state_covariances = kf.filter(measurements)

    return filtered_state_means, filtered_state_covariances

