import numpy as np
from pykalman import KalmanFilter as PyKalmanSmoother

def run_kalman_smoother_on_flight(flight):
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

    # Measurements
    measurements = np.column_stack([flight.data.x, flight.data.y])

    # Initial state (x0, vx0, y0, vy0)
    initial_state = [flight.data.x.iloc[0], 0, flight.data.y.iloc[0], 0]

    # Initial state covariance P
    P = np.eye(4) * 1000  # Large initial uncertainty

    # Process noise covariance Q
    Q = np.eye(4)

    # Observation noise covariance R
    R = np.eye(2) * 100  # Adjust based on measurement noise

    # Create and initialize the Kalman Smoother
    ks = PyKalmanSmoother(transition_matrices=F,
                          observation_matrices=H,
                          initial_state_mean=initial_state,
                          transition_covariance=Q,
                          observation_covariance=R,
                          initial_state_covariance=P)

    # Smooth the measurements and compute the smoothed state estimates
    smoothed_state_means, smoothed_state_covariances = ks.smooth(measurements)

    return smoothed_state_means, smoothed_state_covariances
