#!/usr/bin/env python

from project2_base import *
from filterpy.kalman import KalmanFilter
from src.kalmanfilter import create_kalman_filter
import numpy as np

#############################


def main():
    # Create Kalman filter
    kf = create_kalman_filter(dt=10, sigma_p=1.5, sigma_o=100)

    # Simulate some data
    true_state = np.array([0, 1, 0, 1])  # [x, vx, y, vy]
    measurement_noise = np.random.normal(0, 100, 2)  # [x, y]
    measurement = kf.H @ true_state + measurement_noise

    # Predict next state
    kf.predict()

    # Update state with measurement
    kf.update(measurement)

    print("Estimated state:", kf.x)


#############################

if __name__ == "__main__":
    main()
