#!/usr/bin/env python

from project2_base import *
from filterpy.kalman import KalmanFilter
from src.kalmanfilter import create_kalman_filter
import numpy as np
from traffic.data.samples import belevingsvlucht
from rich.console import Console
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Create Kalman filter
    kf = create_kalman_filter(dt=10, sigma_p=1.5, sigma_o=100)

    # Simulate some data
    true_state = np.array([0, 1, 0, 1])  # [x, vx, y, vy]
    measurement_noise = np.random.normal(0, 100, 2)  # [x, y]
    measurement = kf.H @ true_state + measurement_noise  # Use 'H'

    # Predict next state
    kf.predict()

    # Update state with measurement
    kf.update(measurement)

    print("Estimated state:", kf.x)

    # Get the ground truth data
    ground_truth_data = get_ground_truth_data()

    # Select a random flight
    flight_id, flight = random.choice(list(ground_truth_data.items()))

    print(f"Flight ID: {flight_id}")
    print(flight)

    # Print the flight with rich representation
    console = Console()
    console.print(flight)

    # Print the first 30 minutes of the flight
    print(flight.first(minutes=30))

    # Plot the flight
    fig, ax = plt.subplots()
    flight.plot(ax)

    plt.show()  # Display the plot

if __name__ == "__main__":
    main()