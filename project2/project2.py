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
    flight_id, original_flight = list(ground_truth_data.items())[0]

    #print(f"Flight ID: {flight_id}")
    print(original_flight)

    # Print the flight with rich representation
    console = Console()
    console.print(original_flight)

    # Print the first 30 minutes of the flight
    print(original_flight.first(minutes=30))

    # Get the simulated radar data for the flight
    radar_flights = get_radar_data(ground_truth_data)
    radar_flight = radar_flights[flight_id]

    # Plot the original flight data and the simulated radar data
    fig, ax = plt.subplots()
    original_flight.plot(ax, label='Original')
    radar_flight.plot(ax, label='Radar')

    plt.legend()
    plt.show()  # Display the plot

if __name__ == "__main__":
    main()