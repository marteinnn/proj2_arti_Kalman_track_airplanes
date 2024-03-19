#!/usr/bin/env python

from project2_base import get_ground_truth_data, get_radar_data
from src.kalmanfilter import create_kalman_filter
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    try:
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

        # Get the ground truth data
        ground_truth_data = get_ground_truth_data()

        # Select the first flight from the ground truth data
        flight_id, original_flight = list(ground_truth_data.items())[0]

        # Get the simulated radar data for the selected flight
        radar_flights = get_radar_data(ground_truth_data)
        radar_flight = radar_flights[flight_id]

        # Plot the original flight data and the simulated radar data
        fig, ax = plt.subplots()
        original_flight.plot(ax, label='Original Flight')
        radar_flight.plot(ax, label='Simulated Radar Data', linestyle='-')

        plt.legend()
        plt.show()  # Display the plot

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
