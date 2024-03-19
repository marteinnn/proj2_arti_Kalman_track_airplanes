#!/usr/bin/env python

from project2_base import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
from rich.console import Console
from src.kalmanfilter import create_kalman_filter, run_kalman_filter_on_flight
from src.error_utils import compute_position_error

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    try:
        # Create Kalman filter
        kf = create_kalman_filter(dt=1, sigma_p=1.5, sigma_o=10)

        # Simulate some data
        true_state = np.array([0, 1, 0, 1])  # [x, vx, y, vy]
        measurement_noise = np.random.normal(0, 10, 2)  # [x, y]
        measurement = kf.H @ true_state + measurement_noise

        # Predict next state
        kf.predict()

        # Update state with measurement
        kf.update(measurement)

        print("Estimated state:", kf.x)

        
        # Get the ground truth data
        ground_truth_flights = get_ground_truth_data()

        # Get the simulated radar data for the flight
        radar_data = get_radar_data(ground_truth_flights)
        flight_id, radar_flight = list(radar_data.items())[0]

        # Run Kalman Filter on the flight data
        filtered_state_means, _ = run_kalman_filter_on_flight(radar_flight)

        # Update flight data with filtered positions
        radar_flight.data.x = filtered_state_means[:, 0]
        radar_flight.data.y = filtered_state_means[:, 2]

        # Convert Cartesian coordinates to latitude/longitude
        radar_flight = set_lat_lon_from_x_y(radar_flight)

        # Get original flight data
        original_flight = ground_truth_flights[flight_id]

        print(original_flight)

        # Print the flight with rich representation
        console = Console()
        console.print(original_flight)


        # Plot the results
        fig, ax = plt.subplots()
        original_flight.plot(ax, label='Original Flight')
        radar_flight.plot(ax, label='Filtered Flight')
        plt.legend()

        # Calculate errors
        original_positions = list(zip(original_flight.data.latitude, original_flight.data.longitude))
        errors = compute_position_error(filtered_state_means, original_positions)
        mean_error = sum(errors) / len(errors)
        max_error = max(errors)
        print(f"Mean error: {mean_error} meters")
        print(f"Maximal error: {max_error} meters")

        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
