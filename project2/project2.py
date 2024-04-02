from project2_base import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
from rich.console import Console
from src.kalmanfilter import create_kalman_filter, run_kalman_filter_on_flight
from src.utils import get_filtered_error_measure
import pandas as pd

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    try:
        # Create Kalman filter
        kf = create_kalman_filter(dt=1, sigma_p=1.5, sigma_o=10)

        # Simulate data
        true_state = np.array([0, 1, 0, 1])  # [x, vx, y, vy]
        measurement_noise = np.random.normal(0, 10, 2)  # [x, y]
        measurement = kf.H @ true_state + measurement_noise

        kf.predict()

        # Update state
        kf.update(measurement)

        print("Estimated state:", kf.x)

        
        # Get the ground truth data and the simulated radar data for the flight
        ground_truth_flights = get_ground_truth_data()
        radar_data = get_radar_data(ground_truth_flights)
        first_five_flights = list(radar_data.items())[:5] # First 5 flights


        for flight_id, radar_flight in first_five_flights:
            # Run Kalman Filter on the flight data
            filtered_state_means, _ = run_kalman_filter_on_flight(radar_flight)

            # Update flight data with filtered positions
            radar_flight.data.x = filtered_state_means[:, 0]
            radar_flight.data.y = filtered_state_means[:, 2]

            # Convert Cartesian coordinates to latitude/longitude
            radar_flight = set_lat_lon_from_x_y(radar_flight)

            # Get original flight data
            original_flight = ground_truth_flights[flight_id]

            # Set datetime index
            radar_flight.data.index = pd.to_datetime(radar_flight.data.index)
            original_flight.data.index = pd.to_datetime(original_flight.data.index)

            # Measure error of the filtered positions
            max_distance, mean_distance = get_filtered_error_measure(radar_flight.data, original_flight.data)
            print(f"Maximal distance: {max_distance} meters")
            print(f"Mean distance: {mean_distance} meters")

            # Print the flight with rich representation
            console = Console()
            console.print(original_flight)

            # Plot the results
            fig, ax = plt.subplots()
            original_flight.plot(ax, label='Original Flight')
            radar_flight.plot(ax, label='Filtered Flight')
            plt.legend()
            plt.show()

    except Exception as e:
        print(f"An error occurred: {repr(e)}")

if __name__ == "__main__":
    main()
