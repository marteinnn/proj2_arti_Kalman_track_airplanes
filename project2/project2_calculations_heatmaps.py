from project2_base import *
import numpy as np
import warnings
from src.kalmanfilter import create_kalman_filter, run_kalman_filter_on_flight
from src.utils import get_filtered_error_measure, plot_heatmap
from src.smoothing import run_kalman_smoother_on_flight
import pandas as pd
from src.kalmanfilter import *
import time

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    start_time = time.time()
    try:
        # Get the ground truth data and the simulated radar data for the flight
        ground_truth_flights = get_ground_truth_data()
        radar_data = get_radar_data(ground_truth_flights)

        some_n_flights = list(radar_data.values())[:5]  # First 5 flights

        mean_errors = {}
        max_errors = {}

        for flight in some_n_flights:
            print(f"Processing flight {flight.flight_id}...")
            for sigma_o in [60, 70, 80, 90, 100, 110, 120, 130, 140]: 
                for sigma_p in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]:
                    print(f"Running experiment with sigma_o={sigma_o}, sigma_p={sigma_p}...")
                    # Create Kalman filter
                    kf = create_kalman_filter(dt=1, sigma_p=sigma_p, sigma_o=sigma_o)

                    # Apply filtering to the flight data. To use smoother, uncomment the line below and comment the line below it
                    #filtered_state_means, _ = run_kalman_smoother_on_flight(flight)
                    filtered_state_means, _ = run_kalman_filter_on_flight(flight)

                    # Update flight data with filtered positions
                    flight.data.x = filtered_state_means[:, 0]
                    flight.data.y = filtered_state_means[:, 2]

                    # Check for NaN values
                    if np.isnan(flight.data.x).any() or np.isnan(flight.data.y).any():
                        print(f"Warning: NaN values found in flight data for flight {flight.flight_id}. Skipping this flight.")
                        continue


                    # Convert Cartesian coordinates to latitude/longitude
                    flight = set_lat_lon_from_x_y(flight)

                    # Get original flight data
                    original_flight = ground_truth_flights[flight.flight_id]

                    # Set datetime index
                    flight.data.index = pd.to_datetime(flight.data.index)
                    original_flight.data.index = pd.to_datetime(original_flight.data.index)

                    # Measure error of the filtered positions
                    max_distance, mean_distance = get_filtered_error_measure(flight.data, original_flight.data)

                    # Append the errors to the lists in the dicts
                    mean_errors[(sigma_o, sigma_p)] = mean_errors.get((sigma_o, sigma_p), []) + [mean_distance]
                    max_errors[(sigma_o, sigma_p)] = max_errors.get((sigma_o, sigma_p), []) + [max_distance]

        # Calculate the mean of the mean errors for each sigma_o and sigma_p
        mean_errors = {k: np.mean(v) for k, v in mean_errors.items()}

        # Calculate the mean of the max errors for each sigma_o and sigma_p
        max_errors = {k: np.mean(v) for k, v in max_errors.items()}

        # Plot a heatmap for the mean of the mean errors
        plot_heatmap(mean_errors, title="Mean of Mean Errors")

        # Plot a heatmap for the mean of the max errors
        plot_heatmap(max_errors, title="Mean of Max Errors")

    except Exception as e:
        print(f"An error occurred: {repr(e)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()