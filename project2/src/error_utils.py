from geopy.distance import geodesic
import statistics  # for mean calculation
import numpy as np




# Measure the error of the filtered positions, by computing both the mean and maximal distance of the filtered states to the positions in the original data at the same time point.

#def compute_position_error(filtered_states, original_positions):
#    errors = []
#    for filtered_state, original_position in zip(filtered_states, original_positions):
#        filtered_lat, filtered_lon = filtered_state[0], filtered_state[1]
#        original_lat, original_lon = original_position[0], original_position[1]
#        
#        # Ensure latitude values are within the valid range of [-90, 90]
#        if not (-90 <= filtered_lat <= 90):
#            filtered_lat = max(min(filtered_lat, 90), -90)
#        if not (-90 <= original_lat <= 90):
#            original_lat = max(min(original_lat, 90), -90)
#        
#        filtered_point = (filtered_lat, filtered_lon)
#        original_point = (original_lat, original_lon)
#        error = geodesic(filtered_point, original_point).meters
#        errors.append(error)
#    return errors


def get_filtered_error_measure(filtered_data, unfiltered_data):
    """ Computes and returns the mean and maximal distance of
        the filtered states to the positions in the original data """

    distances = []

    # Select only numeric columns for calculation
    filtered_data_numeric = filtered_data.select_dtypes(include=np.number)
    unfiltered_data_numeric = unfiltered_data.select_dtypes(include=np.number)

    for i in range(min(len(filtered_data_numeric), len(unfiltered_data_numeric))):
        a = (unfiltered_data_numeric.iloc[i]['latitude'], unfiltered_data_numeric.iloc[i]['longitude'])
        b = (filtered_data_numeric.iloc[i]['latitude'], filtered_data_numeric.iloc[i]['longitude'])
        distances.append(geodesic(a, b).m)

    return max(distances), np.mean(distances)

