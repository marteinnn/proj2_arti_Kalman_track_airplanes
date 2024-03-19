from geopy.distance import geodesic


# Measure the error of the filtered positions, by computing both the mean and maximal distance of the filtered states to the positions in the original data at the same time point.
def compute_position_error(filtered_states, original_positions):
    errors = []
    for filtered_state, original_position in zip(filtered_states, original_positions):
        filtered_lat, filtered_lon = filtered_state[0], filtered_state[1]
        original_lat, original_lon = original_position[0], original_position[1]
        
        # Ensure latitude values are within the valid range of [-90, 90]
        if not (-90 <= filtered_lat <= 90):
            filtered_lat = max(min(filtered_lat, 90), -90)
        if not (-90 <= original_lat <= 90):
            original_lat = max(min(original_lat, 90), -90)
        
        filtered_point = (filtered_lat, filtered_lon)
        original_point = (original_lat, original_lon)
        error = geodesic(filtered_point, original_point).meters
        errors.append(error)
    return errors
