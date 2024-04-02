from geopy.distance import geodesic
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




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


def plot_heatmap(data, title): 
    sigma_o_values = sorted(set(k[0] for k in data.keys()))
    sigma_p_values = sorted(set(k[1] for k in data.keys()))
    heatmap_data = np.zeros((len(sigma_o_values), len(sigma_p_values)))
    for i, sigma_o in enumerate(sigma_o_values):
        for j, sigma_p in enumerate(sigma_p_values):
            heatmap_data[i, j] = data[(sigma_o, sigma_p)]

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", xticklabels=sigma_p_values, yticklabels=sigma_o_values)
    plt.title(title)
    plt.xlabel('sigma_p')
    plt.ylabel('sigma_o')
    plt.show()