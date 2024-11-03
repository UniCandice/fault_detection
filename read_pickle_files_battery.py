# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import random
from glob import glob
from tqdm import tqdm
import torch
import pandas as pd

current_dir = 'C:/Users/liyuan.liu/Desktop/HealingBat/detection algorithm'
print(current_dir)

# Set your paths
train_path = os.path.join(current_dir, 'battery_brand1', 'train')
test_path = os.path.join(current_dir, 'battery_brand1', 'test')


# Load all .pkl files
train_pkl_files = glob(os.path.join(train_path, '*.pkl'))
test_pkl_files = glob(os.path.join(test_path, '*.pkl'))

# Initialize lists for time series data and metadata
time_series_dict = {}

# Process files
for each_path in tqdm(train_pkl_files):
    try:
        # Load the .pkl file
        this_pkl_file = torch.load(each_path)

        # Extract the time series data (first part of the tuple)
        time_series_data = this_pkl_file[0]  # Assuming first part is the time series
        metadata = this_pkl_file[1]           # Assuming second part is the metadata

        # Convert the time series data into a DataFrame for easier analysis
        columns = torch.load(os.path.join(os.path.dirname(train_path), "column.pkl"))    
        if columns:
            df = pd.DataFrame(time_series_data, columns=columns)
        else:
            df = pd.DataFrame(time_series_data)  # Use default if columns are not available

        # Store DataFrame and metadata in the dictionary with car number as the key
        car_number = metadata['car']
        time_series_dict[car_number] = {
            'data': df,
            'metadata': metadata
        }

        # Print a preview of the DataFrame
        print(f"Time Series Data for Car {car_number} from {each_path}:")
 #       print(df.head())  # Show the first few rows of the time series data
        print(metadata)    # Print the associated metadata
        print(car_number)

    except Exception as e:
        print(f"Error loading {each_path}: {e}")



import matplotlib.pyplot as plt
from collections import OrderedDict

# Assuming time_series_dict is already defined and contains your data

# Check the length of time_series_dict
length_of_time_series_dict = len(time_series_dict)
print(f"Length of time_series_dict: {length_of_time_series_dict}")

# Iterate over each car entry in the time_series_dict
for car_key, car_data in time_series_dict.items():
    # Extract the DataFrame (df) and metadata
    df = car_data['data']  # DataFrame part
    metadata = car_data['metadata']  # Metadata part
    
    # Extract car number and other metadata
    car_number = metadata['car']
    mileage = metadata['mileage']
    
    # Plot the 'current' vs 'timestamp' for the current car
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['current'], label=f"Car {car_number}")
    
    # Add labels, title, and metadata to the plot
    plt.title(f"Car {car_number} - Mileage: {mileage:.2f}")
    plt.xlabel('Timestamp')
    plt.ylabel('Current (A)')
    plt.legend()
    
    # Display the plot
    plt.show()
    plt.savefig(str(car_number)+'.png', format='png')

