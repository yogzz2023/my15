import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.pf)

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.pf)

    def update_step(self, measurements):
        # Update step with JPDA
        Z = np.array(measurements)
        num_measurements = Z.shape[0]
        num_targets = self.Sf.shape[0] // 2

        # Calculate likelihood matrix
        likelihood_matrix = np.zeros((num_measurements, num_targets))
        for i in range(num_measurements):
            for j in range(num_targets):
                predicted_position = self.Sf[j * 2:j * 2 + 2].reshape(2, 1)
                innovation = Z[i, :2].reshape(2, 1) - predicted_position
                innovation_covariance = np.dot(np.dot(self.H, self.pf), self.H.T) + self.R
                mean = np.zeros(2)  # Adjusted mean vector
                likelihood_matrix[i, j] = multivariate_normal.pdf(innovation.flatten(), mean=mean, cov=innovation_covariance)

        # Calculate conditional probabilities
        conditional_probs = likelihood_matrix / np.sum(likelihood_matrix, axis=0)

        # Calculate marginal probabilities
        marginal_probs = np.prod(conditional_probs, axis=1)

        # Determine the most likely associations
        most_likely_association = np.argmax(marginal_probs)

        # Update filter state with the most likely association
        for i in range(num_targets):
            if i == most_likely_association:
                Inn = Z[most_likely_association, :2] - np.dot(self.H, self.Sf[i * 2:i * 2 + 2])
                S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
                K = np.dot(np.dot(self.pf, self.H.T), np.linalg.inv(S))
                self.Sf[i * 2:i * 2 + 2] = self.Sf[i * 2:i * 2 + 2] + np.dot(K, Inn)
                self.pf = np.dot(np.eye(6) - np.dot(K, self.H), self.pf)
            else:
                self.Sf[i * 2:i * 2 + 2] = self.Sf[i * 2:i * 2 + 2]

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    el = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    az = math.degrees(math.atan2(y, x))

    if x > 0.0:
        az = 90 - az
    else:
        az = 270 - az

    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            print("Cartesian coordinates (x, y, z):", x, y, z)
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            print("Spherical coordinates (r, az, el):", r, az, el)
            measurements.append((r, az, el, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'data_57.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

csv_file_predicted = "data_57.csv"

df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['PT','PR','PA','PE']].values

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Iterate through measurements
for i, (r, az, el, mt) in enumerate(measurements):
    if i == 0:
        # Initialize filter state with the first measurement
        kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
    elif i == 1:
        # Initialize filter state with the second measurement and compute velocity
        prev_r, prev_az, prev_el = measurements[i - 1][:3]
        dt = mt - measurements[i - 1][3]
        vx = (r - prev_r) / dt
        vy = (az - prev_az) / dt
        vz = (el - prev_el) / dt
        kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
    else:
        # Predict step
        kalman_filter.predict_step(mt)

        # Perform JPDA for associating measurements
        kalman_filter.update_step(measurements[:i])

        # Append data for plotting
        time_list.append(mt)
        r_list.append(r)
        az_list.append(az)
        el_list.append(el)

# Plotting
plt.figure(figsize=(10, 6))

# Plot range (r) vs. time
plt.subplot(3, 1, 1)
plt.plot(time_list, r_list, color='green')
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='o')
plt.plot(filtered_values_csv[:, 0], filtered_values_csv[:, 1], color='red', linestyle='--')
plt.scatter(filtered_values_csv[:, 0], filtered_values_csv[:, 1], label='filtered range (track id 57)', color='red',
            marker='*')
plt.xlabel('Time')
plt.ylabel('Range (r)')
plt.title('Range vs. Time')
plt.legend()

# Plot azimuth (az) vs. time
plt.subplot(3, 1, 2)
plt.plot(time_list, az_list, color='green')
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='o')
plt.plot(filtered_values_csv[:, 0], filtered_values_csv[:, 2], color='red', marker='*', linestyle='--')
plt.scatter(filtered_values_csv[:, 0], filtered_values_csv[:, 2], label='filtered azimuth (track id 57)', color='red',
            marker='*')
plt.xlabel('Time')
plt.ylabel('Azimuth (az)')
plt.title('Azimuth vs. Time')
plt.legend()

# Plot elevation (el) vs. time
plt.subplot(3, 1, 3)
plt.plot(time_list, el_list, color='green')
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='o')
plt.plot(filtered_values_csv[:, 0], filtered_values_csv[:, 3], color='red')
plt.scatter(filtered_values_csv[:, 0], filtered_values_csv[:, 3], label='filtered elevation (track id 57)', color='red',
            marker='*')
plt.xlabel('Time')
plt.ylabel('Elevation (el)')
plt.title('Elevation vs. Time')
plt.legend()

plt.tight_layout()
plt.show()
