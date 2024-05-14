import numpy as np
import math
import csv

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.gate_threshold = 0.2  # Association Gate Threshold
        self.most_associated_target = None

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
        num_measurements = len(measurements)

        # Compute association probabilities
        association_probabilities = np.zeros(num_measurements)
        for i, z in enumerate(measurements):
            z_pred = np.dot(self.H, self.Sf)
            innovation = z - z_pred
            innovation_cov = np.dot(np.dot(self.H, self.pf), self.H.T) + self.R
            association_probabilities[i] = self.association_probability(innovation, innovation_cov)

        # Associate measurements
        association_indices = self.associate_measurements(association_probabilities)

        # Update state based on the most likely associated measurement
        if association_indices:
            best_association_index = association_indices[0]
            self.most_associated_target = measurements[best_association_index]  # Capture most associated target
            z = measurements[best_association_index]
            Inn = z - np.dot(self.H, self.Sf)  # Calculate innovation directly
            S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
            K = np.dot(np.dot(self.pf, self.H.T), np.linalg.inv(S))
            self.Sf = self.Sf + np.dot(K, Inn.T)
            self.pf = np.dot(np.eye(6) - np.dot(K, self.H), self.pf)
            print("Updated filter state:")
            print("Sf:", self.Sf)
            print("pf:", self.pf)
        else:
            print("No measurements associated.")

    def association_probability(self, innovation, innovation_cov):
        # Ensure innovation is a column vector
        if innovation.ndim == 1:
            innovation = innovation[:, np.newaxis]

        # Ensure innovation_cov is a 2D matrix
        if innovation_cov.ndim == 1:
            innovation_cov = innovation_cov[np.newaxis, :]

        # Compute association probability using Mahalanobis distance
        mahalanobis_distance = np.sum(innovation * np.linalg.solve(innovation_cov, innovation), axis=0)
        return np.exp(-0.5 * mahalanobis_distance)

    def associate_measurements(self, association_probabilities):
        # Associate measurements using a gate threshold
        association_indices = []
        if isinstance(association_probabilities, np.ndarray):
            sorted_indices = np.argsort(association_probabilities)[::-1]  # Sort in descending order
            for idx in sorted_indices:
                if association_probabilities[idx] > self.gate_threshold:
                    association_indices.append(idx)
        elif association_probabilities > self.gate_threshold:
            association_indices.append(0)
        return association_indices

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
        az = 3.14/180 - az
    else:
        az = 3*3.14/180 - az
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

# Iterate through measurements
for i, (r, az, el, mt) in enumerate(measurements):
    if i == 0:
        # Initialize filter state with the first measurement
        kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
    elif i == 1:
        # Initialize filter state with the second measurement and compute velocity
        prev_r, prev_az, prev_el = measurements[i-1][:3]
        dt = mt - measurements[i-1][3]
        vx = (r - prev_r) / dt
        vy = (az - prev_az) / dt
        vz = (el - prev_el) / dt
        kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
    else:
        # Predict step
        kalman_filter.predict_step(mt)

        # Associate measurements and update step
        kalman_filter.update_step([(r, az, el)])

# Print the most associated target and its spherical coordinates
if kalman_filter.most_associated_target:
    print("Most Associated Target:")
    r, az, el, mt = kalman_filter.most_associated_target
    print("r:", r)
    print("az:", az)
    print("el:", el)
else:
    print("No target associated.")
