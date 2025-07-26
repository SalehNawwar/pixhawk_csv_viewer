# main.py

import pandas as pd
import numpy as np
import os
from ekf import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation # Keep the import for the check

# --- Configuration ---
INPUT_FILENAME = '1.csv'
OUTPUT_FILENAME = 'estimated_states.csv'
STATIONARY_DURATION_S = 320.0 
INITIALIZATION_WARMUP_S = 5.0 
baroBias = -56

def main():
    # --- 1. EKF Initialization ---
    initial_state = np.zeros(22)
    initial_state[6] = 1.0
    initial_covariance = np.eye(22) * 10.0

    # --- MODIFICATION: Define process noise based on sensor characteristics ---
    # These values are typically from the sensor datasheet (Noise Density).
    # Units are (m/s^2)/sqrt(Hz) for accelerometer and (rad/s)/sqrt(Hz) for gyro.
    process_noise = {
        'accel_noise_std': 0.028,  # Example: 100 mg / sqrt(Hz) noise
        'gyro_noise_std': np.deg2rad(0.12) # Example: 0.05 deg/s / sqrt(Hz) noise
    }

    measurement_noise = {
        'height': np.array([[0.5**2]]),
        'magnetometer': np.eye(3) * (0.005**2),
        'zero_velocity': np.eye(3) * (0.03**2),
        'horizontal': np.eye(4) * (np.deg2rad(5)**2),
        'zero_position': np.eye(3)*(1**2)
    }

    # ... (The rest of the main.py script is unchanged) ...
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: Input file '{INPUT_FILENAME}' not found.")
        return
    print(f"Loading data from '{INPUT_FILENAME}'...")
    df = pd.read_csv(INPUT_FILENAME)
    required_cols = ['timestamp_ms', 'w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z', 'm_x', 'm_y', 'm_z']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: For initialization, input CSV must contain columns: {required_cols}")
        return
    estimated_states = []
    output_timestamps = []
    is_initialized = False
    warmup_data = {'accel': [], 'gyro': [], 'mag': []}
    first_timestamp = df['timestamp_ms'].iloc[0] / 1000.0
    first_time = df['timestamp_ms'].iloc[0]
    initial_state[6] = df['q0'].iloc[0]
    initial_state[7] = df['q1'].iloc[0]
    initial_state[8] = df['q2'].iloc[0]
    initial_state[9] = df['q3'].iloc[0]
    initial_state[0] = df['Pn'].iloc[0]
    initial_state[1] = df['Pe'].iloc[0]
    initial_state[2] = df['Pd'].iloc[0]
    initial_state[3] = df['Vn'].iloc[0]
    initial_state[4] = df['Ve'].iloc[0]
    initial_state[5] = df['Vd'].iloc[0]
    
    
    ekf = ExtendedKalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)
    last_timestamp = first_timestamp
    last_time = first_time
    print(f"Starting filter. Warm-up for {INITIALIZATION_WARMUP_S}s, then stationary updates for {STATIONARY_DURATION_S}s.")
    for index, row in df.iterrows():
        current_timestamp = row['timestamp_ms'] / 1000.0
        current_time = row['timestamp_ms']
        elapsed_time = current_timestamp - first_timestamp
        if index%1000==0:print(current_timestamp)
        # if current_timestamp>100:break
        if not is_initialized:
            if elapsed_time < INITIALIZATION_WARMUP_S:
                warmup_data['accel'].append(row[['a_x', 'a_y', 'a_z']].values)
                warmup_data['gyro'].append(row[['w_x', 'w_y', 'w_z']].values)
                warmup_data['mag'].append(row[['m_x', 'm_y', 'm_z']].values)
                continue
            else:
                if not warmup_data['accel']:
                    print("Error: No data collected during warm-up. Check timestamps.")
                    return
                avg_accel = np.mean(warmup_data['accel'], axis=0)
                avg_gyro = np.mean(warmup_data['gyro'], axis=0)
                avg_mag = np.mean(warmup_data['mag'], axis=0)
                ekf.initialize_from_stationary(avg_accel, avg_gyro, avg_mag)
                is_initialized = True
                last_timestamp = current_timestamp
                last_time = current_time
        dt = current_timestamp - last_timestamp
        if dt <= 0: continue
        accel = row[['a_x', 'a_y', 'a_z']].values.astype(float)
        gyro = row[['w_x', 'w_y', 'w_z']].values.astype(float)
        mag = row[['m_x', 'm_y', 'm_z']].values.astype(float)
        ekf.predict(accel, gyro, mag, dt)
        if elapsed_time <= STATIONARY_DURATION_S:
            ekf.update_zero_velocity()
            ekf.update_const_pos()
            q = row[['q0','q1','q2','q3']].values.astype(float)
            ekf.update_horizontal_alignment(q)
        if 'h' in row and pd.notna(row['h']):
            ekf.update_height(row['h']-baroBias)
        if 'm_x' in row and pd.notna(row['m_x']):
            mag = row[['m_x', 'm_y', 'm_z']].values.astype(float)
            if np.linalg.norm(mag) > 1e-6 and ekf.mag_ref_ned is not None:
                ekf.update_magnetometer(mag)
        estimated_states.append(ekf.x.copy())
        output_timestamps.append(last_time)
        last_timestamp = current_timestamp
        last_time = current_time
    print("Filter run complete.")
    if not estimated_states:
        print("No states were estimated.")
        return
    columns = ['custom_Pn','custom_Pe','custom_Pd','custom_Vn','custom_Ve','custom_Vd','custom_q0','custom_q1','custom_q2','custom_q3','custom_bias_ax','custom_bias_ay','custom_bias_az','custom_bias_gx','custom_bias_gy','custom_bias_gz','custom_sf_ax','custom_sf_ay','custom_sf_az','custom_sf_gx','custom_sf_gy','custom_sf_gz']
    results_df = pd.DataFrame(estimated_states, columns=columns)
    results_df.insert(0, 'timestamp_ms', output_timestamps)
    combined_df = pd.merge(df,results_df,on='timestamp_ms',how='outer')
    combined_df.to_csv(OUTPUT_FILENAME, index=False, float_format='%.8f')
    print(f"Successfully saved estimated states to '{OUTPUT_FILENAME}'")
    std_df = pd.DataFrame(data = ekf.std)
    std_df.to_csv("std.csv")
if __name__ == '__main__':
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("This script requires the SciPy library. Please install it using: pip install scipy")
        exit()
    main()