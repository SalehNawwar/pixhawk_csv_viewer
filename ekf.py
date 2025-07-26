# ekf.py

import numpy as np
from scipy.spatial.transform import Rotation

class ExtendedKalmanFilter:
    """
    A 22-state Extended Kalman Filter for IMU sensor fusion.
    Now includes a dedicated initialization routine.
    """
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.x = initial_state
        self.P = initial_covariance
        # process_noise will now contain 'accel_noise_std' and 'gyro_noise_std'
        self.q_proc_noise = process_noise 
        self.R = measurement_noise
        self.g = np.array([0, 0, 9.81])
        self.mag_ref_ned = None
        self.std = [np.sqrt(np.diag(self.P))]
        # self.q_magnet = [1,0,0,0]

    def initialize_from_stationary(self, avg_accel, avg_gyro, avg_mag):
        print("EKF: Performing analytic initialization from stationary data...")
        # self.g = avg_accel
        self.x[13:16] = avg_gyro
        # accel_norm = avg_accel / np.linalg.norm(avg_accel)
        # pitch = np.arcsin(accel_norm[0])
        # roll = -np.arcsin(accel_norm[1] / np.cos(pitch))
        # r_roll = Rotation.from_euler('x', roll).as_matrix()
        # r_pitch = Rotation.from_euler('y', pitch).as_matrix()
        # mag_horizontal = r_pitch @ r_roll @ avg_mag
        # yaw = np.arctan2(-mag_horizontal[1], mag_horizontal[0])
        # self.x[6:10] = Rotation.from_euler('zyx', [yaw, pitch, roll]).as_quat()[[3, 0, 1, 2]]
        g_mag = np.linalg.norm(avg_accel)
        self.g = np.array([0, 0, g_mag])
        print(f"EKF: Estimated gravity magnitude: {g_mag:.4f} m/s^2")
        C_n_b = self._quat_to_rot_matrix(self.x[6:10]).T
        g_body = C_n_b @ self.g
        self.x[10:13] = avg_accel + g_body
        C_b_n = self._quat_to_rot_matrix(self.x[6:10])
        self.mag_ref_ned = C_b_n @ avg_mag
        self.mag_ref_ned /= np.linalg.norm(self.mag_ref_ned)
        print(f"EKF: Estimated North-pointing magnetic vector: {self.mag_ref_ned}")
        self.P[6:10, 6:10] = np.eye(4) * 0.1**2
        self.P[10:16, 10:16] = np.eye(6) * 0.04**2
        
    def update_horizontal_alignment(self,q):
        """
        Applies a measurement update assuming the vehicle is perfectly horizontal.
        This means roll and pitch are zero. The measurement is that qx and qy are zero.
        """
        # Measurement is zero for qx and qy
        # z = np.zeros(2)
        z = q
        # Measurement model: h(x) = [qx, qy]
        # z_pred = self.x[7:9] # qx, qy
        z_pred = self.x[6:10]
        # Jacobian H simply picks out qx and qy from the state vector
        H = np.zeros((4, 22))
        H[:,6:10] = np.eye(4)
        # H[0, 7] = 1.0 # d(qx)/d(qx)
        # H[1, 8] = 1.0 # d(qy)/d(qy)
        
        y = z - z_pred
        S = H @ self.P @ H.T + self.R['horizontal']
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(22) - K @ H) @ self.P
        self._normalize_quaternion()

    # --- Predict method and other updates remain the same ---
    # ... (The rest of the ekf.py code is unchanged from the static gravity version) ...
    def predict(self, accel_raw, gyro_raw, mag_raw, dt):
        """
        Predicts the next state and covariance using IMU data.
        """
        # --- 1. State Propagation (Unchanged) ---
        pos, vel, q = self.x[0:3], self.x[3:6], self.x[6:10]
        bias_a, bias_g = self.x[10:13], self.x[13:16]
        sf_a, sf_g = self.x[16:19], self.x[19:22]
        
        accel = (accel_raw - bias_a) / (1.0 + sf_a)
        gyro = (gyro_raw - bias_g) / (1.0 + sf_g)

        dq = 0.5 * np.array(
            [-q[1]*gyro[0]-q[2]*gyro[1]-q[3]*gyro[2], 
             q[0]*gyro[0]+q[2]*gyro[2]-q[3]*gyro[1], 
             q[0]*gyro[1]-q[1]*gyro[2]+q[3]*gyro[0], 
             q[0]*gyro[2]+q[1]*gyro[1]-q[2]*gyro[0]]
            )
        q_new = q + dq * dt

        # mag_norm = mag_raw/np.linalg.norm(mag_raw)
        # angle1 = np.acos(np.dot(mag_norm,self.mag_ref_ned))
        # axis1 = np.cross(self.mag_ref_ned,mag_norm)
        # axis2 = -axis1
        # axis1 = axis1*np.sin(angle1/2)
        # # angle2 = 2*np.pi - np.acos(np.dot(mag_norm,self.mag_ref_ned))
        # axis2 =  axis2*np.sin(angle1/2)
        # q_new1 = np.array([np.cos(angle1/2),axis1[0],axis1[1],axis1[2]])
        # q_new2 = np.array([np.cos(angle1/2),axis2[0],axis2[1],axis2[2]])
        # n1 = np.linalg.norm(q_new1-q)
        # n2 = np.linalg.norm(q_new2-q)
        # if n1<n2:q_new = q_new1 
        # else :q_new = q_new2

        C_b_n = self._quat_to_rot_matrix(q)
        vel_dot = (C_b_n @ accel) + self.g
        vel_new = vel + vel_dot * dt
        pos_new = pos + vel * dt + 0.5 * vel_dot * dt**2
        
        self.x[0:3], self.x[3:6], self.x[6:10] = pos_new, vel_new, q_new
        self._normalize_quaternion()

        # --- 2. Jacobian of State Transition (Unchanged) ---
        F = np.eye(22)
        F[0:3, 3:6] = np.eye(3)*dt
        F[3:6, 6:10] = self._jacobian_dvel_dq(q, accel) * dt
        F[3:6, 10:13] = -C_b_n * dt
        F[3:6, 16:19] = -C_b_n @ np.diag(accel) * dt
        F[6:10, 6:10] = self._jacobian_dq_dq(gyro) * dt + np.eye(4)
        # F[6:10, 13:16] = -0.5 * self._q_mult_matrix(q)[:, 1:] * dt
        # F[6:10, 19:22] = -0.5 * self._q_mult_matrix(q)[:, 1:] @ np.diag(gyro) * dt
        
        # F[0:3, 3:6] = np.eye(3) * dt
        # F[3:6, 6:10] = self._jacobian_dvel_dq(q, accel) * dt
        # F[3:6, 10:13] = -C_b_n * dt
        # F[3:6, 16:19] = -C_b_n @ np.diag(accel) * dt
        # F[6:10, 6:10] = self._jacobian_dq_dq(gyro) * dt + np.eye(4)
        # F[6:10, 13:16] = -0.5 * self._q_mult_matrix(q)[:, 1:] * dt
        # F[6:10, 19:22] = -0.5 * self._q_mult_matrix(q)[:, 1:] @ np.diag(gyro) * dt
        

        # --- 3. MODIFICATION: Calculate Process Noise Covariance Q ---
        # Based on the physically-driven model: Q = G * Q_w * G^T * dt
        
        # G is the 22x6 Noise Input Matrix
        G = np.zeros((22, 6))
        
        # Gyro noise affects attitude (quaternions)
        # d(quat)/d(noise_g) = 0.5 * Xi(q)
        G[6:10, 3:6] = 0.5 * self._q_mult_matrix(q)[:, 1:]
        
        # Accel noise affects velocity
        # d(vel)/d(noise_a) = C_b_n
        G[3:6, 0:3] = C_b_n
        
        # Q_w is the 6x6 covariance matrix of the IMU noise sources
        var_a = self.q_proc_noise['accel_noise_std']**2
        var_g = self.q_proc_noise['gyro_noise_std']**2
        Q_w = np.diag([var_a, var_a, var_a, var_g, var_g, var_g])
        
        # Final discrete process noise matrix Q
        Q = G @ Q_w @ G.T * dt
        
        # --- 4. Propagate State Covariance ---
        self.P = F @ self.P @ F.T + Q
        self.std.append(np.sqrt(np.diag(self.P)))

    def _normalize_quaternion(self): q_norm = np.linalg.norm(self.x[6:10]); self.x[6:10] = self.x[6:10]/q_norm if q_norm > 0 else self.x[6:10]
    def _skew_symmetric(self, v): return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    def _quat_to_rot_matrix(self, q): 
        qw, qx, qy, qz = q; 
        return np.array(
            [[qw*qw+qx*qx-qy*qy-qz*qz, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)], 
              [2*(qx*qy+qw*qz), qw*qw-qx*qx+qy*qy-qz*qz, 2*(qy*qz-qw*qx)], 
              [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), qw*qw-qx*qx-qy*qy+qz*qz]])
    
    def update_zero_velocity(self): 
        H=np.zeros((3,22)); 
        H[0:3,3:6]=np.eye(3); 
        y=np.zeros(3)-self.x[3:6]; 
        S=H@self.P@H.T+self.R['zero_velocity']; 
        K=self.P@H.T@np.linalg.inv(S); 
        self.x=self.x+K@y; 
        self.P=(np.eye(22)-K@H)@self.P;
        self.std.append(np.sqrt(np.diag(self.P))) 
        self._normalize_quaternion()

    def update_const_pos(self): 
        H=np.zeros((3,22)); 
        H[0:3,0:3]=np.eye(3); 
        y=np.zeros(3)-self.x[0:3]; 
        S=H@self.P@H.T+self.R['zero_position']; 
        K=self.P@H.T@np.linalg.inv(S); 
        self.x=self.x+K@y; 
        self.P=(np.eye(22)-K@H)@self.P;
        self.std.append(np.sqrt(np.diag(self.P))) 
        self._normalize_quaternion()

    def update_height(self, h): 
        H=np.zeros((1,22)); 
        H[0,2]=-1.0; 
        y=h-H@self.x; 
        S=H@self.P@H.T+self.R['height']; 
        K=self.P@H.T@np.linalg.inv(S); 
        # self.x=self.x+K.flatten()*y; 
        self.x=self.x+K@y; 
        
        self.P=(np.eye(22)-K@H)@self.P; 
        self.std.append(np.sqrt(np.diag(self.P)))
        self._normalize_quaternion()

    def update_magnetometer(self, mag): 
        mag_norm=mag/np.linalg.norm(mag); 
        q=self.x[6:10]; 
        C_n_b=self._quat_to_rot_matrix(q).T; 
        z_pred=C_n_b@self.mag_ref_ned; 
        H=np.zeros((3,22)); 
        # zz = self._jacobian_dmag_dq(q,self.mag_ref_ned)
        H[:,6:10]=self._jacobian_dmag_dq(q,self.mag_ref_ned); 
        y=mag_norm-z_pred; 
        S=H@self.P@H.T+self.R['magnetometer']; 
        K=self.P@H.T@np.linalg.inv(S); 
        self.x=self.x+K@y; 
        self.P=(np.eye(22)-K@H)@self.P; 
        self.std.append(np.sqrt(np.diag(self.P)))
        self._normalize_quaternion()

    def _jacobian_dvel_dq(self, q, accel): 
        qw,qx,qy,qz=q; 
        ax,ay,az=accel; 
        return 2*np.array([
            [qw*ax-qz*ay+qy*az, qx*ax+qy*ay+qz*az, -qy*ax+qx*ay+qw*az, -qz*ax-qw*ay+qx*az],
            [qz*ax+qw*ay-qx*az, qy*ax-qx*ay-qw*az, qx*ax+qy*ay+qz*az, +qw*ax-qz*ay+qy*az],
            [-qy*ax+qx*ay+qw*az, qz*ax+qw*ay-qx*az, -qw*ax+qz*ay-qy*az, qx*ax+qy*ay+qz*az]
            ])
    def _jacobian_dq_dq(self, gyro): 
        gx,gy,gz=gyro; 
        return 0.5*np.array([
            [0,-gx,-gy,-gz],
            [gx,0,gz,-gy],
            [gy,-gz,0,gx],
            [gz,gy,-gx,0]
            ])
    def _q_mult_matrix(self,q): 
        qw,qx,qy,qz=q; 
        return np.array([
            [qw,-qx,-qy,-qz],
            [qx,qw,-qz,qy],
            [qy,qz,qw,-qx],
            [qz,-qy,qx,qw]
            ])
    def _jacobian_dmag_dq(self,q,mag_ref): 
        qw,qx,qy,qz=q; 
        mx,my,mz=mag_ref; 
        ret = 2*np.array([
            [qw*mx+qz*my-qy*mz  ,qx*mx+qy*my+qz*mz  ,-qy*mx+qx*my-qw*mz ,-qz*mx+qw*my+qx*mz],
            [-qz*mx+qw*my+qx*mz ,qy*mx-qx*my+qw*mz  ,qx*mx+qy*my+qz*mz  ,-qw*mx-qz*my+qy*mz],
            [qy*mx-qx*my+qw*mz  ,+qz*mx-qw*my-qx*mz ,qw*mx+qz*my-qy*mz  ,qx*mx+qy*my+qz*mz]
            ])
        return ret
    # return 2*np.array([
    #         [qw*mx+qz*my-qy*mz  ,qx*mx+qy*my+qz*mz  ,-qy*mx+qx*my-qw*mz ,-qz*mx+qw*my+qx*mz],
    #         [-qz*mx+qw*my+qx*mz ,qy*mx-qx*my+qw*mz  ,qx*mx+qy*my+qz*mz  ,qw*mx+qz*my-qy*mz],
    #         [qy*mx-qx*my-qw*mz  ,-qz*mx+qw*my+qx*mz ,qw*mx+qz*my-qy*mz  ,qx*mx+qy*my+qz*mz]
    #         ])
    def _q_conj(self,q): return np.array([q[0],-q[1],-q[2],-q[3]])