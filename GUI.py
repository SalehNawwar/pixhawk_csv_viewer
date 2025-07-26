import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTabWidget, QFileDialog, QStatusBar, QMessageBox,
    QSlider, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QFont

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError:
    QMessageBox.critical(None, "Error", "pyqtgraph is not installed. Please run 'pip install pyqtgraph'")
    sys.exit(1)

def calc_error(val1,val2):
    return np.abs(val1-val2)

def to_str(df,val):
    if val in df.columns:
        return str(np.mean(df[val]))
    return ''

class DataProcessingWorker(QObject):
    result_ready = pyqtSignal(pd.DataFrame)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def process_data(self):
        try:
            df = pd.read_csv(self.file_path)
            if 'timestamp_ms' in df.columns:
                df['time_s'] = df['timestamp_ms'] / 1000.0
                df['time_s'] -= df['time_s'].iloc[0]
            else:
                df['time_s'] = pd.to_timedelta(df.index, unit='ms').total_seconds()

            sources = {
                'ekf': {'q_cols': ['q0', 'q1', 'q2', 'q3'], 'prefix': ''},
                'custom': {'q_cols': ['custom_q0', 'custom_q1', 'custom_q2', 'custom_q3'], 'prefix': 'custom_'},
                'tree': {'q_cols': ['tree_q0', 'tree_q1', 'tree_q2', 'tree_q3'], 'prefix': 'tree_'}
            }

            for name, source in sources.items():
                try:
                    q_cols = source['q_cols']
                    if all(c in df.columns for c in q_cols):
                        valid_q_mask = df[q_cols].notna().all(axis=1)
                        q_data = df.loc[valid_q_mask, q_cols].values
                        if len(q_data) > 0:
                            q_scipy = q_data[:, [1, 2, 3, 0]]
                            norms = np.linalg.norm(q_scipy, axis=1)
                            non_zero_norms = norms > 1e-6
                            q_scipy[non_zero_norms] = q_scipy[non_zero_norms] / norms[non_zero_norms, np.newaxis]
                            euler_rad = R.from_quat(q_scipy).as_euler('xyz', degrees=False)
                            prefix = source['prefix']
                            roll_col = f"{prefix}roll" if prefix else "ekf_roll"
                            pitch_col = f"{prefix}pitch" if prefix else "ekf_pitch"
                            yaw_col = f"{prefix}yaw" if prefix else "ekf_yaw"
                            df.loc[valid_q_mask, [roll_col, pitch_col, yaw_col]] = np.rad2deg(euler_rad)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not process quaternions for '{name}'. Error: {e}")
            
            # --- ROBUST ERROR CALCULATION ---
            # The following blocks will only execute if ALL required source columns exist in the CSV.
            # This prevents errors if a file is missing 'custom' or 'tree' data.

            # Calculate Position Errors (vs EKF)
            if all(c in df.columns for c in ['Pn', 'Pe', 'Pd', 'custom_Pn', 'custom_Pe', 'custom_Pd']):
                df['err_custom_Pn'] = calc_error(df['Pn'] , df['custom_Pn']); df['err_custom_Pe'] = calc_error(df['Pe'] , df['custom_Pe']); df['err_custom_Pd'] = calc_error(df['Pd'] , df['custom_Pd'])
            if all(c in df.columns for c in ['Pn', 'Pe', 'Pd', 'tree_Pn', 'tree_Pe', 'tree_Pd']):
                df['err_tree_Pn'] = calc_error(df['Pn'] , df['tree_Pn']); df['err_tree_Pe'] = calc_error(df['Pe'] , df['tree_Pe']); df['err_tree_Pd'] = calc_error(df['Pd'] - df['tree_Pd'])
            
            # Calculate Velocity Errors (vs EKF)
            if all(c in df.columns for c in ['Vn', 'Ve', 'Vd', 'custom_Vn', 'custom_Ve', 'custom_Vd']):
                df['err_custom_Vn'] = calc_error(df['Vn'] , df['custom_Vn']); df['err_custom_Ve'] = calc_error(df['Ve'] , df['custom_Ve']); df['err_custom_Vd'] = calc_error(df['Vd'] , df['custom_Vd'])
            if all(c in df.columns for c in ['Vn', 'Ve', 'Vd', 'tree_Vn', 'tree_Ve', 'tree_Vd']):
                df['err_tree_Vn'] = calc_error(df['Vn'] , df['tree_Vn']); df['err_tree_Ve'] = calc_error(df['Ve'] , df['tree_Ve']); df['err_tree_Vd'] = calc_error(df['Vd'] - df['tree_Vd'])

            # Calculate Orientation Errors (vs EKF)
            if all(c in df.columns for c in ['ekf_roll', 'custom_roll', 'ekf_pitch', 'custom_pitch', 'ekf_yaw', 'custom_yaw']):
                df['err_custom_roll'] = calculate_angle_error(df['ekf_roll'], df['custom_roll']); df['err_custom_pitch'] = calculate_angle_error(df['ekf_pitch'], df['custom_pitch']); df['err_custom_yaw'] = calculate_angle_error(df['ekf_yaw'], df['custom_yaw'])
            if all(c in df.columns for c in ['ekf_roll', 'tree_roll', 'ekf_pitch', 'tree_pitch', 'ekf_yaw', 'tree_yaw']):
                df['err_tree_roll'] = calculate_angle_error(df['ekf_roll'], df['tree_roll']); df['err_tree_pitch'] = calculate_angle_error(df['ekf_pitch'], df['tree_pitch']); df['err_tree_yaw'] = calculate_angle_error(df['ekf_yaw'], df['tree_yaw'])

            self.result_ready.emit(df)
        except Exception as e:
            self.error.emit(f"Failed to process file: {e}")
        finally:
            self.finished.emit()

def calculate_angle_error(angle1, angle2):
    error = np.abs(angle1 - angle2)
    return np.abs((error + 180) % 360 - 180)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixhawk Data Visualizer (Professional Edition)")
        self.setGeometry(100, 100, 1800, 1000)
        self.data = pd.DataFrame()
        self.is_online_mode = False
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.update_render_frame)
        self.render_interval_ms = 33
        self.current_simulation_time = 0.0
        self.last_frame_wall_time = None
        self.is_paused = False
        self.simulation_speed_multiplier = 1.0
        self.last_rendered_index = -1
        self.ekf_track_line = None; self.custom_track_line = None; self.tree_track_line = None
        self.speed_label = None; self.speed_slider = None
        self.total_sim_time = 0.0
        self.sim_progress_slider = None; self.sim_time_label = None
        self.restart_button = None
        self.thread = None; self.worker = None
        self.init_ui()
        self.set_initial_state()

    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget)
        
        control_panel = QWidget(); control_layout = QHBoxLayout(control_panel); main_layout.addWidget(control_panel)
        self.address_input = QLineEdit("udp:127.0.0.1:14550")
        self.connect_button = QPushButton("📡 Connect")
        self.import_button = QPushButton("📄 Import CSV")
        self.export_button = QPushButton("💾 Export CSV")
        
        control_layout.addWidget(QLabel("Pixhawk Address:"))
        control_layout.addWidget(self.address_input)
        control_layout.addWidget(self.connect_button)
        control_layout.addStretch()
        control_layout.addWidget(self.import_button)
        control_layout.addWidget(self.export_button)
        
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        self.setup_3d_view_tab(); self.setup_track_tab(); self.setup_position_tab(); self.setup_euler_tab(); self.setup_quaternion_tab(); self.setup_sensor_tab()
        
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.connection_status_label = QLabel("Status: Disconnected"); self.status_bar.addWidget(self.connection_status_label)
        self.import_button.clicked.connect(self.import_csv_data); self.export_button.clicked.connect(self.export_csv_data)

    def on_processing_finished(self, processed_df):
        self.data = processed_df
        if self.data.empty: self.on_processing_error("Processed data is empty."); return
        self.status_bar.showMessage("Data processed. Plotting...", 2000)
        self.plot_all_data_2d()
        if 'time_s' in self.data.columns:
            self.total_sim_time = self.data['time_s'].iloc[-1]
            if self.sim_progress_slider: self.sim_progress_slider.setMaximum(len(self.data) - 1); self.sim_progress_slider.setEnabled(True)
            if self.sim_time_label: self.sim_time_label.setText(f"Time: 0.00s / {self.total_sim_time:.2f}s")
        self.reset_simulation()
        self.set_ui_busy(False)
        self.restart_button.setEnabled(True)
        self.status_bar.showMessage("Import complete.", 5000)

    def plot_all_data_2d(self):
        df = self.data; time_axis = df['time_s']
        pens = ["#3A23E9","#E50BBD","#12E924","#04C2F6","#F50707","#FCB527","#C5F710","#9E0DF1","#686A6D","#654308",'#ECEFF4',"#FF6D5A"]

        def plot_group(plot_widget, columns, labels):
            plot_widget.clear()
            plot_widget.addLegend(labelTextSize='10pt') 
            for i, col in enumerate(columns):
                if col in df.columns:
                    plot_widget.plot(time_axis, df[col], pen=pens[i % len(pens)], name=labels[i])
        
        # --- DEFINITIVELY FIXED SECTION FOR 2D TRACK PLOT ---
        self.track_plot.clear()
        self.track_plot.addLegend(labelTextSize='10pt')
        
        # Plot EKF Track (if data exists)
        if all(c in df.columns for c in ['Pe', 'Pn']):
            # The Golden Rule: Drop NA, then Reset Index
            clean_ekf_df = df.dropna(subset=['Pe', 'Pn']).reset_index(drop=True)
            self.track_plot.plot(clean_ekf_df['Pe'], clean_ekf_df['Pn'], pen=pens[0], name='EKF Track')

        # Plot Custom Track (if data exists)
        if all(c in df.columns for c in ['custom_Pe', 'custom_Pn']):
            # 1. Drop rows where either column is NaN.
            # 2. Reset the index to be a clean 0, 1, 2... sequence.
            clean_custom_df = df.dropna(subset=['custom_Pe', 'custom_Pn']).reset_index(drop=True)
            self.track_plot.plot(clean_custom_df['custom_Pe'], clean_custom_df['custom_Pn'], pen=pens[1], name='Custom Track')

        # Plot Tree Track (if data exists)
        if all(c in df.columns for c in ['tree_Pe', 'tree_Pn']):
            # Apply the same robust logic here.
            clean_tree_df = df.dropna(subset=['tree_Pe', 'tree_Pn']).reset_index(drop=True)
            self.track_plot.plot(clean_tree_df['tree_Pe'], clean_tree_df['tree_Pn'], pen=pens[2], name='LSTM Track')
        
        # --- END OF FIXED SECTION ---
        
        # The rest of the function can remain the same.
        # Sensor Tab
        plot_group(self.accel_plot, ['a_x', 'a_y', 'a_z'], labels=['Accel X', 'Accel Y', 'Accel Z'])
        plot_group(self.gyro_plot, ['w_x', 'w_y', 'w_z'], labels=['Gyro X', 'Gyro Y', 'Gyro Z'])
        plot_group(self.mag_plot, ['m_x', 'm_y', 'm_z'], labels=['Mag X', 'Mag Y', 'Mag Z'])
        
        # Position Tab
        pos_cols = ['Pn', 'Pe', 'Pd', 'custom_Pn', 'custom_Pe', 'custom_Pd', 'tree_Pn', 'tree_Pe', 'tree_Pd','Vn', 'Ve', 'Vd', 'custom_Vn', 'custom_Ve', 'custom_Vd', 'tree_Vn', 'tree_Ve', 'tree_Vd']
        pos_lbls = ['EKF N', 'EKF E', 'EKF D', 'Custom N', 'Custom E', 'Custom D', 'LSTM N', 'LSTM E', 'LSTM D','EKF Vn', 'EKF Ve', 'EKF Vd', 'Custom Vn', 'Custom Ve', 'Custom Vd', 'LSTM Vn', 'LSTM Ve', 'LSTM Vd']
        plot_group(self.pos_plot, pos_cols, pos_lbls)

        pos_err_cols = ['err_custom_Pn', 'err_custom_Pe', 'err_custom_Pd', 'err_tree_Pn', 'err_tree_Pe', 'err_tree_Pd','err_custom_Vn', 'err_custom_Ve', 'err_custom_Vd', 'err_tree_Vn', 'err_tree_Ve', 'err_tree_Vd']
        pos_err_lbls = ['N Err (Custom), mae = '+to_str(df,pos_err_cols[0]), 'E Err (Custom), mae = '+to_str(df,pos_err_cols[1]), 'D Err (Custom), mae = '+to_str(df,pos_err_cols[2]), 'N Err (LSTM), mae = '+to_str(df,pos_err_cols[3]), 'E Err (LSTM), mae = '+to_str(df,pos_err_cols[4]), 'D Err (LSTM), mae = '+to_str(df,pos_err_cols[5]),'Vn Err (Custom), mae = '+to_str(df,pos_err_cols[6]), 'Ve Err (Custom), mae = '+to_str(df,pos_err_cols[7]), 'Vd Err (Custom), mae = '+to_str(df,pos_err_cols[8]), 'Vn Err (LSTM), mae = '+to_str(df,pos_err_cols[9]), 'Ve Err (LSTM), mae = '+to_str(df,pos_err_cols[10]), 'Vd Err (LSTM), mae = '+to_str(df,pos_err_cols[11])]
        plot_group(self.pos_error_plot, pos_err_cols, pos_err_lbls)

        # Quaternion Tab
        quat_cols = ['q0', 'q1', 'q2', 'q3', 'custom_q0', 'custom_q1', 'custom_q2', 'custom_q3', 'tree_q0', 'tree_q1', 'tree_q2', 'tree_q3']
        quat_lbls = ['EKF q0','EKF q1','EKF q2','EKF q3', 'Custom q0','Custom q1','Custom q2','Custom q3', 'LSTM q0','LSTM q1','LSTM q2','LSTM q3']
        plot_group(self.quat_plot, quat_cols, quat_lbls)
        
        # Euler Tab
        euler_cols = ['ekf_roll', 'ekf_pitch', 'ekf_yaw', 'custom_roll', 'custom_pitch', 'custom_yaw', 'tree_roll', 'tree_pitch', 'tree_yaw']
        euler_lbls = ['EKF Roll', 'EKF Pitch', 'EKF Yaw', 'Custom Roll', 'Custom Pitch', 'Custom Yaw', 'LSTM Roll', 'LSTM Pitch', 'LSTM Yaw']
        plot_group(self.euler_plot, euler_cols, euler_lbls)

        euler_err_cols = ['err_custom_roll', 'err_custom_pitch', 'err_custom_yaw', 'err_tree_roll', 'err_tree_pitch', 'err_tree_yaw']
        euler_err_lbls = ['Roll Err (Custom), mae = '+to_str(df,euler_err_cols[0]), 'Pitch Err (Custom), mae = '+to_str(df,euler_err_cols[1]), 'Yaw Err (Custom), mae = '+to_str(df,euler_err_cols[2]), 'Roll Err (LSTM), mae = '+to_str(df,euler_err_cols[3]), 'Pitch Err (LSTM), mae = '+to_str(df,euler_err_cols[4]), 'Yaw Err (LSTM), mae = '+to_str(df,euler_err_cols[5])]
        plot_group(self.euler_error_plot, euler_err_cols, euler_err_lbls)

    def setup_position_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        self.pos_plot = self.setup_common_plot_style(pg.PlotWidget(), "Position Comparison (NED)")
        self.pos_error_plot = self.setup_common_plot_style(pg.PlotWidget(), "Position Error (vs. EKF)")
        self.pos_plot.getAxis('left').setLabel('Position (m)'); self.pos_error_plot.getAxis('left').setLabel('Error (m)')
        layout.addWidget(self.pos_plot); layout.addWidget(self.pos_error_plot); self.tabs.addTab(tab, "Position")

    def setup_quaternion_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        self.quat_plot = self.setup_common_plot_style(pg.PlotWidget(), "Quaternion Component Comparison")
        self.quat_plot.getAxis('left').setLabel('Component Value')
        layout.addWidget(self.quat_plot); self.tabs.addTab(tab, "Quaternions")

    def setup_euler_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        self.euler_plot = self.setup_common_plot_style(pg.PlotWidget(), "Orientation Comparison (Euler Angles)")
        self.euler_error_plot = self.setup_common_plot_style(pg.PlotWidget(), "Orientation Error (vs. EKF)")
        self.euler_plot.getAxis('left').setLabel('Angle (degrees)'); self.euler_error_plot.getAxis('left').setLabel('Error (degrees)')
        layout.addWidget(self.euler_plot); layout.addWidget(self.euler_error_plot); self.tabs.addTab(tab, "Orientation")

    def setup_3d_view_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); self.view3d = gl.GLViewWidget(); self.view3d.setBackgroundColor('#2E3440'); self.view3d.setCameraPosition(distance=250, elevation=45, azimuth=45); layout.addWidget(self.view3d, 4)
        grid = gl.GLGridItem(); grid.scale(100, 100, 1); self.view3d.addItem(grid); axes = gl.GLAxisItem(); axes.setSize(5, 5, 5); self.view3d.addItem(axes)
        self.ekf_model = self.create_quad_model(color=(0.53, 0.81, 0.92, 0.9), scale=1.5); self.custom_model = self.create_quad_model(color=(0.76, 0.61, 0.95, 0.9), scale=1.5); self.tree_model = self.create_quad_model(color=(0.64, 0.75, 0.55, 0.9), scale=1.5)
        self.view3d.addItem(self.ekf_model); self.view3d.addItem(self.custom_model); self.view3d.addItem(self.tree_model)
        self.ekf_track_line = gl.GLLinePlotItem(color=(0.53, 0.81, 0.92, 0.8), width=4, antialias=True); self.custom_track_line = gl.GLLinePlotItem(color=(0.76, 0.61, 0.95, 0.8), width=4, antialias=True); self.tree_track_line = gl.GLLinePlotItem(color=(0.64, 0.75, 0.55, 0.8), width=4, antialias=True)
        self.view3d.addItem(self.ekf_track_line); self.view3d.addItem(self.custom_track_line); self.view3d.addItem(self.tree_track_line)
        controls_group = QGroupBox("Simulation Controls"); _3d_layout = QVBoxLayout(controls_group); _3d_layout.setAlignment(Qt.AlignTop); buttons_layout = QHBoxLayout(); self.simulate_button = QPushButton("▶ Play"); self.restart_button = QPushButton("⟲ Restart"); buttons_layout.addWidget(self.simulate_button); buttons_layout.addWidget(self.restart_button); _3d_layout.addLayout(buttons_layout)
        self.sim_progress_slider = QSlider(Qt.Horizontal); self.sim_time_label = QLabel("Time: -- / --"); self.sim_time_label.setAlignment(Qt.AlignCenter); self.speed_label = QLabel("Speed: 1.0x"); self.speed_label.setAlignment(Qt.AlignCenter); self.speed_slider = QSlider(Qt.Horizontal); self.speed_slider.setMinimum(10); self.speed_slider.setMaximum(1000); self.speed_slider.setValue(100); self.speed_slider.setTickInterval(10); self.speed_slider.setTickPosition(QSlider.TicksBelow)
        _3d_layout.addWidget(self.sim_progress_slider); _3d_layout.addWidget(self.sim_time_label); _3d_layout.addStretch(); _3d_layout.addWidget(QLabel("Playback Speed")); _3d_layout.addWidget(self.speed_slider); _3d_layout.addWidget(self.speed_label); layout.addWidget(controls_group, 1)
        self.tabs.addTab(tab, "3D View"); self.simulate_button.clicked.connect(self.toggle_simulation); self.restart_button.clicked.connect(self.reset_simulation); self.speed_slider.valueChanged.connect(self.on_speed_slider_changed); self.sim_progress_slider.sliderMoved.connect(self.seek_simulation)

    def import_csv_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import CSV Data", "", "CSV Files (*.csv);;All Files (*)");
        if not file_path: return
        self.set_ui_busy(True); self.clear_all_visuals(); self.thread = QThread(); self.worker = DataProcessingWorker(file_path); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.process_data); self.worker.finished.connect(self.thread.quit); self.worker.finished.connect(self.worker.deleteLater); self.thread.finished.connect(self.thread.deleteLater); self.worker.result_ready.connect(self.on_processing_finished); self.worker.error.connect(self.on_processing_error); self.thread.start()

    def set_initial_state(self):
        self.is_online_mode = False; self.connect_button.setText("📡 Connect"); self.address_input.setEnabled(True); self.import_button.setEnabled(True); self.export_button.setEnabled(False); self.is_paused = False
        if hasattr(self, 'simulate_button'): self.simulate_button.setEnabled(False); self.simulate_button.setText("▶ Play");
        if self.restart_button: self.restart_button.setEnabled(False)
        if self.sim_progress_slider: self.sim_progress_slider.setEnabled(False); self.sim_progress_slider.setValue(0)
        if self.sim_time_label: self.sim_time_label.setText("Time: -- / --")
        if self.speed_slider: self.speed_slider.setEnabled(False); self.speed_slider.setValue(100)
        if self.speed_label: self.speed_label.setText("Speed: 1.0x")
        self.clear_all_visuals()

    def update_render_frame(self):
        if self.last_frame_wall_time is None or self.data.empty: self._internal_stop(); return
        current_wall_time = time.monotonic(); delta_t = current_wall_time - self.last_frame_wall_time; self.last_frame_wall_time = current_wall_time; self.current_simulation_time += delta_t * self.simulation_speed_multiplier
        is_finished = self.current_simulation_time >= self.total_sim_time
        if is_finished: target_index = len(self.data) - 1; sim_time_display = self.total_sim_time
        else: target_index = self.data['time_s'].searchsorted(self.current_simulation_time, side='right') - 1; target_index = max(0, target_index); sim_time_display = self.current_simulation_time
        if self.sim_time_label: self.sim_time_label.setText(f"Time: {sim_time_display:.2f}s / {self.total_sim_time:.2f}s")
        self.sim_progress_slider.blockSignals(True); self.sim_progress_slider.setValue(target_index); self.sim_progress_slider.blockSignals(False)
        if target_index != self.last_rendered_index: self.render_frame_at_index(target_index)
        if is_finished: self._internal_stop(); self.simulate_button.setText("✔ Finished")

    def toggle_simulation(self):
        if self.render_timer.isActive(): self.pause_simulation()
        elif self.is_paused: self.continue_simulation()
        else: self.start_simulation()

    def start_simulation(self):
        if self.data.empty: self.status_bar.showMessage("No data to simulate.", 3000); return
        if self.current_simulation_time >= self.total_sim_time: self.reset_simulation()
        self.is_paused = False; self.last_frame_wall_time = time.monotonic(); self.render_timer.start(self.render_interval_ms); self.simulate_button.setText("❚❚ Pause")

    def pause_simulation(self): self.render_timer.stop(); self.is_paused = True; self.simulate_button.setText("▶ Continue")
    def continue_simulation(self): self.is_paused = False; self.last_frame_wall_time = time.monotonic(); self.render_timer.start(self.render_interval_ms); self.simulate_button.setText("❚❚ Pause")
    def _internal_stop(self): self.render_timer.stop(); self.last_frame_wall_time = None; self.is_paused = False

    def reset_simulation(self):
        self._internal_stop(); self.current_simulation_time = 0.0; self.last_rendered_index = -1; self.is_paused = False
        if hasattr(self, 'simulate_button'): self.simulate_button.setText("▶ Play")
        if self.sim_progress_slider: self.sim_progress_slider.blockSignals(True); self.sim_progress_slider.setValue(0); self.sim_progress_slider.blockSignals(False)
        if self.sim_time_label:
            if not self.data.empty: self.sim_time_label.setText(f"Time: 0.00s / {self.total_sim_time:.2f}s")
            else: self.sim_time_label.setText("Time: -- / --")
        self.render_frame_at_index(0)

    def seek_simulation(self, index):
        if self.data.empty: return
        self.current_simulation_time = self.data['time_s'].iloc[index]
        if self.sim_time_label: self.sim_time_label.setText(f"Time: {self.current_simulation_time:.2f}s / {self.total_sim_time:.2f}s")
        if self.render_timer.isActive(): self.last_frame_wall_time = time.monotonic()
        self.render_frame_at_index(index)

    def render_frame_at_index(self, index):
        if self.data.empty or not (0 <= index < len(self.data)):
            for model in [self.ekf_model, self.custom_model, self.tree_model]: self.update_3d_model(model, [0,0,0], [1,0,0,0])
            for track in [self.ekf_track_line, self.custom_track_line, self.tree_track_line]:
                if track: track.setData(pos=np.array([]))
            return
        row = self.data.iloc[index]
        sources = {'ekf': {'model': self.ekf_model, 'track': self.ekf_track_line, 'pos': ['Pn', 'Pe', 'Pd'], 'quat': ['q0', 'q1', 'q2', 'q3']},'custom': {'model': self.custom_model, 'track': self.custom_track_line, 'pos': ['custom_Pn', 'custom_Pe', 'custom_Pd'], 'quat': ['custom_q0', 'custom_q1', 'custom_q2', 'custom_q3']},'tree': {'model': self.tree_model, 'track': self.tree_track_line, 'pos': ['tree_Pn', 'tree_Pe', 'tree_Pd'], 'quat': ['tree_q0', 'tree_q1', 'tree_q2', 'tree_q3']}}
        for name, source_data in sources.items():
            pos_cols, quat_cols = source_data['pos'], source_data['quat']
            if all(c in row and pd.notna(row[c]) for c in pos_cols + quat_cols): self.update_3d_model(source_data['model'], row[pos_cols].values, row[quat_cols].values)
            else: self.update_3d_model(source_data['model'], [0,0,0], [1,0,0,0])
            self.update_dynamic_track(source_data['track'], pos_cols, index)
        self.last_rendered_index = index

    def update_dynamic_track(self, line_item, pos_columns, end_index):
        if line_item is None or self.data.empty or not all(c in self.data.columns for c in pos_columns): line_item.hide(); return
        pos_data = self.data.iloc[0:end_index + 1][pos_columns].dropna().values
        if len(pos_data) < 2: line_item.hide()
        else: line_item.show(); gl_pos_data = pos_data[:, [1, 0, 2]]; gl_pos_data[:, 2] *= -1; line_item.setData(pos=gl_pos_data)

    def clear_all_visuals(self):
        for plot_widget in self.findChildren(pg.PlotWidget): plot_widget.clear()
        for track_line in [self.ekf_track_line, self.custom_track_line, self.tree_track_line]:
            if track_line: track_line.hide(); track_line.setData(pos=np.array([]))
        
    def on_speed_slider_changed(self, value): self.simulation_speed_multiplier = value / 100.0; self.speed_label.setText(f"Speed: {self.simulation_speed_multiplier:.1f}x")
    def create_quad_model(self, color, scale=1.0): 
        s=scale;body_verts=s*np.array([[-.5,-.5,-.05],[.5,-.5,-.05],[.5,.5,-.05],[-.5,.5,-.05],[-.5,-.5,.05],[.5,-.5,.05],[.5,.5,.05],[-.5,.5,.05]]);arm_verts=s*np.array([[-2,-.1,-.05],[2,-.1,-.05],[2,.1,-.05],[-2,.1,-.05],[-2,-.1,.05],[2,-.1,.05],[2,.1,.05],[-2,.1,.05]]);rot45=R.from_euler('z',45,degrees=True).as_matrix();arm1=arm_verts@rot45.T;arm2=arm_verts@R.from_euler('z',-45,degrees=True).as_matrix().T;prop_verts=s*np.array([[.4,0,.06],[0,.4,.06],[-.4,0,.06],[0,-.4,.06]]);prop_pos=s*2.;motor_positions=[np.array([p,p,0])@rot45.T for p in[prop_pos,-prop_pos]]+[(np.array([p,-p,0])@rot45.T) for p in[prop_pos,-prop_pos]];all_verts=[body_verts,arm1,arm2];all_faces=[];offset=0;base_faces=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[1,5,6],[1,6,2],[0,3,7],[0,7,4]]);[all_faces.append(base_faces+o)for o in[0,8,16]];all_colors=[np.array([color for _ in base_faces])]*3;fpc=np.array([[.9,.4,.4,.9]]*2);rpc=np.array([list(color[:3])+[.9]]*2);
        for i,pos in enumerate(motor_positions):v=prop_verts+pos;f=np.array([[0,1,2],[0,2,3]])+offset+16;c=fpc if i%2==0 else rpc;all_verts.append(v);all_faces.append(f);all_colors.append(c);offset+=len(v);
        return gl.GLMeshItem(vertexes=np.vstack(all_verts),faces=np.vstack(all_faces),faceColors=np.vstack(all_colors),smooth=True,computeNormals=True,drawEdges=True,edgeColor=(1,1,1,.2))
    def update_3d_model(self, model, pos, quat):
        model.resetTransform(); gl_pos = [pos[1], pos[0], -pos[2]]; q_scipy = [quat[1], quat[2], quat[3], quat[0]]
        try:
            norm = np.linalg.norm(q_scipy);
            if norm < 1e-6: return
            q_scipy = np.array(q_scipy) / norm; rot = R.from_quat(q_scipy); axis_angle = rot.as_rotvec(); angle_rad = np.linalg.norm(axis_angle); angle_deg = np.rad2deg(angle_rad)
            if angle_deg > 0: axis_vec = axis_angle / angle_rad; gl_axis = [axis_vec[1], axis_vec[0], -axis_vec[2]]; model.rotate(angle_deg, gl_axis[0], gl_axis[1], gl_axis[2])
            model.translate(gl_pos[0], gl_pos[1], gl_pos[2])
        except (ValueError, ZeroDivisionError): pass
    def on_processing_error(self, error_message): self.set_ui_busy(False); self.status_bar.showMessage(f"Error: {error_message}", 10000); QMessageBox.critical(self, "Processing Error", error_message)
    def set_ui_busy(self, is_busy):
        self.import_button.setEnabled(not is_busy); self.export_button.setEnabled(not is_busy and not self.data.empty); self.connect_button.setEnabled(not is_busy); has_data = not self.data.empty
        if hasattr(self, 'simulate_button'): self.simulate_button.setEnabled(not is_busy and has_data)
        if hasattr(self, 'restart_button'): self.restart_button.setEnabled(not is_busy and has_data)
        if self.sim_progress_slider: self.sim_progress_slider.setEnabled(not is_busy and has_data)
        if self.speed_slider: self.speed_slider.setEnabled(not is_busy and has_data)
        QApplication.setOverrideCursor(Qt.WaitCursor if is_busy else Qt.ArrowCursor)
    def setup_common_plot_style(self, plot_widget, title): plot_widget.setTitle(title, color='#ECEFF4', size='12pt'); plot_widget.getAxis('bottom').setLabel('Time (s)'); plot_widget.getAxis('bottom').setTextPen('#D8DEE9'); plot_widget.getAxis('left').setTextPen('#D8DEE9'); plot_widget.showGrid(x=True, y=True, alpha=0.2); return plot_widget
    def setup_sensor_tab(self): tab = QWidget(); layout = QVBoxLayout(tab); self.accel_plot = self.setup_common_plot_style(pg.PlotWidget(), "Accelerometers (m/s^2)"); self.gyro_plot = self.setup_common_plot_style(pg.PlotWidget(), "Gyroscopes (rad/s)"); self.mag_plot = self.setup_common_plot_style(pg.PlotWidget(), "Magnetometers (Gauss)"); layout.addWidget(self.accel_plot); layout.addWidget(self.gyro_plot); layout.addWidget(self.mag_plot); self.tabs.addTab(tab, "IMU Sensors")
    def setup_track_tab(self): tab = QWidget(); layout = QVBoxLayout(tab); self.track_plot = self.setup_common_plot_style(pg.PlotWidget(), "2D Track Comparison (North vs East)"); self.track_plot.setAspectLocked(True); self.track_plot.setLabel('left', 'North (m)'); self.track_plot.setLabel('bottom', 'East (m)'); layout.addWidget(self.track_plot); self.tabs.addTab(tab, "2D Track")
    def export_csv_data(self):
        if self.data.empty: self.status_bar.showMessage("No data to export.", 3000); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if path:
            try: self.data.to_csv(path, index=False); self.status_bar.showMessage(f"Data saved to {path}", 5000)
            except Exception as e: self.status_bar.showMessage(f"Error saving file: {e}", 10000)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    nord_stylesheet="""QWidget{background-color:#2E3440;color:#D8DEE9;font-family:"Segoe UI",Arial,sans-serif}QMainWindow,QStatusBar{background-color:#2E3440}QStatusBar QLabel{color:#E5E9F0}QTabWidget::pane{border-top:2px solid #4C566A}QTabBar::tab{background-color:#3B4252;color:#D8DEE9;padding:10px 25px;border:1px solid #3B4252;border-bottom:none;border-top-left-radius:5px;border-top-right-radius:5px}QTabBar::tab:hover{background-color:#434C5E}QTabBar::tab:selected{background-color:#88C0D0;color:#2E3440;font-weight:bold}QPushButton{background-color:#4C566A;border:none;padding:8px 16px;border-radius:5px;font-weight:bold}QPushButton:hover{background-color:#5E81AC}QPushButton:pressed{background-color:#81A1C1}QPushButton:disabled{background-color:#3B4252;color:#4C566A}QLineEdit{background-color:#3B4252;border:1px solid #4C566A;padding:8px;border-radius:5px;color:#ECEFF4}QSlider::groove:horizontal{border:1px solid #4C566A;height:4px;background:#3B4252;margin:2px 0;border-radius:2px}QSlider::handle:horizontal{background:#88C0D0;border:none;width:18px;margin:-7px 0;border-radius:9px}QGroupBox{font-weight:bold;border:1px solid #4C566A;border-radius:6px;margin-top:12px;color:#ECEFF4}QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top center;padding:5px 10px}"""
    app.setStyleSheet(nord_stylesheet)
    pg.setConfigOption('background', '#2E3440'); pg.setConfigOption('foreground', '#D8DEE9')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())