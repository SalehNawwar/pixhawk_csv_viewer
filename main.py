# main_pyqt5_final_and_complete.py

import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTabWidget, QFileDialog, QStatusBar, QMessageBox,
    QProgressBar, QSlider
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QColor

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError:
    QMessageBox.critical(None, "Error", "pyqtgraph is not installed. Please run 'pip install pyqtgraph'")
    sys.exit(1)

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
                df['time_s'] = df.index
            q_cols = ['q0', 'q1', 'q2', 'q3']; tree_q_cols = ['tree_q0', 'tree_q1', 'tree_q2', 'tree_q3']
            try:
                if all(c in df.columns for c in q_cols):
                    valid_q_mask = df[q_cols].notna().all(axis=1)
                    q_ekf_data = df.loc[valid_q_mask, q_cols].values
                    if len(q_ekf_data) > 0:
                        q_scipy = q_ekf_data[:, [1, 2, 3, 0]]
                        norms = np.linalg.norm(q_scipy, axis=1)
                        non_zero_norms = norms > 1e-6
                        q_scipy[non_zero_norms] = q_scipy[non_zero_norms] / norms[non_zero_norms, np.newaxis]
                        euler_rad = R.from_quat(q_scipy).as_euler('xyz', degrees=False)
                        df.loc[valid_q_mask, ['ekf_roll', 'ekf_pitch', 'ekf_yaw']] = np.rad2deg(euler_rad)
                if all(c in df.columns for c in tree_q_cols):
                    valid_tree_q_mask = df[tree_q_cols].notna().all(axis=1)
                    q_tree_data = df.loc[valid_tree_q_mask, tree_q_cols].values
                    if len(q_tree_data) > 0:
                        q_scipy = q_tree_data[:, [1, 2, 3, 0]]
                        norms = np.linalg.norm(q_scipy, axis=1)
                        non_zero_norms = norms > 1e-6
                        q_scipy[non_zero_norms] = q_scipy[non_zero_norms] / norms[non_zero_norms, np.newaxis]
                        euler_rad = R.from_quat(q_scipy).as_euler('xyz', degrees=False)
                        df.loc[valid_tree_q_mask, ['tree_roll', 'tree_pitch', 'tree_yaw']] = np.rad2deg(euler_rad)
            except ValueError as e:
                print(f"Warning: Could not process some quaternions. Error: {e}")
            if all(c in df.columns for c in ['Pn', 'Pe', 'Pd', 'tree_Pn', 'tree_Pe', 'tree_Pd']):
                df['err_Pn'] = df['Pn'] - df['tree_Pn']; df['err_Pe'] = df['Pe'] - df['tree_Pe']; df['err_Pd'] = df['Pd'] - df['tree_Pd']
            if all(c in df.columns for c in ['ekf_roll', 'tree_roll', 'ekf_pitch', 'tree_pitch', 'ekf_yaw', 'tree_yaw']):
                df['err_roll'] = calculate_angle_error(df['ekf_roll'], df['tree_roll']); df['err_pitch'] = calculate_angle_error(df['ekf_pitch'], df['tree_pitch']); df['err_yaw'] = calculate_angle_error(df['ekf_yaw'], df['tree_yaw'])
            self.result_ready.emit(df)
        except Exception as e:
            self.error.emit(f"Failed to process file: {e}")
        finally:
            self.finished.emit()

def calculate_angle_error(angle1, angle2):
    error = angle1 - angle2
    return (error + 180) % 360 - 180

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixhawk Data Visualizer (Final)")
        self.setGeometry(100, 100, 1600, 900)
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
        self.ekf_track_line = None; self.tree_track_line = None
        self.speed_label = None; self.speed_slider = None
        self.total_sim_time = 0.0
        self.sim_progress_bar = None; self.sim_time_label = None
        self.thread = None; self.worker = None
        self.init_ui()
        self.set_initial_state()

    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget)
        control_panel = QWidget(); control_layout = QHBoxLayout(control_panel); main_layout.addWidget(control_panel)
        self.address_input = QLineEdit("udp:127.0.0.1:14550"); self.connect_button = QPushButton("Connect"); self.import_button = QPushButton("Import CSV"); self.export_button = QPushButton("Export CSV")
        control_layout.addWidget(QLabel("Pixhawk Address:")); control_layout.addWidget(self.address_input); control_layout.addWidget(self.connect_button); control_layout.addStretch(); control_layout.addWidget(self.import_button); control_layout.addWidget(self.export_button)
        self.tabs = QTabWidget(); main_layout.addWidget(self.tabs)
        self.setup_sensor_tab(); self.setup_track_tab(); self.setup_position_tab(); self.setup_quaternion_tab(); self.setup_euler_tab(); self.setup_3d_view_tab()
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.connection_status_label = QLabel("Status: Disconnected"); self.status_bar.addWidget(self.connection_status_label)
        self.import_button.clicked.connect(self.import_csv_data); self.export_button.clicked.connect(self.export_csv_data)

    def setup_3d_view_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); self.view3d = gl.GLViewWidget(); self.view3d.setBackgroundColor('k'); self.view3d.setCameraPosition(distance=40, elevation=30, azimuth=45); layout.addWidget(self.view3d, 4)
        grid = gl.GLGridItem(); grid.scale(5, 5, 1); self.view3d.addItem(grid); axes = gl.GLAxisItem(); axes.setSize(10, 10, 10); self.view3d.addItem(axes)
        text_color = QColor('white'); self.view3d.addItem(gl.GLTextItem(pos=(11, 0, 0), text='E', color=text_color, font=pg.QtGui.QFont('Helvetica', 12))); self.view3d.addItem(gl.GLTextItem(pos=(0, 11, 0), text='N', color=text_color, font=pg.QtGui.QFont('Helvetica', 12))); self.view3d.addItem(gl.GLTextItem(pos=(0, 0, 11), text='Up', color=text_color, font=pg.QtGui.QFont('Helvetica', 12)))
        self.ekf_model = self.create_quad_model(color=(0, 1, 0, 0.8)); self.tree_model = self.create_quad_model(color=(0, 0, 1, 0.8)); self.view3d.addItem(self.ekf_model); self.view3d.addItem(self.tree_model)
        self.ekf_track_line = gl.GLLinePlotItem(color=(0, 1, 0, 0.6), width=2, antialias=True); self.tree_track_line = gl.GLLinePlotItem(color=(0, 0, 1, 0.6), width=2, antialias=True); self.view3d.addItem(self.ekf_track_line); self.view3d.addItem(self.tree_track_line)
        _3d_controls = QWidget(); _3d_layout = QVBoxLayout(_3d_controls); _3d_layout.setAlignment(Qt.AlignTop)
        self.simulate_button = QPushButton("Start Simulation"); self.sim_progress_bar = QProgressBar(); self.sim_time_label = QLabel("Time: -- / --"); self.sim_time_label.setAlignment(Qt.AlignCenter)
        self.speed_label = QLabel("Speed: 1.0x"); self.speed_label.setAlignment(Qt.AlignCenter); self.speed_slider = QSlider(Qt.Horizontal); self.speed_slider.setMinimum(10); self.speed_slider.setMaximum(400); self.speed_slider.setValue(100); self.speed_slider.setTickInterval(10); self.speed_slider.setTickPosition(QSlider.TicksBelow)
        _3d_layout.addWidget(self.simulate_button); _3d_layout.addWidget(self.sim_progress_bar); _3d_layout.addWidget(self.sim_time_label); _3d_layout.addWidget(self.speed_label); _3d_layout.addWidget(self.speed_slider); layout.addWidget(_3d_controls, 1)
        self.tabs.addTab(tab, "3D View")
        self.simulate_button.clicked.connect(self.toggle_simulation); self.speed_slider.valueChanged.connect(self.on_speed_slider_changed)

    # --- THE MISSING METHODS ARE RESTORED HERE ---

    def import_csv_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import CSV Data", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        self.set_ui_busy(True)
        self.clear_all_visuals()
        self.thread = QThread()
        self.worker = DataProcessingWorker(file_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.process_data)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.result_ready.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.thread.start()

    def on_processing_finished(self, processed_df):
        self.data = processed_df
        if self.data.empty:
            self.on_processing_error("Processed data frame is empty. Check CSV content.")
            return
            
        self.status_bar.showMessage("Data processed. Plotting...", 2000)
        self.plot_all_data_2d()
        
        if not self.data.empty and 'time_s' in self.data.columns:
            self.total_sim_time = self.data['time_s'].iloc[-1]
            if self.sim_progress_bar:
                self.sim_progress_bar.setMaximum(len(self.data) - 1)
                self.sim_progress_bar.setEnabled(True)
            if self.sim_time_label:
                self.sim_time_label.setText(f"Time: 0.00s / {self.total_sim_time:.2f}s")
                
        self.reset_simulation()
        self.set_ui_busy(False)
        self.status_bar.showMessage("Import and plotting complete.", 5000)

    def plot_all_data_2d(self):
        df = self.data
        if df.empty:
            return
        time_axis = df['time_s']
        
        def plot_group(plot_widget, columns, prefix="", labels=None):
            pens = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
            if labels is None:
                labels = [f"{prefix}{col.split('_')[-1].upper()}" for col in columns]
            for i, col in enumerate(columns):
                if col in df.columns:
                    plot_widget.plot(time_axis, df[col].dropna(), pen=pens[i], name=labels[i])
                    
        plot_group(self.accel_plot, ['a_x', 'a_y', 'a_z'])
        plot_group(self.gyro_plot, ['w_x', 'w_y', 'w_z'])
        plot_group(self.mag_plot, ['m_x', 'm_y', 'm_z'])
        
        if all(c in df.columns for c in ['Pe', 'Pn']):
            self.track_plot.plot(df['Pe'].dropna(), df['Pn'].dropna(), pen='g', name='EKF Track')
        if all(c in df.columns for c in ['tree_Pe', 'tree_Pn']):
            self.track_plot.plot(df['tree_Pe'].dropna(), df['tree_Pn'].dropna(), pen='b', name='Tree Track')
            
        plot_group(self.pos_ekf_plot, ['Pn', 'Pe', 'Pd'], 'EKF ')
        plot_group(self.pos_tree_plot, ['tree_Pn', 'tree_Pe', 'tree_Pd'], 'Tree ')
        plot_group(self.pos_error_plot, ['err_Pn', 'err_Pe', 'err_Pd'], 'Err ')
        
        plot_group(self.quat_ekf_plot, ['q0', 'q1', 'q2', 'q3'], 'q')
        plot_group(self.quat_tree_plot, ['tree_q0', 'tree_q1', 'tree_q2', 'tree_q3'], 'tree_q')
        
        euler_labels = ['Roll', 'Pitch', 'Yaw']
        plot_group(self.euler_ekf_plot, ['ekf_roll', 'ekf_pitch', 'ekf_yaw'], labels=[f"EKF {l}" for l in euler_labels])
        plot_group(self.euler_tree_plot, ['tree_roll', 'tree_pitch', 'tree_yaw'], labels=[f"Tree {l}" for l in euler_labels])
        plot_group(self.euler_error_plot, ['err_roll', 'err_pitch', 'err_yaw'], labels=[f"Err {l}" for l in euler_labels])

    # --- ALL OTHER METHODS FOLLOW ---

    def set_initial_state(self):
        self.is_online_mode = False; self.connect_button.setText("Connect"); self.address_input.setEnabled(True); self.import_button.setEnabled(True); self.export_button.setEnabled(False)
        self.is_paused = False
        if hasattr(self, 'simulate_button'):
            self.simulate_button.setEnabled(False); self.simulate_button.setText("Start Simulation")
            if self.sim_progress_bar: self.sim_progress_bar.setEnabled(False); self.sim_progress_bar.setValue(0)
            if self.sim_time_label: self.sim_time_label.setText("Time: -- / --")
            if self.speed_slider: self.speed_slider.setEnabled(False); self.speed_slider.setValue(100)
            if self.speed_label: self.speed_label.setText("Speed: 1.0x")
        self.clear_all_visuals()

    def update_render_frame(self):
        if self.last_frame_wall_time is None or self.data.empty:
            self._internal_stop(); return
        current_wall_time = time.monotonic(); delta_t = current_wall_time - self.last_frame_wall_time; self.last_frame_wall_time = current_wall_time
        self.current_simulation_time += delta_t * self.simulation_speed_multiplier
        target_index = 0; is_finished = self.current_simulation_time >= self.total_sim_time
        if is_finished: target_index = len(self.data) - 1
        else: target_index = self.data['time_s'].searchsorted(self.current_simulation_time, side='right') - 1; target_index = max(0, target_index)
        sim_time_display = min(self.current_simulation_time, self.total_sim_time)
        if self.sim_progress_bar: self.sim_progress_bar.setValue(target_index)
        if self.sim_time_label: self.sim_time_label.setText(f"Time: {sim_time_display:.2f}s / {self.total_sim_time:.2f}s")
        if target_index == self.last_rendered_index and not is_finished: return
        self.last_rendered_index = target_index
        row = self.data.iloc[target_index]
        ekf_pos_cols = ['Pn', 'Pe', 'Pd']; ekf_quat_cols = ['q0', 'q1', 'q2', 'q3']
        if all(c in row and pd.notna(row[c]) for c in ekf_pos_cols + ekf_quat_cols): self.update_3d_model(self.ekf_model, row[ekf_pos_cols].values, row[ekf_quat_cols].values)
        tree_pos_cols = ['tree_Pn', 'tree_Pe', 'tree_Pd']; tree_quat_cols = ['tree_q0', 'tree_q1', 'tree_q2', 'tree_q3']
        if all(c in row and pd.notna(row[c]) for c in tree_pos_cols + tree_quat_cols): self.update_3d_model(self.tree_model, row[tree_pos_cols].values, row[tree_quat_cols].values)
        self.update_dynamic_track(self.ekf_track_line, ekf_pos_cols, target_index); self.update_dynamic_track(self.tree_track_line, tree_pos_cols, target_index)
        if is_finished:
            self._internal_stop()
            if self.simulate_button: self.simulate_button.setText("Simulation Finished")

    def toggle_simulation(self):
        if self.render_timer.isActive(): self.pause_simulation()
        elif self.is_paused: self.continue_simulation()
        else: self.start_simulation()

    def start_simulation(self):
        if self.data.empty: self.status_bar.showMessage("No data loaded for simulation.", 3000); return
        self.reset_simulation(); self.is_paused = False; self.last_frame_wall_time = time.monotonic(); self.render_timer.start(self.render_interval_ms); self.simulate_button.setText("Pause")

    def pause_simulation(self):
        self.render_timer.stop(); self.is_paused = True; self.simulate_button.setText("Continue")

    def continue_simulation(self):
        self.is_paused = False; self.last_frame_wall_time = time.monotonic(); self.render_timer.start(self.render_interval_ms); self.simulate_button.setText("Pause")

    def _internal_stop(self):
        self.render_timer.stop(); self.last_frame_wall_time = None; self.is_paused = False

    def reset_simulation(self):
        self._internal_stop(); self.last_rendered_index = -1; self.current_simulation_time = 0.0
        if hasattr(self, 'simulate_button'): self.simulate_button.setText("Start Simulation")
        if self.ekf_track_line: self.ekf_track_line.hide(); self.ekf_track_line.setData(pos=np.array([]))
        if self.tree_track_line: self.tree_track_line.hide(); self.tree_track_line.setData(pos=np.array([]))
        if self.sim_progress_bar and not self.data.empty: self.sim_progress_bar.setValue(0); self.sim_time_label.setText(f"Time: 0.00s / {self.total_sim_time:.2f}s")
        self.update_3d_model(self.ekf_model, [0,0,0], [1,0,0,0]); self.update_3d_model(self.tree_model, [0,0,0], [1,0,0,0])

    def update_dynamic_track(self, line_item, pos_columns, end_index):
        if line_item is None or self.data.empty: return
        if not all(c in self.data.columns for c in pos_columns): line_item.hide(); return
        pos_data = self.data.iloc[0:end_index + 1][pos_columns].dropna().values
        if len(pos_data) < 2: line_item.hide()
        else: line_item.show(); gl_pos_data = pos_data[:, [1, 0, 2]]; gl_pos_data[:, 2] *= -1; line_item.setData(pos=gl_pos_data)

    def clear_all_visuals(self):
        for plot_widget in self.findChildren(pg.PlotWidget): plot_widget.clear()
        if self.ekf_track_line: self.ekf_track_line.hide(); self.ekf_track_line.setData(pos=np.array([]))
        if self.tree_track_line: self.tree_track_line.hide(); self.tree_track_line.setData(pos=np.array([]))
        
    def on_speed_slider_changed(self, value):
        self.simulation_speed_multiplier = value / 100.0; self.speed_label.setText(f"Speed: {self.simulation_speed_multiplier:.1f}x")

    def create_quad_model(self, color):
        body_verts = np.array([[-0.5,-0.5,-0.1],[0.5,-0.5,-0.1],[0.5,0.5,-0.1],[-0.5,0.5,-0.1],[-0.5,-0.5,0.1],[0.5,-0.5,0.1],[0.5,0.5,0.1],[-0.5,0.5,0.1]]); body_faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]]); body_colors=np.array([color for _ in range(12)])
        arm1_verts = np.array([[-2,-0.2,-0.1],[2,-0.2,-0.1],[2,0.2,-0.1],[-2,0.2,-0.1],[-2,-0.2,0.1],[2,-0.2,0.1],[2,0.2,0.1],[-2,0.2,0.1]]); arm2_verts=np.array([[-0.2,-2,-0.1],[0.2,-2,-0.1],[0.2,2,-0.1],[-0.2,2,-0.1],[-0.2,-2,0.1],[0.2,-2,0.1],[0.2,2,0.1],[-0.2,2,0.1]])
        fwd_verts = np.array([[-0.3,1.5,0.1],[0.3,1.5,0.1],[0.3,2,0.1],[-0.3,2,0.1],[-0.3,1.5,0.3],[0.3,1.5,0.3],[0.3,2,0.3],[-0.3,2,0.3]]); fwd_faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]]); fwd_colors = np.array([[1,0,0,0.8] for _ in range(12)])
        all_verts = np.vstack([body_verts, arm1_verts, arm2_verts, fwd_verts]); all_faces = np.vstack([body_faces, body_faces + 8, body_faces + 16, fwd_faces + 24]); all_colors = np.vstack([body_colors, body_colors, body_colors, fwd_colors])
        return gl.GLMeshItem(vertexes=all_verts, faces=all_faces, faceColors=all_colors, smooth=False, drawEdges=True, edgeColor=(1,1,1,0.3))

    def update_3d_model(self, model, pos, quat):
        model.resetTransform(); gl_pos = [pos[1], pos[0], -pos[2]];  q_scipy = [quat[1], quat[2], quat[3], quat[0]]
        try:
            norm = np.linalg.norm(q_scipy);
            if norm < 1e-6: return
            q_scipy = np.array(q_scipy) / norm; rot = R.from_quat(q_scipy); axis_angle = rot.as_rotvec(); angle_rad = np.linalg.norm(axis_angle); angle_deg = np.rad2deg(angle_rad)
            if angle_deg > 0: axis_vec = axis_angle / angle_rad; gl_axis = [axis_vec[1], axis_vec[0], -axis_vec[2]]; model.rotate(angle_deg, gl_axis[0], gl_axis[1], gl_axis[2])
            model.translate(gl_pos[0], gl_pos[1], gl_pos[2]);
        except (ValueError, ZeroDivisionError): pass

    def on_processing_error(self, error_message):
        self.set_ui_busy(False); self.status_bar.showMessage(f"Error: {error_message}", 10000); QMessageBox.critical(self, "Processing Error", error_message); print(f"Error from worker: {error_message}")

    def set_ui_busy(self, is_busy):
        self.import_button.setEnabled(not is_busy); self.export_button.setEnabled(not is_busy and not self.data.empty); self.connect_button.setEnabled(not is_busy)
        if hasattr(self, 'simulate_button'): self.simulate_button.setEnabled(not is_busy and not self.data.empty)
        if self.sim_progress_bar: self.sim_progress_bar.setEnabled(not is_busy and not self.data.empty)
        if self.speed_slider: self.speed_slider.setEnabled(not is_busy and not self.data.empty)
        if is_busy: self.status_bar.showMessage("Processing data... Please wait."); QApplication.setOverrideCursor(Qt.WaitCursor)
        else: self.status_bar.clearMessage(); QApplication.restoreOverrideCursor()

    def setup_sensor_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); self.accel_plot = pg.PlotWidget(title="Accelerometers (m/s^2)"); self.gyro_plot = pg.PlotWidget(title="Gyroscopes (rad/s)"); self.mag_plot = pg.PlotWidget(title="Magnetometers (Gauss)")
        for p in [self.accel_plot, self.gyro_plot, self.mag_plot]: p.addLegend(); p.getAxis('bottom').setLabel('Time (s)'); layout.addWidget(p)
        self.tabs.addTab(tab, "IMU Sensors")

    def setup_track_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); self.track_plot = pg.PlotWidget(title="2D Track Comparison (North vs East)"); self.track_plot.addLegend(); self.track_plot.setAspectLocked(True); self.track_plot.setLabel('left', 'North (m)'); self.track_plot.setLabel('bottom', 'East (m)'); layout.addWidget(self.track_plot); self.tabs.addTab(tab, "2D Track")

    def setup_position_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); self.pos_ekf_plot = pg.PlotWidget(title="EKF Position (NED)"); self.pos_tree_plot = pg.PlotWidget(title="Tree Position (NED)"); self.pos_error_plot = pg.PlotWidget(title="Position Error (EKF - Tree)")
        for p in [self.pos_ekf_plot, self.pos_tree_plot, self.pos_error_plot]: p.addLegend(); p.getAxis('bottom').setLabel('Time (s)'); p.getAxis('left').setLabel('Position (m)'); layout.addWidget(p)
        self.tabs.addTab(tab, "Position")

    def setup_quaternion_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); self.quat_ekf_plot = pg.PlotWidget(title="EKF Quaternion Components"); self.quat_tree_plot = pg.PlotWidget(title="Tree Quaternion Components")
        for p in [self.quat_ekf_plot, self.quat_tree_plot]: p.addLegend(); p.getAxis('bottom').setLabel('Time (s)'); layout.addWidget(p)
        self.tabs.addTab(tab, "Quaternions")

    def setup_euler_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); self.euler_ekf_plot = pg.PlotWidget(title="EKF Orientation (Euler Angles)"); self.euler_tree_plot = pg.PlotWidget(title="Tree Orientation (Euler Angles)"); self.euler_error_plot = pg.PlotWidget(title="Orientation Error (degrees)")
        for p in [self.euler_ekf_plot, self.euler_tree_plot, self.euler_error_plot]: p.addLegend(); p.getAxis('bottom').setLabel('Time (s)'); p.getAxis('left').setLabel('Angle (degrees)'); layout.addWidget(p)
        self.tabs.addTab(tab, "Orientation")

    def export_csv_data(self):
        if self.data.empty: self.status_bar.showMessage("No data to export.", 3000); return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data as CSV", "", "CSV Files (*.csv)")
        if not file_path: return
        try: self.data.to_csv(file_path, index=False); self.status_bar.showMessage(f"Data successfully saved to {file_path}", 5000)
        except Exception as e: self.status_bar.showMessage(f"Error saving file: {e}", 10000)

if __name__ == '__main__':
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())