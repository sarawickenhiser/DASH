#!/usr/bin/env python3

import sys
import select
import time
import threading
import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, String
from sensor_msgs.msg import JointState

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS


class AutoZeroNode(Node):
    def __init__(self):
        super().__init__('auto_zero_node')

        # -----------------------------
        # ROS parameters
        # -----------------------------
        self.declare_parameter(
            'device_name',
            '/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_F42208A55157375037202020FF122F34-if00'
        )
        self.declare_parameter('baudrate', 57600)

        self.declare_parameter('motor_ids', [1, 2, 3, 4])
        self.declare_parameter('home_positions', [2048, 2048, 1024, 1024])

        # Faster default speeds.
        self.declare_parameter('home_speed_deg_per_sec', 40.0)
        self.declare_parameter('sweep_speed_deg_per_sec', 60.0)
        self.declare_parameter('return_home_speed_deg_per_sec', 60.0)

        # Higher Dynamixel profile velocity for faster motion.
        self.declare_parameter('profile_velocity', 180)

        # Faster command loop.
        self.declare_parameter('motion_loop_sleep', 0.01)

        # More forgiving arrival behavior.
        self.declare_parameter('arrival_tolerance_counts', 70)
        self.declare_parameter('arrival_timeout_sec', 15.0)

        self.declare_parameter('exit_after_sequence', True)

        # True is safest for launch sequencing because the controller reconnects after auto-zero.
        self.declare_parameter('torque_off_on_shutdown', True)

        self.declare_parameter('roll_sweep_deg', 259.0)
        self.declare_parameter('pitch_sweep_deg', 79.0)
        self.declare_parameter('yaw_sweep_deg', 79.0)
        self.declare_parameter('grip_open_deg', 30.0)

        self.device_name = self.get_parameter('device_name').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.motor_ids = list(self.get_parameter('motor_ids').value)
        self.home_positions = list(self.get_parameter('home_positions').value)

        self.home_speed_deg_per_sec = float(self.get_parameter('home_speed_deg_per_sec').value)
        self.sweep_speed_deg_per_sec = float(self.get_parameter('sweep_speed_deg_per_sec').value)
        self.return_home_speed_deg_per_sec = float(self.get_parameter('return_home_speed_deg_per_sec').value)

        self.profile_velocity = int(self.get_parameter('profile_velocity').value)
        self.motion_loop_sleep = float(self.get_parameter('motion_loop_sleep').value)
        self.arrival_tolerance_counts = int(self.get_parameter('arrival_tolerance_counts').value)
        self.arrival_timeout_sec = float(self.get_parameter('arrival_timeout_sec').value)

        self.exit_after_sequence = bool(self.get_parameter('exit_after_sequence').value)
        self.torque_off_on_shutdown = bool(self.get_parameter('torque_off_on_shutdown').value)

        self.roll_sweep_deg = float(self.get_parameter('roll_sweep_deg').value)
        self.pitch_sweep_deg = float(self.get_parameter('pitch_sweep_deg').value)
        self.yaw_sweep_deg = float(self.get_parameter('yaw_sweep_deg').value)
        self.grip_open_deg = float(self.get_parameter('grip_open_deg').value)

        # -----------------------------
        # Dynamixel settings
        # -----------------------------
        self.protocol_version = 2.0

        self.ADDR_OPERATING_MODE = 11
        self.ADDR_HARDWARE_ERROR_STATUS = 70
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132

        self.TORQUE_DISABLE = 0
        self.TORQUE_ENABLE = 1
        self.EXTENDED_POSITION_CONTROL_MODE = 4

        self.COUNTS_PER_DEGREE = 4096.0 / 360.0

        # -----------------------------
        # dVRK coupling matrix
        # -----------------------------
        self.ROLL_D1 = -1.56323325
        self.PITCH_D2 = 1.01857984
        self.YAW_D2 = -0.830634273
        self.YAW_D3 = 0.608862987
        self.YAW_D4 = 0.608862987
        self.GRIP_D3 = -1.21772597
        self.GRIP_D4 = 1.21772597

        self.joint_limits_deg = {
            'roll': (-259.0, 259.0),
            'pitch': (-79.0, 79.0),
            'yaw': (-79.0, 79.0),
            'grip': (0.0, 30.0),
        }

        self.disk_angle_limits_deg = {
            1: (-166.0, 166.0),
            2: (-78.5, 78.5),
            3: (-150.0, 150.0),
            4: (-150.0, 150.0),
        }

        self.motor_position_limits = {}

        for dxl_id, home_position in zip(self.motor_ids, self.home_positions):
            min_deg, max_deg = self.disk_angle_limits_deg[dxl_id]
            min_count = int(home_position + min_deg * self.COUNTS_PER_DEGREE)
            max_count = int(home_position + max_deg * self.COUNTS_PER_DEGREE)
            self.motor_position_limits[dxl_id] = (min_count, max_count)

        self.goal_positions = {}

        self.current_joint_targets_deg = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'grip': 0.0,
        }

        self.no_tool_confirmed = False
        self.tool_inserted_confirmed = False
        self.sequence_started = False
        self.auto_zero_complete = False
        self.emergency_stop_requested = False

        self.loop_counter = 0
        self.loop_rate_timer = time.time()

        self.dxl_lock = threading.Lock()

        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(self.protocol_version)

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        self.no_tool_sub = self.create_subscription(
            Bool,
            '/h4hr/confirm_no_tool',
            self.confirm_no_tool_callback,
            10
        )

        self.confirm_tool_sub = self.create_subscription(
            Bool,
            '/h4hr/confirm_tool_insertion',
            self.confirm_tool_callback,
            10
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/h4hr/emergency_stop',
            self.emergency_stop_callback,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/h4hr/auto_zero_status',
            10
        )

        self.ready_pub = self.create_publisher(
            Bool,
            '/h4hr/auto_zero_ready',
            10
        )

        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        self.joint_state_timer = self.create_timer(
            0.05,
            self.publish_joint_states
        )

        self.connect_to_motors()
        self.print_instructions()

    # -----------------------------
    # ROS callbacks/status
    # -----------------------------
    def confirm_no_tool_callback(self, msg):
        if msg.data:
            self.no_tool_confirmed = True
            self.publish_status('NO TOOL confirmation received from ROS topic.')

    def confirm_tool_callback(self, msg):
        if msg.data:
            self.tool_inserted_confirmed = True
            self.publish_status('Tool insertion confirmation received from ROS topic.')

    def emergency_stop_callback(self, msg):
        if msg.data:
            self.emergency_stop_requested = True
            self.publish_status('EMERGENCY STOP received from ROS topic.')
            self.emergency_torque_off()

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def publish_ready(self):
        msg = Bool()
        msg.data = True
        self.ready_pub.publish(msg)
        self.auto_zero_complete = True
        self.publish_status('AUTO ZERO READY: Tool control may now start.')

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = [
            'roll_joint',
            'pitch_joint',
            'yaw_joint',
            'grip_joint',
        ]

        msg.position = [
            math.radians(self.current_joint_targets_deg['roll']),
            math.radians(self.current_joint_targets_deg['pitch']),
            math.radians(self.current_joint_targets_deg['yaw']),
            math.radians(self.current_joint_targets_deg['grip']),
        ]

        self.joint_state_pub.publish(msg)

    # -----------------------------
    # Connection and communication
    # -----------------------------
    def connect_to_motors(self):
        if not self.port_handler.openPort():
            raise RuntimeError(f'Failed to open port: {self.device_name}')

        self.get_logger().info(f'Opened port: {self.device_name}')

        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f'Failed to set baudrate: {self.baudrate}')

        self.get_logger().info(f'Set baudrate: {self.baudrate}')
        self.get_logger().info(f'Motor position limits: {self.motor_position_limits}')

        for dxl_id in self.motor_ids:
            self.ping_motor(dxl_id)
            self.read_hardware_error_status(dxl_id)
            self.goal_positions[dxl_id] = self.get_home_position(dxl_id)

        self.set_all_extended_position_mode()

        for dxl_id in self.motor_ids:
            self.sync_goal_to_present_position(dxl_id)

    def check_result(self, comm_result, dxl_error, action):
        if comm_result != COMM_SUCCESS:
            self.get_logger().error(
                f'{action} failed: {self.packet_handler.getTxRxResult(comm_result)}'
            )
            return False

        if dxl_error != 0:
            self.get_logger().error(
                f'{action} error: {self.packet_handler.getRxPacketError(dxl_error)}'
            )
            return False

        return True

    def ping_motor(self, dxl_id):
        with self.dxl_lock:
            model_number, comm_result, dxl_error = self.packet_handler.ping(
                self.port_handler,
                dxl_id
            )

        if self.check_result(comm_result, dxl_error, f'Ping motor {dxl_id}'):
            self.get_logger().info(f'Motor {dxl_id} found. Model: {model_number}')
            return True

        self.get_logger().warn(f'Motor {dxl_id} not found.')
        return False

    def write_1_byte(self, dxl_id, address, value, action):
        with self.dxl_lock:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler,
                dxl_id,
                address,
                int(value)
            )

        return self.check_result(comm_result, dxl_error, action)

    def write_4_byte(self, dxl_id, address, value, action):
        with self.dxl_lock:
            comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
                self.port_handler,
                dxl_id,
                address,
                int(value)
            )

        return self.check_result(comm_result, dxl_error, action)

    def read_1_byte(self, dxl_id, address, action):
        with self.dxl_lock:
            value, comm_result, dxl_error = self.packet_handler.read1ByteTxRx(
                self.port_handler,
                dxl_id,
                address
            )

        if self.check_result(comm_result, dxl_error, action):
            return value

        return None

    def read_4_byte(self, dxl_id, address, action):
        with self.dxl_lock:
            value, comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
                self.port_handler,
                dxl_id,
                address
            )

        if self.check_result(comm_result, dxl_error, action):
            return value

        return None

    def read_hardware_error_status(self, dxl_id):
        value = self.read_1_byte(
            dxl_id,
            self.ADDR_HARDWARE_ERROR_STATUS,
            f'Read hardware error status motor {dxl_id}'
        )

        if value is None:
            return None

        if value != 0:
            self.get_logger().error(
                f'Motor {dxl_id} Hardware Error Status = {value}. '
                'Power-cycle motor supply/OpenRB if this persists.'
            )
        else:
            self.get_logger().info(f'Motor {dxl_id} Hardware Error Status = 0.')

        return value

    # -----------------------------
    # Signed / unsigned conversion
    # -----------------------------
    def signed_to_unsigned_32(self, value):
        return int(value) & 0xFFFFFFFF

    def unsigned_to_signed_32(self, value):
        value = int(value)

        if value > 0x7FFFFFFF:
            value -= 0x100000000

        return value

    # -----------------------------
    # Position helpers
    # -----------------------------
    def get_home_position(self, dxl_id):
        if dxl_id not in self.motor_ids:
            return 0

        index = self.motor_ids.index(dxl_id)
        return int(self.home_positions[index])

    def read_present_position(self, dxl_id):
        present = self.read_4_byte(
            dxl_id,
            self.ADDR_PRESENT_POSITION,
            f'Read present position motor {dxl_id}'
        )

        if present is None:
            return None

        return self.unsigned_to_signed_32(present)

    def sync_goal_to_present_position(self, dxl_id):
        present = self.read_present_position(dxl_id)

        if present is None:
            return False

        self.goal_positions[dxl_id] = present
        return True

    def disk_angle_to_motor_position(self, dxl_id, disk_angle_deg):
        home = self.get_home_position(dxl_id)
        return int(home + disk_angle_deg * self.COUNTS_PER_DEGREE)

    def motor_position_to_disk_angle(self, dxl_id, position):
        home = self.get_home_position(dxl_id)
        return float(position - home) / self.COUNTS_PER_DEGREE

    def is_position_inside_bounds(self, dxl_id, position):
        if dxl_id not in self.motor_position_limits:
            return False

        min_position, max_position = self.motor_position_limits[dxl_id]
        return min_position <= position <= max_position

    def command_position(self, dxl_id, position, enforce_bounds=True, log=True):
        position = int(position)

        if self.emergency_stop_requested:
            self.get_logger().error(
                f'Motor {dxl_id} command refused because emergency stop is active.'
            )
            return False

        if enforce_bounds and not self.is_position_inside_bounds(dxl_id, position):
            min_position, max_position = self.motor_position_limits[dxl_id]
            self.get_logger().error(
                f'Motor {dxl_id} command {position} outside safe bounds '
                f'[{min_position}, {max_position}]. Command refused.'
            )
            return False

        self.goal_positions[dxl_id] = position

        ok = self.write_4_byte(
            dxl_id,
            self.ADDR_GOAL_POSITION,
            self.signed_to_unsigned_32(position),
            f'Set goal position motor {dxl_id}'
        )

        if ok and log:
            disk_angle = self.motor_position_to_disk_angle(dxl_id, position)
            self.get_logger().info(
                f'Motor {dxl_id} / Disk {dxl_id}: '
                f'{position} counts, {disk_angle:.1f} deg from home'
            )

        return ok

    # -----------------------------
    # Coupling matrix helpers
    # -----------------------------
    def joints_to_disks(self, joints_deg):
        roll = float(joints_deg['roll'])
        pitch = float(joints_deg['pitch'])
        yaw = float(joints_deg['yaw'])
        grip = float(joints_deg['grip'])

        d1 = roll / self.ROLL_D1
        d2 = pitch / self.PITCH_D2

        d3_plus_d4 = (yaw - self.YAW_D2 * d2) / self.YAW_D3
        d4_minus_d3 = grip / self.GRIP_D4

        d3 = 0.5 * (d3_plus_d4 - d4_minus_d3)
        d4 = 0.5 * (d3_plus_d4 + d4_minus_d3)

        return {
            1: d1,
            2: d2,
            3: d3,
            4: d4,
        }

    def validate_joint_targets(self, joints_deg):
        for joint_name, value in joints_deg.items():
            lower, upper = self.joint_limits_deg[joint_name]

            if value < lower or value > upper:
                self.get_logger().error(
                    f'{joint_name} target {value:.1f} outside limit '
                    f'[{lower:.1f}, {upper:.1f}]'
                )
                return False

        return True

    def validate_disk_angles(self, disk_angles_deg):
        for dxl_id, disk_angle in disk_angles_deg.items():
            lower, upper = self.disk_angle_limits_deg[dxl_id]

            if disk_angle < lower or disk_angle > upper:
                self.get_logger().error(
                    f'Disk {dxl_id} target {disk_angle:.1f} outside safe limit '
                    f'[{lower:.1f}, {upper:.1f}]'
                )
                return False

        return True

    def joint_targets_to_motor_positions(self, joints_deg):
        if not self.validate_joint_targets(joints_deg):
            return None

        disk_angles = self.joints_to_disks(joints_deg)

        if not self.validate_disk_angles(disk_angles):
            return None

        motor_positions = {}

        for dxl_id, disk_angle in disk_angles.items():
            motor_positions[dxl_id] = self.disk_angle_to_motor_position(
                dxl_id,
                disk_angle
            )

        return motor_positions

    def log_joint_and_disk_targets(self, joints_deg):
        disk_angles = self.joints_to_disks(joints_deg)

        self.get_logger().info(
            'Joint targets: '
            f"Roll={joints_deg['roll']:.1f}, "
            f"Pitch={joints_deg['pitch']:.1f}, "
            f"Yaw={joints_deg['yaw']:.1f}, "
            f"Grip={joints_deg['grip']:.1f} deg"
        )

        self.get_logger().info(
            'Disk targets: '
            f"D1={disk_angles[1]:.1f}, "
            f"D2={disk_angles[2]:.1f}, "
            f"D3={disk_angles[3]:.1f}, "
            f"D4={disk_angles[4]:.1f} deg"
        )

    # -----------------------------
    # Mode setup
    # -----------------------------
    def set_motor_extended_position_mode(self, dxl_id):
        self.write_1_byte(
            dxl_id,
            self.ADDR_TORQUE_ENABLE,
            self.TORQUE_DISABLE,
            f'Disable torque motor {dxl_id}'
        )

        self.write_1_byte(
            dxl_id,
            self.ADDR_OPERATING_MODE,
            self.EXTENDED_POSITION_CONTROL_MODE,
            f'Set extended position mode motor {dxl_id}'
        )

        self.write_4_byte(
            dxl_id,
            self.ADDR_PROFILE_VELOCITY,
            self.profile_velocity,
            f'Set profile velocity motor {dxl_id}'
        )

        self.write_1_byte(
            dxl_id,
            self.ADDR_TORQUE_ENABLE,
            self.TORQUE_ENABLE,
            f'Enable torque motor {dxl_id}'
        )

    def set_all_extended_position_mode(self):
        for dxl_id in self.motor_ids:
            self.set_motor_extended_position_mode(dxl_id)

        self.get_logger().info('Set all motors to EXTENDED POSITION control mode.')

    # -----------------------------
    # Runtime keyboard commands
    # -----------------------------
    def check_keyboard_runtime_commands(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)

        if not readable:
            return True

        line = sys.stdin.readline().strip().lower()

        if line == '1':
            self.emergency_stop_requested = True
            self.emergency_torque_off()
            self.publish_status('Emergency torque off requested from keyboard.')
            return False

        return True

    # -----------------------------
    # Motion engine
    # -----------------------------
    def log_loop_rate_if_needed(self):
        self.loop_counter += 1
        now = time.time()

        elapsed = now - self.loop_rate_timer

        if elapsed >= 2.0:
            rate = self.loop_counter / elapsed
            self.get_logger().info(f'Auto-zero motion loop rate: {rate:.1f} Hz')
            self.loop_counter = 0
            self.loop_rate_timer = now

    def run_independent_motor_sequences(self, sequences, speed_deg_per_sec, enforce_bounds=True):
        states = {}
        speed_deg_per_sec = abs(float(speed_deg_per_sec))

        if speed_deg_per_sec <= 0.0:
            self.get_logger().error('Speed must be greater than 0 deg/sec.')
            return False

        speed_counts_per_sec = speed_deg_per_sec * self.COUNTS_PER_DEGREE

        for dxl_id, waypoints in sequences.items():
            if dxl_id not in self.motor_ids:
                self.get_logger().warn(f'Motor {dxl_id} ignored: not in motor_ids.')
                continue

            if not waypoints:
                self.get_logger().warn(f'Motor {dxl_id} ignored: no waypoints.')
                continue

            valid_waypoints = []

            for target_position in waypoints:
                target_position = int(target_position)

                if enforce_bounds and not self.is_position_inside_bounds(dxl_id, target_position):
                    min_position, max_position = self.motor_position_limits[dxl_id]
                    self.get_logger().error(
                        f'Motor {dxl_id} waypoint {target_position} outside bounds '
                        f'[{min_position}, {max_position}].'
                    )
                    valid_waypoints = []
                    break

                valid_waypoints.append(target_position)

            if not valid_waypoints:
                continue

            present = self.read_present_position(dxl_id)

            if present is None:
                self.get_logger().error(f'Could not read motor {dxl_id}.')
                continue

            if enforce_bounds and not self.is_position_inside_bounds(dxl_id, present):
                min_position, max_position = self.motor_position_limits[dxl_id]
                self.get_logger().error(
                    f'Motor {dxl_id} present position {present} outside bounds '
                    f'[{min_position}, {max_position}].'
                )
                continue

            if not enforce_bounds:
                self.get_logger().warn(
                    f'Motor {dxl_id} move is running with disk bounds disabled. '
                    f'This should only be used for out-of-bounds recovery with NO TOOL inserted.'
                )

            self.goal_positions[dxl_id] = present

            states[dxl_id] = {
                'waypoints': valid_waypoints,
                'index': 0,
                'last_update_time': time.time(),
                'waypoint_start_time': time.time(),
                'done': False,
            }

        if not states:
            self.get_logger().error('No valid motor sequences to run.')
            return False

        self.loop_counter = 0
        self.loop_rate_timer = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)

            if self.emergency_stop_requested:
                self.get_logger().error('Motion stopped because emergency stop is active.')
                return False

            if not self.check_keyboard_runtime_commands():
                return False

            self.log_loop_rate_if_needed()

            all_done = True
            now = time.time()

            for dxl_id, state in states.items():
                if state['done']:
                    continue

                all_done = False

                waypoints = state['waypoints']
                waypoint_index = state['index']
                target_position = waypoints[waypoint_index]

                present = self.read_present_position(dxl_id)

                if present is None:
                    self.get_logger().error(
                        f'Motor {dxl_id} stopped: could not read present position.'
                    )
                    state['done'] = True
                    continue

                present_error = target_position - present

                if abs(present_error) <= self.arrival_tolerance_counts:
                    self.command_position(
                        dxl_id,
                        target_position,
                        enforce_bounds=enforce_bounds,
                        log=False
                    )

                    state['index'] += 1

                    if state['index'] >= len(waypoints):
                        state['done'] = True
                        continue

                    state['waypoint_start_time'] = now
                    state['last_update_time'] = now
                    continue

                elapsed_at_waypoint = now - state['waypoint_start_time']

                if elapsed_at_waypoint > self.arrival_timeout_sec:
                    self.get_logger().error(
                        f'Motor {dxl_id} timeout at waypoint '
                        f'{waypoint_index + 1}/{len(waypoints)}. '
                        f'Target {target_position}, present {present}.'
                    )
                    state['done'] = True
                    continue

                dt = now - state['last_update_time']
                state['last_update_time'] = now

                step_counts = max(1, int(speed_counts_per_sec * dt))

                current_command = int(self.goal_positions.get(dxl_id, present))
                command_error = target_position - current_command

                if abs(command_error) <= step_counts:
                    next_command = target_position
                elif command_error > 0:
                    next_command = current_command + step_counts
                else:
                    next_command = current_command - step_counts

                if enforce_bounds and not self.is_position_inside_bounds(dxl_id, next_command):
                    self.get_logger().error(
                        f'Motor {dxl_id} stopped: next command {next_command} '
                        f'would leave bounds.'
                    )
                    state['done'] = True
                    continue

                self.command_position(
                    dxl_id,
                    next_command,
                    enforce_bounds=enforce_bounds,
                    log=False
                )

            if all_done:
                return True

            time.sleep(self.motion_loop_sleep)

        return False

    def move_to_motor_positions(self, motor_positions, speed_deg_per_sec, enforce_bounds=True):
        sequences = {}

        for dxl_id, position in motor_positions.items():
            sequences[dxl_id] = [position]

        return self.run_independent_motor_sequences(
            sequences,
            speed_deg_per_sec,
            enforce_bounds=enforce_bounds
        )

    def move_to_joint_targets(self, joints_deg, speed_deg_per_sec):
        motor_positions = self.joint_targets_to_motor_positions(joints_deg)

        if motor_positions is None:
            self.get_logger().error('Joint move refused.')
            return False

        self.log_joint_and_disk_targets(joints_deg)

        ok = self.move_to_motor_positions(
            motor_positions,
            speed_deg_per_sec,
            enforce_bounds=True
        )

        if ok:
            self.current_joint_targets_deg = dict(joints_deg)

        return ok

    # -----------------------------
    # Confirmation helpers
    # -----------------------------
    def wait_for_line_or_topic(self, prompt_text, accepted_words, topic_flag_name):
        self.publish_status(prompt_text)

        print()
        print(prompt_text)
        print('Use the Hand for Humanoid Robot UI node, or type one of these and press Enter:')
        print(f'  {accepted_words}')
        print('Type 1 and press Enter for emergency torque off.')
        print()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.emergency_stop_requested:
                return False

            if getattr(self, topic_flag_name):
                return True

            readable, _, _ = select.select([sys.stdin], [], [], 0.05)

            if readable:
                line = sys.stdin.readline().strip().lower()

                if line == '1':
                    self.emergency_stop_requested = True
                    self.emergency_torque_off()
                    self.publish_status('Emergency torque off requested from keyboard.')
                    return False

                if line in accepted_words:
                    setattr(self, topic_flag_name, True)
                    self.publish_status(f'Confirmation received from keyboard: {line}')
                    return True

                print()
                print(f'Input "{line}" was not accepted.')
                print(f'Type one of these and press Enter: {accepted_words}')
                print()

            time.sleep(0.05)

        return False

    # -----------------------------
    # Auto-zero sequence
    # -----------------------------
    def confirm_no_tool_inserted(self):
        return self.wait_for_line_or_topic(
            'STEP 0: Confirm that NO TOOL is inserted. This must be true before homing motors.',
            ['n', 'no', 'no tool', 'none'],
            'no_tool_confirmed'
        )

    def move_all_motors_home_no_tool(self):
        self.publish_status(
            'STEP 1: Moving all motors to home positions. '
            'Motors already inside bounds will remain bound-enforced. '
            'Only motors currently outside bounds will use recovery homing.'
        )

        motor_positions = {}

        for dxl_id in self.motor_ids:
            motor_positions[dxl_id] = self.get_home_position(dxl_id)

        motors_in_bounds = {}

        for dxl_id in self.motor_ids:
            present = self.read_present_position(dxl_id)

            if present is None:
                motors_in_bounds[dxl_id] = False
                self.get_logger().error(
                    f'Motor {dxl_id}: cannot read present position. '
                    'Treating as out of bounds recovery candidate.'
                )
                continue

            motors_in_bounds[dxl_id] = self.is_position_inside_bounds(dxl_id, present)

            if motors_in_bounds[dxl_id]:
                self.get_logger().info(
                    f'Motor {dxl_id} is inside bounds at {present}. '
                    'Homing with bounds enforced.'
                )
            else:
                self.get_logger().warn(
                    f'Motor {dxl_id} is outside safe bounds at {present}. '
                    'Bounds will be disabled for this motor during homing only.'
                )

        in_bounds_ids = [dxl_id for dxl_id, inside in motors_in_bounds.items() if inside]
        out_of_bounds_ids = [dxl_id for dxl_id, inside in motors_in_bounds.items() if not inside]

        if in_bounds_ids:
            in_bounds_positions = {
                dxl_id: motor_positions[dxl_id]
                for dxl_id in in_bounds_ids
            }

            ok = self.move_to_motor_positions(
                in_bounds_positions,
                self.home_speed_deg_per_sec,
                enforce_bounds=True
            )

            if not ok:
                self.publish_status('Failed to home in-bounds motors with bounds enforced.')
                return False

        if out_of_bounds_ids:
            self.publish_status(
                'Recovering out-of-bounds motors. This should only happen with NO TOOL inserted.'
            )

            out_of_bounds_positions = {
                dxl_id: motor_positions[dxl_id]
                for dxl_id in out_of_bounds_ids
            }

            ok = self.move_to_motor_positions(
                out_of_bounds_positions,
                self.home_speed_deg_per_sec,
                enforce_bounds=False
            )

            if not ok:
                self.publish_status('Failed to recover out-of-bounds motors.')
                return False

        self.current_joint_targets_deg = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'grip': 0.0,
        }

        self.publish_status('All motors reached home position.')
        return True

    def confirm_tool_inserted(self):
        return self.wait_for_line_or_topic(
            'STEP 2: Insert the tool now, then confirm tool insertion.',
            ['y', 'yes', 'tool', 'inserted'],
            'tool_inserted_confirmed'
        )

    def sweep_all_joints_with_coupling_matrix(self):
        self.publish_status('STEP 3: Starting FAST combined coupling-matrix joint sweep.')

        zero = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'grip': 0.0,
        }

        # Fast combined sweep:
        # This checks roll, pitch, yaw, and grip in fewer moves.
        # It is faster than sweeping each joint separately.
        sweep_targets = [
            zero,

            {
                'roll': self.roll_sweep_deg,
                'pitch': self.pitch_sweep_deg,
                'yaw': self.yaw_sweep_deg,
                'grip': self.grip_open_deg,
            },

            zero,

            {
                'roll': -self.roll_sweep_deg,
                'pitch': -self.pitch_sweep_deg,
                'yaw': -self.yaw_sweep_deg,
                'grip': 0.0,
            },

            zero,
        ]

        for index, target in enumerate(sweep_targets):
            if self.emergency_stop_requested:
                self.publish_status('Sweep stopped by emergency stop.')
                return False

            self.publish_status(
                f'Fast sweep waypoint {index + 1}/{len(sweep_targets)}: '
                f"roll={target['roll']:.1f}, "
                f"pitch={target['pitch']:.1f}, "
                f"yaw={target['yaw']:.1f}, "
                f"grip={target['grip']:.1f}"
            )

            ok = self.move_to_joint_targets(
                target,
                self.sweep_speed_deg_per_sec
            )

            if not ok:
                self.publish_status(
                    'Fast sweep stopped because a waypoint failed. '
                    'Attempting safe return to zero.'
                )

                self.try_return_to_zero_after_failure()
                return False

            time.sleep(0.10)

        self.publish_status('FAST combined coupling-matrix joint sweep complete.')
        return True

    def try_return_to_zero_after_failure(self):
        if self.emergency_stop_requested:
            self.publish_status('Return to zero skipped because emergency stop is active.')
            return False

        zero = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'grip': 0.0,
        }

        self.publish_status('Attempting return to zero after sweep failure.')

        ok = self.move_to_joint_targets(
            zero,
            self.return_home_speed_deg_per_sec
        )

        if ok:
            self.publish_status('Returned to zero after failure.')
        else:
            self.publish_status('Could not return to zero after failure.')

        return ok

    def run_auto_zero_sequence(self):
        if self.sequence_started:
            self.publish_status('Auto-zero sequence already started.')
            return False

        self.sequence_started = True

        self.publish_status('Auto-zero sequence started.')

        ok = self.confirm_no_tool_inserted()

        if not ok:
            self.publish_status('Auto-zero stopped before homing.')
            return False

        ok = self.move_all_motors_home_no_tool()

        if not ok:
            self.publish_status('Auto-zero failed during home motion.')
            return False

        ok = self.confirm_tool_inserted()

        if not ok:
            self.publish_status('Auto-zero stopped before sweep.')
            return False

        ok = self.sweep_all_joints_with_coupling_matrix()

        if ok:
            self.publish_status('Auto-zero sequence complete.')
            self.publish_ready()
            return True

        self.publish_status('Auto-zero sequence failed during sweep.')
        return False

    # -----------------------------
    # Safety
    # -----------------------------
    def emergency_torque_off(self):
        self.get_logger().warn('EMERGENCY STOP: disabling torque on all motors.')

        for dxl_id in self.motor_ids:
            self.write_1_byte(
                dxl_id,
                self.ADDR_TORQUE_ENABLE,
                self.TORQUE_DISABLE,
                f'Emergency torque off motor {dxl_id}'
            )

    def shutdown(self):
        if self.torque_off_on_shutdown:
            self.emergency_torque_off()
        else:
            self.get_logger().warn('Leaving motor torque enabled on shutdown.')

        self.port_handler.closePort()
        self.get_logger().info('Closed Dynamixel port.')

    # -----------------------------
    # UI
    # -----------------------------
    def print_instructions(self):
        print()
        print('H4HR Auto-Zero Node with FAST Combined dVRK Coupling Matrix Sweep')
        print('----------------------------------------------------------------')
        print('Sequence:')
        print('  0. Confirm NO TOOL is inserted.')
        print('  1. Motors move to home positions.')
        print('  2. Insert tool.')
        print('  3. Confirm tool insertion.')
        print('  4. Node performs fast combined joint-space sweep using coupling matrix.')
        print('  5. After completion, RViz control can start.')
        print()
        print('Recommended: use the Hand for Humanoid Robot UI node from another terminal.')
        print()
        print('Keyboard confirmations require Enter:')
        print('  No tool inserted: type n then press Enter')
        print('  Tool inserted:    type y then press Enter')
        print('  Emergency:        type 1 then press Enter')
        print()
        print('ROS topic confirmations:')
        print('  Confirm NO TOOL:')
        print('    ros2 topic pub --once /h4hr/confirm_no_tool std_msgs/msg/Bool "{data: true}"')
        print()
        print('  Confirm TOOL INSERTED:')
        print('    ros2 topic pub --once /h4hr/confirm_tool_insertion std_msgs/msg/Bool "{data: true}"')
        print()
        print('  Emergency stop:')
        print('    ros2 topic pub --once /h4hr/emergency_stop std_msgs/msg/Bool "{data: true}"')
        print()
        print('Motor/Disk mapping:')
        print('  Motor 1 = Disk 1')
        print('  Motor 2 = Disk 2')
        print('  Motor 3 = Disk 3')
        print('  Motor 4 = Disk 4')
        print()
        print('Home positions:')
        for dxl_id, home in zip(self.motor_ids, self.home_positions):
            print(f'  Motor {dxl_id}: {home}')
        print()
        print('Disk safety limits during sweep:')
        for dxl_id in self.motor_ids:
            lower, upper = self.disk_angle_limits_deg[dxl_id]
            print(f'  Disk {dxl_id}: {lower:.1f} deg to {upper:.1f} deg')
        print()
        print('Fast joint sweep:')
        print(f'  Waypoint 1: zero')
        print(f'  Waypoint 2: +Roll, +Pitch, +Yaw, Grip open')
        print(f'  Waypoint 3: zero')
        print(f'  Waypoint 4: -Roll, -Pitch, -Yaw, Grip closed')
        print(f'  Waypoint 5: zero')
        print()
        print('Sweep magnitudes:')
        print(f'  Roll:  +/-{self.roll_sweep_deg:.1f} deg')
        print(f'  Pitch: +/-{self.pitch_sweep_deg:.1f} deg')
        print(f'  Yaw:   +/-{self.yaw_sweep_deg:.1f} deg')
        print(f'  Grip:  0 to {self.grip_open_deg:.1f} deg')
        print()


def main(args=None):
    rclpy.init(args=args)

    node = AutoZeroNode()

    try:
        success = node.run_auto_zero_sequence()

        if node.exit_after_sequence:
            if success:
                node.get_logger().info(
                    'Auto-zero finished successfully. Exiting so launch can start control nodes.'
                )
                return

            node.get_logger().error(
                'Auto-zero did not complete successfully. Staying alive so launch will NOT continue.'
            )
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt.')

    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()