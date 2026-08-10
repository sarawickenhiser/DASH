#!/usr/bin/env python3

import math
import threading
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

from std_srvs.srv import Trigger
from std_msgs.msg import Bool, String

import sys
import signal
from std_msgs.msg import Float64MultiArray

class DynamixelControllerNode(Node):
    def __init__(self):
        super().__init__('dynamixel_controller_node')

        # -----------------------------
        # ROS parameters
        # -----------------------------
        self.declare_parameter(
            'device_name',
            '/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_F42208A55157375037202020FF122F34-if00'
        )
        self.declare_parameter('baudrate', 57600)

        # Motor IDs correspond to Disk 1, Disk 2, Disk 3, Disk 4.
        self.declare_parameter('motor_ids', [1, 2, 3, 4])

        # Home positions should match your auto-zero home positions.
        self.declare_parameter('home_positions', [2048, 2048, 1024, 1024])

        # Actual operation speed.
        self.declare_parameter('command_speed_deg_per_sec', 50.0)

        # Dynamixel internal profile velocity.
        self.declare_parameter('profile_velocity', 80)

        # Motion update rate.
        self.declare_parameter('control_period_sec', 0.02)

        # Start enabled after auto-zero/manual launch.
        # Set false if you want to require /h4hr/enable_control true.
        self.declare_parameter('start_enabled', True)

        # Joint limits.
        self.declare_parameter('max_roll_deg', 259.0)
        self.declare_parameter('max_pitch_deg', 79.0)
        self.declare_parameter('max_yaw_deg', 79.0)

        # Generic fallback gripper range, used until a tool is scanned via
        # RFID and /h4hr/update_limits provides the real per-tool range.
        # Current mechanical setup:
        #   open  = -15 deg
        #   close = 38 deg
        self.declare_parameter('min_grip_deg', -15.0)
        self.declare_parameter('max_grip_deg', 38.0)

        self.device_name = self.get_parameter('device_name').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.motor_ids = list(self.get_parameter('motor_ids').value)
        self.home_positions = list(self.get_parameter('home_positions').value)

        self.command_speed_deg_per_sec = float(
            self.get_parameter('command_speed_deg_per_sec').value
        )
        self.profile_velocity = int(self.get_parameter('profile_velocity').value)
        self.control_period_sec = float(self.get_parameter('control_period_sec').value)
        self.control_enabled = bool(self.get_parameter('start_enabled').value)

        self.max_roll_deg = float(self.get_parameter('max_roll_deg').value)
        self.max_pitch_deg = float(self.get_parameter('max_pitch_deg').value)
        self.max_yaw_deg = float(self.get_parameter('max_yaw_deg').value)
        self.min_grip_deg = float(self.get_parameter('min_grip_deg').value)
        self.max_grip_deg = float(self.get_parameter('max_grip_deg').value)

        # -----------------------------
        # Dynamixel settings
        # -----------------------------
        self.protocol_version = 2.0

        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132

        self.TORQUE_DISABLE = 0
        self.TORQUE_ENABLE = 1
        self.EXTENDED_POSITION_CONTROL_MODE = 4

        # 4096 counts = 360 deg.
        self.COUNTS_PER_DEGREE = 4096.0 / 360.0

        # CURRENT SENSING
        self.ADDR_PRESENT_CURRENT = 126
        self.CURRENT_LIMIT_MA = 300      # mA threshold to detect limit hit
        self.CURRENT_CONSECUTIVE = 3     # consecutive readings before triggering
        self.BACKOFF_DEG = 5.0           # degrees to reverse

        # Motor direction correction.
        # Motors 3 and 4 are reversed because their physical clockwise/counterclockwise
        # directions are opposite of what the coupling matrix expects.
        self.motor_direction = {
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
        }

        # -----------------------------
        # dVRK coupling matrix
        # -----------------------------
        # Joint = Matrix * Disk
        #
        # Roll  = -1.56323325  * Disk1
        # Pitch =  1.01857984  * Disk2
        # Yaw   = -0.830634273 * Disk2 + 0.608862987 * Disk3 + 0.608862987 * Disk4
        # Grip  = -1.21772597  * Disk3 + 1.21772597  * Disk4
        self.ROLL_D1 = -1.56323325
        self.PITCH_D2 = 1.01857984
        self.YAW_D2 = -0.830634273 
        self.YAW_D3 = 0.608862987
        self.YAW_D4 = 0.608862987 
        self.GRIP_D3 = -1.21772597
        self.GRIP_D4 = 1.21772597

        # Disk-level safety limits relative to home.
        self.disk_angle_limits_deg = {
            1: (-166.0, 166.0),
            2: (-78.5, 78.5),
            3: (-150.0, 150.0),
            4: (-153.0, 153.0),
        }

        self.motor_position_limits = {}

        for dxl_id, home_position in zip(self.motor_ids, self.home_positions):
            min_deg, max_deg = self.disk_angle_limits_deg[dxl_id]

            min_count_raw = self.disk_angle_to_motor_position(dxl_id, min_deg)
            max_count_raw = self.disk_angle_to_motor_position(dxl_id, max_deg)

            self.motor_position_limits[dxl_id] = (
                min(min_count_raw, max_count_raw),
                max(min_count_raw, max_count_raw),
            )

        # RLock: the SIGINT/SIGTERM handler runs on this same (main) thread and
        # must be able to force a torque-off write even if the control loop was
        # interrupted mid-write while holding this lock. A plain Lock would
        # deadlock in that case, which prevented torque-off on Ctrl-C.
        self.dxl_lock = threading.RLock()
        self.state_lock = threading.Lock()

        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(self.protocol_version)

        self.goal_positions = {}

        self.current_joint_deg = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'grip': 0.0,
        }

        self.target_joint_deg = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'grip': 0.0,
        }

        self.last_log_time = 0.0

        #current reading time
        self.current_readings = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.current_hit_count = {1: 0, 2: 0, 3: 0, 4: 0}

        # Per-motor limit state so two motors can be limit-blocked at once
        # (a single shared flag let a second motor's hit silently clobber
        # the first motor's block).
        self.limit_hit = {dxl_id: False for dxl_id in self.motor_ids}
        self.limit_hit_direction = {}    # {motor_id: +1 or -1} direction that caused the hit

        self.connect_to_motors()

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        self.joint_command_sub = self.create_subscription(
            JointState,
            '/h4hr/joint_command',
            self.joint_command_callback,
            10
        )

        self.enable_sub = self.create_subscription(
            Bool,
            '/h4hr/enable_control',
            self.enable_callback,
            10
        )

        self.estop_sub = self.create_subscription(
            Bool,
            '/h4hr/estop',
            self.estop_callback,
            10
        )

        self.clear_limit_sub = self.create_subscription(
            Bool,
            '/h4hr/clear_limit',
            self.clear_limit_callback,
            10
        )

        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        self.coupling_sub = self.create_subscription(
            Float64MultiArray, '/h4hr/update_coupling', self.update_coupling_callback, 10
        )
        
        self.limits_sub = self.create_subscription(
            Float64MultiArray, '/h4hr/update_limits', self.update_limits_callback, 10
        )

        #for current
        self.limit_pub = self.create_publisher(String, '/h4hr/limit_hit', 10)
        self.current_pub = self.create_publisher(String, '/h4hr/motor_currents', 10)
        self.current_timer = self.create_timer(0.05, self.current_sensing_loop)

        self.control_timer = self.create_timer(
            self.control_period_sec,
            self.control_loop
        )

        self.joint_state_timer = self.create_timer(
            0.05,
            self.publish_joint_states
        )

        self.get_logger().warn('Dynamixel controller ready for ACTUAL OPERATION.')
        self.get_logger().warn('Make sure auto-zero was completed before using RViz control.')
        self.get_logger().warn('Emergency stop topic: /h4hr/estop std_msgs/Bool true')
        self.get_logger().info(
            f'Grip range: {self.min_grip_deg:.1f} deg to {self.max_grip_deg:.1f} deg'
        )
        self.get_logger().info(
            f'Motor direction correction: {self.motor_direction}'
        )

        self.release_port_srv = self.create_service(
            Trigger,
            '/h4hr/release_port',
            self.release_port_callback
        )

        self.reclaim_port_srv = self.create_service(
            Trigger,
            '/h4hr/reclaim_port',
            self.reclaim_port_callback
        )

    # -----------------------------
    # Connection and Dynamixel I/O
    # -----------------------------
    def connect_to_motors(self):
        if len(self.motor_ids) != len(self.home_positions):
            raise RuntimeError('motor_ids and home_positions must be same length.')

        if not self.port_handler.openPort():
            raise RuntimeError(f'Failed to open port: {self.device_name}')

        self.get_logger().info(f'Opened port: {self.device_name}')

        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(f'Failed to set baudrate: {self.baudrate}')

        self.get_logger().info(f'Set baudrate: {self.baudrate}')

        self.port_handler.ser.timeout = 0.1

        for dxl_id in self.motor_ids:
            self.ping_motor(dxl_id)

        self.set_all_extended_position_mode()

        for dxl_id in self.motor_ids:
            present = self.read_present_position(dxl_id)

            if present is None:
                present = self.get_home_position(dxl_id)

            self.goal_positions[dxl_id] = present

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
    
    #current readings
    def read_2_byte(self, dxl_id, address, action):
        with self.dxl_lock:
            value, comm_result, dxl_error = self.packet_handler.read2ByteTxRx(
                self.port_handler, dxl_id, address)
        if self.check_result(comm_result, dxl_error, action):
            return value
        return None

    def read_present_current(self, dxl_id):
        val = self.read_2_byte(
            dxl_id, self.ADDR_PRESENT_CURRENT,
            f'Read current motor {dxl_id}')
        if val is None:
            return None
        if val > 32767:
            val -= 65536
        return val  # in mA for XC330

    def signed_to_unsigned_32(self, value):
        return int(value) & 0xFFFFFFFF

    def unsigned_to_signed_32(self, value):
        value = int(value)

        if value > 0x7FFFFFFF:
            value -= 0x100000000

        return value

    def read_present_position(self, dxl_id):
        present = self.read_4_byte(
            dxl_id,
            self.ADDR_PRESENT_POSITION,
            f'Read present position motor {dxl_id}'
        )

        if present is None:
            return None

        return self.unsigned_to_signed_32(present)

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
    # Position and coupling helpers
    # -----------------------------
    def get_home_position(self, dxl_id):
        index = self.motor_ids.index(dxl_id)
        return int(self.home_positions[index])

    def get_motor_direction(self, dxl_id):
        return float(self.motor_direction.get(dxl_id, 1.0))

    def disk_angle_to_motor_position(self, dxl_id, disk_angle_deg):
        direction = self.get_motor_direction(dxl_id)

        return int(
            self.get_home_position(dxl_id)
            + direction * disk_angle_deg * self.COUNTS_PER_DEGREE
        )

    def motor_position_to_disk_angle(self, dxl_id, position):
        direction = self.get_motor_direction(dxl_id)

        return float(position - self.get_home_position(dxl_id)) / (
            direction * self.COUNTS_PER_DEGREE
        )

    def is_position_inside_bounds(self, dxl_id, position):
        if dxl_id not in self.motor_position_limits:
            return False

        min_position, max_position = self.motor_position_limits[dxl_id]
        return min_position <= int(position) <= max_position

    def clamp_joint_targets(self, targets):
        targets['roll'] = max(
            -self.max_roll_deg,
            min(self.max_roll_deg, targets['roll'])
        )

        targets['pitch'] = max(
            -self.max_pitch_deg,
            min(self.max_pitch_deg, targets['pitch'])
        )

        targets['yaw'] = max(
            -self.max_yaw_deg,
            min(self.max_yaw_deg, targets['yaw'])
        )

        targets['grip'] = max(
            self.min_grip_deg,
            min(self.max_grip_deg, targets['grip'])
        )

        return targets

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

    def validate_disk_angles(self, disk_angles):
        for dxl_id, angle_deg in disk_angles.items():
            lower, upper = self.disk_angle_limits_deg[dxl_id]

            if angle_deg < lower or angle_deg > upper:
                self.get_logger().error(
                    f'Disk {dxl_id} command {angle_deg:.1f} deg outside limit '
                    f'[{lower:.1f}, {upper:.1f}] deg. Command refused.'
                )
                return False

        return True

    # -----------------------------
    # ROS callbacks
    # -----------------------------
    def joint_command_callback(self, msg):
        if not self.control_enabled:
            return

        targets = dict(self.target_joint_deg)

        for name, position_rad in zip(msg.name, msg.position):
            position_deg = math.degrees(position_rad)

            if name == 'roll_joint':
                targets['roll'] = position_deg
            elif name == 'pitch_joint':
                targets['pitch'] = position_deg
            elif name == 'yaw_joint':
                targets['yaw'] = position_deg
            elif name == 'grip_joint':
                targets['grip'] = position_deg

        targets = self.clamp_joint_targets(targets)

        disk_angles = self.joints_to_disks(targets)

        if not self.validate_disk_angles(disk_angles):
            return

        with self.state_lock:
            self.target_joint_deg = targets

        now = time.time()

        if now - self.last_log_time > 0.5:
            self.last_log_time = now
            self.get_logger().info(
                f"Target joint deg: R={targets['roll']:.1f}, "
                f"P={targets['pitch']:.1f}, "
                f"Y={targets['yaw']:.1f}, "
                f"G={targets['grip']:.1f} | "
                f"Disk deg: D1={disk_angles[1]:.1f}, "
                f"D2={disk_angles[2]:.1f}, "
                f"D3={disk_angles[3]:.1f}, "
                f"D4={disk_angles[4]:.1f}"
            )

    def enable_callback(self, msg):
        self.control_enabled = bool(msg.data)

        if self.control_enabled:
            self.get_logger().warn('Control enabled.')
        else:
            self.get_logger().warn('Control disabled. Holding current commanded position.')

    def estop_callback(self, msg):
        if msg.data:
            self.get_logger().error('ESTOP received.')
            self.control_enabled = False
            self.emergency_torque_off()

    def clear_limit_callback(self, msg):
        if msg.data:
            for dxl_id in self.limit_hit:
                self.limit_hit[dxl_id] = False
            self.get_logger().info('Limit cleared. Motion resumed.')

    # -----------------------------
    # Control loop
    # -----------------------------
    def step_value(self, current, target, max_step):
        error = target - current

        if abs(error) <= max_step:
            return target

        if error > 0:
            return current + max_step

        return current - max_step

    def control_loop(self):
        if not self.control_enabled:
            return

        with self.state_lock:
            target = dict(self.target_joint_deg)
            current = dict(self.current_joint_deg)

        max_step = self.command_speed_deg_per_sec * self.control_period_sec

        next_joint = {
            'roll': self.step_value(current['roll'], target['roll'], max_step),
            'pitch': self.step_value(current['pitch'], target['pitch'], max_step),
            'yaw': self.step_value(current['yaw'], target['yaw'], max_step),
            'grip': self.step_value(current['grip'], target['grip'], max_step),
        }

        disk_angles = self.joints_to_disks(next_joint)

        if not self.validate_disk_angles(disk_angles):
            return

        for dxl_id, disk_angle in disk_angles.items():
            motor_position = self.disk_angle_to_motor_position(dxl_id, disk_angle)

            if not self.is_position_inside_bounds(dxl_id, motor_position):
                self.get_logger().error(
                    f'Motor {dxl_id} position {motor_position} outside safe bounds. '
                    f'Command refused.'
                )
                return

        for dxl_id, disk_angle in disk_angles.items():
            motor_position = self.disk_angle_to_motor_position(dxl_id, disk_angle)

            # If this motor hit a limit, only allow movement away from it
            if self.limit_hit.get(dxl_id, False):
                current_pos = self.goal_positions.get(dxl_id, self.get_home_position(dxl_id))
                hit_dir = self.limit_hit_direction.get(dxl_id, 0)
                movement = motor_position - current_pos
                if hit_dir > 0 and movement > 0:
                    continue  # block positive movement
                elif hit_dir < 0 and movement < 0:
                    continue  # block negative movement
                else:
                    # Moving away from limit — clear the flag for this motor
                    self.limit_hit[dxl_id] = False
                    self.get_logger().info(f'Motor {dxl_id} moving away from limit — resuming.')

            self.command_position(dxl_id, motor_position)

        with self.state_lock:
            self.current_joint_deg = next_joint

    def command_position(self, dxl_id, position):
        self.goal_positions[dxl_id] = int(position)

        return self.write_4_byte(
            dxl_id,
            self.ADDR_GOAL_POSITION,
            self.signed_to_unsigned_32(position),
            f'Set goal position motor {dxl_id}'
        )

    def publish_joint_states(self):
        with self.state_lock:
            current = dict(self.current_joint_deg)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = [
            'roll_joint',
            'pitch_joint',
            'yaw_joint',
            'grip_joint',
        ]

        msg.position = [
            math.radians(current['roll']),
            math.radians(current['pitch']),
            math.radians(current['yaw']),
            math.radians(current['grip']),
        ]

        self.joint_state_pub.publish(msg)

    # -----------------------------
    # Safety/shutdown
    # -----------------------------
    def emergency_torque_off(self):
        self.get_logger().warn('Disabling torque on all motors.')

        for dxl_id in self.motor_ids:
            self.write_1_byte(
                dxl_id,
                self.ADDR_TORQUE_ENABLE,
                self.TORQUE_DISABLE,
                f'Torque off motor {dxl_id}'
            )

    def current_sensing_loop(self):
        if not self.control_enabled:
            return

        currents = {}
        for dxl_id in self.motor_ids:
            current_ma = self.read_present_current(dxl_id)
            if current_ma is None:
                currents[dxl_id] = 0
                continue

            currents[dxl_id] = abs(current_ma)
            self.current_readings[dxl_id] = abs(current_ma)

            if abs(current_ma) > self.CURRENT_LIMIT_MA:
                self.current_hit_count[dxl_id] += 1
            else:
                self.current_hit_count[dxl_id] = 0

            if self.current_hit_count[dxl_id] >= self.CURRENT_CONSECUTIVE:
                self.current_hit_count[dxl_id] = 0
                self.handle_limit_hit(dxl_id)

        # Publish current readings
        msg = String()
        msg.data = (f"M1:{currents.get(1,0):.0f}mA  "
                    f"M2:{currents.get(2,0):.0f}mA  "
                    f"M3:{currents.get(3,0):.0f}mA  "
                    f"M4:{currents.get(4,0):.0f}mA")
        self.current_pub.publish(msg)

    def handle_limit_hit(self, dxl_id):
        self.get_logger().warn(
            f'Motor {dxl_id} limit hit! Backing off {self.BACKOFF_DEG} deg.')

        msg = String()
        msg.data = f'Motor {dxl_id} limit hit'
        self.limit_pub.publish(msg)

        self.limit_hit[dxl_id] = True

        current_pos = self.goal_positions.get(dxl_id, self.get_home_position(dxl_id))
        backoff_counts = int(self.BACKOFF_DEG * self.COUNTS_PER_DEGREE)

        if current_pos > self.get_home_position(dxl_id):
            new_pos = current_pos - backoff_counts
            self.limit_hit_direction[dxl_id] = +1  # was moving positive
        else:
            new_pos = current_pos + backoff_counts
            self.limit_hit_direction[dxl_id] = -1  # was moving negative

        self.command_position(dxl_id, new_pos)
        self.goal_positions[dxl_id] = new_pos

        with self.state_lock:
            self.target_joint_deg = dict(self.current_joint_deg)

    def _reenable_control(self):
        self.control_enabled = True
        self.get_logger().info('Control re-enabled after limit backoff.')

    def update_coupling_callback(self, msg):
        m = msg.data
        self.ROLL_D1  = m[0]
        self.PITCH_D2 = m[5]
        self.YAW_D2   = m[9]
        self.YAW_D3   = m[10]
        self.YAW_D4   = m[11]
        self.GRIP_D3  = m[14]
        self.GRIP_D4  = m[15]
        self.get_logger().info('Coupling matrix updated.')

    def update_limits_callback(self, msg):
        d = msg.data
        self.max_roll_deg  = max(abs(d[0]), abs(d[1]))
        self.max_pitch_deg = max(abs(d[2]), abs(d[3]))
        self.max_yaw_deg   = max(abs(d[4]), abs(d[5]))
        self.min_grip_deg  = d[6]
        self.max_grip_deg  = d[7]
        self.get_logger().info(
            f'Limits updated: roll=±{self.max_roll_deg:.1f}, '
            f'pitch=±{self.max_pitch_deg:.1f}, '
            f'yaw=±{self.max_yaw_deg:.1f}, '
            f'grip=[{self.min_grip_deg:.1f}, {self.max_grip_deg:.1f}]')

    #def shutdown(self):
        #self.control_enabled = False
        #self.limit_hit = False
        #try:
            #for dxl_id in self.motor_ids:
                #self.write_1_byte(
                    #dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE,
                    #f'Torque off motor {dxl_id}')
            #self.port_handler.closePort()
        #except Exception as e:
            #self.get_logger().info(f'Shutdown cleanup: {e}')
        #self.get_logger().info('Closed Dynamixel port.')
       


    def release_port_callback(self, request, response):
        self.get_logger().info('Releasing serial port for RFID scan...')
        self.control_enabled = False
    # Disable torque on all motors first
        for dxl_id in self.motor_ids:
            self.write_1_byte(
            dxl_id,
            self.ADDR_TORQUE_ENABLE,
            self.TORQUE_DISABLE,
            f'Torque off motor {dxl_id}'
        )
    # Close the port
        self.port_handler.closePort()
        self.get_logger().info('Port released.')
        response.success = True
        response.message = 'Port released'
        return response

    def reclaim_port_callback(self, request, response):
        self.get_logger().info('Reclaiming serial port...')
        try:
            self.connect_to_motors()
            self.control_enabled = True
            self.get_logger().info('Port reclaimed, motors ready.')
            response.success = True
            response.message = 'Port reclaimed'
        except Exception as e:
            self.get_logger().error(f'Failed to reclaim port: {e}')
            response.success = False
            response.message = str(e)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = DynamixelControllerNode()

    def cleanup():
        try:
            for dxl_id in node.motor_ids:
                node.write_1_byte(dxl_id, node.ADDR_TORQUE_ENABLE, node.TORQUE_DISABLE, '')
            node.port_handler.closePort()
        except:
            pass

    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rclpy.spin(node)
    finally:
        cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()