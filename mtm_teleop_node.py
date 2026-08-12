#!/usr/bin/env python3
"""
dVRK MTM teleoperation node.
Converts MTM handle pose to roll/pitch/yaw motor commands, replacing the
retired Quest path now in quest/quest_teleop_node.py.

Topics (all pre-existing -- this node creates none):
  Sub: /MTMR/measured_js          (sensor_msgs/JointState)     100 Hz
  Sub: /MTMR/measured_cp          (geometry_msgs/PoseStamped)  100 Hz
  Sub: /MTMR/gripper/measured_js  (sensor_msgs/JointState)     100 Hz
  Sub: /footpedals/clutch         (sensor_msgs/Joy)
  Sub: /footpedals/coag           (sensor_msgs/Joy)
  Pub: /h4hr/joint_command        (sensor_msgs/JointState)

Where the angles come from -- orientation_source:

  'joints' (default)  Read the MTM handle gimbal's own wrist_roll /
      wrist_pitch / wrist_yaw straight off /<arm>/measured_js. They are
      already independent scalar joint angles, so there is no decomposition
      to do: no gimbal lock, and no coupling of roll into pitch/yaw. This is
      the direct 1:1 wrist-to-wrist map and what you almost certainly want.

  'quaternion'        Reconstruct the angles from the /<arm>/measured_cp
      orientation instead, via a swing-twist split (see swing_twist below).
      Useful only if you need the handle's orientation in Cartesian space
      rather than the gimbal's joint angles -- e.g. if a base-frame rotation
      or a registration offset has to be applied first.

Either way the angles are taken relative to a home reference, scaled,
rate-limited, and clamped to the tool's mechanical limits before publishing.

Controls:
  MTM handle orientation → roll/pitch/yaw
  MTM gripper            → grip (squeeze to close)
  Clutch pedal (hold)    → freeze the tool and reposition the handle freely;
                           releasing resumes from wherever the handle is, with
                           no jump
  Coag pedal (tap)       → set home: current handle pose becomes the zero
                           reference

Usage:
  python3 mtm_teleop_node.py
  python3 mtm_teleop_node.py --ros-args -p arm:=MTML -p debug_axes:=true
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Joy

# ---------- Quaternion math ----------

def quat_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_inverse(q):
    """Inverse of a unit quaternion [x, y, z, w]."""
    x, y, z, w = q
    return np.array([-x, -y, -z, w])

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_canonical(q):
    """Force w >= 0 so interpolation always takes the short way round."""
    return -q if q[3] < 0.0 else q

def quat_from_rotvec(v):
    """Rotation vector (axis * angle, radians) -> quaternion [x, y, z, w]."""
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = v / theta
    s = math.sin(theta / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(theta / 2.0)])

def quat_to_rotvec(q):
    """Quaternion -> rotation vector. Inverse of quat_from_rotvec."""
    q = quat_canonical(quat_normalize(q))
    w = max(-1.0, min(1.0, float(q[3])))
    s = math.sqrt(max(0.0, 1.0 - w*w))
    if s < 1e-8:          # angle ~ 0; axis is meaningless and magnitude is too
        return np.zeros(3)
    return (q[:3] / s) * (2.0 * math.acos(w))

def quat_slerp(q0, q1, t):
    """Interpolate along the shortest geodesic between two orientations."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:           # opposite hemisphere: negate so we take the short arc
        q1 = -q1
        d = -d
    d = max(-1.0, min(1.0, d))
    if d > 0.9995:        # nearly identical; slerp is ill-conditioned, lerp suffices
        return quat_normalize(q0 + t * (q1 - q0))
    theta = math.acos(d) * t
    q_perp = quat_normalize(q1 - q0 * d)
    return q0 * math.cos(theta) + q_perp * math.sin(theta)

def quat_angle_between(q0, q1):
    """Geodesic angle (radians) between two orientations, always in [0, pi]."""
    d = abs(float(np.dot(quat_normalize(q0), quat_normalize(q1))))
    return 2.0 * math.acos(min(1.0, d))

def swing_twist(q, axis):
    """
    Split q into (swing, twist) about `axis`, such that q = swing * twist.

    twist is the rotation about `axis` (the handle's roll); swing is everything
    else, and its axis is perpendicular to `axis` by construction -- so swing
    carries exactly two degrees of freedom and maps onto pitch/yaw with no
    third angle leaking in.

    This is why the measured_cp quaternion is never converted to an Euler RPY
    triple directly: Euler would couple handle roll into pitch/yaw and is
    singular at pitch = 90 deg, uncomfortably close to this tool's 79 deg pitch
    limit. Swing-twist only degenerates at 180 deg of swing, far outside any
    reachable bound.
    """
    proj = float(np.dot(q[:3], axis)) * axis
    twist = np.array([proj[0], proj[1], proj[2], q[3]])
    n = float(np.linalg.norm(twist))
    if n < 1e-8:          # 180 deg swing: twist is undefined, call it zero
        return quat_normalize(q), np.array([0.0, 0.0, 0.0, 1.0])
    twist = twist / n
    return quat_multiply(q, quat_inverse(twist)), twist

def twist_angle(twist, axis):
    """Signed rotation angle of a pure-twist quaternion, wrapped to [-pi, pi]."""
    twist = quat_canonical(twist)
    return 2.0 * math.atan2(float(np.dot(twist[:3], axis)), float(twist[3]))

def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}

# dVRK MTM gimbal joint names, as published on /<arm>/measured_js.
DEFAULT_ROLL_JOINT  = 'wrist_roll'
DEFAULT_PITCH_JOINT = 'wrist_pitch'
DEFAULT_YAW_JOINT   = 'wrist_yaw'


def clamp(value, low, high):
    return max(low, min(high, value))


def step_toward(current, target, max_step):
    """Move current toward target by at most max_step."""
    error = target - current
    if abs(error) <= max_step:
        return target
    return current + max_step if error > 0 else current - max_step


def extract_gimbal(name_list, position_list,
                   roll_name=DEFAULT_ROLL_JOINT,
                   pitch_name=DEFAULT_PITCH_JOINT,
                   yaw_name=DEFAULT_YAW_JOINT):
    """
    Pull the three gimbal angles out of a measured_js message by NAME.

    Indexed by name rather than array position on purpose: the order in
    measured_js is a dVRK implementation detail, the names are the contract.
    Raises KeyError naming the joint that was missing.
    """
    lookup = dict(zip(name_list, position_list))
    return (
        float(lookup[roll_name]),
        float(lookup[pitch_name]),
        float(lookup[yaw_name]),
    )


class MTMWristMapper:
    """
    MTM handle gimbal -> tool roll/pitch/yaw/grip. Radians throughout.

    Lives here, outside the Node, so the monitor and the dynamixel controller
    can `import mtm_teleop_node` and reuse the exact same behaviour rather than
    each keeping their own copy. That matters on a teleoperated instrument:
    three drifting copies means the monitor eventually shows angles the motors
    are not actually being sent.

    The gimbal joints are already three independent scalar angles, so this is a
    per-axis map -- no quaternion, no Euler decomposition, hence no gimbal lock
    near the tool's 79 deg pitch limit and no roll bleeding into pitch/yaw.

    Per incoming sample:

        mapper.update_gripper(raw_gripper_rad)      # whenever it arrives
        out = mapper.update(roll, pitch, yaw, dt)   # None while frozen
        if out is not None:
            roll_cmd, pitch_cmd, yaw_cmd, grip_cmd = out

    `update` returns None when there is nothing to command -- on the first
    sample (which becomes the home reference) and while the clutch is held.
    Hold the last command in that case.
    """

    def __init__(self,
                 max_roll, max_pitch, max_yaw,
                 open_grip, close_grip,
                 mtm_grip_open, mtm_grip_closed,
                 motion_scale=1.0,
                 max_angular_speed=math.radians(180.0),
                 roll_sign=1.0, pitch_sign=1.0, yaw_sign=1.0):
        if abs(mtm_grip_open - mtm_grip_closed) < 1e-6:
            raise ValueError('mtm_grip_open and mtm_grip_closed must differ')

        self.max_roll  = max_roll
        self.max_pitch = max_pitch
        self.max_yaw   = max_yaw
        self.open_grip  = open_grip
        self.close_grip = close_grip
        self.mtm_grip_open   = mtm_grip_open
        self.mtm_grip_closed = mtm_grip_closed
        self.motion_scale = motion_scale
        self.max_angular_speed = max_angular_speed
        self.roll_sign  = roll_sign
        self.pitch_sign = pitch_sign
        self.yaw_sign   = yaw_sign

        self.ref = None             # gimbal angles at home
        self.at_clutch_start = None
        self.clutched = False
        self.home_requested = False

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.grip = open_grip
        self.raw_grip = None

    def request_home(self):
        """Re-zero: the next sample's gimbal pose becomes the new reference."""
        self.home_requested = True

    def set_clutch(self, engaged):
        """True freezes the tool and lets the handle be repositioned."""
        self.clutched = bool(engaged)

    def update_gripper(self, raw_rad):
        """
        MTM gripper angle -> tool grip.

        The MTM reads large when open (~1.07 rad untouched) and falls toward 0
        as it is squeezed, while the tool's grip runs the other way (-15 deg
        open, +38 deg closed), so this is a deliberately inverted map.
        """
        self.raw_grip = float(raw_rad)
        closed_fraction = (self.mtm_grip_open - self.raw_grip) / (
            self.mtm_grip_open - self.mtm_grip_closed
        )
        closed_fraction = clamp(closed_fraction, 0.0, 1.0)
        self.grip = self.open_grip + closed_fraction * (
            self.close_grip - self.open_grip
        )
        return self.grip

    def update(self, raw_roll, raw_pitch, raw_yaw, dt):
        """
        Feed one gimbal sample. Returns (roll, pitch, yaw, grip) in radians,
        or None if there is nothing to command yet (homing or clutched).
        """
        raw = (float(raw_roll), float(raw_pitch), float(raw_yaw))

        # Bootstrap, or an explicit re-zero: this pose becomes the new home, so
        # the tool snaps to neutral.
        if self.ref is None or self.home_requested:
            self.ref = raw
            self.roll = self.pitch = self.yaw = 0.0
            self.home_requested = False
            self.at_clutch_start = None
            return None

        # Clutch held: freeze, and remember where the gimbal was when it
        # engaged so the release can be made seamless.
        if self.clutched:
            if self.at_clutch_start is None:
                self.at_clutch_start = raw
            return None

        # Clutch just released: shift the reference by exactly how far the
        # gimbal moved while frozen, so the tool resumes where it left off
        # instead of jumping to match the handle's new pose.
        if self.at_clutch_start is not None:
            self.ref = tuple(
                r + (n - c)
                for r, n, c in zip(self.ref, raw, self.at_clutch_start)
            )
            self.at_clutch_start = None

        delta = [(n - r) * self.motion_scale for n, r in zip(raw, self.ref)]
        targets = (
            self.roll_sign  * delta[0],
            self.pitch_sign * delta[1],
            self.yaw_sign   * delta[2],
        )

        # Rate-limit per axis, then clamp to the mechanical limits. Storing the
        # clamped values (not the raw target) is what stops the tracker winding
        # up past a limit and lagging when it reverses.
        max_step = self.max_angular_speed * dt
        self.roll = clamp(
            step_toward(self.roll, targets[0], max_step),
            -self.max_roll, self.max_roll)
        self.pitch = clamp(
            step_toward(self.pitch, targets[1], max_step),
            -self.max_pitch, self.max_pitch)
        self.yaw = clamp(
            step_toward(self.yaw, targets[2], max_step),
            -self.max_yaw, self.max_yaw)

        return self.command()

    def command(self):
        """Last computed (roll, pitch, yaw, grip), radians."""
        return (
            self.roll,
            self.pitch,
            self.yaw,
            clamp(self.grip,
                  min(self.open_grip, self.close_grip),
                  max(self.open_grip, self.close_grip)),
        )

    @property
    def homed(self):
        return self.ref is not None


class MTMTeleopNode(Node):
    def __init__(self):
        super().__init__('mtm_teleop_node')

        # MTMR or MTML. Only changes which existing dVRK topics are read.
        self.declare_parameter('arm', 'MTMR')
        self.arm = str(self.get_parameter('arm').value)

        # 'joints' (default) or 'quaternion' -- see the module docstring.
        self.declare_parameter('orientation_source', 'joints')
        self.orientation_source = str(
            self.get_parameter('orientation_source').value
        ).lower()
        if self.orientation_source not in ('joints', 'quaternion'):
            raise ValueError(
                "orientation_source must be 'joints' or 'quaternion', got "
                f"'{self.orientation_source}'"
            )

        # Which MTM gimbal joint feeds each tool joint. Indexed by name rather
        # than by position in the array: measured_js order is a dVRK
        # implementation detail, the names are the contract.
        self.declare_parameter('roll_joint_name',  'wrist_roll')
        self.declare_parameter('pitch_joint_name', 'wrist_pitch')
        self.declare_parameter('yaw_joint_name',   'wrist_yaw')
        self.mtm_roll_name  = str(self.get_parameter('roll_joint_name').value)
        self.mtm_pitch_name = str(self.get_parameter('pitch_joint_name').value)
        self.mtm_yaw_name   = str(self.get_parameter('yaw_joint_name').value)

        # Joint limits (degrees). Matched to dynamixel_controller_node.py, which
        # clamps to these anyway -- keeping them equal means the value published
        # here is the value the tool actually reaches, so the internal tracker
        # never winds up past a limit the controller silently trimmed.
        self.declare_parameter('max_roll_deg',   255.0)
        self.declare_parameter('max_pitch_deg',   79.0)
        self.declare_parameter('max_yaw_deg',     79.0)

        # Where the MTM gripper's two extremes map to, in the controller's
        # sense: open = -200 deg, closed = +38 deg (see
        # dynamixel_controller_node.py).
        #
        # This is a mapping endpoint, not a safety limit -- the limit is
        # min_grip_deg on the controller. The MTM handle only travels about
        # 60 deg, so it is being stretched across 238 deg of tool grip here:
        # roughly 4 deg of jaw per 1 deg of handle. Reduce open_grip_deg if
        # that is too coarse to control finely; the tool can still reach -200
        # by other inputs either way.
        self.declare_parameter('open_grip_deg', -200.0)
        self.declare_parameter('close_grip_deg',  38.0)

        # MTM gripper angle (rad) at its extremes, as read from
        # /<arm>/gripper/measured_js. Measured on this console the untouched
        # handle rests at ~1.07 rad and squeezing shut drives it toward 0.
        # Squeeze fully and read the raw value in the log line to retune.
        self.declare_parameter('mtm_grip_open_rad',   1.05)
        self.declare_parameter('mtm_grip_closed_rad', 0.0)

        # Caps how fast the commanded rotation may change (deg/sec), regardless
        # of how fast the handle is moved.
        self.declare_parameter('max_angular_speed_deg_per_sec', 180.0)

        # 1.0 = the tool rotates exactly as much as the handle does. The MTM is
        # a master manipulator built for this, so unlike the Quest (which used
        # 0.3) a direct mapping is the natural default; lower it if the +-79 deg
        # pitch/yaw limits saturate too easily for the task.
        self.declare_parameter('motion_scale', 1.0)
        self.motion_scale = float(self.get_parameter('motion_scale').value)

        # Which handle axis drives which tool joint, with independent sign
        # flips. The MTM tip frame's z runs along the handle, so twisting the
        # handle about its own length is the tool's roll. If a handle twist
        # comes out as pitch/yaw, or an axis runs backwards, launch with
        # -p debug_axes:=true, move one axis at a time, and set these from what
        # the log shows -- no code change needed.
        self.declare_parameter('roll_axis',  'z')
        self.declare_parameter('pitch_axis', 'x')
        self.declare_parameter('yaw_axis',   'y')
        self.declare_parameter('roll_sign',   1.0)
        # Negated: the MTM handle's wrist_pitch runs opposite to the tool's
        # pitch. Corrected here, at the operator-mapping layer, rather than by
        # flipping motor_direction[2] or PITCH_D2 in the controller -- disk 2
        # drives both pitch AND yaw, so reversing it physically would leave the
        # controller solving d3/d4 against the wrong d2 and corrupt yaw (a
        # +40 deg yaw command would come out near +105 deg). Only the sign of
        # the commanded joint angle belongs here.
        self.declare_parameter('pitch_sign', -1.0)
        self.declare_parameter('yaw_sign',    1.0)
        self.declare_parameter('debug_axes', False)

        # Optional: drive pitch/yaw by *translating* the handle rather than
        # rotating your wrist (roll still comes from handle twist). Off by
        # default -- the tool's wrist has no translational joint, so position
        # is otherwise read but unused.
        self.declare_parameter('position_to_pitch_yaw', False)
        self.declare_parameter('position_gain_deg_per_m', 300.0)

        roll_axis_name  = str(self.get_parameter('roll_axis').value).lower()
        pitch_axis_name = str(self.get_parameter('pitch_axis').value).lower()
        yaw_axis_name   = str(self.get_parameter('yaw_axis').value).lower()
        for label, name in (('roll_axis', roll_axis_name),
                            ('pitch_axis', pitch_axis_name),
                            ('yaw_axis', yaw_axis_name)):
            if name not in AXIS_INDEX:
                raise ValueError(f"{label} must be one of x/y/z, got '{name}'")
        if len({roll_axis_name, pitch_axis_name, yaw_axis_name}) != 3:
            raise ValueError('roll_axis, pitch_axis and yaw_axis must all differ')

        self.roll_idx  = AXIS_INDEX[roll_axis_name]
        self.pitch_idx = AXIS_INDEX[pitch_axis_name]
        self.yaw_idx   = AXIS_INDEX[yaw_axis_name]
        self.roll_vec = np.zeros(3)
        self.roll_vec[self.roll_idx] = 1.0

        self.roll_sign  = float(self.get_parameter('roll_sign').value)
        self.pitch_sign = float(self.get_parameter('pitch_sign').value)
        self.yaw_sign   = float(self.get_parameter('yaw_sign').value)
        self.debug_axes = bool(self.get_parameter('debug_axes').value)

        self.max_roll  = math.radians(self.get_parameter('max_roll_deg').value)
        self.max_pitch = math.radians(self.get_parameter('max_pitch_deg').value)
        self.max_yaw   = math.radians(self.get_parameter('max_yaw_deg').value)
        self.open_grip  = math.radians(self.get_parameter('open_grip_deg').value)
        self.close_grip = math.radians(self.get_parameter('close_grip_deg').value)

        self.mtm_grip_open   = float(self.get_parameter('mtm_grip_open_rad').value)
        self.mtm_grip_closed = float(self.get_parameter('mtm_grip_closed_rad').value)
        if abs(self.mtm_grip_open - self.mtm_grip_closed) < 1e-6:
            raise ValueError('mtm_grip_open_rad and mtm_grip_closed_rad must differ')

        self.max_angular_speed = math.radians(
            self.get_parameter('max_angular_speed_deg_per_sec').value
        )

        self.use_position = bool(self.get_parameter('position_to_pitch_yaw').value)
        self.position_gain = math.radians(
            float(self.get_parameter('position_gain_deg_per_m').value)
        )

        # State
        self.ref_quat = None   # handle orientation at home (the zero reference)
        self.ref_pos  = None   # handle position at home
        # Rate-limited/clamped target, tracked as a single SO(3) rotation and
        # decomposed to joint angles only at publish time. Interpolating here
        # rather than per-axis means a diagonal move follows the geodesic and
        # respects max_angular_speed as a true angular rate.
        self.current_quat = IDENTITY_QUAT.copy()
        # Roll is unwrapped across the +-180 deg quaternion branch cut so the
        # tool can reach its full +-255 deg range; a quaternion alone cannot
        # distinguish +200 deg from -160 deg.
        self.unwrapped_roll = 0.0
        self.prev_roll_raw  = None
        self.current_roll  = 0.0
        self.current_pitch = 0.0
        self.current_yaw   = 0.0
        self.grip = self.open_grip
        self.raw_grip = None            # last raw MTM gripper angle, for tuning
        self.clutched = False           # True while clutch pedal held
        self.pose_at_clutch_start = None
        self.pos_at_clutch_start  = None
        self.coag_prev = False          # for rising-edge detection
        self.home_reset_requested = False
        self.last_pose_time = None

        # The joints path delegates entirely to this, so the monitor and the
        # dynamixel controller can build the same object and get identical
        # behaviour. The quaternion path keeps its own SO(3) tracking above.
        self.mapper = MTMWristMapper(
            max_roll=self.max_roll,
            max_pitch=self.max_pitch,
            max_yaw=self.max_yaw,
            open_grip=self.open_grip,
            close_grip=self.close_grip,
            mtm_grip_open=self.mtm_grip_open,
            mtm_grip_closed=self.mtm_grip_closed,
            motion_scale=self.motion_scale,
            max_angular_speed=self.max_angular_speed,
            roll_sign=self.roll_sign,
            pitch_sign=self.pitch_sign,
            yaw_sign=self.yaw_sign,
        )
        self.missing_joints_warned = False

        # Publisher / Subscribers -- every topic below already exists.
        self.cmd_pub = self.create_publisher(JointState, '/h4hr/joint_command', 10)

        # Only the selected source is subscribed: both callbacks consume the
        # home-reset flag and publish, so running them together would race.
        if self.orientation_source == 'joints':
            self.joints_sub = self.create_subscription(
                JointState, f'/{self.arm}/measured_js',
                self.joints_callback, 10)
        else:
            self.pose_sub = self.create_subscription(
                PoseStamped, f'/{self.arm}/measured_cp',
                self.pose_callback, 10)

        self.gripper_sub = self.create_subscription(
            JointState, f'/{self.arm}/gripper/measured_js',
            self.gripper_callback, 10)

        self.clutch_sub = self.create_subscription(
            Joy, '/footpedals/clutch', self.clutch_callback, 10)

        self.coag_sub = self.create_subscription(
            Joy, '/footpedals/coag', self.coag_callback, 10)

        self.get_logger().info(
            f'MTM teleop node ready, following {self.arm} '
            f'(orientation from {self.orientation_source}).'
        )
        self.get_logger().info('Tap the coag pedal to set home (zero the reference pose).')
        self.get_logger().info('Hold the clutch pedal to freeze the tool and reposition the handle.')
        self.get_logger().info('Squeeze the MTM gripper to close the tool grip.')
        if self.use_position:
            self.get_logger().info('Handle translation drives pitch/yaw; twist drives roll.')

    def clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    def step_value(self, current, target, max_step):
        error = target - current
        if abs(error) <= max_step:
            return target
        return current + max_step if error > 0 else current - max_step

    # ---------- Pedals ----------

    def clutch_callback(self, msg):
        """Clutch pedal held → freeze the tool, let the handle be repositioned."""
        if not msg.buttons:
            return
        clutched = bool(msg.buttons[0])
        if clutched != self.clutched:
            if clutched:
                self.get_logger().info('Clutch engaged — tool frozen, reposition freely.')
            else:
                self.get_logger().info('Clutch released — control resumed.')
            self.clutched = clutched
            self.mapper.set_clutch(clutched)

    def coag_callback(self, msg):
        """Coag pedal tap → re-zero the reference at the current handle pose."""
        if not msg.buttons:
            return
        pressed = bool(msg.buttons[0])
        if pressed and not self.coag_prev:
            self.home_reset_requested = True   # consumed by the quaternion path
            self.mapper.request_home()
            self.get_logger().info('Home reset requested.')
        self.coag_prev = pressed

    # ---------- Gripper ----------

    def gripper_callback(self, msg):
        """
        MTM gripper angle → tool grip.

        The MTM reads large when open (~1.07 rad untouched) and falls toward 0
        as it is squeezed, while the tool's grip runs the other way (-15 deg
        open, +38 deg closed), so this is a deliberately inverted map.
        """
        if not msg.position:
            return
        raw = float(msg.position[0])
        self.raw_grip = raw
        # The mapper owns the inversion; the quaternion path reads self.grip.
        self.grip = self.mapper.update_gripper(raw)

    # ---------- Orientation from the MTM's own gimbal joints ----------

    def joints_callback(self, msg):
        """
        MTM wrist gimbal joints → tool roll/pitch/yaw, with no quaternion in
        the path at all.

        wrist_roll/pitch/yaw are already the three independent angles of the
        handle's gimbal, so this is a straight per-axis map. That is the whole
        advantage over the measured_cp route: nothing to decompose means no
        gimbal lock near the 79 deg pitch limit and no roll bleeding into
        pitch/yaw.
        """
        try:
            raw = extract_gimbal(
                msg.name, msg.position,
                self.mtm_roll_name, self.mtm_pitch_name, self.mtm_yaw_name)
        except KeyError:
            if not self.missing_joints_warned:
                self.get_logger().error(
                    f'{self.arm}/measured_js has no '
                    f'{self.mtm_roll_name}/{self.mtm_pitch_name}/'
                    f'{self.mtm_yaw_name}. Published names: {list(msg.name)}. '
                    'Set the *_joint_name parameters to match.'
                )
                self.missing_joints_warned = True
            return

        if self.debug_axes:
            self.get_logger().info(
                'mtm gimbal deg: '
                f'{self.mtm_roll_name}={math.degrees(raw[0]):7.2f} '
                f'{self.mtm_pitch_name}={math.degrees(raw[1]):7.2f} '
                f'{self.mtm_yaw_name}={math.degrees(raw[2]):7.2f}',
                throttle_duration_sec=0.2
            )

        now = self.get_clock().now()
        if self.last_pose_time is None:
            dt = 0.0
        else:
            dt = (now - self.last_pose_time).nanoseconds * 1e-9
        self.last_pose_time = now

        out = self.mapper.update(raw[0], raw[1], raw[2], dt)
        if out is None:          # homing on this sample, or clutched: hold
            return

        roll, pitch, yaw, _grip = out
        self.current_roll  = roll
        self.current_pitch = pitch
        self.current_yaw   = yaw

        self.publish_command(roll, pitch, yaw)

    # ---------- Command out ----------

    def publish_command(self, roll, pitch, yaw):
        """Clamp the grip and publish the four tool joints, in radians."""
        grip = self.clamp(
            self.grip,
            min(self.open_grip, self.close_grip),
            max(self.open_grip, self.close_grip),
        )

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = ['roll_joint', 'pitch_joint', 'yaw_joint', 'grip_joint']
        cmd.position = [roll, pitch, yaw, grip]
        self.cmd_pub.publish(cmd)

        raw_grip_str = 'n/a' if self.raw_grip is None else f'{self.raw_grip:.3f}'
        self.get_logger().info(
            f'R={math.degrees(roll):.1f}° '
            f'P={math.degrees(pitch):.1f}° '
            f'Y={math.degrees(yaw):.1f}° '
            f'G={math.degrees(grip):.1f}° '
            f'(mtm grip {raw_grip_str} rad)',
            throttle_duration_sec=0.5
        )

    # ---------- Orientation from the measured_cp quaternion ----------

    def pose_callback(self, msg):
        """Convert MTM handle pose to a joint command."""
        ros_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        norm = float(np.linalg.norm(ros_quat))
        if not np.isfinite(norm) or norm < 1e-6:
            self.get_logger().warn(
                'Ignoring degenerate quaternion from measured_cp.',
                throttle_duration_sec=2.0
            )
            return
        ros_quat = ros_quat / norm

        ros_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        # Bootstrap (no reference yet) or explicit coag "set home": the current
        # handle pose becomes the new zero, so the tool target snaps to neutral
        # (0,0,0). This is a deliberate recenter, unlike the clutch below.
        if self.ref_quat is None or self.home_reset_requested:
            self.ref_quat = ros_quat.copy()
            self.ref_pos = ros_pos.copy()
            self.current_quat = IDENTITY_QUAT.copy()
            self.unwrapped_roll = 0.0
            self.prev_roll_raw = None
            self.current_roll = 0.0
            self.current_pitch = 0.0
            self.current_yaw = 0.0
            self.last_pose_time = self.get_clock().now()
            self.home_reset_requested = False
            self.pose_at_clutch_start = None
            self.pos_at_clutch_start = None
            return

        # Frozen while the clutch is held: remember the handle pose at the
        # moment it engaged and publish nothing, so the tool holds its last
        # command -- grip included.
        if self.clutched:
            if self.pose_at_clutch_start is None:
                self.pose_at_clutch_start = ros_quat.copy()
                self.pos_at_clutch_start = ros_pos.copy()
            self.last_pose_time = self.get_clock().now()
            return

        # Just released the clutch: rebase the reference to discount exactly
        # the motion the handle made *while frozen*, so the tool continues from
        # where it was instead of snapping back toward zero.
        if self.pose_at_clutch_start is not None:
            self.ref_quat = quat_multiply(
                quat_multiply(ros_quat, quat_inverse(self.pose_at_clutch_start)),
                self.ref_quat
            )
            self.ref_quat = self.ref_quat / np.linalg.norm(self.ref_quat)
            self.ref_pos = self.ref_pos + (ros_pos - self.pos_at_clutch_start)
            self.pose_at_clutch_start = None
            self.pos_at_clutch_start = None

        # Rotation of the handle relative to home, expressed in the home frame.
        rel_quat = quat_multiply(quat_inverse(self.ref_quat), ros_quat)
        rel_quat = rel_quat / np.linalg.norm(rel_quat)

        now = self.get_clock().now()
        if self.last_pose_time is None:
            dt = 0.0
        else:
            dt = (now - self.last_pose_time).nanoseconds * 1e-9
        self.last_pose_time = now

        # Scale the operator's rotation by slerping out from identity, which
        # shrinks the rotation angle and leaves its axis untouched.
        q_target = quat_slerp(IDENTITY_QUAT, rel_quat, self.motion_scale)

        # Rate-limit along the geodesic: one angular speed cap for the whole
        # rotation rather than one per axis, so a diagonal move neither exceeds
        # max_angular_speed nor bows off the shortest path.
        max_step = self.max_angular_speed * dt
        angle = quat_angle_between(self.current_quat, q_target)
        if angle <= max_step or angle < 1e-9:
            self.current_quat = q_target
        else:
            self.current_quat = quat_slerp(
                self.current_quat, q_target, max_step / angle
            )

        # Decompose once, at the last point before the joint command.
        swing, twist = swing_twist(self.current_quat, self.roll_vec)
        rotvec = quat_to_rotvec(swing)
        pitch = self.pitch_sign * float(rotvec[self.pitch_idx])
        yaw   = self.yaw_sign   * float(rotvec[self.yaw_idx])

        if self.debug_axes:
            self.get_logger().info(
                'swing rotvec deg: '
                f'x={math.degrees(rotvec[0]):7.2f} '
                f'y={math.degrees(rotvec[1]):7.2f} '
                f'z={math.degrees(rotvec[2]):7.2f} | '
                f'twist about {"xyz"[self.roll_idx]}='
                f'{math.degrees(twist_angle(twist, self.roll_vec)):7.2f}',
                throttle_duration_sec=0.2
            )

        # Roll from the handle's twist, unwrapped across the +-180 deg branch
        # cut so the tool's full +-255 deg range stays reachable.
        roll_raw = self.roll_sign * twist_angle(twist, self.roll_vec)
        if self.prev_roll_raw is None:
            self.unwrapped_roll = roll_raw
        else:
            self.unwrapped_roll += wrap_to_pi(roll_raw - self.prev_roll_raw)
        self.prev_roll_raw = roll_raw

        # Optional: handle translation drives pitch/yaw instead of wrist swing.
        # These replace the swing values, so they miss the geodesic rate limit
        # above and get their own per-axis one -- otherwise a fast hand
        # translation would step pitch/yaw arbitrarily far in a single message.
        if self.use_position:
            d_pos = ros_pos - self.ref_pos
            pitch_target = self.pitch_sign * self.position_gain * float(d_pos[2])
            yaw_target   = self.yaw_sign   * self.position_gain * float(d_pos[1])
            pitch = self.step_value(self.current_pitch, pitch_target, max_step)
            yaw   = self.step_value(self.current_yaw, yaw_target, max_step)

        # Clamp in joint space -- the limits are mechanical, so this is the one
        # place they legitimately apply.
        roll  = self.clamp(self.unwrapped_roll, -self.max_roll, self.max_roll)
        pitch = self.clamp(pitch, -self.max_pitch, self.max_pitch)
        yaw   = self.clamp(yaw, -self.max_yaw, self.max_yaw)

        # Fold the clamped result back into the tracked rotation. Without this
        # the quaternion keeps integrating past what the tool can reach, and
        # reversing direction lags until it unwinds.
        self.unwrapped_roll = roll
        swing_back = np.zeros(3)
        swing_back[self.pitch_idx] = pitch * self.pitch_sign
        swing_back[self.yaw_idx]   = yaw * self.yaw_sign
        swing_c = quat_from_rotvec(swing_back)
        twist_c = quat_from_rotvec(self.roll_vec * (roll * self.roll_sign))
        self.current_quat = quat_normalize(quat_multiply(swing_c, twist_c))
        self.prev_roll_raw = wrap_to_pi(roll * self.roll_sign) * self.roll_sign

        self.current_roll  = roll
        self.current_pitch = pitch
        self.current_yaw   = yaw

        self.publish_command(roll, pitch, yaw)


def main(args=None):
    rclpy.init(args=args)
    node = MTMTeleopNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C, or SIGTERM from a launch/timeout wrapper. Both are normal
        # exits; without catching the latter every kill prints a traceback.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():          # SIGTERM already tore the context down
            rclpy.shutdown()


if __name__ == '__main__':
    main()
