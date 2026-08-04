#!/usr/bin/env python3
"""
Quest2 controller teleoperation node.
Converts Quest2 controller pose to roll/pitch/yaw motor commands.

Topics:
  Sub: /q2r_right_hand_pose  (geometry_msgs/PoseStamped)
  Sub: /q2r_right_hand_inputs (quest2ros/OVR2ROSInputs)
  Pub: /h4hr/joint_command   (sensor_msgs/JointState)

Controls:
  Controller orientation → pitch/yaw
  Thumbstick horizontal   → roll (rate control: deflect to spin, matches the
                             tool's large +-255 deg continuous-rotation range)
  Index trigger           → grip
  A+B buttons (tap)       → set home: current pose becomes the zero reference
  Side grip trigger (hold) → clutch: freeze the tool and reposition your hand
                              freely; releasing resumes motion from wherever
                              your hand is, with no jump

Usage:
  ros2 run hand_for_humanoid_robot quest_teleop_node
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from quest2ros.msg import OVR2ROSInputs

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

# Superseded by swing_twist() + quat_to_rotvec() below. Kept for reference and
# for the rviz/debug path: a ZYX decomposition is singular at pitch = 90 deg,
# uncomfortably close to this tool's 79 deg pitch limit, and it couples roll
# into pitch/yaw -- both of which the swing-twist split avoids.
#
# def quat_to_euler(q):
#     """Convert quaternion [x, y, z, w] to roll, pitch, yaw (radians)."""
#     x, y, z, w = q
#     # Roll (x-axis)
#     sinr_cosp = 2 * (w * x + y * z)
#     cosr_cosp = 1 - 2 * (x * x + y * y)
#     roll = math.atan2(sinr_cosp, cosr_cosp)
#     # Pitch (y-axis)
#     sinp = 2 * (w * y - z * x)
#     sinp = max(-1.0, min(1.0, sinp))
#     pitch = math.asin(sinp)
#     # Yaw (z-axis)
#     siny_cosp = 2 * (w * z + x * y)
#     cosy_cosp = 1 - 2 * (y * y + z * z)
#     yaw = math.atan2(siny_cosp, cosy_cosp)
#     return roll, pitch, yaw

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

    twist is the rotation about `axis` (the tool's roll); swing is everything
    else, and its axis is perpendicular to `axis` by construction -- so swing
    carries exactly two degrees of freedom and maps onto pitch/yaw with no
    third angle leaking in. Unlike a ZYX decomposition this has no singularity
    anywhere near the tool's 79 deg pitch limit; it only degenerates when the
    swing reaches 180 deg, far outside any reachable bound.
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

# Tool roll axis in the reference-relative frame.
ROLL_AXIS = np.array([1.0, 0.0, 0.0])
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])

class QuestTeleopNode(Node):
    def __init__(self):
        super().__init__('quest_teleop_node')

        # Joint limits (degrees)
        self.declare_parameter('max_roll_deg',   255.0)
        self.declare_parameter('max_pitch_deg',   70.0)
        self.declare_parameter('max_yaw_deg',     90.0)
        self.declare_parameter('min_grip_deg',   -20.0)
        self.declare_parameter('max_grip_deg',    80.0)

        # Caps how fast each of roll/pitch/yaw may change (deg/sec),
        # regardless of how far/fast the raw controller pose jumps between
        # messages.
        self.declare_parameter('max_angular_speed_deg_per_sec', 180.0)

        # Scales the operator's real wrist rotation down before mapping it to
        # the tool (e.g. 0.3 means a 90 deg hand rotation becomes 27 deg of
        # tool rotation), so a comfortable full hand range doesn't immediately
        # saturate the tool's much smaller joint limits.
        self.declare_parameter('motion_scale', 0.3)
        self.motion_scale = float(self.get_parameter('motion_scale').value)

        # Side grip trigger threshold (0-1) above which the clutch engages.
        self.declare_parameter('clutch_trigger_threshold', 0.5)
        self.clutch_trigger_threshold = float(
            self.get_parameter('clutch_trigger_threshold').value
        )

        # Roll is driven by the thumbstick as a rate control (deg/sec at
        # full deflection) rather than tracked from hand orientation, so it
        # can't couple with pitch/yaw the way a 3-axis Euler decomposition
        # inherently does for a real 3-joint roll/pitch/yaw wrist.
        self.declare_parameter('roll_stick_speed_deg_per_sec', 60.0)
        self.roll_stick_speed = math.radians(
            self.get_parameter('roll_stick_speed_deg_per_sec').value
        )
        self.declare_parameter('roll_stick_deadband', 0.1)
        self.roll_stick_deadband = float(
            self.get_parameter('roll_stick_deadband').value
        )

        # 'quaternion': roll comes from the twist component of the hand's
        #   rotation, so all three axes track orientation directly. The
        #   swing-twist split keeps it from coupling into pitch/yaw, which is
        #   what made the Euler version unusable and motivated the thumbstick.
        # 'thumbstick': previous behaviour, kept as a fallback -- roll is rate
        #   controlled and hand roll is ignored entirely.
        self.declare_parameter('roll_source', 'quaternion')
        self.roll_source = str(self.get_parameter('roll_source').value)

        self.max_roll  = math.radians(self.get_parameter('max_roll_deg').value)
        self.max_pitch = math.radians(self.get_parameter('max_pitch_deg').value)
        self.max_yaw   = math.radians(self.get_parameter('max_yaw_deg').value)
        self.min_grip  = math.radians(self.get_parameter('min_grip_deg').value)
        self.max_grip  = math.radians(self.get_parameter('max_grip_deg').value)

        self.max_angular_speed = math.radians(
            self.get_parameter('max_angular_speed_deg_per_sec').value
        )

        # State
        self.ref_quat     = None   # reference quaternion (zero pose)
        # Rate-limited/clamped target, tracked as a single SO(3) rotation and
        # decomposed to joint angles only at publish time. Interpolating here
        # rather than per-axis means a diagonal move follows the geodesic and
        # respects max_angular_speed as a true angular rate, instead of each
        # axis independently stepping at that rate.
        self.current_quat = IDENTITY_QUAT.copy()
        # Roll is unwrapped across the +-180 deg quaternion branch cut so the
        # tool can reach its full +-255 deg continuous-rotation range; a
        # quaternion alone cannot distinguish +200 deg from -160 deg.
        self.unwrapped_roll = 0.0
        self.prev_roll_raw  = None
        self.current_roll  = 0.0
        self.current_pitch = 0.0
        self.current_yaw   = 0.0
        self.grip         = self.min_grip
        self.clutched      = False  # True while side grip trigger held: tool frozen
        self.pose_at_clutch_start = None  # hand pose when clutch was engaged
        self.ab_prev       = False  # previous A+B combined state, for edge detection
        self.home_reset_requested = False  # set by A+B tap, consumed in pose_callback
        self.last_pose_time = None  # for rate-limiting each axis's step
        self.roll_stick = 0.0  # latest thumbstick horizontal, -1..1

        # Publishers / Subscribers
        self.cmd_pub = self.create_publisher(JointState, '/h4hr/joint_command', 10)

        self.pose_sub = self.create_subscription(
            PoseStamped, '/q2r_right_hand_pose', self.pose_callback, 10)

        self.input_sub = self.create_subscription(
            OVR2ROSInputs, '/q2r_right_hand_inputs', self.input_callback, 10)

        self.get_logger().info('Quest teleop node ready.')
        self.get_logger().info('Tap A+B to set home (zero the reference pose).')
        self.get_logger().info('Hold the side grip trigger to clutch (freeze tool, reposition hand).')
        self.get_logger().info('Index trigger controls grip.')

    def clamp(self, val, lo, hi):
        return max(lo, min(hi, val))

    def step_value(self, current, target, max_step):
        error = target - current
        if abs(error) <= max_step:
            return target
        return current + max_step if error > 0 else current - max_step

    def input_callback(self, msg):
        """Handle button presses and trigger."""

        # A+B tap → request an explicit home/reference reset
        ab_pressed = bool(msg.button_upper and msg.button_lower)
        if ab_pressed and not self.ab_prev:
            self.home_reset_requested = True
            self.get_logger().info('Home reset requested.')
        self.ab_prev = ab_pressed

        # Side grip trigger (hold) → clutch
        clutched = float(msg.press_middle) > self.clutch_trigger_threshold
        if clutched != self.clutched:
            if clutched:
                self.get_logger().info('Clutch engaged — tool frozen, reposition freely.')
            else:
                self.get_logger().info('Clutch released — control resumed.')
            self.clutched = clutched

        # Index trigger → grip (works regardless of clutch state)
        trigger = float(msg.press_index)
        self.grip = self.min_grip + trigger * (self.max_grip - self.min_grip)

        # Thumbstick horizontal → roll rate
        self.roll_stick = float(msg.thumb_stick_horizontal)

    def pose_callback(self, msg):
        """Convert controller pose to joint command."""
        # quest2ros already publishes this pose in ROS (right-handed) convention
        # (confirmed empirically: a pure left/right controller turn shows up
        # cleanly as yaw and a pure up/down tilt shows up cleanly as pitch,
        # only when the raw quaternion is used unmodified).
        ros_quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        ros_quat = ros_quat / np.linalg.norm(ros_quat)

        # Bootstrap (never had a reference yet) or explicit A+B "set home":
        # current hand pose becomes the new zero, so the tool target snaps
        # to neutral (0,0,0). This is a deliberate recenter, unlike the
        # clutch below.
        if self.ref_quat is None or self.home_reset_requested:
            self.ref_quat = ros_quat.copy()
            self.current_quat = IDENTITY_QUAT.copy()
            self.unwrapped_roll = 0.0
            self.prev_roll_raw = None
            self.current_roll = 0.0
            self.current_pitch = 0.0
            self.current_yaw = 0.0
            self.last_pose_time = self.get_clock().now()
            self.home_reset_requested = False
            self.pose_at_clutch_start = None
            return

        # Frozen while the clutch is held: remember the hand pose at the
        # moment it engaged, publish nothing (tool holds its last command).
        if self.clutched:
            if self.pose_at_clutch_start is None:
                self.pose_at_clutch_start = ros_quat.copy()
            self.last_pose_time = self.get_clock().now()
            return

        # Just released the clutch: rebase the reference to discount exactly
        # the rotation the hand did *while frozen*, so the tool continues
        # from wherever it was instead of snapping back toward zero.
        if self.pose_at_clutch_start is not None:
            self.ref_quat = quat_multiply(
                quat_multiply(ros_quat, quat_inverse(self.pose_at_clutch_start)),
                self.ref_quat
            )
            self.ref_quat = self.ref_quat / np.linalg.norm(self.ref_quat)
            self.pose_at_clutch_start = None

        # Compute relative rotation from reference
        rel_quat = quat_multiply(quat_inverse(self.ref_quat), ros_quat)
        rel_quat = rel_quat / np.linalg.norm(rel_quat)

        # --- Previous per-axis Euler pipeline, superseded by the SO(3) one
        # --- below. Kept for reference / quick revert.
        #
        # # Extract pitch/yaw and treat them as independent scalar channels
        # # (roll is intentionally not derived from hand orientation -- see
        # # the thumbstick handling below).
        # _unused_roll, raw_pitch, raw_yaw = quat_to_euler(rel_quat)
        #
        # target_pitch = raw_pitch * self.motion_scale
        # target_yaw   = raw_yaw * self.motion_scale
        #
        # # Rate-limit each axis independently, and clamp to joint limits
        # # in-place (not just on the published value) so the internal
        # # tracker never "winds up" past the limit -- otherwise reversing
        # # direction after saturating would lag until the internal value
        # # ratchets back within range.
        # max_step = self.max_angular_speed * dt
        #
        # self.current_pitch = self.clamp(
        #     self.step_value(self.current_pitch, target_pitch, max_step),
        #     -self.max_pitch, self.max_pitch
        # )
        # self.current_yaw = self.clamp(
        #     self.step_value(self.current_yaw, target_yaw, max_step),
        #     -self.max_yaw, self.max_yaw
        # )
        #
        # # Roll: driven directly by the thumbstick as a rate control, with a
        # # deadband so a resting stick that isn't exactly zero doesn't drift.
        # stick = self.roll_stick
        # if abs(stick) < self.roll_stick_deadband:
        #     stick = 0.0
        # self.current_roll = self.clamp(
        #     self.current_roll + stick * self.roll_stick_speed * dt,
        #     -self.max_roll, self.max_roll
        # )

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
        # rotation rather than one per axis, so a diagonal move no longer
        # exceeds max_angular_speed or bows off the shortest path.
        max_step = self.max_angular_speed * dt
        angle = quat_angle_between(self.current_quat, q_target)
        if angle <= max_step or angle < 1e-9:
            self.current_quat = q_target
        else:
            self.current_quat = quat_slerp(
                self.current_quat, q_target, max_step / angle
            )

        # Decompose once, at the last point before the joint command.
        swing, twist = swing_twist(self.current_quat, ROLL_AXIS)
        rotvec = quat_to_rotvec(swing)
        pitch, yaw = float(rotvec[1]), float(rotvec[2])

        if self.roll_source == 'thumbstick':
            # Fallback: ignore hand roll entirely, integrate the stick as a rate.
            stick = self.roll_stick
            if abs(stick) < self.roll_stick_deadband:
                stick = 0.0
            self.unwrapped_roll += stick * self.roll_stick_speed * dt
        else:
            # Track hand roll from the twist component, unwrapping across the
            # +-180 deg branch cut so the tool's full +-255 deg range stays
            # reachable -- a quaternion cannot tell +200 deg from -160 deg.
            roll_raw = twist_angle(twist, ROLL_AXIS)
            if self.prev_roll_raw is None:
                self.unwrapped_roll = roll_raw
            else:
                self.unwrapped_roll += wrap_to_pi(roll_raw - self.prev_roll_raw)
            self.prev_roll_raw = roll_raw

        # Clamp in joint space -- the limits are mechanical, so this is the one
        # place they legitimately apply.
        roll  = self.clamp(self.unwrapped_roll, -self.max_roll, self.max_roll)
        pitch = self.clamp(pitch, -self.max_pitch, self.max_pitch)
        yaw   = self.clamp(yaw, -self.max_yaw, self.max_yaw)

        # Fold the clamped result back into the tracked rotation. Without this
        # the quaternion keeps integrating past what the tool can reach, and
        # reversing direction lags until it unwinds -- the same wind-up the
        # per-axis version guarded against, just harder to see in SO(3).
        self.unwrapped_roll = roll
        swing_c = quat_from_rotvec(np.array([0.0, pitch, yaw]))
        twist_c = quat_from_rotvec(ROLL_AXIS * roll)
        self.current_quat = quat_normalize(quat_multiply(swing_c, twist_c))
        if self.roll_source != 'thumbstick':
            self.prev_roll_raw = wrap_to_pi(roll)

        self.current_roll  = roll
        self.current_pitch = pitch
        self.current_yaw   = yaw

        roll  = self.current_roll
        pitch = self.current_pitch
        yaw   = self.current_yaw
        grip  = self.clamp(self.grip, self.min_grip, self.max_grip)

        # Publish joint command
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = ['roll_joint', 'pitch_joint', 'yaw_joint', 'grip_joint']
        cmd.position = [roll, pitch, yaw, grip]
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'R={math.degrees(roll):.1f}° '
            f'P={math.degrees(pitch):.1f}° '
            f'Y={math.degrees(yaw):.1f}° '
            f'G={math.degrees(grip):.1f}°',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    node = QuestTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()