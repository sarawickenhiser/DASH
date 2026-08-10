#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class JointCommandBridgeNode(Node):
    def __init__(self):
        super().__init__('joint_command_bridge_node')

        self.declare_parameter('max_roll_deg', 259.0)
        self.declare_parameter('max_pitch_deg', 79.0)
        self.declare_parameter('max_yaw_deg', 79.0)
        self.declare_parameter('min_grip_deg', -15.0)
        self.declare_parameter('max_grip_deg', 38.0)

        self.max_roll_deg = float(self.get_parameter('max_roll_deg').value)
        self.max_pitch_deg = float(self.get_parameter('max_pitch_deg').value)
        self.max_yaw_deg = float(self.get_parameter('max_yaw_deg').value)
        self.min_grip_deg = float(self.get_parameter('min_grip_deg').value)
        self.max_grip_deg = float(self.get_parameter('max_grip_deg').value)

        self.sub = self.create_subscription(
            JointState,
            '/target_tool_joints',
            self.target_joint_callback,
            10
        )

        self.pub = self.create_publisher(
            JointState,
            '/h4hr/joint_command',
            10
        )

        self.limits_sub = self.create_subscription(
            Float64MultiArray,
            '/h4hr/update_limits',
            self.update_limits_callback,
            10)

        self.get_logger().info('joint_command_bridge_node started.')
        self.get_logger().info('Listening to /target_tool_joints')
        self.get_logger().info('Publishing to /h4hr/joint_command')

    def update_limits_callback(self, msg):
        d = msg.data
        self.min_grip_deg = d[6]
        self.max_grip_deg = d[7]
        self.get_logger().info(
            f'Grip range updated: open={self.min_grip_deg:.1f} deg, close={self.max_grip_deg:.1f} deg')

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def target_joint_callback(self, msg):
        if len(msg.position) < 4:
            self.get_logger().warn(
                'Received /target_tool_joints with fewer than 4 positions.'
            )
            return

        roll_deg  = math.degrees(msg.position[0])
        pitch_deg = math.degrees(msg.position[1])
        yaw_deg   = math.degrees(msg.position[2])
        grip_deg  = math.degrees(msg.position[3])

        roll_deg  = self.clamp(roll_deg,  -self.max_roll_deg,  self.max_roll_deg)
        pitch_deg = self.clamp(pitch_deg, -self.max_pitch_deg, self.max_pitch_deg)
        yaw_deg   = self.clamp(yaw_deg,   -self.max_yaw_deg,   self.max_yaw_deg)
        grip_deg  = self.clamp(grip_deg,   self.min_grip_deg,  self.max_grip_deg)

        command_msg = JointState()
        command_msg.header.stamp = self.get_clock().now().to_msg()
        command_msg.name = ['roll_joint', 'pitch_joint', 'yaw_joint', 'grip_joint']
        command_msg.position = [
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
            math.radians(grip_deg)
        ]

        self.pub.publish(command_msg)

        self.get_logger().info(
            f'Published /h4hr/joint_command deg: '
            f'R={roll_deg:.1f}, '
            f'P={pitch_deg:.1f}, '
            f'Y={yaw_deg:.1f}, '
            f'G={grip_deg:.1f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()