#!/usr/bin/env python3
"""
Interactive joint command sender.
Prompts for roll pitch yaw grip in degrees, sends to /h4hr/joint_command.
Type 'q' to quit.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointCommandSender(Node):
    def __init__(self):
        super().__init__('joint_command_sender')
        self.pub = self.create_publisher(JointState, '/h4hr/joint_command', 10)

    def send(self, roll, pitch, yaw, grip=0.0):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['roll_joint', 'pitch_joint', 'yaw_joint', 'grip_joint']
        msg.position = [
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw),
            math.radians(grip),
        ]
        self.pub.publish(msg)
        self.get_logger().info(
            f'Sent → roll={roll}°  pitch={pitch}°  yaw={yaw}°  grip={grip}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandSender()

    # Let publisher connect
    import time
    time.sleep(0.5)

    print('\nJoint command sender ready.')
    print('Enter: roll pitch yaw [grip]  (degrees)')
    print('Example: 20 15 8   or   90 0 0 10')
    print('Type q to quit.\n')

    while rclpy.ok():
        try:
            raw = input('Enter desired values: ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if raw.lower() == 'q':
            break

        parts = raw.split()

        if len(parts) < 3:
            print('  Need at least 3 values: roll pitch yaw')
            continue

        try:
            roll  = float(parts[0])
            pitch = float(parts[1])
            yaw   = float(parts[2])
            grip  = float(parts[3]) if len(parts) >= 4 else 0.0
        except ValueError:
            print('  Invalid input — enter numbers only.')
            continue

        node.send(roll, pitch, yaw, grip)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()