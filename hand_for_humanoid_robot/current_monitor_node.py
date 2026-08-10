#!/usr/bin/env python3
"""
Monitors motor currents from /h4hr/motor_currents topic.
Usage: ros2 run hand_for_humanoid_robot current_monitor_node
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CurrentMonitorNode(Node):
    def __init__(self):
        super().__init__('current_monitor_node')
        self.create_subscription(
            String,
            '/h4hr/motor_currents',
            self.callback,
            10)
        self.get_logger().info('Monitoring motor currents...')

    def callback(self, msg):
        print(f'\r{msg.data}    ', end='', flush=True)

def main(args=None):
    rclpy.init(args=args)
    node = CurrentMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print()

if __name__ == '__main__':
    main()