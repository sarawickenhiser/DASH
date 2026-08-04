#!/usr/bin/env python3
import os
import math
import json
import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
import ament_index_python.packages

TOOL_CONFIG_DIR = os.path.join(
    ament_index_python.packages.get_package_share_directory('hand_for_humanoid_robot'),
    'tool_configs'
)
REGISTRY_FILE = os.path.join(TOOL_CONFIG_DIR, 'rfid_registry.json')

def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'(?<!:)//[^\n]*', '', text)
    return text

def load_json_with_comments(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    text = strip_comments(text)
    return json.loads(text)

class ToolLoaderNode(Node):
    def __init__(self):
        super().__init__('tool_loader_node')
        self.last_uid = None
        self.last_config = None
        self.last_tool_name = None

        self.tool_pub = self.create_publisher(String, '/h4hr/tool_loaded', 10)
        self.coupling_pub = self.create_publisher(Float64MultiArray, '/h4hr/update_coupling', 10)
        self.limits_pub = self.create_publisher(Float64MultiArray, '/h4hr/update_limits', 10)
        self.uid_sub = self.create_subscription(String, '/rfid/uid', self.uid_callback, 10)

        try:
            with open(REGISTRY_FILE, 'r') as f:
                self.registry = json.load(f)
            self.get_logger().info(f'Loaded registry with {len(self.registry)} tools.')
        except Exception as e:
            self.get_logger().error(f'Failed to load registry: {e}')
            self.registry = {}

        self.get_logger().info('Waiting for RFID scans...')

    def uid_callback(self, msg):
        uid = msg.data.strip()
        if uid == self.last_uid:
            return
        self.last_uid = uid
        self.get_logger().info(f'UID received: {uid}')

        if uid not in self.registry:
            self.get_logger().warn(f'UID {uid} not in registry — unknown tool.')
            return

        entry = self.registry[uid]
        tool_name = entry['name']
        config_file = os.path.join(TOOL_CONFIG_DIR, entry['config'])

        try:
            config = load_json_with_comments(config_file)
        except Exception as e:
            self.get_logger().error(f'Failed to load {config_file}: {e}')
            return

        self.last_config = config
        self.last_tool_name = tool_name

        self.publish_tool(config, tool_name)

        # Republish after 2 seconds to catch late subscribers
        self.create_timer(2.0, self.republish_tool)

    def republish_tool(self):
        if self.last_config is not None:
            self.publish_tool(self.last_config, self.last_tool_name)

    def publish_tool(self, config, tool_name):
        try:
            matrix = config['coupling']['ActuatorToJointPosition']
            flat = [v for row in matrix for v in row]
            coupling_msg = Float64MultiArray()
            coupling_msg.data = flat
            self.coupling_pub.publish(coupling_msg)
            self.get_logger().info(f'Published coupling matrix for {tool_name}')
        except Exception as e:
            self.get_logger().error(f'Failed to publish coupling: {e}')
            return

        try:
            joints = config['DH']['joints']
            jaw = config.get('jaw', {})
            limits = []
            for joint in joints:
                limits.extend([math.degrees(joint['qmin']), math.degrees(joint['qmax'])])
            limits.extend([
                math.degrees(jaw.get('qmin', -0.349066)),
                math.degrees(jaw.get('qmax', 1.39626))
            ])
            limits_msg = Float64MultiArray()
            limits_msg.data = limits
            self.limits_pub.publish(limits_msg)
            self.get_logger().info(f'Published limits for {tool_name}')
        except Exception as e:
            self.get_logger().error(f'Failed to publish limits: {e}')
            return

        msg = String()
        msg.data = tool_name
        self.tool_pub.publish(msg)
        self.get_logger().info(f'Tool loaded: {tool_name}')


def main(args=None):
    rclpy.init(args=args)
    node = ToolLoaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()