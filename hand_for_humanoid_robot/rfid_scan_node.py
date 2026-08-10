#!/usr/bin/env python3
import time
import threading
import serial
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from std_srvs.srv import Trigger

PORT = "/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_F42208A55157375037202020FF122F34-if00"
BAUD = 57600
SCAN_TIMEOUT = 1.5

class RFIDScanNode(Node):
    def __init__(self):
        super().__init__('rfid_scan_node')
        self.cb_group = ReentrantCallbackGroup()
        self.uid_pub = self.create_publisher(String, '/rfid/uid', 10)
        self.scan_srv = self.create_service(
            Trigger, '/rfid/scan', self.scan_callback,
            callback_group=self.cb_group)
        self.release_client = self.create_client(
            Trigger, '/h4hr/release_port',
            callback_group=self.cb_group)
        self.reclaim_client = self.create_client(
            Trigger, '/h4hr/reclaim_port',
            callback_group=self.cb_group)
        self.scan_lock = threading.Lock()
        self.last_uid = 'None'

        # Scan thread — runs every 2 seconds
        self.auto_scan_thread = threading.Thread(target=self.auto_scan_loop, daemon=True)
        self.auto_scan_thread.start()

        # Print thread — prints every 1 second
        self.print_thread = threading.Thread(target=self.print_loop, daemon=True)
        self.print_thread.start()

        self.get_logger().info('Ready. Scanning every 2s, printing every 1s.')

    def call_service(self, client, name):
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f'{name} service not available')
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() and future.result().success:
            return True
        self.get_logger().error(f'{name} failed')
        return False

    def do_scan(self):
        if not self.call_service(self.release_client, 'release_port'):
            return None

        time.sleep(0.1)
        uid = None

        try:
            ser = serial.Serial(PORT, BAUD, timeout=2.0, dsrdtr=False, rtscts=False)
            ser.dtr = False
            ser.reset_input_buffer()
            ser.write(b'SCAN\n')

            deadline = time.time() + SCAN_TIMEOUT
            while time.time() < deadline:
                line = ser.readline().decode(errors='ignore').strip()
                if line.startswith('UID:'):
                    uid_val = line.replace('UID:', '').strip()
                    if uid_val != 'NONE':
                        uid = uid_val
                    break

            ser.close()

        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}')

        self.call_service(self.reclaim_client, 'reclaim_port')
        return uid

    def auto_scan_loop(self):
        while rclpy.ok():
            with self.scan_lock:
                uid = self.do_scan()
                if uid:
                    self.last_uid = uid
                    msg = String()
                    msg.data = uid
                    self.uid_pub.publish(msg)
                else:
                    self.last_uid = 'None'
            time.sleep(2.0)

    def print_loop(self):
        while rclpy.ok():
            self.get_logger().info(f'RFID: {self.last_uid}')
            time.sleep(1.0)

    def scan_callback(self, request, response):
        with self.scan_lock:
            uid = self.do_scan()
            if uid:
                self.last_uid = uid
                msg = String()
                msg.data = uid
                self.uid_pub.publish(msg)
                self.get_logger().info(f'Tag detected: {uid}')
                response.success = True
                response.message = uid
            else:
                self.get_logger().info('No tag detected.')
                response.success = False
                response.message = 'NONE'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RFIDScanNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()