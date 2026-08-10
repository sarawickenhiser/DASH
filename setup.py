import os
from glob import glob

from setuptools import find_packages, setup
from glob import glob

package_name = 'hand_for_humanoid_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')
        ),
        (
            'share/' + package_name + '/tool_configs', glob('tool_configs/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arclab',
    maintainer_email='arclab@example.com',
    description='Hand for Humanoid Robot ROS2 package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'auto_zero_node = hand_for_humanoid_robot.auto_zero_node:main',
            'dynamixel_controller_node = hand_for_humanoid_robot.dynamixel_controller_node:main',
            'keyboard_control_node = hand_for_humanoid_robot.keyboard_control_node:main',
            'motor_current_monitor_node = hand_for_humanoid_robot.motor_current_monitor_node:main',
            'raw_joint_keyboard_node = hand_for_humanoid_robot.raw_joint_keyboard_node:main',
            'tool_keyboard_control_node = hand_for_humanoid_robot.tool_keyboard_control_node:main',
            'hand_for_humanoid_robot_ui_node = hand_for_humanoid_robot.hand_for_humanoid_robot_ui_node:main',
	        'degree_command_node = hand_for_humanoid_robot.degree_command_node:main',
            'send_joint_command = hand_for_humanoid_robot.send_joint_command:main',
            'rfid_scan_node = hand_for_humanoid_robot.rfid_scan_node:main',
            'current_monitor_node = hand_for_humanoid_robot.current_monitor_node:main',
            'tool_loader_node = hand_for_humanoid_robot.tool_loader_node:main',
            ]
        ,
    },
)
