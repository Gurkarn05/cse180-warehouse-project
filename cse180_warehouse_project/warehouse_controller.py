#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math
import numpy as np

class WarehouseController(Node):
    def __init__(self):
        super().__init__('warehouse_controller')
        
        self.person1_expected = {'x': 1.00, 'y': -1.00, 'name': 'Person 1'}
        self.person2_expected = {'x': -12.00, 'y': 15.00, 'name': 'Person 2'}
        self.robot_start = {'x': 2.12, 'y': -21.3, 'yaw': 1.57}
        
        self.person1_found = None
        self.person2_found = None
        self.person1_new_location = None
        self.person2_new_location = None
        self.latest_scan = None
        self.current_pose = None
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        self.navigator = BasicNavigator()
        
        self.get_logger().info('Warehouse Controller Started!')
        self.get_logger().info(f'Expected Person 1 at: ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        self.get_logger().info(f'Expected Person 2 at: ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        
        self.create_timer(3.0, self.start_mission)
    
    def scan_callback(self, msg):
        self.latest_scan = msg
    
    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose
    
    def start_mission(self):
        self.get_logger().info('===== STARTING HUMAN DETECTION MISSION =====')
        
        # SET INITIAL POSE - THIS IS THE KEY FIX
        self.get_logger().info('Setting initial robot pose...')
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = self.robot_start['x']
        initial_pose.pose.position.y = self.robot_start['y']
        initial_pose.pose.position.z = 0.0
        initial_pose.pose.orientation.z = math.sin(self.robot_start['yaw'] / 2.0)
        initial_pose.pose.orientation.w = math.cos(self.robot_start['yaw'] / 2.0)
        
        self.navigator.setInitialPose(initial_pose)
        self.get_logger().info('Initial pose set!')
        
        # Wait for Nav2
        self.get_logger().info('Waiting for Nav2 to activate...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('Navigation system ready!')
        
        # Check Person 1
        self.get_logger().info('\n--- Checking Person 1 ---')
        self.check_person(self.person1_expected, 'person1')
        
        # Check Person 2
        self.get_logger().info('\n--- Checking Person 2 ---')
        self.check_person(self.person2_expected, 'person2')
        
        self.report_findings()
        
        if self.person1_found == False or self.person2_found == False:
            self.get_logger().info('\n--- Starting Search Pattern ---')
            self.search_warehouse()
        
        self.final_report()
    
    def check_person(self, expected_pos, person_id):
        goal_pose = self.create_pose_stamped(expected_pos['x'], expected_pos['y'], 0.0)
        self.get_logger().info(f'Navigating to expected position: ({expected_pos["x"]}, {expected_pos["y"]})')
        
        self.navigator.goToPose(goal_pose)
        
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        result = self.navigator.getResult()
        
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Arrived! Scanning...')
            rclpy.spin_once(self, timeout_sec=1.0)
            
            person_detected = self.detect_person_nearby()
            
            if person_id == 'person1':
                self.person1_found = person_detected
            else:
                self.person2_found = person_detected
            
            if person_detected:
                self.get_logger().info(f'✓ {expected_pos["name"]} FOUND!')
            else:
                self.get_logger().warn(f'✗ {expected_pos["name"]} MOVED!')
        else:
            self.get_logger().error(f'Navigation failed: {result}')
    
    def detect_person_nearby(self):
        if self.latest_scan is None:
            return False
        
        ranges = np.array(self.latest_scan.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]
        
        if len(valid_ranges) == 0:
            return False
        
        human_range_detections = valid_ranges[(valid_ranges > 0.5) & (valid_ranges < 2.5)]
        
        if len(human_range_detections) > 20:
            min_dist = np.min(human_range_detections)
            self.get_logger().info(f'Detected obstacle at {min_dist:.2f}m')
            return True
        return False
    
    def search_warehouse(self):
        self.get_logger().info('Searching warehouse...')
        
        search_points = [
            {'x': -5.0, 'y': 5.0}, {'x': -10.0, 'y': 10.0}, {'x': -15.0, 'y': 5.0},
            {'x': -10.0, 'y': 0.0}, {'x': -5.0, 'y': -5.0}, {'x': 0.0, 'y': -10.0},
        ]
        
        for i, point in enumerate(search_points):
            self.get_logger().info(f'Waypoint {i+1}/{len(search_points)}: ({point["x"]}, {point["y"]})')
            
            goal_pose = self.create_pose_stamped(point['x'], point['y'], 0.0)
            self.navigator.goToPose(goal_pose)
            
            while not self.navigator.isTaskComplete():
                rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.detect_person_nearby():
                self.get_logger().info(f'Found someone at ({point["x"]}, {point["y"]})!')
                
                if self.person1_found == False and self.person1_new_location is None:
                    self.person1_new_location = point
                elif self.person2_found == False and self.person2_new_location is None:
                    self.person2_new_location = point
    
    def report_findings(self):
        self.get_logger().info('\n===== INITIAL CHECK RESULTS =====')
        
        if self.person1_found:
            self.get_logger().info(f'Person 1: FOUND at ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        else:
            self.get_logger().warn(f'Person 1: MOVED from ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        
        if self.person2_found:
            self.get_logger().info(f'Person 2: FOUND at ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        else:
            self.get_logger().warn(f'Person 2: MOVED from ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
    
    def final_report(self):
        self.get_logger().info('\n\n╔════════════════════════════════╗')
        self.get_logger().info('║   FINAL MISSION REPORT         ║')
        self.get_logger().info('╚════════════════════════════════╝')
        
        if self.person1_found:
            self.get_logger().info(f'✓ Person 1: Original location ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        elif self.person1_new_location:
            self.get_logger().info(f'→ Person 1: MOVED to ({self.person1_new_location["x"]}, {self.person1_new_location["y"]})')
        else:
            self.get_logger().warn('? Person 1: MOVED but not found')
        
        if self.person2_found:
            self.get_logger().info(f'✓ Person 2: Original location ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        elif self.person2_new_location:
            self.get_logger().info(f'→ Person 2: MOVED to ({self.person2_new_location["x"]}, {self.person2_new_location["y"]})')
        else:
            self.get_logger().warn('? Person 2: MOVED but not found')
        
        self.get_logger().info('════════════════════════════════\n')
    
    def create_pose_stamped(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

def main(args=None):
    rclpy.init(args=args)
    controller = WarehouseController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.navigator.lifecycleShutdown()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
