#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math
import numpy as np

class WarehouseController(Node):
    def __init__(self):
        super().__init__('warehouse_controller')
        
        # Known initial positions of the two humans from map
        self.person1_expected = {'x': 1.00, 'y': -1.00, 'name': 'Person 1'}
        self.person2_expected = {'x': -12.00, 'y': 15.00, 'name': 'Person 2'}
        
        # Robot starting position
        self.robot_start = {'x': 2.12, 'y': -21.3, 'yaw': 1.57}
        
        # State tracking
        self.person1_found = None
        self.person2_found = None
        self.person1_new_location = None
        self.person2_new_location = None
        
        # Scanner data
        self.latest_scan = None
        self.current_pose = None
        
        # Subscribe to laser scan
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Subscribe to robot pose
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        
        # Initialize navigator
        self.navigator = BasicNavigator()
        
        self.get_logger().info('Warehouse Controller Started!')
        self.get_logger().info(f'Expected Person 1 at: ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        self.get_logger().info(f'Expected Person 2 at: ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        
        # Start the main task after a short delay
        self.create_timer(3.0, self.start_mission)
    
    def scan_callback(self, msg):
        """Store latest laser scan data"""
        self.latest_scan = msg
    
    def pose_callback(self, msg):
        """Store current robot pose"""
        self.current_pose = msg.pose.pose
    
    def start_mission(self):
        """Main mission logic"""
        self.get_logger().info('===== STARTING HUMAN DETECTION MISSION =====')
        
        # Wait for initial pose to be set
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('Navigation system ready!')
        
        # Check Person 1
        self.get_logger().info('\n--- Checking Person 1 ---')
        self.check_person(self.person1_expected, 'person1')
        
        # Check Person 2
        self.get_logger().info('\n--- Checking Person 2 ---')
        self.check_person(self.person2_expected, 'person2')
        
        # Analyze results
        self.report_findings()
        
        # If anyone moved, search for them
        if self.person1_found == False or self.person2_found == False:
            self.get_logger().info('\n--- Starting Search Pattern ---')
            self.search_warehouse()
        
        # Final report
        self.final_report()
    
    def check_person(self, expected_pos, person_id):
        """Navigate to expected position and check if person is there"""
        # Create goal pose
        goal_pose = self.create_pose_stamped(expected_pos['x'], expected_pos['y'], 0.0)
        
        self.get_logger().info(f'Navigating to expected position: ({expected_pos["x"]}, {expected_pos["y"]})')
        
        # Navigate to position
        self.navigator.goToPose(goal_pose)
        
        # Wait for navigation to complete
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        result = self.navigator.getResult()
        
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Arrived at expected position. Scanning...')
            
            # Wait a moment for scan to stabilize
            rclpy.spin_once(self, timeout_sec=1.0)
            
            # Check if person is detected
            person_detected = self.detect_person_nearby()
            
            if person_id == 'person1':
                self.person1_found = person_detected
            else:
                self.person2_found = person_detected
            
            if person_detected:
                self.get_logger().info(f'✓ {expected_pos["name"]} FOUND at expected location!')
            else:
                self.get_logger().warn(f'✗ {expected_pos["name"]} NOT FOUND - Person has moved!')
        else:
            self.get_logger().error(f'Failed to reach position: {result}')
    
    def detect_person_nearby(self):
        """
        Detect if a person is nearby using laser scan data.
        Looks for obstacles at typical human distance (0.5-2.0 meters)
        """
        if self.latest_scan is None:
            self.get_logger().warn('No scan data available')
            return False
        
        ranges = np.array(self.latest_scan.ranges)
        
        # Filter out inf and nan values
        valid_ranges = ranges[np.isfinite(ranges)]
        
        if len(valid_ranges) == 0:
            return False
        
        # Check for obstacles in human detection range (0.5 to 2.5 meters)
        human_range_detections = valid_ranges[(valid_ranges > 0.5) & (valid_ranges < 2.5)]
        
        # If we detect multiple points in human range, likely a person
        detection_threshold = 20
        
        if len(human_range_detections) > detection_threshold:
            min_dist = np.min(human_range_detections)
            self.get_logger().info(f'Detected obstacle at {min_dist:.2f}m - likely a person')
            return True
        else:
            self.get_logger().info(f'No significant obstacles detected (only {len(human_range_detections)} points)')
            return False
    
    def search_warehouse(self):
        """Search the warehouse systematically for moved humans"""
        self.get_logger().info('Searching warehouse for moved persons...')
        
        # Define search waypoints covering the warehouse
        search_points = [
            {'x': -5.0, 'y': 5.0},
            {'x': -10.0, 'y': 10.0},
            {'x': -15.0, 'y': 5.0},
            {'x': -10.0, 'y': 0.0},
            {'x': -5.0, 'y': -5.0},
            {'x': 0.0, 'y': -10.0},
            {'x': 5.0, 'y': -5.0},
        ]
        
        for i, point in enumerate(search_points):
            self.get_logger().info(f'Searching waypoint {i+1}/{len(search_points)}: ({point["x"]}, {point["y"]})')
            
            goal_pose = self.create_pose_stamped(point['x'], point['y'], 0.0)
            self.navigator.goToPose(goal_pose)
            
            while not self.navigator.isTaskComplete():
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if we found anyone
            if self.detect_person_nearby():
                self.get_logger().info(f'Found a person at search point ({point["x"]}, {point["y"]})!')
                
                # Record location for missing person
                if self.person1_found == False and self.person1_new_location is None:
                    self.person1_new_location = point
                    self.get_logger().info('Recorded as Person 1 new location')
                elif self.person2_found == False and self.person2_new_location is None:
                    self.person2_new_location = point
                    self.get_logger().info('Recorded as Person 2 new location')
    
    def report_findings(self):
        """Report initial check results"""
        self.get_logger().info('\n===== INITIAL CHECK RESULTS =====')
        
        if self.person1_found:
            self.get_logger().info(f'Person 1: FOUND at expected location ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        else:
            self.get_logger().warn(f'Person 1: MOVED from expected location ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        
        if self.person2_found:
            self.get_logger().info(f'Person 2: FOUND at expected location ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        else:
            self.get_logger().warn(f'Person 2: MOVED from expected location ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
    
    def final_report(self):
        """Print final mission report"""
        self.get_logger().info('\n\n╔════════════════════════════════════════════╗')
        self.get_logger().info('║     FINAL MISSION REPORT                   ║')
        self.get_logger().info('╚════════════════════════════════════════════╝')
        
        # Person 1 status
        if self.person1_found:
            self.get_logger().info(f'✓ Person 1: At original location ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        elif self.person1_new_location:
            self.get_logger().info(f'→ Person 1: MOVED to ({self.person1_new_location["x"]}, {self.person1_new_location["y"]})')
        else:
            self.get_logger().warn('? Person 1: MOVED but new location not found')
        
        # Person 2 status
        if self.person2_found:
            self.get_logger().info(f'✓ Person 2: At original location ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        elif self.person2_new_location:
            self.get_logger().info(f'→ Person 2: MOVED to ({self.person2_new_location["x"]}, {self.person2_new_location["y"]})')
        else:
            self.get_logger().warn('? Person 2: MOVED but new location not found')
        
        self.get_logger().info('════════════════════════════════════════════\n')
        self.get_logger().info('Mission complete!')
    
    def create_pose_stamped(self, x, y, yaw):
        """Create a PoseStamped message for navigation"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
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
