"""
===============================================================================
WAREHOUSE CONTROLLER - BALANCED TUNING
===============================================================================
Fixes:
1. RELAXED TOLERANCE: Reduced SEARCH_TOLERANCE from 8 -> 4.
   (Prevents deleting people who are standing close to walls).
2. RADIUS: Increased back to 1.5m to be more forgiving.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math
import numpy as np
import time
import os
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

STATE_FILE = '/tmp/warehouse_controller_state.json'

# --- TARGETS ---
PERSON1_EXPECTED = {'x': 1.00, 'y': -1.00, 'name': 'Person 1'}
PERSON1_SAFE_SPOT = {'x': -1.00, 'y': 1.00, 'yaw': -0.785} # Facing SE

PERSON2_EXPECTED = {'x': -12.00, 'y': 15.00, 'name': 'Person 2'}
PERSON2_SAFE_SPOT = {'x': -13.00, 'y': 15.00, 'yaw': 0.0} # Facing East

ROBOT_START = {'x': 2.12, 'y': -21.3, 'yaw': 1.57}

# --- BALANCED DETECTION SETTINGS ---
DETECTION_RADIUS = 1.5      # Increased to 1.5m (was 1.0m)
MAX_OBJECT_WIDTH = 1.2      # Increased to 1.2m (was 1.0m)
MAP_THRESHOLD = 50          
SEARCH_TOLERANCE = 4        # Reduced to 4 pixels (~12cm) to avoid deleting people near walls

class WarehouseController(Node):
    def __init__(self):
        super().__init__('warehouse_controller')
        
        self.map_data = None
        self.map_info = None
        self.latest_scan = None
        self.person1_found = False
        self.person2_found = False
        self.person1_new_location = None
        self.person2_new_location = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.navigator = BasicNavigator()
        self.handle_reset_on_restart()

        self.startup_timer = self.create_timer(2.0, self.startup_check)
        self.get_logger().info('Initialized BALANCED TUNING. Waiting for Map...')

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.get_logger().info('Map Received!')

    def scan_callback(self, msg):
        self.latest_scan = msg

    # =========================================================================
    # HELPERS
    # =========================================================================

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (x, y, yaw)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def create_pose(self, x, y, yaw):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.orientation.z = math.sin(yaw / 2.0)
        p.pose.orientation.w = math.cos(yaw / 2.0)
        return p

    def navigate_to(self, x, y, yaw):
        self.get_logger().info(f'Navigating to Safe Spot ({x:.2f}, {y:.2f})...')
        goal = self.create_pose(x, y, yaw)
        self.navigator.goToPose(goal)
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.navigator.getResult() == TaskResult.SUCCEEDED

    def handle_reset_on_restart(self):
        if os.path.exists(STATE_FILE):
            self.get_logger().info('Resetting robot pose...')
            init_pose = self.create_pose(ROBOT_START['x'], ROBOT_START['y'], ROBOT_START['yaw'])
            self.navigator.setInitialPose(init_pose)
            self.navigator.clearAllCostmaps()
            time.sleep(3.0) 
        with open(STATE_FILE, 'w') as f:
            json.dump({'status': 'running'}, f)

    # =========================================================================
    # ROBUST DETECTION LOGIC
    # =========================================================================

    def is_static_object(self, wx, wy):
        if self.map_data is None: return False
        
        mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)

        if not (0 <= mx < self.map_info.width and 0 <= my < self.map_info.height):
            return False

        r = SEARCH_TOLERANCE
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                check_x, check_y = mx + dx, my + dy
                if 0 <= check_x < self.map_info.width and 0 <= check_y < self.map_info.height:
                        if self.map_data[check_y, check_x] > MAP_THRESHOLD:
                            return True
        return False

    def detect_person(self, expected_x, expected_y, name):
        self.get_logger().info(f'Scanning for {name} at ({expected_x}, {expected_y})...')
        
        pose = self.get_robot_pose()
        if not pose or not self.latest_scan:
            return False, None

        rx, ry, ryaw = pose
        dynamic_points = []
        
        for i, r in enumerate(self.latest_scan.ranges):
            if r < 0.5 or r > 8.0 or not math.isfinite(r): continue

            angle = ryaw + self.latest_scan.angle_min + (i * self.latest_scan.angle_increment)
            wx = rx + r * math.cos(angle)
            wy = ry + r * math.sin(angle)

            if not self.is_static_object(wx, wy):
                dynamic_points.append((wx, wy))

        nearby_points = []
        for wx, wy in dynamic_points:
            dist = math.sqrt((wx - expected_x)**2 + (wy - expected_y)**2)
            if dist < DETECTION_RADIUS:
                nearby_points.append((wx, wy))

        if len(nearby_points) > 3:
            min_x = min(p[0] for p in nearby_points)
            max_x = max(p[0] for p in nearby_points)
            min_y = min(p[1] for p in nearby_points)
            max_y = max(p[1] for p in nearby_points)
            width = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)

            if width > MAX_OBJECT_WIDTH:
                self.get_logger().info(f'Ignored object: Too wide ({width:.2f}m)')
                return False, None
            else:
                cx = sum(p[0] for p in nearby_points) / len(nearby_points)
                cy = sum(p[1] for p in nearby_points) / len(nearby_points)
                self.get_logger().info(f'FOUND {name} at ({cx:.2f}, {cy:.2f}) (Width: {width:.2f}m)')
                return True, (cx, cy)
        else:
            self.get_logger().info(f'{name} NOT FOUND (Area is empty or matches wall)')
            return False, None

    # =========================================================================
    # MISSION
    # =========================================================================

    def startup_check(self):
        if self.map_data is not None and self.get_robot_pose() is not None:
            self.startup_timer.cancel()
            self.navigator.waitUntilNav2Active()
            self.run_mission()
        else:
            self.get_logger().info('Waiting for Map/TF...')

    def run_mission(self):
        self.get_logger().info('--- STARTING MISSION ---')

        # 1. Person 1
        if self.navigate_to(PERSON1_SAFE_SPOT['x'], PERSON1_SAFE_SPOT['y'], PERSON1_SAFE_SPOT['yaw']):
            time.sleep(1.0)
            found, loc = self.detect_person(PERSON1_EXPECTED['x'], PERSON1_EXPECTED['y'], 'Person 1')
            self.person1_found = found
            if not found and loc: self.person1_new_location = loc

        # 2. Person 2
        if self.navigate_to(PERSON2_SAFE_SPOT['x'], PERSON2_SAFE_SPOT['y'], PERSON2_SAFE_SPOT['yaw']):
            time.sleep(1.0)
            found, loc = self.detect_person(PERSON2_EXPECTED['x'], PERSON2_EXPECTED['y'], 'Person 2')
            self.person2_found = found
            if not found and loc: self.person2_new_location = loc

        # 3. Search
        if not self.person1_found or not self.person2_found:
            self.search_warehouse()

        self.get_logger().info('Mission Complete. Returning Home.')
        self.navigate_to(ROBOT_START['x'], ROBOT_START['y'], ROBOT_START['yaw'])
        if os.path.exists(STATE_FILE): os.remove(STATE_FILE)

    def search_warehouse(self):
        self.get_logger().info('Searching Warehouse...')
        waypoints = [
            (0.0, -10.0), (0.0, 0.0), (0.0, 10.0), (-5.0, 10.0), (-5.0, 0.0)
        ]
        for wx, wy in waypoints:
            self.navigate_to(wx, wy, 0.0)

def main(args=None):
    rclpy.init(args=args)
    controller = WarehouseController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
