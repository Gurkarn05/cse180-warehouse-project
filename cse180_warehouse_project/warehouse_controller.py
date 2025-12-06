#!/usr/bin/env python3
"""
Map-Aware Warehouse Human Detection Controller (FIXED VERSION)
===============================================================

FIXES APPLIED:
1. Added TF listener to get robot pose reliably (AMCL doesn't publish frequently)
2. Added fallback pose estimation from navigation
3. Improved scan timing to ensure fresh data
4. Added more robust pose waiting

Author: ROS2 Tutorial
For: ROS2 Jazzy + Nav2 + Gazebo
"""

# =============================================================================
# IMPORTS
# =============================================================================

import rclpy                                    # ROS2 Python library
from rclpy.node import Node                     # Base class for nodes
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Message types
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

# TF2 for getting robot pose from transform tree (MORE RELIABLE than AMCL topic!)
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# Nav2 Simple Commander
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# Standard Python libraries
import math
import numpy as np
import time


# =============================================================================
# MAIN CONTROLLER CLASS
# =============================================================================

class MapAwareWarehouseController(Node):
    """
    Complete warehouse human detection system with FIXED pose handling.
    """
    
    def __init__(self):
        """Initialize the node, subscribers, and parameters."""
        super().__init__('map_aware_warehouse_controller')
        
        # =====================================================================
        # CONFIGURATION PARAMETERS
        # =====================================================================
        
        # Expected person positions (from your Gazebo screenshots)
        self.person1_expected = {
            'x': 1.00,
            'y': -1.00,
            'name': 'Person 1 - Standing'
        }
        
        self.person2_expected = {
            'x': -12.00,
            'y': 15.00,
            'name': 'Person 2 - Walking'
        }
        
        # Robot's starting position
        self.robot_start = {
            'x': 2.12,
            'y': -21.3,
            'yaw': 1.57
        }
        
        # Detection parameters - TUNED for better detection
        self.detection_config = {
            # Navigation
            'approach_distance': 2.0,       # Reduced from 2.5 to get closer
            
            # Laser scan filtering
            'scan_range_min': 0.3,
            'scan_range_max': 5.0,          # Increased from 4.0 for wider detection
            
            # Human identification - WIDENED ranges
            'human_width_min': 0.10,        # Reduced from 0.15
            'human_width_max': 1.2,         # Increased from 0.9
            'min_cluster_points': 3,        # Reduced from 4
            'max_cluster_points': 100,      # Increased from 80
            'cluster_gap_threshold': 8,     # Increased from 5
            
            # Map comparison
            'map_obstacle_threshold': 50,
            'dynamic_object_buffer': 0.15,  # Increased buffer
            
            # Detection confidence
            'num_scans_to_check': 5,
            'detection_threshold': 2,       # Reduced from 3 for easier detection
        }
        
        # Search waypoints
        self.search_waypoints = [
            {'x': 0.0, 'y': 0.0},
            {'x': 5.0, 'y': 0.0},
            {'x': 5.0, 'y': 5.0},
            {'x': 0.0, 'y': 5.0},
            {'x': -5.0, 'y': 5.0},
            {'x': -5.0, 'y': 10.0},
            {'x': -10.0, 'y': 10.0},
            {'x': -10.0, 'y': 5.0},
            {'x': -10.0, 'y': 0.0},
            {'x': -5.0, 'y': 0.0},
            {'x': -5.0, 'y': -5.0},
            {'x': 0.0, 'y': -5.0},
            {'x': 0.0, 'y': -10.0},
            {'x': 5.0, 'y': -10.0},
        ]
        
        # =====================================================================
        # STATE VARIABLES
        # =====================================================================
        
        self.person1_found = None
        self.person2_found = None
        self.person1_new_location = None
        self.person2_new_location = None
        
        self.map_data = None
        self.map_info = None
        self.latest_scan = None
        self.robot_pose = None              # From AMCL topic (backup)
        self.last_known_pose = None         # Fallback pose storage
        
        self.mission_started = False
        self.map_received = False
        
        # =====================================================================
        # TF2 BUFFER AND LISTENER (THE FIX!)
        # =====================================================================
        # TF (Transform) system provides real-time robot pose
        # This is MORE RELIABLE than waiting for AMCL topic publications
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info('TF listener initialized for pose tracking')
        
        # =====================================================================
        # SUBSCRIBERS
        # =====================================================================
        
        # Map subscriber with transient local QoS
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )
        self.get_logger().info('Subscribed to /map topic')
        
        # Laser scan subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.get_logger().info('Subscribed to /scan topic')
        
        # AMCL pose subscriber (as backup)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )
        self.get_logger().info('Subscribed to /amcl_pose topic')
        
        # =====================================================================
        # NAVIGATOR
        # =====================================================================
        
        self.navigator = BasicNavigator()
        
        # =====================================================================
        # STARTUP
        # =====================================================================
        
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  MAP-AWARE WAREHOUSE CONTROLLER (FIXED)')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Person 1 expected: ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        self.get_logger().info(f'  Person 2 expected: ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        self.get_logger().info('=' * 60)
        self.get_logger().info('')
        
        self.create_timer(5.0, self.start_mission_once)
    
    
    # =========================================================================
    # CALLBACK FUNCTIONS
    # =========================================================================
    
    def map_callback(self, msg):
        """Store map data when received."""
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        
        if not self.map_received:
            self.map_received = True
            self.get_logger().info(
                f'✓ Map received: {msg.info.width} x {msg.info.height} pixels, '
                f'resolution: {msg.info.resolution:.3f} m/pixel'
            )
            self.get_logger().info(
                f'  Map origin: ({msg.info.origin.position.x:.2f}, '
                f'{msg.info.origin.position.y:.2f})'
            )
    
    
    def scan_callback(self, msg):
        """Store latest laser scan."""
        self.latest_scan = msg
    
    
    def pose_callback(self, msg):
        """Store pose from AMCL (used as backup)."""
        self.robot_pose = msg.pose.pose
        self.last_known_pose = msg.pose.pose  # Keep as fallback
        self.get_logger().debug(f'AMCL pose received: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})')
    
    
    # =========================================================================
    # POSE RETRIEVAL (THE KEY FIX!)
    # =========================================================================
    
    def get_robot_pose_from_tf(self):
        """
        Get robot pose from TF transform tree.
        
        This is MORE RELIABLE than the AMCL topic because:
        - TF is updated continuously
        - AMCL only publishes when pose changes significantly
        
        Returns:
            tuple: (x, y, yaw) or None if not available
        """
        try:
            # Look up transform from 'map' frame to 'base_link' (robot) frame
            # This tells us where the robot is in the map
            transform = self.tf_buffer.lookup_transform(
                'map',           # Target frame (where we want the pose)
                'base_link',     # Source frame (the robot)
                rclpy.time.Time(),  # Get latest available transform
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            
            # Extract position
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Extract yaw from quaternion
            q = transform.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return (x, y, yaw)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None
    
    
    def get_current_pose(self):
        """
        Get current robot pose using multiple fallback methods.
        
        Priority:
        1. TF transform (most reliable, updated continuously)
        2. AMCL topic (only updated on significant pose change)
        3. Last known pose (stale but better than nothing)
        4. Starting position (last resort)
        
        Returns:
            tuple: (x, y, yaw)
        """
        # Method 1: Try TF (best option)
        tf_pose = self.get_robot_pose_from_tf()
        if tf_pose is not None:
            return tf_pose
        
        # Method 2: Use AMCL pose from topic
        if self.robot_pose is not None:
            q = self.robot_pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (self.robot_pose.position.x, self.robot_pose.position.y, yaw)
        
        # Method 3: Use last known pose
        if self.last_known_pose is not None:
            q = self.last_known_pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (self.last_known_pose.position.x, self.last_known_pose.position.y, yaw)
        
        # Method 4: Fallback to starting position
        self.get_logger().warn('Using starting position as fallback')
        return (self.robot_start['x'], self.robot_start['y'], self.robot_start['yaw'])
    
    
    def wait_for_valid_pose(self, timeout=5.0):
        """
        Wait until we have a valid robot pose.
        
        Parameters:
            timeout (float): Maximum time to wait in seconds
            
        Returns:
            bool: True if pose obtained, False if timeout
        """
        self.get_logger().info('  Waiting for valid robot pose...')
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Process callbacks to get fresh data
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Try to get pose
            tf_pose = self.get_robot_pose_from_tf()
            if tf_pose is not None:
                self.get_logger().info(f'  ✓ Got pose from TF: ({tf_pose[0]:.2f}, {tf_pose[1]:.2f})')
                return True
            
            if self.robot_pose is not None:
                self.get_logger().info(f'  ✓ Got pose from AMCL: ({self.robot_pose.position.x:.2f}, {self.robot_pose.position.y:.2f})')
                return True
        
        self.get_logger().warn('  ⚠ Timeout waiting for pose - using fallback')
        return False
    
    
    # =========================================================================
    # COORDINATE TRANSFORMATION FUNCTIONS
    # =========================================================================
    
    def world_to_map(self, world_x, world_y):
        """Convert world coordinates (meters) to map pixel coordinates."""
        if self.map_info is None:
            return None
        
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        resolution = self.map_info.resolution
        
        map_x = int((world_x - origin_x) / resolution)
        map_y = int((world_y - origin_y) / resolution)
        
        if 0 <= map_x < self.map_info.width and 0 <= map_y < self.map_info.height:
            return (map_x, map_y)
        else:
            return None
    
    
    def map_to_world(self, map_x, map_y):
        """Convert map pixel coordinates to world coordinates (meters)."""
        if self.map_info is None:
            return None
        
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        resolution = self.map_info.resolution
        
        world_x = origin_x + (map_x + 0.5) * resolution
        world_y = origin_y + (map_y + 0.5) * resolution
        
        return (world_x, world_y)
    
    
    # =========================================================================
    # MAP QUERY FUNCTIONS
    # =========================================================================
    
    def is_occupied_in_map(self, world_x, world_y):
        """Check if a world position is marked as OCCUPIED in the static map."""
        if self.map_data is None:
            return False
        
        map_coords = self.world_to_map(world_x, world_y)
        if map_coords is None:
            return False
        
        map_x, map_y = map_coords
        occupancy_value = self.map_data[map_y, map_x]
        
        threshold = self.detection_config['map_obstacle_threshold']
        return occupancy_value > threshold
    
    
    def is_near_static_obstacle(self, world_x, world_y):
        """Check if a position is near a static obstacle in the map."""
        if self.map_data is None:
            return False
        
        buffer = self.detection_config['dynamic_object_buffer']
        
        for dx in [-buffer, 0, buffer]:
            for dy in [-buffer, 0, buffer]:
                if self.is_occupied_in_map(world_x + dx, world_y + dy):
                    return True
        
        return False
    
    
    # =========================================================================
    # HUMAN DETECTION FUNCTIONS (FIXED!)
    # =========================================================================
    
    def detect_humans_with_map(self):
        """
        Main detection function: Find humans by comparing scan to map.
        
        FIXED: Now uses get_current_pose() which has multiple fallback methods.
        """
        if self.latest_scan is None:
            self.get_logger().warn('No laser scan data available')
            return []
        
        if self.map_data is None:
            self.get_logger().warn('No map data available - using basic detection')
            return self.detect_humans_basic()
        
        # GET POSE USING NEW RELIABLE METHOD
        pose = self.get_current_pose()
        robot_x, robot_y, robot_yaw = pose
        
        self.get_logger().info(f'  Robot pose: ({robot_x:.2f}, {robot_y:.2f}, yaw={math.degrees(robot_yaw):.1f}°)')
        
        scan = self.latest_scan
        ranges = np.array(scan.ranges)
        
        range_min = self.detection_config['scan_range_min']
        range_max = self.detection_config['scan_range_max']
        gap_threshold = self.detection_config['cluster_gap_threshold']
        min_points = self.detection_config['min_cluster_points']
        
        # =====================================================================
        # STEP 1: Find dynamic points (in scan but NOT in static map)
        # =====================================================================
        
        dynamic_points = []
        
        for i, distance in enumerate(ranges):
            if not np.isfinite(distance):
                continue
            if distance < range_min or distance > range_max:
                continue
            
            laser_angle = scan.angle_min + i * scan.angle_increment
            world_angle = robot_yaw + laser_angle
            
            hit_x = robot_x + distance * math.cos(world_angle)
            hit_y = robot_y + distance * math.sin(world_angle)
            
            is_static = self.is_near_static_obstacle(hit_x, hit_y)
            
            if not is_static:
                dynamic_points.append({
                    'x': hit_x,
                    'y': hit_y,
                    'distance': distance,
                    'angle': laser_angle,
                    'index': i
                })
        
        self.get_logger().info(f'  Found {len(dynamic_points)} dynamic points (not in map)')
        
        # =====================================================================
        # STEP 2: Cluster dynamic points into objects
        # =====================================================================
        
        if len(dynamic_points) < min_points:
            return []
        
        dynamic_points.sort(key=lambda p: p['index'])
        
        clusters = []
        current_cluster = [dynamic_points[0]]
        
        for i in range(1, len(dynamic_points)):
            prev_idx = dynamic_points[i - 1]['index']
            curr_idx = dynamic_points[i]['index']
            
            if curr_idx - prev_idx <= gap_threshold:
                current_cluster.append(dynamic_points[i])
            else:
                if len(current_cluster) >= min_points:
                    clusters.append(current_cluster)
                current_cluster = [dynamic_points[i]]
        
        if len(current_cluster) >= min_points:
            clusters.append(current_cluster)
        
        self.get_logger().info(f'  Formed {len(clusters)} clusters')
        
        # =====================================================================
        # STEP 3: Analyze clusters to identify human-shaped objects
        # =====================================================================
        
        humans = []
        
        for cluster in clusters:
            human = self.analyze_cluster_for_human(cluster)
            if human:
                humans.append(human)
        
        return humans
    
    
    def analyze_cluster_for_human(self, cluster):
        """Analyze a cluster of points to determine if it's human-shaped."""
        min_points = self.detection_config['min_cluster_points']
        max_points = self.detection_config['max_cluster_points']
        min_width = self.detection_config['human_width_min']
        max_width = self.detection_config['human_width_max']
        
        if len(cluster) < min_points:
            return None
        if len(cluster) > max_points:
            self.get_logger().debug(f'  Cluster too large: {len(cluster)} points (max {max_points})')
            return None
        
        x_coords = [p['x'] for p in cluster]
        y_coords = [p['y'] for p in cluster]
        distances = [p['distance'] for p in cluster]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        avg_distance = np.mean(distances)
        
        width_x = max(x_coords) - min(x_coords)
        width_y = max(y_coords) - min(y_coords)
        width = math.sqrt(width_x**2 + width_y**2)
        
        if min_width < width < max_width:
            self.get_logger().info(
                f'  → HUMAN DETECTED: pos=({center_x:.2f}, {center_y:.2f}), '
                f'width={width:.2f}m, dist={avg_distance:.2f}m, '
                f'points={len(cluster)} ✓'
            )
            
            return {
                'x': center_x,
                'y': center_y,
                'width': width,
                'distance': avg_distance,
                'num_points': len(cluster)
            }
        else:
            self.get_logger().debug(
                f'  → Object width={width:.2f}m not in range [{min_width}, {max_width}]'
            )
            return None
    
    
    def detect_humans_basic(self):
        """Fallback detection without map comparison."""
        if self.latest_scan is None:
            return []
        
        scan = self.latest_scan
        ranges = np.array(scan.ranges)
        
        ranges = np.where(np.isinf(ranges), scan.range_max, ranges)
        ranges = np.where(np.isnan(ranges), scan.range_max, ranges)
        
        range_min = self.detection_config['scan_range_min']
        range_max = self.detection_config['scan_range_max']
        min_width = self.detection_config['human_width_min']
        max_width = self.detection_config['human_width_max']
        min_points = self.detection_config['min_cluster_points']
        max_points = self.detection_config['max_cluster_points']
        gap_threshold = self.detection_config['cluster_gap_threshold']
        
        in_range = (ranges > range_min) & (ranges < range_max)
        indices = np.where(in_range)[0]
        
        if len(indices) < min_points:
            return []
        
        clusters = []
        current = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] <= gap_threshold:
                current.append(indices[i])
            else:
                if len(current) >= min_points:
                    clusters.append(current)
                current = [indices[i]]
        
        if len(current) >= min_points:
            clusters.append(current)
        
        humans = []
        
        for cluster in clusters:
            if len(cluster) > max_points:
                continue
            
            start_angle = scan.angle_min + cluster[0] * scan.angle_increment
            end_angle = scan.angle_min + cluster[-1] * scan.angle_increment
            avg_dist = np.mean(ranges[cluster])
            width = avg_dist * abs(end_angle - start_angle)
            
            if min_width < width < max_width:
                center_idx = cluster[len(cluster) // 2]
                center_angle = scan.angle_min + center_idx * scan.angle_increment
                
                humans.append({
                    'x': avg_dist * math.cos(center_angle),
                    'y': avg_dist * math.sin(center_angle),
                    'width': width,
                    'distance': avg_dist,
                    'num_points': len(cluster)
                })
        
        return humans
    
    
    def check_for_human_with_confidence(self):
        """Take multiple scans and use majority voting for reliable detection."""
        num_scans = self.detection_config['num_scans_to_check']
        threshold = self.detection_config['detection_threshold']
        
        # Wait for valid pose first
        self.wait_for_valid_pose(timeout=3.0)
        
        detection_count = 0
        
        for i in range(num_scans):
            # Give time for fresh scan data
            for _ in range(3):  # Spin multiple times to get fresh data
                rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.2)
            
            humans = self.detect_humans_with_map()
            
            if len(humans) > 0:
                detection_count += 1
                self.get_logger().info(f'  Scan {i+1}/{num_scans}: {len(humans)} human(s) detected ✓')
            else:
                self.get_logger().info(f'  Scan {i+1}/{num_scans}: No humans detected')
        
        confident = detection_count >= threshold
        self.get_logger().info(
            f'  Detection result: {detection_count}/{num_scans} positive '
            f'(threshold: {threshold}) → {"CONFIRMED ✓" if confident else "NOT CONFIRMED"}'
        )
        
        return confident
    
    
    # =========================================================================
    # NAVIGATION FUNCTIONS
    # =========================================================================
    
    def create_pose_stamped(self, x, y, yaw):
        """Create a PoseStamped message for navigation goals."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        return pose
    
    
    def navigate_to(self, x, y, yaw=0.0):
        """Navigate to a position and wait for completion."""
        goal = self.create_pose_stamped(x, y, yaw)
        
        self.get_logger().info(f'  Navigating to ({x:.2f}, {y:.2f})...')
        self.navigator.goToPose(goal)
        
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        result = self.navigator.getResult()
        
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('  ✓ Navigation succeeded')
            # Wait a moment for pose to update
            time.sleep(0.5)
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.1)
            return True
        else:
            self.get_logger().warn(f'  ✗ Navigation failed: {result}')
            return False
    
    
    def check_person_at_location(self, person_x, person_y, person_name):
        """Navigate near a person's expected location and scan for them."""
        self.get_logger().info(f'\n  Checking for {person_name}')
        self.get_logger().info(f'  Expected position: ({person_x}, {person_y})')
        
        # Get current pose using our robust method
        robot_x, robot_y, _ = self.get_current_pose()
        
        dx = person_x - robot_x
        dy = person_y - robot_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist > 0:
            dx /= dist
            dy /= dist
        
        approach_dist = self.detection_config['approach_distance']
        obs_x = person_x - dx * approach_dist
        obs_y = person_y - dy * approach_dist
        obs_yaw = math.atan2(dy, dx)
        
        self.get_logger().info(f'  Observation point: ({obs_x:.2f}, {obs_y:.2f})')
        
        if not self.navigate_to(obs_x, obs_y, obs_yaw):
            return False
        
        # Wait for pose to stabilize after navigation
        self.get_logger().info('  Waiting for pose to stabilize...')
        time.sleep(1.0)
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info('  Scanning for human...')
        found = self.check_for_human_with_confidence()
        
        if found:
            self.get_logger().info(f'  ✓ {person_name} FOUND at expected location!')
        else:
            self.get_logger().warn(f'  ✗ {person_name} NOT at expected location')
        
        return found
    
    
    # =========================================================================
    # MISSION CONTROL
    # =========================================================================
    
    def start_mission_once(self):
        """Ensure mission only starts once."""
        if self.mission_started:
            return
        self.mission_started = True
        self.run_mission()
    
    
    def run_mission(self):
        """Main mission sequence."""
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  STARTING HUMAN DETECTION MISSION')
        self.get_logger().info('=' * 60)
        
        # Step 1: Set initial pose
        self.get_logger().info('\n[Step 1] Setting initial robot pose...')
        
        initial_pose = self.create_pose_stamped(
            self.robot_start['x'],
            self.robot_start['y'],
            self.robot_start['yaw']
        )
        self.navigator.setInitialPose(initial_pose)
        self.get_logger().info('  Initial pose set')
        
        # Step 2: Wait for Nav2
        self.get_logger().info('\n[Step 2] Waiting for Nav2...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('  Nav2 is active')
        
        # Step 3: Wait for map
        self.get_logger().info('\n[Step 3] Waiting for map data...')
        timeout = 10.0
        start_time = time.time()
        
        while not self.map_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.5)
        
        if self.map_received:
            self.get_logger().info('  Map received')
        else:
            self.get_logger().warn('  Map not received - using basic detection')
        
        # Wait for localization AND pose to stabilize
        self.get_logger().info('  Waiting for localization to stabilize...')
        time.sleep(3.0)
        
        # Spin to get pose updates
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Verify we have a pose
        self.wait_for_valid_pose(timeout=5.0)
        
        # Step 4: Check Person 1
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('[Step 4] CHECKING PERSON 1')
        self.get_logger().info('=' * 60)
        
        self.person1_found = self.check_person_at_location(
            self.person1_expected['x'],
            self.person1_expected['y'],
            self.person1_expected['name']
        )
        
        # Step 5: Check Person 2
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('[Step 5] CHECKING PERSON 2')
        self.get_logger().info('=' * 60)
        
        self.person2_found = self.check_person_at_location(
            self.person2_expected['x'],
            self.person2_expected['y'],
            self.person2_expected['name']
        )
        
        # Step 6: Report initial findings
        self.print_initial_report()
        
        # Step 7: Search if needed
        if not self.person1_found or not self.person2_found:
            self.get_logger().info('\n' + '=' * 60)
            self.get_logger().info('[Step 6] SEARCHING FOR MISSING PERSON(S)')
            self.get_logger().info('=' * 60)
            self.search_warehouse()
        
        # Step 8: Final report
        self.print_final_report()
    
    
    def search_warehouse(self):
        """Search predefined waypoints for missing people."""
        missing = []
        if not self.person1_found:
            missing.append('Person 1')
        if not self.person2_found:
            missing.append('Person 2')
        
        self.get_logger().info(f'  Missing: {", ".join(missing)}')
        self.get_logger().info(f'  Searching {len(self.search_waypoints)} waypoints...')
        
        for i, wp in enumerate(self.search_waypoints):
            if self.person1_found and self.person2_found:
                self.get_logger().info('  All persons found! Ending search.')
                break
            
            self.get_logger().info(f'\n  Waypoint {i+1}/{len(self.search_waypoints)}: ({wp["x"]}, {wp["y"]})')
            
            if not self.navigate_to(wp['x'], wp['y'], 0.0):
                self.get_logger().warn('  Could not reach waypoint, skipping...')
                continue
            
            # Wait for stable pose
            time.sleep(0.5)
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.1)
            
            humans = self.detect_humans_with_map()
            
            if len(humans) > 0:
                self.get_logger().info(f'  🔍 Found {len(humans)} person(s) here!')
                
                if not self.person1_found and self.person1_new_location is None:
                    self.person1_new_location = {'x': wp['x'], 'y': wp['y']}
                    self.get_logger().info('  → Recorded as Person 1 new location')
                elif not self.person2_found and self.person2_new_location is None:
                    self.person2_new_location = {'x': wp['x'], 'y': wp['y']}
                    self.get_logger().info('  → Recorded as Person 2 new location')
    
    
    def print_initial_report(self):
        """Print results after checking expected locations."""
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('  INITIAL CHECK RESULTS')
        self.get_logger().info('=' * 60)
        
        if self.person1_found:
            self.get_logger().info(
                f'  ✓ Person 1: FOUND at ({self.person1_expected["x"]}, {self.person1_expected["y"]})'
            )
        else:
            self.get_logger().warn(
                f'  ✗ Person 1: NOT at ({self.person1_expected["x"]}, {self.person1_expected["y"]})'
            )
        
        if self.person2_found:
            self.get_logger().info(
                f'  ✓ Person 2: FOUND at ({self.person2_expected["x"]}, {self.person2_expected["y"]})'
            )
        else:
            self.get_logger().warn(
                f'  ✗ Person 2: NOT at ({self.person2_expected["x"]}, {self.person2_expected["y"]})'
            )
    
    
    def print_final_report(self):
        """Print final mission results."""
        self.get_logger().info('\n')
        self.get_logger().info('╔' + '═' * 58 + '╗')
        self.get_logger().info('║' + '  FINAL MISSION REPORT'.center(58) + '║')
        self.get_logger().info('╠' + '═' * 58 + '╣')
        
        if self.person1_found:
            msg = f'✓ Person 1: Original location ({self.person1_expected["x"]}, {self.person1_expected["y"]})'
        elif self.person1_new_location:
            msg = f'→ Person 1: MOVED to ({self.person1_new_location["x"]}, {self.person1_new_location["y"]})'
        else:
            msg = '? Person 1: MOVED - new location unknown'
        self.get_logger().info('║  ' + msg.ljust(56) + '║')
        
        if self.person2_found:
            msg = f'✓ Person 2: Original location ({self.person2_expected["x"]}, {self.person2_expected["y"]})'
        elif self.person2_new_location:
            msg = f'→ Person 2: MOVED to ({self.person2_new_location["x"]}, {self.person2_new_location["y"]})'
        else:
            msg = '? Person 2: MOVED - new location unknown'
        self.get_logger().info('║  ' + msg.ljust(56) + '║')
        
        self.get_logger().info('╚' + '═' * 58 + '╝')
        self.get_logger().info('')


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    controller = MapAwareWarehouseController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down...')
    finally:
        controller.navigator.lifecycleShutdown()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()