#!/usr/bin/env python3
"""
Map-Aware Warehouse Human Detection Controller (FULLY FIXED VERSION)
=====================================================================

FIXES APPLIED:
1. TF listener for reliable robot pose (AMCL doesn't publish frequently)
2. Fallback pose estimation from navigation
3. Improved scan timing to ensure fresh data
4. More robust pose waiting
5. FIXED: Navigation now approaches from robot's side (not through walls!)
6. FIXED: Proper localization reset for re-runs (no more hallucinated walls)

Author: ROS2 Tutorial
For: ROS2 Jazzy + Nav2 + Gazebo
"""

# =============================================================================
# IMPORTS
# =============================================================================

import rclpy                                    # ROS2 Python client library - the main interface for ROS2
from rclpy.node import Node                     # Base class that all ROS2 nodes inherit from
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # Quality of Service settings for subscribers/publishers

# Message types - these define the structure of data sent between nodes
from geometry_msgs.msg import PoseStamped                   # A pose (position + orientation) with a timestamp and frame
from geometry_msgs.msg import PoseWithCovarianceStamped     # Pose with uncertainty information (from AMCL)
from sensor_msgs.msg import LaserScan                       # Laser scanner data (distances at various angles)
from nav_msgs.msg import OccupancyGrid                      # 2D map data (grid of occupied/free/unknown cells)

# TF2 for getting robot pose from transform tree
# TF (Transform) is ROS's system for tracking coordinate frames over time
from tf2_ros import Buffer, TransformListener              # Buffer stores transforms, Listener subscribes to /tf
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException  # Possible TF errors

# Nav2 Simple Commander - high-level navigation interface
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# Standard Python libraries
import math                 # Mathematical functions (sin, cos, atan2, sqrt)
import numpy as np          # Numerical operations on arrays
import time                 # Time delays and timestamps


# =============================================================================
# MAIN CONTROLLER CLASS
# =============================================================================

class MapAwareWarehouseController(Node):
    """
    Complete warehouse human detection system with FIXED pose handling
    and proper re-run support.
    
    This node:
    1. Navigates to expected person locations
    2. Uses laser scans + map comparison to detect humans (dynamic objects)
    3. Searches the warehouse if people aren't at expected locations
    4. Reports findings
    
    Inherits from Node - the base class for all ROS2 Python nodes.
    """
    
    def __init__(self):
        """
        Initialize the node, subscribers, and parameters.
        
        This is called once when the node is created.
        Sets up all ROS2 communication (subscribers, publishers, timers)
        and initializes state variables.
        """
        # Initialize the parent Node class with our node name
        # This name appears in 'ros2 node list' and log messages
        super().__init__('map_aware_warehouse_controller')
        
        # =====================================================================
        # CONFIGURATION PARAMETERS
        # =====================================================================
        # These values control the robot's behavior
        # You can tune these based on your environment
        
        # Expected person positions (from your Gazebo screenshots)
        # These are the locations where we expect to find people
        self.person1_expected = {
            'x': 1.00,          # X coordinate in meters (map frame)
            'y': -1.00,         # Y coordinate in meters (map frame)
            'name': 'Person 1 - Standing'  # Human-readable name for logging
        }
        
        self.person2_expected = {
            'x': -12.00,        # X coordinate in meters
            'y': 15.00,         # Y coordinate in meters
            'name': 'Person 2 - Walking'
        }
        
        # Robot's starting position (must match Gazebo spawn point)
        # Used for initial localization and as fallback pose
        self.robot_start = {
            'x': 2.12,          # Starting X position
            'y': -21.3,         # Starting Y position
            'yaw': 1.57         # Starting orientation (radians) - 1.57 ≈ 90° = facing +Y
        }
        
        # Detection parameters - TUNED for better detection
        # These control how we identify humans from laser scan data
        self.detection_config = {
            # Navigation settings
            'approach_distance': 2.0,       # How far from person to stop (meters)
            
            # Laser scan filtering - ignore readings outside this range
            'scan_range_min': 0.3,          # Minimum valid distance (meters) - filters out noise
            'scan_range_max': 5.0,          # Maximum detection distance (meters)
            
            # Human identification thresholds
            # Humans typically appear as clusters of points with specific width
            'human_width_min': 0.10,        # Minimum width to be considered human (meters)
            'human_width_max': 1.2,         # Maximum width (larger = furniture/wall)
            'min_cluster_points': 3,        # Minimum laser points in a cluster
            'max_cluster_points': 100,      # Maximum points (larger = wall, not person)
            'cluster_gap_threshold': 8,     # Max index gap between points in same cluster
            
            # Map comparison settings
            'map_obstacle_threshold': 50,   # Occupancy value above this = obstacle (0-100 scale)
            'dynamic_object_buffer': 0.15,  # Buffer zone around static obstacles (meters)
            
            # Detection confidence settings
            'num_scans_to_check': 5,        # Number of scans for majority voting
            'detection_threshold': 2,       # Minimum positive detections to confirm
        }
        
        # Search waypoints - places to check if people aren't at expected locations
        # These should cover the navigable areas of your warehouse
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
        # These track the current state of the mission
        
        self.person1_found = None           # True/False/None - was person 1 at expected location?
        self.person2_found = None           # True/False/None - was person 2 at expected location?
        self.person1_new_location = None    # Dict with x,y if found during search
        self.person2_new_location = None    # Dict with x,y if found during search
        
        self.map_data = None                # 2D numpy array of occupancy values
        self.map_info = None                # Metadata about the map (resolution, origin, size)
        self.latest_scan = None             # Most recent LaserScan message
        self.robot_pose = None              # Pose from AMCL topic (backup method)
        self.last_known_pose = None         # Fallback pose storage for reliability
        
        self.mission_started = False        # Prevents mission from running twice
        self.map_received = False           # True once we've received map data
        self.localization_initialized = False  # True once localization is verified
        
        # =====================================================================
        # TF2 BUFFER AND LISTENER
        # =====================================================================
        # TF (Transform) system provides real-time robot pose
        # This is MORE RELIABLE than waiting for AMCL topic publications
        #
        # The TF tree looks like:
        #   map -> odom -> base_link
        # Where:
        #   - map: Fixed world frame (from SLAM/localization)
        #   - odom: Odometry frame (drifts over time)
        #   - base_link: Robot's body frame
        #
        # We want map -> base_link to know robot's position in the world
        
        self.tf_buffer = Buffer(
            cache_time=rclpy.duration.Duration(seconds=10.0)  # Shorter cache for re-runs
        )
        # TransformListener automatically subscribes to /tf and /tf_static
        # and populates the buffer with transforms
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info('TF listener initialized for pose tracking')
        
        # =====================================================================
        # SUBSCRIBERS
        # =====================================================================
        # Subscribers receive messages published by other nodes
        
        # Map subscriber with transient local QoS
        # QoS (Quality of Service) settings control message delivery guarantees
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,      # Guaranteed delivery
            durability=DurabilityPolicy.TRANSIENT_LOCAL, # Late subscribers get last message
            depth=1                                       # Only keep 1 message in queue
        )
        
        # Subscribe to the /map topic (published by map_server or SLAM)
        # When a message arrives, map_callback() is called
        self.map_sub = self.create_subscription(
            OccupancyGrid,          # Message type
            '/map',                 # Topic name
            self.map_callback,      # Callback function
            map_qos                 # QoS profile
        )
        self.get_logger().info('Subscribed to /map topic')
        
        # Laser scan subscriber - uses default QoS (best effort, volatile)
        # Laser data arrives at high frequency (10-40 Hz typically)
        self.scan_sub = self.create_subscription(
            LaserScan,              # Message type containing distance measurements
            '/scan',                # Topic name (from lidar driver)
            self.scan_callback,     # Callback function
            10                      # Queue depth (integer = default QoS with this depth)
        )
        self.get_logger().info('Subscribed to /scan topic')
        
        # AMCL pose subscriber (as backup to TF)
        # AMCL publishes when the pose estimate changes significantly
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,  # Pose with uncertainty covariance matrix
            '/amcl_pose',               # Topic from AMCL localization
            self.pose_callback,         # Callback function
            10                          # Queue depth
        )
        self.get_logger().info('Subscribed to /amcl_pose topic')
        
        # =====================================================================
        # NAVIGATOR
        # =====================================================================
        # BasicNavigator is a high-level interface to Nav2
        # It handles path planning, obstacle avoidance, and recovery behaviors
        
        self.navigator = BasicNavigator()
        
        # =====================================================================
        # STARTUP
        # =====================================================================
        # Print startup banner and schedule mission start
        
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  MAP-AWARE WAREHOUSE CONTROLLER (FULLY FIXED)')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Person 1 expected: ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        self.get_logger().info(f'  Person 2 expected: ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        self.get_logger().info('=' * 60)
        self.get_logger().info('')
        
        # Create a one-shot timer to start the mission after 5 seconds
        # This gives time for other nodes to start and publish data
        self.create_timer(5.0, self.start_mission_once)
    
    
    # =========================================================================
    # CALLBACK FUNCTIONS
    # =========================================================================
    # Callbacks are called automatically when messages arrive on subscribed topics
    
    def map_callback(self, msg):
        """
        Store map data when received.
        
        Called automatically when a message is published to /map.
        
        Parameters:
            msg (OccupancyGrid): The map message containing:
                - info: Metadata (width, height, resolution, origin)
                - data: Flat array of occupancy values (-1=unknown, 0=free, 100=occupied)
        """
        # Store the metadata (resolution, size, origin position)
        self.map_info = msg.info
        
        # Convert flat array to 2D numpy array for easier indexing
        # msg.data is a flat list, we reshape it to (height, width)
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        
        # Log info only on first reception
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
        """
        Store latest laser scan.
        
        Called automatically when a message is published to /scan.
        We just store it; processing happens when we actively scan for humans.
        
        Parameters:
            msg (LaserScan): Laser scan data containing:
                - ranges: Array of distance measurements
                - angle_min/max: Start and end angles of scan
                - angle_increment: Angle between consecutive readings
        """
        self.latest_scan = msg
    
    
    def pose_callback(self, msg):
        """
        Store pose from AMCL (used as backup to TF).
        
        Called when AMCL publishes a pose update (not every frame!).
        AMCL only publishes when the estimated pose changes significantly.
        
        Parameters:
            msg (PoseWithCovarianceStamped): Contains:
                - pose.pose: The estimated pose (position + orientation)
                - pose.covariance: 6x6 uncertainty matrix
        """
        self.robot_pose = msg.pose.pose
        self.last_known_pose = msg.pose.pose  # Keep as fallback
        self.get_logger().debug(
            f'AMCL pose received: ({msg.pose.pose.position.x:.2f}, '
            f'{msg.pose.pose.position.y:.2f})'
        )
    
    
    # =========================================================================
    # LOCALIZATION RESET (FIX FOR RE-RUN ISSUES!)
    # =========================================================================
    
    def reset_localization(self):
        """
        Properly reset the localization system before starting.
        
        This clears old state and ensures AMCL re-initializes correctly.
        
        WHY THIS IS NEEDED:
        - When you re-run the controller, Gazebo resets the robot position
        - But AMCL still thinks the robot is at its LAST position
        - The TF buffer has cached old transforms
        - Costmaps have obstacles placed based on old (wrong) poses
        
        This function:
        1. Clears the TF buffer
        2. Resets cached poses
        3. Publishes initial pose to AMCL multiple times
        4. Waits for AMCL to converge
        5. Clears costmaps
        6. Verifies the pose is correct
        """
        self.get_logger().info('Resetting localization system...')
        
        # ---------------------------------------------------------------------
        # Step 1: Clear the TF buffer (remove stale transforms)
        # ---------------------------------------------------------------------
        # Old transforms from previous run will give wrong poses
        self.tf_buffer.clear()
        self.get_logger().info('  ✓ TF buffer cleared')
        
        # ---------------------------------------------------------------------
        # Step 2: Reset stored poses
        # ---------------------------------------------------------------------
        # Clear any cached pose data from previous run
        self.robot_pose = None
        self.last_known_pose = None
        self.get_logger().info('  ✓ Cached poses cleared')
        
        # ---------------------------------------------------------------------
        # Step 3: Set initial pose MULTIPLE times
        # ---------------------------------------------------------------------
        # AMCL uses a particle filter - publishing initial pose resets particles
        # Publishing multiple times helps ensure AMCL receives and processes it
        initial_pose = self.create_pose_stamped(
            self.robot_start['x'],
            self.robot_start['y'],
            self.robot_start['yaw']
        )
        
        # Publish initial pose several times to ensure AMCL receives it
        for i in range(3):
            self.navigator.setInitialPose(initial_pose)
            time.sleep(0.5)  # Wait between publications
            rclpy.spin_once(self, timeout_sec=0.1)  # Process any callbacks
        self.get_logger().info('  ✓ Initial pose published (3x)')
        
        # ---------------------------------------------------------------------
        # Step 4: Wait for AMCL to re-initialize particles
        # ---------------------------------------------------------------------
        # AMCL needs time to:
        # - Receive the initial pose
        # - Spread particles around the initial pose
        # - Converge particles based on sensor data
        self.get_logger().info('  Waiting for AMCL to converge...')
        time.sleep(3.0)  # Give AMCL time to process
        
        # ---------------------------------------------------------------------
        # Step 5: Spin to process incoming messages and update TF
        # ---------------------------------------------------------------------
        # This processes callbacks to get fresh pose data
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # ---------------------------------------------------------------------
        # Step 6: Clear costmaps (removes hallucinated walls!)
        # ---------------------------------------------------------------------
        # Costmaps contain obstacles from laser scans placed at OLD (wrong) poses
        # We need to clear them so they rebuild with correct pose
        self.get_logger().info('  Clearing costmaps...')
        try:
            self.navigator.clearAllCostmaps()
            time.sleep(1.0)  # Wait for costmaps to clear
            self.get_logger().info('  ✓ Costmaps cleared')
        except Exception as e:
            self.get_logger().warn(f'  ⚠ Could not clear costmaps: {e}')
        
        # ---------------------------------------------------------------------
        # Step 7: Verify we have a valid pose near the expected start
        # ---------------------------------------------------------------------
        pose = self.get_robot_pose_from_tf()
        if pose:
            x, y, yaw = pose
            expected_x = self.robot_start['x']
            expected_y = self.robot_start['y']
            error = math.sqrt((x - expected_x)**2 + (y - expected_y)**2)
            
            if error > 1.0:  # More than 1 meter off
                self.get_logger().warn(
                    f'  ⚠ Pose error: {error:.2f}m - AMCL may not have converged!'
                )
                self.get_logger().warn(
                    f'    Expected: ({expected_x:.2f}, {expected_y:.2f})'
                )
                self.get_logger().warn(
                    f'    Got: ({x:.2f}, {y:.2f})'
                )
                self.get_logger().warn(
                    '    TIP: Try running the controller again, or reset Gazebo'
                )
            else:
                self.get_logger().info(
                    f'  ✓ Pose verified: ({x:.2f}, {y:.2f}), error: {error:.2f}m'
                )
        else:
            self.get_logger().warn('  ⚠ Could not verify pose from TF')
        
        self.localization_initialized = True
        self.get_logger().info('  ✓ Localization reset complete')
    
    
    # =========================================================================
    # POSE RETRIEVAL
    # =========================================================================
    
    def get_robot_pose_from_tf(self):
        """
        Get robot pose from TF transform tree.
        
        This is MORE RELIABLE than the AMCL topic because:
        - TF is updated continuously by AMCL
        - AMCL topic only publishes when pose changes significantly
        
        The TF tree maintains transforms between coordinate frames:
        - 'map' frame: Fixed world reference
        - 'base_link' frame: Robot's body
        
        We look up the transform from map to base_link to get robot's
        position in the world.
        
        Returns:
            tuple: (x, y, yaw) in meters and radians, or None if not available
        """
        try:
            # Look up transform from 'map' frame to 'base_link' (robot) frame
            # This tells us where the robot is in the map
            transform = self.tf_buffer.lookup_transform(
                'map',           # Target frame (where we want the pose expressed)
                'base_link',     # Source frame (the robot's body)
                rclpy.time.Time(),  # Get latest available transform (time=0)
                timeout=rclpy.duration.Duration(seconds=0.5)  # Wait up to 0.5s
            )
            
            # Extract position from the transform
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Extract yaw (rotation around Z axis) from quaternion
            # Quaternions represent 3D rotations as (x, y, z, w)
            # For 2D navigation, we only care about yaw (rotation in XY plane)
            q = transform.transform.rotation
            # Formula to extract yaw from quaternion:
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return (x, y, yaw)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # These exceptions mean the transform isn't available
            # LookupException: Frame doesn't exist
            # ConnectivityException: No path between frames
            # ExtrapolationException: Requested time is outside available data
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None
    
    
    def get_current_pose(self):
        """
        Get current robot pose using multiple fallback methods.
        
        This is a robust method that tries multiple sources for the pose.
        
        Priority (best to worst):
        1. TF transform - Most reliable, updated continuously
        2. AMCL topic - Only updated on significant pose change
        3. Last known pose - Stale but better than nothing
        4. Starting position - Last resort fallback
        
        Returns:
            tuple: (x, y, yaw) - always returns something
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
        
        This is important because:
        - TF data may not be immediately available at startup
        - AMCL needs time to initialize
        - We should not navigate until we know where we are
        
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
            
            # Try to get pose from TF
            tf_pose = self.get_robot_pose_from_tf()
            if tf_pose is not None:
                self.get_logger().info(
                    f'  ✓ Got pose from TF: ({tf_pose[0]:.2f}, {tf_pose[1]:.2f})'
                )
                return True
            
            # Try AMCL topic as backup
            if self.robot_pose is not None:
                self.get_logger().info(
                    f'  ✓ Got pose from AMCL: '
                    f'({self.robot_pose.position.x:.2f}, {self.robot_pose.position.y:.2f})'
                )
                return True
        
        self.get_logger().warn('  ⚠ Timeout waiting for pose - using fallback')
        return False
    
    
    # =========================================================================
    # COORDINATE TRANSFORMATION FUNCTIONS
    # =========================================================================
    
    def world_to_map(self, world_x, world_y):
        """
        Convert world coordinates (meters) to map pixel coordinates.
        
        The map is stored as a 2D grid of pixels. To look up a world position
        in the map, we need to convert from meters to pixel indices.
        
        The conversion uses:
        - Map origin: World position of pixel (0,0)
        - Resolution: Meters per pixel
        
        Parameters:
            world_x (float): X position in meters (map frame)
            world_y (float): Y position in meters (map frame)
            
        Returns:
            tuple: (map_x, map_y) pixel coordinates, or None if outside map
        """
        if self.map_info is None:
            return None
        
        # Get map parameters
        origin_x = self.map_info.origin.position.x  # World X of pixel (0,0)
        origin_y = self.map_info.origin.position.y  # World Y of pixel (0,0)
        resolution = self.map_info.resolution       # Meters per pixel
        
        # Convert: subtract origin, divide by resolution
        map_x = int((world_x - origin_x) / resolution)
        map_y = int((world_y - origin_y) / resolution)
        
        # Check bounds
        if 0 <= map_x < self.map_info.width and 0 <= map_y < self.map_info.height:
            return (map_x, map_y)
        else:
            return None  # Outside map bounds
    
    
    def map_to_world(self, map_x, map_y):
        """
        Convert map pixel coordinates to world coordinates (meters).
        
        Inverse of world_to_map().
        
        Parameters:
            map_x (int): Pixel X coordinate
            map_y (int): Pixel Y coordinate
            
        Returns:
            tuple: (world_x, world_y) in meters, or None if no map
        """
        if self.map_info is None:
            return None
        
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        resolution = self.map_info.resolution
        
        # Convert: multiply by resolution, add origin
        # +0.5 to get center of pixel instead of corner
        world_x = origin_x + (map_x + 0.5) * resolution
        world_y = origin_y + (map_y + 0.5) * resolution
        
        return (world_x, world_y)
    
    
    # =========================================================================
    # MAP QUERY FUNCTIONS
    # =========================================================================
    
    def is_occupied_in_map(self, world_x, world_y):
        """
        Check if a world position is marked as OCCUPIED in the static map.
        
        The map contains occupancy values:
        - -1: Unknown (not seen by mapping)
        - 0: Free space (definitely empty)
        - 100: Occupied (definitely an obstacle)
        - Values in between represent probability
        
        Parameters:
            world_x (float): X position in meters
            world_y (float): Y position in meters
            
        Returns:
            bool: True if position is occupied in the static map
        """
        if self.map_data is None:
            return False
        
        map_coords = self.world_to_map(world_x, world_y)
        if map_coords is None:
            return False  # Outside map = not occupied
        
        map_x, map_y = map_coords
        
        # Note: numpy array is indexed as [row, col] = [y, x]
        occupancy_value = self.map_data[map_y, map_x]
        
        # Compare to threshold (default 50)
        threshold = self.detection_config['map_obstacle_threshold']
        return occupancy_value > threshold
    
    
    def is_near_static_obstacle(self, world_x, world_y):
        """
        Check if a position is near a static obstacle in the map.
        
        This checks the position AND surrounding area, providing a buffer
        to account for:
        - Map resolution (might miss thin obstacles)
        - Laser accuracy
        - Robot footprint
        
        Parameters:
            world_x (float): X position in meters
            world_y (float): Y position in meters
            
        Returns:
            bool: True if near a static obstacle
        """
        if self.map_data is None:
            return False
        
        buffer = self.detection_config['dynamic_object_buffer']
        
        # Check 3x3 grid of points around the position
        for dx in [-buffer, 0, buffer]:
            for dy in [-buffer, 0, buffer]:
                if self.is_occupied_in_map(world_x + dx, world_y + dy):
                    return True
        
        return False
    
    
    # =========================================================================
    # HUMAN DETECTION FUNCTIONS
    # =========================================================================
    
    def detect_humans_with_map(self):
        """
        Main detection function: Find humans by comparing scan to map.
        
        ALGORITHM:
        1. Get robot's current pose
        2. Transform each laser point to world coordinates
        3. Check if that point is in the static map (obstacle)
        4. Points NOT in static map are "dynamic" = potential humans
        5. Cluster nearby dynamic points
        6. Analyze clusters for human-like shape (correct width)
        
        This distinguishes humans (dynamic) from walls/shelves (static).
        
        Returns:
            list: List of detected humans, each dict with x, y, width, etc.
        """
        if self.latest_scan is None:
            self.get_logger().warn('No laser scan data available')
            return []
        
        if self.map_data is None:
            self.get_logger().warn('No map data available - using basic detection')
            return self.detect_humans_basic()
        
        # GET POSE USING ROBUST METHOD
        pose = self.get_current_pose()
        robot_x, robot_y, robot_yaw = pose
        
        self.get_logger().info(
            f'  Robot pose: ({robot_x:.2f}, {robot_y:.2f}, '
            f'yaw={math.degrees(robot_yaw):.1f}°)'
        )
        
        scan = self.latest_scan
        ranges = np.array(scan.ranges)
        
        # Get configuration parameters
        range_min = self.detection_config['scan_range_min']
        range_max = self.detection_config['scan_range_max']
        gap_threshold = self.detection_config['cluster_gap_threshold']
        min_points = self.detection_config['min_cluster_points']
        
        # =====================================================================
        # STEP 1: Find dynamic points (in scan but NOT in static map)
        # =====================================================================
        
        dynamic_points = []
        
        for i, distance in enumerate(ranges):
            # Skip invalid readings
            if not np.isfinite(distance):
                continue
            if distance < range_min or distance > range_max:
                continue
            
            # Calculate world position of this laser hit
            # laser_angle: angle relative to robot's forward direction
            laser_angle = scan.angle_min + i * scan.angle_increment
            # world_angle: angle in world frame (add robot's yaw)
            world_angle = robot_yaw + laser_angle
            
            # Convert polar (distance, angle) to Cartesian (x, y)
            hit_x = robot_x + distance * math.cos(world_angle)
            hit_y = robot_y + distance * math.sin(world_angle)
            
            # Check if this point is a STATIC obstacle (in the map)
            is_static = self.is_near_static_obstacle(hit_x, hit_y)
            
            # If NOT static, it's a dynamic object (potential human!)
            if not is_static:
                dynamic_points.append({
                    'x': hit_x,
                    'y': hit_y,
                    'distance': distance,
                    'angle': laser_angle,
                    'index': i  # Keep track of scan index for clustering
                })
        
        self.get_logger().info(
            f'  Found {len(dynamic_points)} dynamic points (not in map)'
        )
        
        # =====================================================================
        # STEP 2: Cluster dynamic points into objects
        # =====================================================================
        # Consecutive laser indices hitting the same object should cluster
        
        if len(dynamic_points) < min_points:
            return []
        
        # Sort by scan index
        dynamic_points.sort(key=lambda p: p['index'])
        
        # Group points with consecutive indices
        clusters = []
        current_cluster = [dynamic_points[0]]
        
        for i in range(1, len(dynamic_points)):
            prev_idx = dynamic_points[i - 1]['index']
            curr_idx = dynamic_points[i]['index']
            
            # If indices are close, same cluster
            if curr_idx - prev_idx <= gap_threshold:
                current_cluster.append(dynamic_points[i])
            else:
                # Gap too large - save current cluster, start new one
                if len(current_cluster) >= min_points:
                    clusters.append(current_cluster)
                current_cluster = [dynamic_points[i]]
        
        # Don't forget the last cluster!
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
        """
        Analyze a cluster of points to determine if it's human-shaped.
        
        Humans typically appear as:
        - A cluster of 3-100 points (depending on distance)
        - Width between 0.1m and 1.2m (shoulder width to extended arms)
        
        Parameters:
            cluster (list): List of point dicts with x, y, distance, etc.
            
        Returns:
            dict: Human detection info, or None if not human-shaped
        """
        min_points = self.detection_config['min_cluster_points']
        max_points = self.detection_config['max_cluster_points']
        min_width = self.detection_config['human_width_min']
        max_width = self.detection_config['human_width_max']
        
        # Check point count
        if len(cluster) < min_points:
            return None
        if len(cluster) > max_points:
            self.get_logger().debug(
                f'  Cluster too large: {len(cluster)} points (max {max_points})'
            )
            return None
        
        # Extract coordinates
        x_coords = [p['x'] for p in cluster]
        y_coords = [p['y'] for p in cluster]
        distances = [p['distance'] for p in cluster]
        
        # Calculate cluster center and dimensions
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        avg_distance = np.mean(distances)
        
        # Calculate width (diagonal of bounding box)
        width_x = max(x_coords) - min(x_coords)
        width_y = max(y_coords) - min(y_coords)
        width = math.sqrt(width_x**2 + width_y**2)
        
        # Check if width is human-like
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
                f'  → Object width={width:.2f}m not in range '
                f'[{min_width}, {max_width}]'
            )
            return None
    
    
    def detect_humans_basic(self):
        """
        Fallback detection without map comparison.
        
        Used when map data isn't available. Less reliable because
        it can't distinguish humans from walls - just looks for
        human-sized clusters of points.
        
        Returns:
            list: List of potential human detections
        """
        if self.latest_scan is None:
            return []
        
        scan = self.latest_scan
        ranges = np.array(scan.ranges)
        
        # Replace inf/nan with max range
        ranges = np.where(np.isinf(ranges), scan.range_max, ranges)
        ranges = np.where(np.isnan(ranges), scan.range_max, ranges)
        
        # Get config
        range_min = self.detection_config['scan_range_min']
        range_max = self.detection_config['scan_range_max']
        min_width = self.detection_config['human_width_min']
        max_width = self.detection_config['human_width_max']
        min_points = self.detection_config['min_cluster_points']
        max_points = self.detection_config['max_cluster_points']
        gap_threshold = self.detection_config['cluster_gap_threshold']
        
        # Find points in valid range
        in_range = (ranges > range_min) & (ranges < range_max)
        indices = np.where(in_range)[0]
        
        if len(indices) < min_points:
            return []
        
        # Cluster consecutive indices
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
        
        # Analyze clusters
        humans = []
        
        for cluster in clusters:
            if len(cluster) > max_points:
                continue
            
            # Calculate angular width
            start_angle = scan.angle_min + cluster[0] * scan.angle_increment
            end_angle = scan.angle_min + cluster[-1] * scan.angle_increment
            avg_dist = np.mean(ranges[cluster])
            
            # Arc length = radius * angle
            width = avg_dist * abs(end_angle - start_angle)
            
            if min_width < width < max_width:
                center_idx = cluster[len(cluster) // 2]
                center_angle = scan.angle_min + center_idx * scan.angle_increment
                
                humans.append({
                    'x': avg_dist * math.cos(center_angle),  # Relative to robot
                    'y': avg_dist * math.sin(center_angle),
                    'width': width,
                    'distance': avg_dist,
                    'num_points': len(cluster)
                })
        
        return humans
    
    
    def check_for_human_with_confidence(self):
        """
        Take multiple scans and use majority voting for reliable detection.
        
        Single scans can give false positives/negatives. By taking multiple
        scans and requiring a majority to detect something, we get more
        reliable results.
        
        Returns:
            bool: True if human confidently detected
        """
        num_scans = self.detection_config['num_scans_to_check']
        threshold = self.detection_config['detection_threshold']
        
        # Wait for valid pose first
        self.wait_for_valid_pose(timeout=3.0)
        
        detection_count = 0
        
        for i in range(num_scans):
            # Give time for fresh scan data
            # Spin multiple times to ensure we get new laser data
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.2)  # Wait for scan to update
            
            humans = self.detect_humans_with_map()
            
            if len(humans) > 0:
                detection_count += 1
                self.get_logger().info(
                    f'  Scan {i+1}/{num_scans}: {len(humans)} human(s) detected ✓'
                )
            else:
                self.get_logger().info(
                    f'  Scan {i+1}/{num_scans}: No humans detected'
                )
        
        # Check if we met the threshold
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
        """
        Create a PoseStamped message for navigation goals.
        
        PoseStamped is the standard ROS2 message for specifying a position
        and orientation in a particular coordinate frame.
        
        Parameters:
            x (float): X position in meters
            y (float): Y position in meters
            yaw (float): Orientation in radians (0 = facing +X, π/2 = facing +Y)
            
        Returns:
            PoseStamped: The goal pose message
        """
        pose = PoseStamped()
        
        # Header specifies the coordinate frame and timestamp
        pose.header.frame_id = 'map'  # Goal is in map frame
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        
        # Position (in meters)
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0  # Ground level
        
        # Orientation as quaternion
        # For 2D navigation, we only set Z and W components
        # This represents rotation around the Z axis (yaw)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)  # Quaternion formula
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        return pose
    
    
    def navigate_to(self, x, y, yaw=0.0):
        """
        Navigate to a position and wait for completion.
        
        Uses Nav2's BasicNavigator to plan and execute a path to the goal.
        Blocks until navigation completes or fails.
        
        Parameters:
            x (float): Goal X position in meters
            y (float): Goal Y position in meters
            yaw (float): Goal orientation in radians
            
        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        goal = self.create_pose_stamped(x, y, yaw)
        
        self.get_logger().info(f'  Navigating to ({x:.2f}, {y:.2f})...')
        
        # Send goal to Nav2
        self.navigator.goToPose(goal)
        
        # Wait for completion, processing callbacks meanwhile
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Check result
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
    
    
    def is_approach_point_clear(self, x, y, check_radius=0.5):
        """
        Check if an approach point is in clear space (not blocked by obstacles).
        
        Parameters:
            x, y: World coordinates to check
            check_radius: Radius around point to verify is clear
            
        Returns:
            bool: True if the area appears navigable
        """
        if self.map_data is None:
            return True  # No map data, assume it's fine
        
        # Check a grid of points around the target
        step = 0.2
        for dx in np.arange(-check_radius, check_radius + step, step):
            for dy in np.arange(-check_radius, check_radius + step, step):
                # Only check points within the radius (circular check)
                if dx*dx + dy*dy <= check_radius * check_radius:
                    if self.is_occupied_in_map(x + dx, y + dy):
                        return False
        return True
    
    
    def find_clear_approach_point(self, person_x, person_y):
        """
        Find a clear approach point around the person.
        
        Tries multiple angles around the person to find one that's
        not blocked by walls/shelves.
        
        Parameters:
            person_x, person_y: Person's location
            
        Returns:
            tuple: (obs_x, obs_y, obs_yaw) or None if no clear point found
        """
        robot_x, robot_y, _ = self.get_current_pose()
        approach_dist = self.detection_config['approach_distance']
        
        # Calculate base angle (from person toward robot)
        dx = robot_x - person_x
        dy = robot_y - person_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist > 0:
            base_angle = math.atan2(dy, dx)
        else:
            base_angle = 0
        
        # Try different angle offsets from the base direction
        # Start with robot's direction (0°), then try sides
        angle_offsets_deg = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]
        
        for offset_deg in angle_offsets_deg:
            angle = base_angle + math.radians(offset_deg)
            
            # Calculate candidate approach point
            cand_x = person_x + approach_dist * math.cos(angle)
            cand_y = person_y + approach_dist * math.sin(angle)
            
            # Check if this point is clear
            if self.is_approach_point_clear(cand_x, cand_y, check_radius=0.4):
                # Calculate yaw to face the person
                face_yaw = math.atan2(person_y - cand_y, person_x - cand_x)
                
                self.get_logger().info(
                    f'  Found clear approach at {offset_deg}° offset: '
                    f'({cand_x:.2f}, {cand_y:.2f})'
                )
                return (cand_x, cand_y, face_yaw)
            else:
                self.get_logger().debug(f'  Approach at {offset_deg}° blocked')
        
        # Fallback: return the direct approach even if it might be blocked
        # (Nav2 will try to plan around obstacles)
        self.get_logger().warn('  No clear approach found, using direct approach')
        if dist > 0:
            dx /= dist
            dy /= dist
        else:
            dx, dy = 1.0, 0.0
        
        obs_x = person_x + dx * approach_dist
        obs_y = person_y + dy * approach_dist
        obs_yaw = math.atan2(-dy, -dx)
        
        return (obs_x, obs_y, obs_yaw)
    
    
    def check_person_at_location(self, person_x, person_y, person_name):
        """
        Navigate near a person's expected location and scan for them.
        
        IMPROVED VERSION: Tries multiple approach angles to find clear space,
        avoiding walls and shelves.
        
        Parameters:
            person_x (float): Expected X position of person
            person_y (float): Expected Y position of person
            person_name (str): Human-readable name for logging
            
        Returns:
            bool: True if person was detected at this location
        """
        self.get_logger().info(f'\n  Checking for {person_name}')
        self.get_logger().info(f'  Expected position: ({person_x}, {person_y})')
        
        # Find a clear approach point (tries multiple angles!)
        approach = self.find_clear_approach_point(person_x, person_y)
        
        if approach is None:
            self.get_logger().error('  Could not find any approach point!')
            return False
        
        obs_x, obs_y, obs_yaw = approach
        
        self.get_logger().info(f'  Observation point: ({obs_x:.2f}, {obs_y:.2f})')
        self.get_logger().info(
            f'  (Facing person at yaw={math.degrees(obs_yaw):.1f}°)'
        )
        
        # Navigate to observation point
        if not self.navigate_to(obs_x, obs_y, obs_yaw):
            return False
        
        # Wait for pose to stabilize after navigation
        self.get_logger().info('  Waiting for pose to stabilize...')
        time.sleep(1.0)
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Perform human detection scan
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
        """
        Ensure mission only starts once.
        
        This is called by a timer, but we only want to run the mission
        once (not repeatedly every 5 seconds).
        """
        if self.mission_started:
            return
        self.mission_started = True
        self.run_mission()
    
    
    def run_mission(self):
        """
        Main mission sequence.
        
        This is the high-level control flow:
        1. Initialize Nav2
        2. Reset localization (CRITICAL for re-runs!)
        3. Wait for map data
        4. Check expected person locations
        5. Search warehouse if people are missing
        6. Report results
        """
        self.get_logger().info('')
        self.get_logger().info('=' * 60)
        self.get_logger().info('  STARTING HUMAN DETECTION MISSION')
        self.get_logger().info('=' * 60)
        
        # =====================================================================
        # Step 1: Wait for Nav2 to be active
        # =====================================================================
        self.get_logger().info('\n[Step 1] Waiting for Nav2...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('  ✓ Nav2 is active')
        
        # =====================================================================
        # Step 2: Reset localization (THE FIX FOR RE-RUN ISSUES!)
        # =====================================================================
        self.get_logger().info('\n[Step 2] Resetting localization...')
        self.reset_localization()
        
        # =====================================================================
        # Step 3: Wait for map data
        # =====================================================================
        self.get_logger().info('\n[Step 3] Waiting for map data...')
        timeout = 10.0
        start_time = time.time()
        
        while not self.map_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.5)
        
        if self.map_received:
            self.get_logger().info('  ✓ Map received')
        else:
            self.get_logger().warn('  ⚠ Map not received - using basic detection')
        
        # =====================================================================
        # Step 4: Final verification before starting mission
        # =====================================================================
        self.get_logger().info('\n[Step 4] Final pose verification...')
        if not self.wait_for_valid_pose(timeout=5.0):
            self.get_logger().error('  ✗ Could not get valid pose! Continuing anyway...')
        
        pose = self.get_current_pose()
        self.get_logger().info(f'  ✓ Starting from: ({pose[0]:.2f}, {pose[1]:.2f})')
        
        # =====================================================================
        # Step 5: Check Person 1
        # =====================================================================
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('[Step 5] CHECKING PERSON 1')
        self.get_logger().info('=' * 60)
        
        self.person1_found = self.check_person_at_location(
            self.person1_expected['x'],
            self.person1_expected['y'],
            self.person1_expected['name']
        )
        
        # =====================================================================
        # Step 6: Check Person 2
        # =====================================================================
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('[Step 6] CHECKING PERSON 2')
        self.get_logger().info('=' * 60)
        
        self.person2_found = self.check_person_at_location(
            self.person2_expected['x'],
            self.person2_expected['y'],
            self.person2_expected['name']
        )
        
        # =====================================================================
        # Step 7: Report initial findings
        # =====================================================================
        self.print_initial_report()
        
        # =====================================================================
        # Step 8: Search if needed
        # =====================================================================
        if not self.person1_found or not self.person2_found:
            self.get_logger().info('\n' + '=' * 60)
            self.get_logger().info('[Step 7] SEARCHING FOR MISSING PERSON(S)')
            self.get_logger().info('=' * 60)
            self.search_warehouse()
        
        # =====================================================================
        # Step 9: Final report
        # =====================================================================
        self.print_final_report()
    
    
    def search_warehouse(self):
        """
        Search predefined waypoints for missing people.
        
        If people weren't found at their expected locations, we search
        the warehouse by visiting waypoints and scanning for humans.
        """
        missing = []
        if not self.person1_found:
            missing.append('Person 1')
        if not self.person2_found:
            missing.append('Person 2')
        
        self.get_logger().info(f'  Missing: {", ".join(missing)}')
        self.get_logger().info(
            f'  Searching {len(self.search_waypoints)} waypoints...'
        )
        
        for i, wp in enumerate(self.search_waypoints):
            # Stop if we've found everyone
            if self.person1_found and self.person2_found:
                self.get_logger().info('  All persons found! Ending search.')
                break
            
            self.get_logger().info(
                f'\n  Waypoint {i+1}/{len(self.search_waypoints)}: '
                f'({wp["x"]}, {wp["y"]})'
            )
            
            if not self.navigate_to(wp['x'], wp['y'], 0.0):
                self.get_logger().warn('  Could not reach waypoint, skipping...')
                continue
            
            # Wait for stable pose
            time.sleep(0.5)
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # Scan for humans
            humans = self.detect_humans_with_map()
            
            if len(humans) > 0:
                self.get_logger().info(f'  🔍 Found {len(humans)} person(s) here!')
                
                # Record the location
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
                f'  ✓ Person 1: FOUND at '
                f'({self.person1_expected["x"]}, {self.person1_expected["y"]})'
            )
        else:
            self.get_logger().warn(
                f'  ✗ Person 1: NOT at '
                f'({self.person1_expected["x"]}, {self.person1_expected["y"]})'
            )
        
        if self.person2_found:
            self.get_logger().info(
                f'  ✓ Person 2: FOUND at '
                f'({self.person2_expected["x"]}, {self.person2_expected["y"]})'
            )
        else:
            self.get_logger().warn(
                f'  ✗ Person 2: NOT at '
                f'({self.person2_expected["x"]}, {self.person2_expected["y"]})'
            )
    
    
    def print_final_report(self):
        """Print final mission results."""
        self.get_logger().info('\n')
        self.get_logger().info('╔' + '═' * 58 + '╗')
        self.get_logger().info('║' + '  FINAL MISSION REPORT'.center(58) + '║')
        self.get_logger().info('╠' + '═' * 58 + '╣')
        
        # Person 1 status
        if self.person1_found:
            msg = (f'✓ Person 1: Original location '
                   f'({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        elif self.person1_new_location:
            msg = (f'→ Person 1: MOVED to '
                   f'({self.person1_new_location["x"]}, {self.person1_new_location["y"]})')
        else:
            msg = '? Person 1: MOVED - new location unknown'
        self.get_logger().info('║  ' + msg.ljust(56) + '║')
        
        # Person 2 status
        if self.person2_found:
            msg = (f'✓ Person 2: Original location '
                   f'({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        elif self.person2_new_location:
            msg = (f'→ Person 2: MOVED to '
                   f'({self.person2_new_location["x"]}, {self.person2_new_location["y"]})')
        else:
            msg = '? Person 2: MOVED - new location unknown'
        self.get_logger().info('║  ' + msg.ljust(56) + '║')
        
        self.get_logger().info('╚' + '═' * 58 + '╝')
        self.get_logger().info('')


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args=None):
    """
    Main function - entry point for the node.
    
    This is called when you run the script or use 'ros2 run'.
    """
    # Initialize the ROS2 Python client library
    # This must be called before creating any nodes
    rclpy.init(args=args)
    
    # Create an instance of our controller node
    controller = MapAwareWarehouseController()
    
    try:
        # Spin the node - this keeps it running and processing callbacks
        # spin() blocks until the node is shut down
        rclpy.spin(controller)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        controller.get_logger().info('Shutting down...')
    finally:
        # Clean up
        controller.navigator.lifecycleShutdown()  # Properly shut down Nav2
        controller.destroy_node()                  # Destroy the node
        rclpy.shutdown()                          # Shut down ROS2


# This runs only if the script is executed directly (not imported)
if __name__ == '__main__':
    main()