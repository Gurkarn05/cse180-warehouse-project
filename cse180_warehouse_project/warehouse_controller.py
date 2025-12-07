#!/usr/bin/env python3
"""
===============================================================================
WAREHOUSE CONTROLLER - Complete Human Detection and Search System
===============================================================================

This ROS2 node performs the following tasks:
1. Navigate to Person 1 and Person 2's expected locations
2. Use TF listener to get the robot's current location reliably
3. Compare laser scans with the static map to detect dynamic objects (people)
4. If people are not at expected locations, search the entire warehouse
5. Reset robot position and map memory when re-run after cancellation

Key Features:
- TF2 listener for reliable robot pose (more reliable than AMCL topic)
- Map comparison to distinguish static obstacles from dynamic objects (people)
- Comprehensive search pattern for the entire warehouse
- Proper reset/re-initialization when re-running the controller
- Extensively commented for ROS2 beginners

Coordinates from your Gazebo setup:
- Person 1 (Standing): X=1.00m, Y=-1.00m, Yaw=1.57 rad
- Person 2 (Walking): X=-12.00m, Y=15.00m, Yaw=0.00 rad
- Robot Start: X=2.12m, Y=-21.3m, Yaw=1.57 rad
- Map: Resolution ~0.03m, Origin (-15.1, -25.0)

Author: ROS2 Tutorial
For: ROS2 Jazzy + Nav2 + Gazebo
"""

# =============================================================================
# IMPORTS - Each import is explained for beginners
# =============================================================================

# ROS2 Core Libraries
import rclpy                                    # ROS2 Python client library - the main interface for ROS2
from rclpy.node import Node                     # Base class that all ROS2 nodes inherit from
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # Quality of Service settings

# Message Types - These define the data structures for ROS2 communication
from geometry_msgs.msg import PoseStamped       # Position + orientation with timestamp and frame
from geometry_msgs.msg import PoseWithCovarianceStamped  # Pose with uncertainty (from AMCL)
from geometry_msgs.msg import Twist             # Linear and angular velocity commands
from sensor_msgs.msg import LaserScan           # Laser scanner data (array of distances)
from nav_msgs.msg import OccupancyGrid          # 2D map data (grid of occupied/free/unknown)
from std_srvs.srv import Empty                  # Empty service request/response

# TF2 - ROS2's coordinate transform system
# TF tracks how different coordinate frames relate to each other over time
from tf2_ros import Buffer, TransformListener   # Buffer stores transforms, Listener subscribes to /tf
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException  # TF errors

# Nav2 Simple Commander - High-level navigation interface
# This makes it easy to send navigation goals without manually managing action clients
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# Standard Python Libraries
import math                 # Mathematical functions (sin, cos, atan2, sqrt)
import numpy as np          # Numerical operations on arrays (essential for map/scan processing)
import time                 # Time delays and timestamps
import os                   # Operating system interface (for file operations)
import json                 # JSON encoding/decoding (for saving/loading state)


# =============================================================================
# CONFIGURATION CONSTANTS - Easy to modify parameters
# =============================================================================

# File path for saving/loading robot state between runs
STATE_FILE = '/tmp/warehouse_controller_state.json'

# Person locations from Gazebo (hardcoded from your screenshots)
PERSON1_EXPECTED = {'x': 1.00, 'y': -1.00, 'name': 'Person 1 - Standing'}
PERSON2_EXPECTED = {'x': -12.00, 'y': 15.00, 'name': 'Person 2 - Walking'}

# Robot's initial starting position (from your description)
ROBOT_START = {'x': 2.12, 'y': -21.3, 'yaw': 1.57}

# Detection parameters
APPROACH_DISTANCE = 2.0     # How far from person to stop for detection (meters)
DETECTION_RADIUS = 1.5      # How close a detected object must be to expected position
HUMAN_SIZE_MIN = 0.2        # Minimum width of human-like object (meters)
HUMAN_SIZE_MAX = 1.5        # Maximum width of human-like object (meters)
SCAN_TIMEOUT = 5.0          # How long to wait for fresh scan data (seconds)

# Map parameters (from your screenshot)
MAP_RESOLUTION = 0.03       # meters per pixel
MAP_ORIGIN_X = -15.1        # X coordinate of map origin
MAP_ORIGIN_Y = -25.0        # Y coordinate of map origin

# Search grid parameters (for warehouse search)
SEARCH_GRID_SPACING = 3.0   # Distance between search waypoints (meters)
SEARCH_AREA_MIN_X = -14.0   # Minimum X coordinate of search area
SEARCH_AREA_MAX_X = 12.0    # Maximum X coordinate of search area
SEARCH_AREA_MIN_Y = -23.0   # Minimum Y coordinate of search area
SEARCH_AREA_MAX_Y = 20.0    # Maximum Y coordinate of search area


# =============================================================================
# MAIN CONTROLLER CLASS
# =============================================================================

class WarehouseController(Node):
    """
    Complete warehouse human detection and search controller.
    
    This class inherits from Node, which is the base class for all ROS2 nodes.
    A node is a participant in the ROS2 graph - it can publish topics,
    subscribe to topics, provide services, and call actions.
    
    Key responsibilities:
    1. Subscribe to /map, /scan, and TF transforms
    2. Navigate to expected person locations
    3. Detect if people are at expected positions using map comparison
    4. Search warehouse if people have moved
    5. Reset properly when re-run
    """
    
    def __init__(self):
        """
        Initialize the warehouse controller node.
        
        This method is called once when the node is created. It sets up:
        - Subscribers for sensor data
        - TF listener for robot pose
        - Navigator for autonomous movement
        - State variables for tracking mission progress
        """
        # Initialize the Node with a unique name
        # This name appears in 'ros2 node list' and log messages
        super().__init__('warehouse_controller')
        
        self.get_logger().info('='*60)
        self.get_logger().info('WAREHOUSE CONTROLLER INITIALIZING')
        self.get_logger().info('='*60)
        
        # ---------------------------------------------------------------------
        # STEP 1: Initialize state variables
        # ---------------------------------------------------------------------
        # These track the mission progress and store received data
        
        # Store the static map from /map topic
        self.map_data = None            # The occupancy grid array
        self.map_info = None            # Metadata (resolution, origin, size)
        
        # Store laser scan data
        self.latest_scan = None         # Most recent LaserScan message
        self.scan_received_time = None  # When we last received a scan
        
        # Robot pose from AMCL (backup method)
        self.robot_pose = None          # PoseWithCovarianceStamped message
        self.last_known_pose = None     # Fallback pose if TF fails
        
        # Person tracking
        self.person1_expected = PERSON1_EXPECTED
        self.person2_expected = PERSON2_EXPECTED
        self.person1_found = False      # Is Person 1 at expected location?
        self.person2_found = False      # Is Person 2 at expected location?
        self.person1_new_location = None  # New location if moved
        self.person2_new_location = None  # New location if moved
        
        # Robot starting position
        self.robot_start = ROBOT_START
        
        # Mission flags
        self.mission_started = False    # Has the mission begun?
        self.mission_complete = False   # Is the mission finished?
        
        self.get_logger().info(f'Expected Person 1 at: ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        self.get_logger().info(f'Expected Person 2 at: ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        self.get_logger().info(f'Robot start position: ({self.robot_start["x"]}, {self.robot_start["y"]})')
        
        # ---------------------------------------------------------------------
        # STEP 2: Set up TF2 Buffer and Listener
        # ---------------------------------------------------------------------
        # TF2 (Transform Library 2) tracks coordinate frame transformations.
        # The 'map' frame is the world coordinate system.
        # The 'base_link' frame is attached to the robot.
        # By looking up the transform from 'map' to 'base_link', we get the
        # robot's position in world coordinates.
        #
        # WHY USE TF instead of AMCL topic?
        # - AMCL only publishes when the robot's pose changes significantly
        # - TF is updated continuously and is always available
        # - TF is the authoritative source for robot pose in Nav2
        
        self.tf_buffer = Buffer()       # Stores recent transforms (default: 10 seconds)
        self.tf_listener = TransformListener(self.tf_buffer, self)  # Subscribes to /tf and /tf_static
        self.get_logger().info('TF listener initialized')
        
        # ---------------------------------------------------------------------
        # STEP 3: Set up Quality of Service (QoS) profiles
        # ---------------------------------------------------------------------
        # QoS defines how messages are delivered between publishers and subscribers.
        # Different topics have different QoS requirements:
        #
        # - /map: Uses "transient local" durability - the map is published once
        #         and late subscribers should still receive it
        # - /scan: Uses "reliable" or "best effort" depending on sensor
        # - /amcl_pose: Uses default reliability
        
        # QoS for map topic (transient local = get last published message)
        map_qos = QoSProfile(
            depth=1,                                    # Only keep 1 message in queue
            reliability=ReliabilityPolicy.RELIABLE,     # Guaranteed delivery
            durability=DurabilityPolicy.TRANSIENT_LOCAL # New subscribers get last message
        )
        
        # QoS for sensor topics (best effort = low latency, may drop messages)
        sensor_qos = QoSProfile(
            depth=10,                                   # Keep up to 10 messages
            reliability=ReliabilityPolicy.BEST_EFFORT   # May drop messages for speed
        )
        
        # ---------------------------------------------------------------------
        # STEP 4: Create subscribers
        # ---------------------------------------------------------------------
        # Subscribers receive messages from topics published by other nodes.
        # Each subscriber has:
        # - Topic name (string)
        # - Message type (defines data structure)
        # - Callback function (called when message arrives)
        # - QoS profile (delivery settings)
        
        # Subscribe to the static map
        # The map is an OccupancyGrid: a 2D array where each cell is:
        # - 0 = free space
        # - 100 = occupied (wall/obstacle)
        # - -1 = unknown
        self.map_subscription = self.create_subscription(
            OccupancyGrid,              # Message type
            '/map',                     # Topic name
            self.map_callback,          # Function to call when message received
            map_qos                     # Quality of Service settings
        )
        self.get_logger().info('Subscribed to /map')
        
        # Subscribe to laser scan data
        # LaserScan contains an array of distance measurements at different angles
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )
        self.get_logger().info('Subscribed to /scan')
        
        # Subscribe to AMCL pose (backup for robot position)
        # AMCL (Adaptive Monte Carlo Localization) estimates robot pose
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10  # Queue depth
        )
        self.get_logger().info('Subscribed to /amcl_pose')
        
        # ---------------------------------------------------------------------
        # STEP 5: Initialize Nav2 BasicNavigator
        # ---------------------------------------------------------------------
        # BasicNavigator provides a simple Python API for Nav2 navigation.
        # It handles:
        # - Sending navigation goals
        # - Monitoring navigation progress
        # - Setting initial pose
        # - Clearing costmaps
        
        self.navigator = BasicNavigator()
        self.get_logger().info('Navigator initialized')
        
        # ---------------------------------------------------------------------
        # STEP 6: Check for previous state and handle reset
        # ---------------------------------------------------------------------
        # If this controller was run before and cancelled, we need to:
        # - Reset the robot to starting position
        # - Clear any cached state
        # - Re-initialize the localization system
        
        self.handle_reset_on_restart()
        
        # ---------------------------------------------------------------------
        # STEP 7: Create a timer to start the mission after initialization
        # ---------------------------------------------------------------------
        # We use a one-shot timer to delay mission start until:
        # - Map is received
        # - TF transforms are available
        # - Nav2 is ready
        
        self.startup_timer = self.create_timer(3.0, self.startup_check)
        self.get_logger().info('Waiting for data and Nav2 to be ready...')
    
    
    # =========================================================================
    # CALLBACK FUNCTIONS - Called when messages are received
    # =========================================================================
    
    def map_callback(self, msg):
        """
        Callback for /map topic.
        
        This is called whenever a new OccupancyGrid message is received.
        The map is used as "ground truth" to distinguish static obstacles
        (walls, shelves) from dynamic objects (people).
        
        Parameters:
            msg (OccupancyGrid): The map message containing:
                - info: MapMetaData (resolution, width, height, origin)
                - data: list of cell values (0=free, 100=occupied, -1=unknown)
        """
        # Store map metadata
        self.map_info = msg.info
        
        # Convert the flat list to a 2D numpy array for easier processing
        # The map is stored as a flat list in row-major order
        self.map_data = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width)
        )
        
        self.get_logger().info(
            f'Map received: {msg.info.width}x{msg.info.height} pixels, '
            f'resolution={msg.info.resolution:.4f} m/pixel'
        )
    
    
    def scan_callback(self, msg):
        """
        Callback for /scan topic.
        
        This is called whenever new laser scan data arrives.
        The laser scan contains distance measurements in a fan pattern
        around the robot.
        
        Parameters:
            msg (LaserScan): The scan message containing:
                - angle_min, angle_max: Angular range of the scan (radians)
                - angle_increment: Angular resolution (radians per ray)
                - range_min, range_max: Valid distance range (meters)
                - ranges: Array of distance measurements
        """
        self.latest_scan = msg
        self.scan_received_time = time.time()
    
    
    def pose_callback(self, msg):
        """
        Callback for /amcl_pose topic.
        
        AMCL publishes the robot's estimated pose in the map frame.
        This is a backup method - TF is preferred for getting robot pose.
        
        Parameters:
            msg (PoseWithCovarianceStamped): Pose with uncertainty estimate
        """
        self.robot_pose = msg.pose.pose
        # Also save as last known pose (useful as fallback)
        self.last_known_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'yaw': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }
    
    
    # =========================================================================
    # TF LISTENER FUNCTIONS - Get robot pose from transforms
    # =========================================================================
    
    def get_robot_pose_from_tf(self):
        """
        Get the robot's current pose using TF2 transforms.
        
        This is the PREFERRED method for getting robot pose because:
        1. TF is always available (unlike AMCL topic which only publishes on change)
        2. TF is the authoritative source used by Nav2
        3. TF provides the most up-to-date pose
        
        Returns:
            tuple: (x, y, yaw) in map frame, or None if transform not available
        
        How it works:
        1. Look up the transform from 'map' frame to 'base_link' frame
        2. Extract the translation (x, y position)
        3. Convert quaternion rotation to yaw angle
        """
        try:
            # lookup_transform(target_frame, source_frame, time)
            # - target_frame: 'map' (world coordinates)
            # - source_frame: 'base_link' (robot's body)
            # - time: rclpy.time.Time() means "latest available"
            # - timeout: How long to wait for the transform
            transform = self.tf_buffer.lookup_transform(
                'map',                      # Target frame
                'base_link',                # Source frame
                rclpy.time.Time(),          # Latest available time
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            
            # Extract position from the transform
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Convert quaternion to yaw angle
            yaw = self.quaternion_to_yaw(transform.transform.rotation)
            
            # Update last known pose
            self.last_known_pose = {'x': x, 'y': y, 'yaw': yaw}
            
            return (x, y, yaw)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Transform not available - log warning and return None
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None
    
    
    def wait_for_robot_pose(self, timeout=10.0):
        """
        Wait until we can get a valid robot pose.
        
        This function repeatedly tries to get the robot pose using multiple
        methods until one succeeds or timeout is reached.
        
        Parameters:
            timeout (float): Maximum time to wait in seconds
        
        Returns:
            tuple: (x, y, yaw) or None if timeout reached
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Method 1: Try TF (preferred)
            pose = self.get_robot_pose_from_tf()
            if pose is not None:
                return pose
            
            # Method 2: Try AMCL topic
            if self.robot_pose is not None:
                x = self.robot_pose.position.x
                y = self.robot_pose.position.y
                yaw = self.quaternion_to_yaw(self.robot_pose.orientation)
                return (x, y, yaw)
            
            # Method 3: Use last known pose
            if self.last_known_pose is not None:
                return (
                    self.last_known_pose['x'],
                    self.last_known_pose['y'],
                    self.last_known_pose['yaw']
                )
            
            # Spin once to process callbacks and wait
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Timeout reached - use starting position as fallback
        self.get_logger().warn('Timeout waiting for pose - using starting position')
        return (self.robot_start['x'], self.robot_start['y'], self.robot_start['yaw'])
    
    
    # =========================================================================
    # RESET AND STATE MANAGEMENT
    # =========================================================================
    
    def handle_reset_on_restart(self):
        """
        Handle reset when the controller is re-run after cancellation.
        
        This function:
        1. Checks if there's a saved state from a previous run
        2. If so, resets the robot to starting position
        3. Clears the map memory (costmaps)
        4. Saves new state file to indicate this run started
        
        WHY IS THIS NEEDED?
        When you cancel the controller (Ctrl+C) and re-run it:
        - The robot may be at a different position than the start
        - The costmaps may have obstacles marked from the previous run
        - AMCL may think the robot is somewhere else
        
        This ensures a clean restart every time.
        """
        self.get_logger().info('Checking for previous run state...')
        
        # Check if state file exists from previous run
        if os.path.exists(STATE_FILE):
            self.get_logger().info('Previous run detected - performing full reset')
            
            try:
                # Read previous state (for logging purposes)
                with open(STATE_FILE, 'r') as f:
                    previous_state = json.load(f)
                self.get_logger().info(f'Previous state: {previous_state}')
            except Exception as e:
                self.get_logger().warn(f'Could not read previous state: {e}')
            
            # Perform full reset
            self.reset_to_start_position()
        else:
            self.get_logger().info('Fresh start - no previous run detected')
        
        # Save current state (indicating this run has started)
        self.save_state({'status': 'running', 'start_time': time.time()})
    
    
    def reset_to_start_position(self):
        """
        Reset the robot to its starting position and clear map memory.
        
        This performs a comprehensive reset:
        1. Clear TF buffer (remove stale transforms)
        2. Reset cached poses
        3. Set initial pose to starting position (tells AMCL where we are)
        4. Wait for localization to stabilize
        5. Clear costmaps (removes any obstacles from previous run)
        6. Verify pose is correct
        
        This is called when re-running the controller after cancellation.
        """
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('RESETTING ROBOT TO START POSITION')
        self.get_logger().info('='*60)
        
        # Step 1: Clear TF buffer
        # Old transforms from previous run can cause confusion
        self.tf_buffer.clear()
        self.get_logger().info('  ✓ TF buffer cleared')
        
        # Step 2: Reset cached poses
        self.robot_pose = None
        self.last_known_pose = None
        self.get_logger().info('  ✓ Cached poses cleared')
        
        # Step 3: Reset person tracking
        self.person1_found = False
        self.person2_found = False
        self.person1_new_location = None
        self.person2_new_location = None
        self.get_logger().info('  ✓ Person tracking reset')
        
        # Step 4: Create initial pose message
        initial_pose = self.create_pose_stamped(
            self.robot_start['x'],
            self.robot_start['y'],
            self.robot_start['yaw']
        )
        
        # Step 5: Publish initial pose multiple times
        # AMCL uses a particle filter - publishing initial pose resets particles
        # Publishing multiple times helps ensure AMCL receives and processes it
        self.get_logger().info('  Publishing initial pose to AMCL...')
        for i in range(5):
            self.navigator.setInitialPose(initial_pose)
            time.sleep(0.3)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('  ✓ Initial pose published (5x)')
        
        # Step 6: Wait for AMCL to converge
        self.get_logger().info('  Waiting for localization to stabilize...')
        for i in range(50):  # 5 seconds of spinning
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if TF is available
            if i % 10 == 0:
                pose = self.get_robot_pose_from_tf()
                if pose:
                    x, y, _ = pose
                    error = math.sqrt(
                        (x - self.robot_start['x'])**2 +
                        (y - self.robot_start['y'])**2
                    )
                    self.get_logger().info(
                        f'    TF pose: ({x:.2f}, {y:.2f}), error from start: {error:.2f}m'
                    )
                    if error < 2.0:
                        break
        
        # Step 7: Clear costmaps
        # Costmaps store obstacles detected by sensors
        # After reset, they may have "hallucinated" obstacles from previous run
        self.get_logger().info('  Clearing costmaps (removing old obstacle memory)...')
        try:
            self.navigator.clearAllCostmaps()
            time.sleep(1.0)
            self.get_logger().info('  ✓ Costmaps cleared')
        except Exception as e:
            self.get_logger().warn(f'  Could not clear costmaps: {e}')
        
        # Step 8: Final spin to update everything
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Step 9: Verify final pose
        final_pose = self.wait_for_robot_pose(timeout=3.0)
        if final_pose:
            x, y, yaw = final_pose
            self.get_logger().info(f'  ✓ Final pose: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°)')
        
        self.get_logger().info('='*60)
        self.get_logger().info('RESET COMPLETE')
        self.get_logger().info('='*60)
        self.get_logger().info('')
    
    
    def save_state(self, state):
        """
        Save controller state to file.
        
        This allows the controller to detect if it was run before
        and needs to perform a reset.
        
        Parameters:
            state (dict): State information to save
        """
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.get_logger().warn(f'Could not save state: {e}')
    
    
    def cleanup_state_file(self):
        """
        Remove state file on clean exit.
        
        If the mission completes successfully, we remove the state file
        so the next run doesn't think it needs to reset.
        """
        try:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
                self.get_logger().info('State file cleaned up')
        except Exception as e:
            self.get_logger().warn(f'Could not remove state file: {e}')
    
    
    # =========================================================================
    # COORDINATE TRANSFORMATION FUNCTIONS
    # =========================================================================
    
    def quaternion_to_yaw(self, q):
        """
        Convert a quaternion orientation to yaw angle.
        
        Quaternions are 4D vectors (x, y, z, w) that represent 3D rotation.
        Yaw is the rotation around the Z axis (left-right turning).
        
        Parameters:
            q: Quaternion with x, y, z, w attributes
        
        Returns:
            float: Yaw angle in radians (-π to π)
        
        The formula uses the atan2 function to extract the yaw component
        from the quaternion representation.
        """
        # Standard formula for extracting yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    
    def world_to_map(self, world_x, world_y):
        """
        Convert world coordinates (meters) to map pixel coordinates.
        
        The map is a grid of pixels. To convert world coordinates to
        map pixels:
        1. Subtract the map origin (corner of the map in world coords)
        2. Divide by resolution (meters per pixel)
        
        Parameters:
            world_x (float): X position in world frame (meters)
            world_y (float): Y position in world frame (meters)
        
        Returns:
            tuple: (map_x, map_y) pixel coordinates, or None if outside map
        """
        if self.map_info is None:
            return None
        
        # Get map metadata
        origin_x = self.map_info.origin.position.x  # -15.1 from your setup
        origin_y = self.map_info.origin.position.y  # -25.0 from your setup
        resolution = self.map_info.resolution        # ~0.03 m/pixel
        
        # Convert world to map coordinates
        map_x = int((world_x - origin_x) / resolution)
        map_y = int((world_y - origin_y) / resolution)
        
        # Check if within map bounds
        if 0 <= map_x < self.map_info.width and 0 <= map_y < self.map_info.height:
            return (map_x, map_y)
        else:
            return None
    
    
    def map_to_world(self, map_x, map_y):
        """
        Convert map pixel coordinates to world coordinates (meters).
        
        Parameters:
            map_x (int): X pixel coordinate
            map_y (int): Y pixel coordinate
        
        Returns:
            tuple: (world_x, world_y) in meters
        """
        if self.map_info is None:
            return None
        
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        resolution = self.map_info.resolution
        
        # Convert map to world coordinates (center of pixel)
        world_x = origin_x + (map_x + 0.5) * resolution
        world_y = origin_y + (map_y + 0.5) * resolution
        
        return (world_x, world_y)
    
    
    # =========================================================================
    # NAVIGATION FUNCTIONS
    # =========================================================================
    
    def create_pose_stamped(self, x, y, yaw):
        """
        Create a PoseStamped message for navigation.
        
        PoseStamped contains:
        - Header with timestamp and coordinate frame
        - Pose with position (x, y, z) and orientation (quaternion)
        
        Parameters:
            x (float): X position in meters
            y (float): Y position in meters
            yaw (float): Heading angle in radians
        
        Returns:
            PoseStamped: Message ready to send to Nav2
        """
        pose = PoseStamped()
        
        # Header - specifies when and in what frame
        pose.header.frame_id = 'map'        # World coordinate frame
        pose.header.stamp = self.get_clock().now().to_msg()
        
        # Position
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0          # 2D navigation - z is always 0
        
        # Orientation - convert yaw to quaternion
        # For 2D navigation, only yaw matters (rotation around Z axis)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        return pose
    
    
    def navigate_to(self, x, y, yaw=0.0):
        """
        Navigate the robot to a specified position.
        
        This uses Nav2's navigation stack to:
        1. Plan a path from current position to goal
        2. Execute the path while avoiding obstacles
        3. Return True if goal reached, False if failed
        
        Parameters:
            x (float): Target X position in meters
            y (float): Target Y position in meters
            yaw (float): Target heading in radians (default: facing +X)
        
        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        self.get_logger().info(f'Navigating to ({x:.2f}, {y:.2f})...')
        
        # Create goal pose
        goal_pose = self.create_pose_stamped(x, y, yaw)
        
        # Send goal to Nav2
        self.navigator.goToPose(goal_pose)
        
        # Wait for navigation to complete
        while not self.navigator.isTaskComplete():
            # Check navigation feedback (optional - could log progress)
            feedback = self.navigator.getFeedback()
            
            # Process callbacks while waiting
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Get result
        result = self.navigator.getResult()
        
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('  ✓ Navigation succeeded!')
            return True
        elif result == TaskResult.CANCELED:
            self.get_logger().warn('  Navigation was canceled')
            return False
        elif result == TaskResult.FAILED:
            self.get_logger().warn('  Navigation failed!')
            return False
        else:
            self.get_logger().warn(f'  Navigation result: {result}')
            return False
    
    
    def calculate_observation_point(self, person_x, person_y):
        """
        Calculate a good position to observe a person from.
        
        We don't want to navigate TO the person - we want to stop
        a safe distance away where we can observe them with the laser.
        
        The observation point is calculated as:
        1. Get robot's current position
        2. Calculate direction from robot to person
        3. Stop APPROACH_DISTANCE meters before the person
        
        This ensures the robot approaches from its current side,
        avoiding obstacles behind the person.
        
        Parameters:
            person_x (float): Person's X position
            person_y (float): Person's Y position
        
        Returns:
            tuple: (obs_x, obs_y, obs_yaw) - observation position and heading
        """
        # Get current robot position
        robot_pose = self.wait_for_robot_pose(timeout=5.0)
        if robot_pose is None:
            robot_x, robot_y = self.robot_start['x'], self.robot_start['y']
        else:
            robot_x, robot_y, _ = robot_pose
        
        # Calculate direction vector from robot to person
        dx = person_x - robot_x
        dy = person_y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # If already close enough, stay here
        if distance < APPROACH_DISTANCE:
            # Just face the person
            yaw = math.atan2(dy, dx)
            return (robot_x, robot_y, yaw)
        
        # Normalize direction vector
        dx /= distance
        dy /= distance
        
        # Calculate observation point (APPROACH_DISTANCE before person)
        obs_x = person_x - dx * APPROACH_DISTANCE
        obs_y = person_y - dy * APPROACH_DISTANCE
        
        # Face toward the person
        obs_yaw = math.atan2(dy, dx)
        
        return (obs_x, obs_y, obs_yaw)
    
    
    # =========================================================================
    # DETECTION FUNCTIONS - Using map comparison
    # =========================================================================
    
    def get_fresh_scan(self, timeout=SCAN_TIMEOUT):
        """
        Wait for a fresh laser scan.
        
        We want the LATEST scan data, not an old cached one.
        This function waits until a new scan arrives after calling it.
        
        Parameters:
            timeout (float): Maximum time to wait in seconds
        
        Returns:
            LaserScan: Fresh scan message, or None if timeout
        """
        # Mark current time
        start_time = time.time()
        wait_start = time.time()
        
        # Clear old scan
        self.scan_received_time = None
        
        # Wait for new scan
        while (time.time() - wait_start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if we got a new scan
            if self.scan_received_time is not None and self.scan_received_time > start_time:
                return self.latest_scan
        
        self.get_logger().warn('Timeout waiting for fresh scan')
        return self.latest_scan  # Return old scan as fallback
    
    
    def scan_to_points(self, scan, robot_x, robot_y, robot_yaw):
        """
        Convert laser scan to world coordinates.
        
        The laser scan gives distances at different angles relative to
        the robot. This function converts each reading to world coordinates.
        
        Parameters:
            scan (LaserScan): The laser scan message
            robot_x, robot_y (float): Robot position in world frame
            robot_yaw (float): Robot heading in world frame
        
        Returns:
            list: List of (world_x, world_y) points for valid readings
        """
        points = []
        
        for i, distance in enumerate(scan.ranges):
            # Skip invalid readings (inf, nan, out of range)
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue
            
            # Calculate angle of this ray in world frame
            # angle_min is the starting angle (relative to robot front)
            ray_angle_robot = scan.angle_min + i * scan.angle_increment
            ray_angle_world = robot_yaw + ray_angle_robot
            
            # Calculate world coordinates of the detected point
            world_x = robot_x + distance * math.cos(ray_angle_world)
            world_y = robot_y + distance * math.sin(ray_angle_world)
            
            points.append((world_x, world_y, distance))
        
        return points
    
    
    def is_point_in_static_map(self, world_x, world_y, tolerance=2):
        """
        Check if a point corresponds to a static obstacle in the map.
        
        This is the key to distinguishing people from walls/shelves:
        - If a point is marked as occupied in the static map -> it's a wall
        - If a point is NOT in the static map -> it's a dynamic object (person!)
        
        Parameters:
            world_x, world_y (float): Point in world coordinates
            tolerance (int): How many nearby pixels to check
        
        Returns:
            bool: True if point is a static obstacle, False if dynamic
        """
        if self.map_data is None:
            return False
        
        # Convert to map coordinates
        map_coords = self.world_to_map(world_x, world_y)
        if map_coords is None:
            return False  # Outside map bounds
        
        map_x, map_y = map_coords
        
        # Check this pixel and neighbors (tolerance allows for slight misalignment)
        for dx in range(-tolerance, tolerance+1):
            for dy in range(-tolerance, tolerance+1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                # Bounds check
                if 0 <= check_x < self.map_info.width and 0 <= check_y < self.map_info.height:
                    # Check if occupied in static map (100 = occupied)
                    if self.map_data[check_y, check_x] > 50:
                        return True
        
        return False
    
    
    def detect_person_at_location(self, expected_x, expected_y, person_name):
        """
        Detect if a person is at the expected location.
        
        This is the main detection function. It:
        1. Gets fresh laser scan data
        2. Converts scan to world coordinates
        3. Filters out static obstacles using the map
        4. Looks for dynamic objects near the expected position
        5. Determines if the dynamic object is human-sized
        
        Parameters:
            expected_x, expected_y (float): Expected person position
            person_name (str): Name for logging
        
        Returns:
            tuple: (found, location) where found is bool and location is (x,y) or None
        """
        self.get_logger().info(f'  Detecting {person_name} at ({expected_x}, {expected_y})...')
        
        # Get robot pose
        robot_pose = self.wait_for_robot_pose(timeout=5.0)
        if robot_pose is None:
            self.get_logger().warn('  Could not get robot pose!')
            return (False, None)
        
        robot_x, robot_y, robot_yaw = robot_pose
        self.get_logger().info(f'  Robot at: ({robot_x:.2f}, {robot_y:.2f})')
        
        # Get fresh scan
        scan = self.get_fresh_scan()
        if scan is None:
            self.get_logger().warn('  No scan data available!')
            return (False, None)
        
        # Convert scan to world points
        points = self.scan_to_points(scan, robot_x, robot_y, robot_yaw)
        self.get_logger().info(f'  Got {len(points)} scan points')
        
        # Filter out static obstacles (keep only dynamic objects)
        dynamic_points = []
        for wx, wy, dist in points:
            if not self.is_point_in_static_map(wx, wy):
                dynamic_points.append((wx, wy, dist))
        
        self.get_logger().info(f'  Found {len(dynamic_points)} dynamic (non-map) points')
        
        # Look for dynamic points near expected location
        nearby_points = []
        for wx, wy, dist in dynamic_points:
            distance_to_expected = math.sqrt(
                (wx - expected_x)**2 + (wy - expected_y)**2
            )
            if distance_to_expected < DETECTION_RADIUS * 2:
                nearby_points.append((wx, wy))
        
        self.get_logger().info(f'  {len(nearby_points)} dynamic points near expected location')
        
        if len(nearby_points) >= 3:  # Require multiple points for a person
            # Calculate centroid of nearby points
            centroid_x = sum(p[0] for p in nearby_points) / len(nearby_points)
            centroid_y = sum(p[1] for p in nearby_points) / len(nearby_points)
            
            # Check if centroid is close to expected position
            distance_to_expected = math.sqrt(
                (centroid_x - expected_x)**2 + (centroid_y - expected_y)**2
            )
            
            self.get_logger().info(
                f'  Detected object at ({centroid_x:.2f}, {centroid_y:.2f}), '
                f'{distance_to_expected:.2f}m from expected'
            )
            
            if distance_to_expected < DETECTION_RADIUS:
                self.get_logger().info(f'  ✓ {person_name} FOUND at expected location!')
                return (True, (centroid_x, centroid_y))
            else:
                self.get_logger().info(f'  Object detected but too far from expected position')
                return (False, (centroid_x, centroid_y))
        else:
            self.get_logger().info(f'  ✗ No person detected at expected location')
            return (False, None)
    
    
    # =========================================================================
    # WAREHOUSE SEARCH FUNCTIONS
    # =========================================================================
    
    def generate_search_waypoints(self):
        """
        Generate a grid of waypoints to search the entire warehouse.
        
        This creates a lawn-mower pattern of waypoints that covers
        the searchable area of the warehouse.
        
        Returns:
            list: List of (x, y) waypoint positions
        """
        waypoints = []
        
        # Generate grid
        y = SEARCH_AREA_MIN_Y
        row = 0
        while y <= SEARCH_AREA_MAX_Y:
            if row % 2 == 0:
                # Left to right
                x = SEARCH_AREA_MIN_X
                while x <= SEARCH_AREA_MAX_X:
                    waypoints.append((x, y))
                    x += SEARCH_GRID_SPACING
            else:
                # Right to left (creates zigzag pattern)
                x = SEARCH_AREA_MAX_X
                while x >= SEARCH_AREA_MIN_X:
                    waypoints.append((x, y))
                    x -= SEARCH_GRID_SPACING
            
            y += SEARCH_GRID_SPACING
            row += 1
        
        return waypoints
    
    
    def search_for_moved_people(self):
        """
        Search the entire warehouse for people who have moved.
        
        This function:
        1. Generates search waypoints
        2. Navigates to each waypoint
        3. Scans for dynamic objects at each location
        4. Reports any people found
        
        Returns:
            list: List of detected person locations [(x, y), ...]
        """
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('SEARCHING WAREHOUSE FOR MOVED PEOPLE')
        self.get_logger().info('='*60)
        
        # Generate search waypoints
        waypoints = self.generate_search_waypoints()
        self.get_logger().info(f'Generated {len(waypoints)} search waypoints')
        
        # Track all detected people
        detected_people = []
        
        # Visit each waypoint
        for i, (wx, wy) in enumerate(waypoints):
            self.get_logger().info(f'\nSearch waypoint {i+1}/{len(waypoints)}: ({wx:.1f}, {wy:.1f})')
            
            # Try to navigate to waypoint
            success = self.navigate_to(wx, wy)
            if not success:
                self.get_logger().warn(f'  Could not reach waypoint, skipping')
                continue
            
            # Wait a moment for robot to settle
            time.sleep(0.5)
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # Scan for people
            people_here = self.scan_for_people_here()
            
            for px, py in people_here:
                # Check if this is a new detection (not already found)
                is_new = True
                for prev_x, prev_y in detected_people:
                    if math.sqrt((px-prev_x)**2 + (py-prev_y)**2) < 1.0:
                        is_new = False
                        break
                
                if is_new:
                    self.get_logger().info(f'  ★ NEW PERSON FOUND at ({px:.2f}, {py:.2f})')
                    detected_people.append((px, py))
        
        self.get_logger().info('')
        self.get_logger().info(f'Search complete. Found {len(detected_people)} people.')
        return detected_people
    
    
    def scan_for_people_here(self):
        """
        Scan for people at the current location.
        
        Similar to detect_person_at_location but looks for ANY
        human-sized dynamic objects, not just at a specific position.
        
        Returns:
            list: List of (x, y) positions of detected people
        """
        # Get robot pose
        robot_pose = self.wait_for_robot_pose(timeout=3.0)
        if robot_pose is None:
            return []
        
        robot_x, robot_y, robot_yaw = robot_pose
        
        # Get fresh scan
        scan = self.get_fresh_scan(timeout=2.0)
        if scan is None:
            return []
        
        # Convert scan to world points
        points = self.scan_to_points(scan, robot_x, robot_y, robot_yaw)
        
        # Filter out static obstacles
        dynamic_points = []
        for wx, wy, dist in points:
            if not self.is_point_in_static_map(wx, wy):
                # Only consider points within reasonable distance
                if dist < 10.0:
                    dynamic_points.append((wx, wy))
        
        if len(dynamic_points) < 3:
            return []
        
        # Cluster dynamic points to find people
        people = self.cluster_points(dynamic_points)
        
        return people
    
    
    def cluster_points(self, points, cluster_distance=0.5):
        """
        Cluster nearby points to identify distinct objects.
        
        Points that are close together likely belong to the same object.
        This simple clustering groups points within cluster_distance.
        
        Parameters:
            points (list): List of (x, y) points
            cluster_distance (float): Maximum distance between cluster members
        
        Returns:
            list: List of cluster centroids (x, y)
        """
        if len(points) == 0:
            return []
        
        # Simple clustering: assign points to clusters
        clusters = []
        used = [False] * len(points)
        
        for i, (x1, y1) in enumerate(points):
            if used[i]:
                continue
            
            # Start new cluster
            cluster = [(x1, y1)]
            used[i] = True
            
            # Find all nearby points
            for j, (x2, y2) in enumerate(points):
                if used[j]:
                    continue
                
                # Check distance to any point in cluster
                for cx, cy in cluster:
                    if math.sqrt((x2-cx)**2 + (y2-cy)**2) < cluster_distance:
                        cluster.append((x2, y2))
                        used[j] = True
                        break
            
            # Only keep clusters that are human-sized
            if len(cluster) >= 3:
                centroid_x = sum(p[0] for p in cluster) / len(cluster)
                centroid_y = sum(p[1] for p in cluster) / len(cluster)
                
                # Check cluster width (rough estimate)
                width = max(p[0] for p in cluster) - min(p[0] for p in cluster)
                depth = max(p[1] for p in cluster) - min(p[1] for p in cluster)
                size = max(width, depth)
                
                if HUMAN_SIZE_MIN < size < HUMAN_SIZE_MAX:
                    clusters.append((centroid_x, centroid_y))
        
        return clusters
    
    
    # =========================================================================
    # MISSION CONTROL
    # =========================================================================
    
    def startup_check(self):
        """
        Check if we're ready to start the mission.
        
        This is called by the startup timer. It verifies:
        1. Map has been received
        2. TF transforms are available
        3. Nav2 is ready
        
        Once all conditions are met, it starts the mission.
        """
        # Cancel the timer after first successful check
        self.startup_timer.cancel()
        
        self.get_logger().info('Checking startup conditions...')
        
        # Wait for map
        if self.map_data is None:
            self.get_logger().info('  Waiting for map...')
            self.startup_timer = self.create_timer(1.0, self.startup_check)
            return
        
        # Wait for TF
        pose = self.get_robot_pose_from_tf()
        if pose is None:
            self.get_logger().info('  Waiting for TF transforms...')
            self.startup_timer = self.create_timer(1.0, self.startup_check)
            return
        
        # Wait for Nav2
        self.get_logger().info('  Waiting for Nav2 to activate...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('  Nav2 is ready!')
        
        # All ready - start mission
        self.get_logger().info('')
        self.start_mission()
    
    
    def start_mission(self):
        """
        Main mission logic.
        
        This function:
        1. Checks Person 1's expected location
        2. Checks Person 2's expected location
        3. If either person moved, searches the warehouse
        4. Reports final results
        """
        if self.mission_started:
            return
        
        self.mission_started = True
        
        self.get_logger().info('='*60)
        self.get_logger().info('STARTING HUMAN DETECTION MISSION')
        self.get_logger().info('='*60)
        
        # ---------------------------------------------------------------------
        # PHASE 1: Check Person 1
        # ---------------------------------------------------------------------
        self.get_logger().info('')
        self.get_logger().info('-'*60)
        self.get_logger().info('PHASE 1: Checking Person 1')
        self.get_logger().info('-'*60)
        
        # Calculate observation point
        obs_x, obs_y, obs_yaw = self.calculate_observation_point(
            self.person1_expected['x'],
            self.person1_expected['y']
        )
        self.get_logger().info(f'Observation point: ({obs_x:.2f}, {obs_y:.2f})')
        
        # Navigate to observation point
        if self.navigate_to(obs_x, obs_y, obs_yaw):
            # Wait for robot to settle
            time.sleep(1.0)
            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # Detect person
            found, location = self.detect_person_at_location(
                self.person1_expected['x'],
                self.person1_expected['y'],
                'Person 1'
            )
            
            self.person1_found = found
            if not found and location is not None:
                self.person1_new_location = {'x': location[0], 'y': location[1]}
        else:
            self.get_logger().warn('Could not navigate to Person 1 observation point')
        
        # ---------------------------------------------------------------------
        # PHASE 2: Check Person 2
        # ---------------------------------------------------------------------
        self.get_logger().info('')
        self.get_logger().info('-'*60)
        self.get_logger().info('PHASE 2: Checking Person 2')
        self.get_logger().info('-'*60)
        
        # Calculate observation point
        obs_x, obs_y, obs_yaw = self.calculate_observation_point(
            self.person2_expected['x'],
            self.person2_expected['y']
        )
        self.get_logger().info(f'Observation point: ({obs_x:.2f}, {obs_y:.2f})')
        
        # Navigate to observation point
        if self.navigate_to(obs_x, obs_y, obs_yaw):
            # Wait for robot to settle
            time.sleep(1.0)
            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # Detect person
            found, location = self.detect_person_at_location(
                self.person2_expected['x'],
                self.person2_expected['y'],
                'Person 2'
            )
            
            self.person2_found = found
            if not found and location is not None:
                self.person2_new_location = {'x': location[0], 'y': location[1]}
        else:
            self.get_logger().warn('Could not navigate to Person 2 observation point')
        
        # ---------------------------------------------------------------------
        # PHASE 3: Search for moved people (if needed)
        # ---------------------------------------------------------------------
        people_to_find = []
        if not self.person1_found:
            people_to_find.append('Person 1')
        if not self.person2_found:
            people_to_find.append('Person 2')
        
        if len(people_to_find) > 0:
            self.get_logger().info('')
            self.get_logger().info('-'*60)
            self.get_logger().info(f'PHASE 3: Searching for {", ".join(people_to_find)}')
            self.get_logger().info('-'*60)
            
            detected_people = self.search_for_moved_people()
            
            # Try to assign detected people to missing persons
            for i, (px, py) in enumerate(detected_people):
                if not self.person1_found and self.person1_new_location is None:
                    self.person1_new_location = {'x': px, 'y': py}
                    self.get_logger().info(f'Assigned detection {i+1} to Person 1')
                elif not self.person2_found and self.person2_new_location is None:
                    self.person2_new_location = {'x': px, 'y': py}
                    self.get_logger().info(f'Assigned detection {i+1} to Person 2')
        
        # ---------------------------------------------------------------------
        # PHASE 4: Return to start and report
        # ---------------------------------------------------------------------
        self.get_logger().info('')
        self.get_logger().info('-'*60)
        self.get_logger().info('PHASE 4: Returning to start position')
        self.get_logger().info('-'*60)
        
        self.navigate_to(
            self.robot_start['x'],
            self.robot_start['y'],
            self.robot_start['yaw']
        )
        
        # Print final report
        self.print_mission_report()
        
        # Clean up state file (mission completed successfully)
        self.mission_complete = True
        self.cleanup_state_file()
        
        self.get_logger().info('')
        self.get_logger().info('Mission complete! You can now Ctrl+C to exit.')
    
    
    def print_mission_report(self):
        """
        Print a formatted summary of mission results.
        """
        self.get_logger().info('')
        self.get_logger().info('╔' + '═'*58 + '╗')
        self.get_logger().info('║' + '  FINAL MISSION REPORT'.center(58) + '║')
        self.get_logger().info('╠' + '═'*58 + '╣')
        
        # Person 1 status
        if self.person1_found:
            msg = f'✓ Person 1: At expected location ({self.person1_expected["x"]}, {self.person1_expected["y"]})'
        elif self.person1_new_location:
            msg = f'→ Person 1: MOVED to ({self.person1_new_location["x"]:.2f}, {self.person1_new_location["y"]:.2f})'
        else:
            msg = '? Person 1: MOVED - new location unknown'
        self.get_logger().info('║  ' + msg.ljust(56) + '║')
        
        # Person 2 status
        if self.person2_found:
            msg = f'✓ Person 2: At expected location ({self.person2_expected["x"]}, {self.person2_expected["y"]})'
        elif self.person2_new_location:
            msg = f'→ Person 2: MOVED to ({self.person2_new_location["x"]:.2f}, {self.person2_new_location["y"]:.2f})'
        else:
            msg = '? Person 2: MOVED - new location unknown'
        self.get_logger().info('║  ' + msg.ljust(56) + '║')
        
        self.get_logger().info('╚' + '═'*58 + '╝')
        self.get_logger().info('')


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(args=None):
    """
    Main function - entry point for the node.
    
    This function:
    1. Initializes ROS2
    2. Creates the WarehouseController node
    3. Spins the node (processes callbacks until shutdown)
    4. Cleans up on exit
    """
    # Initialize ROS2 client library
    rclpy.init(args=args)
    
    # Create the controller node
    controller = WarehouseController()
    
    try:
        # Spin the node - this blocks and processes callbacks
        # The mission runs via timers and callbacks
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        controller.get_logger().info('')
        controller.get_logger().info('Keyboard interrupt - shutting down...')
        
        # If mission wasn't complete, state file remains for next run
        if not controller.mission_complete:
            controller.get_logger().info('Mission was interrupted - will reset on next run')
        
    finally:
        # Clean shutdown
        try:
            controller.navigator.lifecycleShutdown()
        except:
            pass
        
        controller.destroy_node()
        rclpy.shutdown()
        
        print('\nWarehouse Controller exited.')


# This allows running the script directly with: python3 warehouse_controller.py
if __name__ == '__main__':
    main()