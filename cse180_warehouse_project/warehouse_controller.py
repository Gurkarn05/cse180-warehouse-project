#!/usr/bin/env python3
"""
Warehouse Human Detection Controller (VERSION 3 - SMART APPROACH)
==================================================================

KEY FIX: Robot now finds an OPEN SPACE approach point around the person,
instead of blindly calculating a point that might be behind a shelf/wall.

The robot checks multiple directions around the person and picks the
first one that is:
1. Not blocked by obstacles in the map
2. Navigable by the robot

Author: ROS2 Tutorial
For: ROS2 Jazzy + Nav2 + Gazebo
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import math
import numpy as np
import time


class WarehouseController(Node):
    """
    Warehouse human detection with SMART approach point selection.
    """
    
    def __init__(self):
        super().__init__('warehouse_controller')
        
        self.shutting_down = False
        
        # =====================================================================
        # CONFIGURATION
        # =====================================================================
        
        self.person1_expected = {'x': 1.00, 'y': -1.00, 'name': 'Person 1'}
        self.person2_expected = {'x': -12.00, 'y': 15.00, 'name': 'Person 2'}
        
        self.robot_start = {'x': 2.12, 'y': -21.3, 'yaw': 1.57}
        
        self.detection_config = {
            # Approach settings - INCREASED distance for safety
            'approach_distance': 3.0,       # Stay 3m away from person
            'min_approach_distance': 2.0,   # Minimum if 3m is blocked
            
            # Angles to try around the person (degrees from +X axis)
            # Starts with common open directions, then tries all around
            'approach_angles_deg': [
                180,    # Behind (relative to typical shelf layout)
                -90,    # Left
                90,     # Right  
                -135,   # Back-left
                135,    # Back-right
                -45,    # Front-left
                45,     # Front-right
                0,      # In front
            ],
            
            # Laser/detection settings
            'scan_range_min': 0.3,
            'scan_range_max': 5.0,
            'human_width_min': 0.10,
            'human_width_max': 1.2,
            'min_cluster_points': 3,
            'max_cluster_points': 100,
            'cluster_gap_threshold': 8,
            'map_obstacle_threshold': 50,
            'dynamic_object_buffer': 0.15,
            'num_scans_to_check': 5,
            'detection_threshold': 2,
        }
        
        # =====================================================================
        # STATE
        # =====================================================================
        
        self.person1_found = None
        self.person2_found = None
        self.map_data = None
        self.map_info = None
        self.latest_scan = None
        self.robot_pose = None
        self.map_received = False
        self.mission_started = False
        
        # =====================================================================
        # TF2
        # =====================================================================
        
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # =====================================================================
        # SUBSCRIBERS
        # =====================================================================
        
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        
        # =====================================================================
        # NAVIGATOR
        # =====================================================================
        
        self.navigator = BasicNavigator()
        
        # =====================================================================
        # STARTUP
        # =====================================================================
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('  WAREHOUSE CONTROLLER v3 - SMART APPROACH')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Person 1: ({self.person1_expected["x"]}, {self.person1_expected["y"]})')
        self.get_logger().info(f'  Person 2: ({self.person2_expected["x"]}, {self.person2_expected["y"]})')
        self.get_logger().info('=' * 60)
        
        self.create_timer(3.0, self.start_mission_once)
    
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        if not self.map_received:
            self.map_received = True
            self.get_logger().info(f'✓ Map received: {msg.info.width}x{msg.info.height}')
    
    def scan_callback(self, msg):
        self.latest_scan = msg
    
    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose
    
    
    # =========================================================================
    # LOCALIZATION
    # =========================================================================
    
    def reset_localization(self):
        """Reset AMCL localization."""
        self.get_logger().info('Resetting localization...')
        
        initial_pose = self.create_pose_stamped(
            self.robot_start['x'],
            self.robot_start['y'],
            self.robot_start['yaw']
        )
        
        for _ in range(5):
            self.navigator.setInitialPose(initial_pose)
            time.sleep(0.3)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info('  Waiting for AMCL...')
        for i in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if i % 10 == 0:
                pose = self.get_pose_from_tf()
                if pose:
                    self.get_logger().info(f'  ✓ TF ready: ({pose[0]:.2f}, {pose[1]:.2f})')
                    break
        
        try:
            self.navigator.clearAllCostmaps()
            time.sleep(1.0)
            self.get_logger().info('  ✓ Costmaps cleared')
        except:
            pass
        
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info('  ✓ Localization ready')
    
    
    def get_pose_from_tf(self):
        """Get robot pose from TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            return (x, y, yaw)
        except:
            return None
    
    
    def get_current_pose(self):
        """Get pose with fallbacks."""
        pose = self.get_pose_from_tf()
        if pose:
            return pose
        
        if self.robot_pose:
            q = self.robot_pose.orientation
            yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            return (self.robot_pose.position.x, self.robot_pose.position.y, yaw)
        
        return (self.robot_start['x'], self.robot_start['y'], self.robot_start['yaw'])
    
    
    def wait_for_pose(self, timeout=10.0):
        """Wait for valid pose."""
        start = time.time()
        while (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.get_pose_from_tf() or self.robot_pose:
                return True
        return False
    
    
    # =========================================================================
    # MAP FUNCTIONS
    # =========================================================================
    
    def world_to_map(self, wx, wy):
        """Convert world coords to map pixels."""
        if not self.map_info:
            return None
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        res = self.map_info.resolution
        mx = int((wx - ox) / res)
        my = int((wy - oy) / res)
        if 0 <= mx < self.map_info.width and 0 <= my < self.map_info.height:
            return (mx, my)
        return None
    
    
    def is_occupied(self, wx, wy):
        """Check if world position is occupied."""
        if self.map_data is None:
            return False
        coords = self.world_to_map(wx, wy)
        if coords is None:
            return True  # Outside map = occupied
        mx, my = coords
        return self.map_data[my, mx] > self.detection_config['map_obstacle_threshold']
    
    
    def is_area_clear(self, wx, wy, radius=0.5):
        """
        Check if an area around a point is clear of obstacles.
        
        This is KEY for finding good approach points!
        """
        if self.map_data is None:
            return True
        
        # Check grid of points
        step = 0.15  # Check every 15cm
        for dx in np.arange(-radius, radius + step, step):
            for dy in np.arange(-radius, radius + step, step):
                if dx*dx + dy*dy <= radius*radius:  # Circle check
                    if self.is_occupied(wx + dx, wy + dy):
                        return False
        return True
    
    
    def is_path_clear(self, x1, y1, x2, y2, step=0.2):
        """
        Check if straight line path between two points is clear.
        
        This helps verify robot can actually reach the approach point.
        """
        if self.map_data is None:
            return True
        
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if dist < step:
            return self.is_area_clear(x2, y2, 0.3)
        
        steps = int(dist / step) + 1
        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if self.is_occupied(x, y):
                return False
        return True
    
    
    def is_near_static_obstacle(self, wx, wy):
        """Check if near static obstacle."""
        buf = self.detection_config['dynamic_object_buffer']
        for dx in [-buf, 0, buf]:
            for dy in [-buf, 0, buf]:
                if self.is_occupied(wx + dx, wy + dy):
                    return True
        return False
    
    
    # =========================================================================
    # SMART APPROACH POINT FINDER (THE KEY FIX!)
    # =========================================================================
    
    def find_clear_approach_point(self, person_x, person_y):
        """
        Find an approach point around the person that is in CLEAR SPACE.
        
        This is the KEY FIX for the "going to wall/shelf" problem!
        
        Strategy:
        1. Try multiple angles around the person
        2. For each angle, check if that position is:
           - Not occupied in the map
           - Has clear area around it (robot footprint)
           - Has relatively clear path from current position
        3. Return the first valid point found
        
        Returns:
            tuple: (x, y, yaw) for approach point, or None
        """
        robot_x, robot_y, _ = self.get_current_pose()
        
        approach_dist = self.detection_config['approach_distance']
        min_dist = self.detection_config['min_approach_distance']
        angles_deg = self.detection_config['approach_angles_deg']
        
        self.get_logger().info(f'  Finding clear approach to ({person_x}, {person_y})...')
        
        # Try each angle
        for angle_deg in angles_deg:
            angle_rad = math.radians(angle_deg)
            
            # Try different distances (start far, move closer if blocked)
            for dist in [approach_dist, min_dist + 0.5, min_dist]:
                # Calculate candidate point
                cand_x = person_x + dist * math.cos(angle_rad)
                cand_y = person_y + dist * math.sin(angle_rad)
                
                # Check 1: Is the point itself clear?
                if not self.is_area_clear(cand_x, cand_y, radius=0.4):
                    self.get_logger().debug(
                        f'    {angle_deg}° @ {dist}m: Point blocked')
                    continue
                
                # Check 2: Is there a reasonably clear path?
                # (Nav2 will do proper planning, this is just a quick check)
                if not self.is_path_clear(robot_x, robot_y, cand_x, cand_y, step=0.5):
                    self.get_logger().debug(
                        f'    {angle_deg}° @ {dist}m: Path blocked')
                    continue
                
                # Found a valid point!
                # Calculate yaw to face the person
                face_yaw = math.atan2(person_y - cand_y, person_x - cand_x)
                
                self.get_logger().info(
                    f'  ✓ Found clear approach: ({cand_x:.2f}, {cand_y:.2f})')
                self.get_logger().info(
                    f'    Angle: {angle_deg}°, Distance: {dist}m')
                
                return (cand_x, cand_y, face_yaw)
        
        # No clear point found - return a fallback
        self.get_logger().warn('  ⚠ No clear approach found, using fallback')
        
        # Fallback: try to get as close as possible from robot's direction
        dx = robot_x - person_x
        dy = robot_y - person_y
        d = math.sqrt(dx*dx + dy*dy)
        if d > 0:
            dx, dy = dx/d, dy/d
        
        fall_x = person_x + dx * approach_dist
        fall_y = person_y + dy * approach_dist
        fall_yaw = math.atan2(-dy, -dx)
        
        return (fall_x, fall_y, fall_yaw)
    
    
    # =========================================================================
    # HUMAN DETECTION
    # =========================================================================
    
    def detect_humans(self):
        """Detect humans by finding dynamic objects."""
        if not self.latest_scan or self.map_data is None:
            return []
        
        robot_x, robot_y, robot_yaw = self.get_current_pose()
        scan = self.latest_scan
        ranges = np.array(scan.ranges)
        
        cfg = self.detection_config
        
        # Find dynamic points
        dynamic = []
        for i, dist in enumerate(ranges):
            if not np.isfinite(dist):
                continue
            if dist < cfg['scan_range_min'] or dist > cfg['scan_range_max']:
                continue
            
            angle = scan.angle_min + i * scan.angle_increment
            world_angle = robot_yaw + angle
            
            hx = robot_x + dist * math.cos(world_angle)
            hy = robot_y + dist * math.sin(world_angle)
            
            if not self.is_near_static_obstacle(hx, hy):
                dynamic.append({'x': hx, 'y': hy, 'dist': dist, 'idx': i})
        
        if len(dynamic) < cfg['min_cluster_points']:
            return []
        
        # Cluster
        dynamic.sort(key=lambda p: p['idx'])
        clusters = []
        curr = [dynamic[0]]
        
        for i in range(1, len(dynamic)):
            if dynamic[i]['idx'] - dynamic[i-1]['idx'] <= cfg['cluster_gap_threshold']:
                curr.append(dynamic[i])
            else:
                if len(curr) >= cfg['min_cluster_points']:
                    clusters.append(curr)
                curr = [dynamic[i]]
        if len(curr) >= cfg['min_cluster_points']:
            clusters.append(curr)
        
        # Analyze
        humans = []
        for cluster in clusters:
            if len(cluster) > cfg['max_cluster_points']:
                continue
            
            xs = [p['x'] for p in cluster]
            ys = [p['y'] for p in cluster]
            
            cx, cy = np.mean(xs), np.mean(ys)
            width = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
            
            if cfg['human_width_min'] < width < cfg['human_width_max']:
                self.get_logger().info(
                    f'  → HUMAN: ({cx:.2f}, {cy:.2f}), w={width:.2f}m ✓')
                humans.append({'x': cx, 'y': cy, 'width': width})
        
        return humans
    
    
    def check_for_human_confident(self):
        """Multiple scan check."""
        num = self.detection_config['num_scans_to_check']
        thresh = self.detection_config['detection_threshold']
        
        count = 0
        for i in range(num):
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.2)
            
            if len(self.detect_humans()) > 0:
                count += 1
                self.get_logger().info(f'  Scan {i+1}/{num}: Detected ✓')
            else:
                self.get_logger().info(f'  Scan {i+1}/{num}: None')
        
        result = count >= thresh
        self.get_logger().info(
            f'  Result: {count}/{num} → {"CONFIRMED ✓" if result else "NOT FOUND"}')
        return result
    
    
    # =========================================================================
    # NAVIGATION
    # =========================================================================
    
    def create_pose_stamped(self, x, y, yaw):
        """Create goal pose."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose
    
    
    def navigate_to(self, x, y, yaw=0.0):
        """Navigate to position."""
        if self.shutting_down:
            return False
        
        self.get_logger().info(f'  Navigating to ({x:.2f}, {y:.2f})...')
        self.navigator.goToPose(self.create_pose_stamped(x, y, yaw))
        
        while not self.navigator.isTaskComplete():
            if self.shutting_down:
                self.navigator.cancelTask()
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('  ✓ Arrived')
            time.sleep(0.5)
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.1)
            return True
        else:
            self.get_logger().warn(f'  ✗ Nav failed: {result}')
            return False
    
    
    def check_person_at_location(self, person_x, person_y, name):
        """Check for person using SMART approach."""
        self.get_logger().info(f'\n  Checking for {name}')
        self.get_logger().info(f'  Expected: ({person_x}, {person_y})')
        
        # Find clear approach point (THE FIX!)
        approach = self.find_clear_approach_point(person_x, person_y)
        if not approach:
            self.get_logger().error('  No approach point found!')
            return False
        
        obs_x, obs_y, obs_yaw = approach
        
        # Navigate
        if not self.navigate_to(obs_x, obs_y, obs_yaw):
            self.get_logger().warn('  Nav failed, scanning anyway...')
        
        # Stabilize
        time.sleep(1.0)
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Scan
        self.get_logger().info('  Scanning...')
        found = self.check_for_human_confident()
        
        if found:
            self.get_logger().info(f'  ✓ {name} FOUND!')
        else:
            self.get_logger().warn(f'  ✗ {name} not found')
        
        return found
    
    
    # =========================================================================
    # MISSION
    # =========================================================================
    
    def start_mission_once(self):
        if self.mission_started or self.shutting_down:
            return
        self.mission_started = True
        self.run_mission()
    
    
    def run_mission(self):
        """Main mission."""
        self.get_logger().info('\n' + '=' * 60)
        self.get_logger().info('  STARTING MISSION')
        self.get_logger().info('=' * 60)
        
        # Setup
        self.get_logger().info('\n[1] Waiting for Nav2...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('  ✓ Nav2 ready')
        
        self.get_logger().info('\n[2] Resetting localization...')
        self.reset_localization()
        
        self.get_logger().info('\n[3] Checking map...')
        if not self.map_received:
            for _ in range(50):
                rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f'  Map: {"✓ Ready" if self.map_received else "⚠ Not available"}')
        
        self.get_logger().info('\n[4] Verifying pose...')
        self.wait_for_pose(timeout=10.0)
        pose = self.get_current_pose()
        self.get_logger().info(f'  Position: ({pose[0]:.2f}, {pose[1]:.2f})')
        
        # Check persons
        if not self.shutting_down:
            self.get_logger().info('\n' + '=' * 60)
            self.get_logger().info('[5] CHECKING PERSON 1')
            self.get_logger().info('=' * 60)
            self.person1_found = self.check_person_at_location(
                self.person1_expected['x'],
                self.person1_expected['y'],
                self.person1_expected['name']
            )
        
        if not self.shutting_down:
            self.get_logger().info('\n' + '=' * 60)
            self.get_logger().info('[6] CHECKING PERSON 2')
            self.get_logger().info('=' * 60)
            self.person2_found = self.check_person_at_location(
                self.person2_expected['x'],
                self.person2_expected['y'],
                self.person2_expected['name']
            )
        
        # Report
        self.print_report()
    
    
    def print_report(self):
        """Print results."""
        self.get_logger().info('\n')
        self.get_logger().info('╔' + '═' * 50 + '╗')
        self.get_logger().info('║' + '  FINAL REPORT'.center(50) + '║')
        self.get_logger().info('╠' + '═' * 50 + '╣')
        
        p1 = '✓ FOUND' if self.person1_found else '✗ NOT FOUND'
        p2 = '✓ FOUND' if self.person2_found else '✗ NOT FOUND'
        
        self.get_logger().info('║  ' + f'Person 1: {p1}'.ljust(48) + '║')
        self.get_logger().info('║  ' + f'Person 2: {p2}'.ljust(48) + '║')
        self.get_logger().info('╚' + '═' * 50 + '╝')
    
    
    def shutdown(self):
        self.shutting_down = True


def main(args=None):
    rclpy.init(args=args)
    controller = WarehouseController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        try:
            controller.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()