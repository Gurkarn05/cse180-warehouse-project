import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math
import numpy as np
import time
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- TARGETS ---
PERSON1_EXPECTED = {'x': 1.00, 'y': -1.00}
PERSON1_SAFE_SPOT = {'x': -1.00, 'y': 1.00, 'yaw': -0.785} 
PERSON2_EXPECTED = {'x': -12.00, 'y': 15.00}
PERSON2_SAFE_SPOT = {'x': -14.00, 'y': 15.00, 'yaw': 0.0}
ROBOT_START = {'x': 2.12, 'y': -21.3, 'yaw': 1.57}

# --- DETECTION SETTINGS ---
DETECTION_RADIUS = 2.5
MAP_THRESHOLD = 40
SEARCH_TOLERANCE = 5
CLUSTER_TOLERANCE = 0.5
PERSON_WIDTH_MAX = 1.0
MIN_POINTS_CLUSTER = 6

class WarehouseUnifiedController(Node):
    def __init__(self):
        super().__init__('warehouse_unified_controller')
        
        # State Tracking
        self.map_data = None
        self.map_info = None
        self.latest_scan = None
        self.person1_found = False
        self.person2_found = False
        self.detected_extra_people = []

        # TF Buffer for coordinates
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS for Map (Transient Local is required for static maps)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.navigator = BasicNavigator()
        
        # Start mission when map and robot are ready
        self.startup_timer = self.create_timer(2.0, self.startup_check)
        self.get_logger().info('Unified Controller Online. Waiting for Map/Nav2...')

    # =========================================================================
    # CORE MISSION LOGIC
    # =========================================================================

    def startup_check(self):
        if self.map_data is not None and self.get_robot_pose() is not None:
            self.startup_timer.cancel()
            self.navigator.waitUntilNav2Active()
            self.run_master_mission()
        else:
            self.get_logger().info('Waiting for sensor readiness...')

    def run_master_mission(self):
        self.get_logger().info('--- PHASE 1: TARGET CHECK ---')

        # Check Person 1
        self.navigate_to(PERSON1_SAFE_SPOT['x'], PERSON1_SAFE_SPOT['y'], PERSON1_SAFE_SPOT['yaw'])
        time.sleep(1.0) # Let laser stabilize
        self.person1_found, _ = self.detect_person(PERSON1_EXPECTED['x'], PERSON1_EXPECTED['y'], "Person 1")

        # Check Person 2
        self.navigate_to(PERSON2_SAFE_SPOT['x'], PERSON2_SAFE_SPOT['y'], PERSON2_SAFE_SPOT['yaw'])
        time.sleep(1.0) # Let laser stabilize
        self.person2_found, _ = self.detect_person(PERSON2_EXPECTED['x'], PERSON2_EXPECTED['y'], "Person 2")

        # PHASE 2: Comprehensive Search (Only if missing someone)
        if not self.person1_found or not self.person2_found:
            self.get_logger().warn('--- PHASE 2: STARTING FULL TRAVERSAL ---')
            self.search_entire_warehouse()
        else:
            self.get_logger().info('Success! Both primary targets verified.')

        self.get_logger().info('Mission Finished. Returning to Base.')
        self.navigate_to(ROBOT_START['x'], ROBOT_START['y'], ROBOT_START['yaw'])

    def search_entire_warehouse(self):
        """Generates lawn-mower waypoints based on map size"""
        margin = 1.0
        min_x = self.map_info.origin.position.x + margin
        max_x = min_x + (self.map_info.width * self.map_info.resolution) - 2*margin
        min_y = self.map_info.origin.position.y + margin
        max_y = min_y + (self.map_info.height * self.map_info.resolution) - 2*margin
        
        spacing = 6.0  # Efficient 6m spacing
        
        x = min_x
        y_dir = 1 # 1 for up, -1 for down
        
        while x < max_x:
            # Generate Y coordinates for this column
            if y_dir > 0:
                y_target = min_y
                while y_target < max_y:
                    self.process_waypoint(x, y_target)
                    y_target += spacing
            else:
                y_target = max_y
                while y_target > min_y:
                    self.process_waypoint(x, y_target)
                    y_target -= spacing
            
            x += spacing
            y_dir *= -1

    def process_waypoint(self, x, y):
        self.get_logger().info(f'Traversing to ({x:.2f}, {y:.2f})')
        self.navigate_to(x, y, 0.0)
        self.check_for_unknown_discrepancies()

    def check_for_unknown_discrepancies(self):
        # Look for unknown objects (clusters) anywhere nearby
        found, loc = self.detect_person(0, 0, "Unknown Zone", ignore_target_dist=True)
        if found:
            # Check if this loc is near already found locs to avoid spam
            is_new = True
            for d in self.detected_extra_people:
                if math.dist(loc, d) < 2.0: is_new = False
            
            if is_new:
                self.detected_extra_people.append(loc)
                self.get_logger().info(f"!!! NEW OBJECT DETECTED at {loc} !!!")

    # =========================================================================
    # HELPERS & CALLBACKS
    # =========================================================================

    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def scan_callback(self, msg):
        self.latest_scan = msg

    def navigate_to(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self.navigator.goToPose(goal)
        
        while not self.navigator.isTaskComplete():
            # Keep spinning to ensure map/scan callbacks happen
            rclpy.spin_once(self, timeout_sec=0.1)
            
        return self.navigator.getResult() == TaskResult.SUCCEEDED

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            # Manual Euler conversion to get Yaw
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (x, y, yaw)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def is_static_object(self, wx, wy):
        if self.map_data is None: return False
        
        # Convert World (meters) -> Map (indices)
        mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        
        # Check bounds
        if not (0 <= mx < self.map_info.width and 0 <= my < self.map_info.height): return False

        # Check area around point for walls
        r = SEARCH_TOLERANCE
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                check_x, check_y = mx + dx, my + dy
                if 0 <= check_x < self.map_info.width and 0 <= check_y < self.map_info.height:
                    if self.map_data[check_y, check_x] > MAP_THRESHOLD: return True
        return False

    def detect_person(self, target_x, target_y, name, ignore_target_dist=False):
        pose = self.get_robot_pose()
        if not pose or not self.latest_scan: return False, None
        
        rx, ry, ryaw = pose
        
        # 1. Filter Non-Static Points (Dynamic Object Detection)
        dynamic_pts = []
        for i, r in enumerate(self.latest_scan.ranges):
            # Filter bad readings
            if r < 0.5 or r > 10.0 or not math.isfinite(r): continue
            
            # Correct Angle Math: Robot Yaw + Scan Angle
            angle_in_world = ryaw + self.latest_scan.angle_min + (i * self.latest_scan.angle_increment)
            
            wx = rx + r * math.cos(angle_in_world)
            wy = ry + r * math.sin(angle_in_world)
            
            if not self.is_static_object(wx, wy):
                dynamic_pts.append((wx, wy))

        if not dynamic_pts: return False, None

        # 2. Cluster Points (Simple Euclidean Clustering)
        clusters = []
        if dynamic_pts:
            current_cluster = [dynamic_pts[0]]
            for i in range(1, len(dynamic_pts)):
                px, py = dynamic_pts[i]
                prev_x, prev_y = dynamic_pts[i-1]
                dist = math.sqrt((px - prev_x)**2 + (py - prev_y)**2)
                
                if dist < CLUSTER_TOLERANCE:
                    current_cluster.append((px, py))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(px, py)]
            clusters.append(current_cluster)

        # 3. Analyze Clusters
        for cluster in clusters:
            if len(cluster) < MIN_POINTS_CLUSTER: continue 
            
            # Centroid
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            
            # Size
            min_x = min(p[0] for p in cluster); max_x = max(p[0] for p in cluster)
            min_y = min(p[1] for p in cluster); max_y = max(p[1] for p in cluster)
            width = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
            
            if width > PERSON_WIDTH_MAX: continue

            # Check if this is the specific target we are looking for
            if not ignore_target_dist:
                dist_to_target = math.sqrt((cx - target_x)**2 + (cy - target_y)**2)
                if dist_to_target < DETECTION_RADIUS:
                    self.get_logger().info(f'FOUND {name} at ({cx:.2f}, {cy:.2f})')
                    return True, (cx, cy)
            else:
                # In search mode, any valid cluster counts
                return True, (cx, cy)

        self.get_logger().info(f'{name} NOT FOUND in current view.')
        return False, None

def main(args=None):
    rclpy.init(args=args)
    node = WarehouseUnifiedController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()