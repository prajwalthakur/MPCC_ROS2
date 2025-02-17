import rclpy #python api for ros2
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange, IntegerRange, SetParametersResult
from rclpy.parameter import Parameter
import rclpy.time, rclpy.duration

from tf2_ros import Buffer, TransformListener
import time
from builtin_interfaces.msg import Time
import copy
import threading
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import math
import numpy as np
from tf_transformations import quaternion_from_euler ,euler_from_quaternion
from scipy.spatial.transform import Rotation
from typing import List, Tuple

from visualization_msgs.msg import Marker, MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter
from f110_msgs.msg import WpntArray, ObstacleArray, Obstacle as ObstacleMessage
from bisect import bisect_left
Point2D  = Tuple[float , float]
from .helper import racingTrack
latching_qos = QoSProfile(depth=1 , durability = QoSDurabilityPolicy.TRANSIENT_LOCAL , history = HistoryPolicy.KEEP_LAST) 

def from_vector3_msg(msg):
    return np.r_[msg.x,msg.y,msg.z]

def from_quat_msg(msg):
    return Rotation.from_quat([msg.x, msg.y,msg.z , msg.w])

import csv        
def load_map():
    file = open("workspace/src/maps/e7_floor5/e7_floor5_square_standard_format.csv", "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    waypoints =np.zeros((0,4))
    for i in range(1,len(data)-1):
        cx = float(data[i][0])
        cy = float(data[i][1])
        w_right = float(data[i][2])
        w_left = float(data[i][3])
        waypoints = np.vstack((waypoints,np.asarray((cx,cy,w_right,w_left))))
    return waypoints         
from dataclasses import dataclass
class Obstacle : 
    "implements the properties of the obstacles"
    current_id = 0
    def __init__(self, x, y , size , theta )->None:
        self.center_x = x
        self.center_y = y
        self.size = size
        self.id = None
        self.theta = theta # orientation
    
    def squaredDist( self, obstacle ) : 
        return (self.center_x - obstacle.center_x )**2 + (self.center_y - obstacle.center_y)**2

@dataclass
class carState:
    x:np.float32
    y:np.float32
    yaw:np.float32
    speed:np.float32 = 0.0
    progress:np.float32 = 0.0


# node class detect ...
class Detect(Node):
    #TODO:seperate class for just detecting the track ?
    """
    class implements a ros node that detects the obstacles on the track  
    subscribes to the the following topics :
    - '/scan': publishes the lidar scans
    - '/global_waypoints' : publishes the global waypoints 
    - '/odom_fernet' : publishes the car state in the ferenet frame ... not needed
    """
    """
        The Node publishes :
        -> '/breakpoints_markers': Publishes the breakpoint markers of the obstacles
        -> '/detect_bound' : publishes the detect boundaries 
        -> '/raw_obstacles': Publishes teh detected obstacles
        -> '/obstacles_markers_new': Publihses the markers of the detected obstacles
    """
    def __init__(self)-> None:
        
        self.bold_text = '\033[1m'
        self.reset_text = '\033[0m'
        self.red_color = '\033[31m'
        self.reset_color = '\033[0m'
        self.green_color = '\033[32m'
        
        # initialize the node , subsribe to the topics , and create publishers and services proxies
        super().__init__('detection_node' , allow_undeclared_parameters = True  , automatically_declare_parameters_from_overrides = True)
        self.converter = None
        # --- Node properties ---
        self.measuring  = True #self.get_parameter("/measure").get_parameter_value().bool_value
        self.from_bag = self.get_parameter("/from_bag").get_parameter_value().bool_value
        self.get_logger().info(f"Measuring:{self.measuring}")
        self.get_logger().info(f"From ros-Bag : {self.from_bag}")
        
        # --- subsribers --- 
        msg_cb_group = ReentrantCallbackGroup()
        self.scan_sub = self.create_subscription(LaserScan , '/scan' , self.laserCb , 10 , callback_group = msg_cb_group )
        # -- publisher --
        
        self.breakpoints_markers_pub = self.create_publisher(MarkerArray , '/perception/breakpoints_markers' , 5)
        self.boundaries_pub = self.create_publisher(Marker , '/perception/detect_bound' , qos_profile = latching_qos)
        self.obstacles_msg_pub = self.create_publisher(ObstacleArray , '/perception/detection/raw_obstacles', qos_profile = latching_qos)
        self.obstacles_marker_pub = self.create_publisher(MarkerArray , '/perception/obstacles_markers_new', 5 )
 
        # track variables 
        # dyn params--
        self.max_obs_size = 0.80
        self.min_obs_size = 0.10
        self.max_viewing_distance = 9.0 #meters
        self.boundaries_inflation = 0.2 #meters
        
        
        self.waypoints = None
        self.biggest_d = None
        self.smallest_d = None
        self.d_right_array = None
        self.d_left_array = None
        self.track_length  = None
        self.s_array = None
        self.pathCb()
         
        
        #self.glob_wpts_sub = self.create_subscription(WpntArray  , '/global_waypoints',self.pathCb , 10 , callback_group = msg_cb_group)
        self.car_state_sub = self.create_subscription(Odometry , '/ego_racecar/odom' ,  self.carStateCb  , 10  , callback_group = msg_cb_group)
        self.declare_parameter("rate" , 10 , descriptor = ParameterDescriptor(description = "rate at which node is running" ))
        self.declare_parameter("lambda" , 10 , descriptor = ParameterDescriptor(description = "minimum reliables detection angle in degress" ))  # incidence angle ie (angle made by the ray to the normal of the surface )
        self.declare_parameter("sigma" , 0.03 , descriptor = ParameterDescriptor(description = "std deviation of the noise from the lidar ranges in m" ))
        self.declare_parameter("min_2_points_dist" , 0.01 , descriptor = ParameterDescriptor(description = "minimum distance b/w two points" ))

        # tunable parameters
        self.rate = self.get_parameter("rate").get_parameter_value().integer_value
        self.lambda_angle = self.get_parameter("lambda").get_parameter_value().integer_value * math.pi/180
        self.sigma = self.get_parameter("sigma").get_parameter_value().double_value
        self.min_2_points_dist = self.get_parameter("min2_points_dist").get_parameter_value().double_value
        


        param_dicts = [{'name' : 'min_obs_size',
                        'default' : self.min_obs_size,
                        'descriptor' : ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, integer_range=[IntegerRange(from_value=5, to_value=40, step=1)])},
                       {'name' : 'max_obs_size',
                        'default' : self.max_obs_size,
                        'descriptor' : ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.1, to_value=1.0, step=0.01)])},
                       {'name' : 'max_viewing_distance',
                        'default' : self.max_viewing_distance,
                        'descriptor' : ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=3.0, to_value=10.0, step=0.1)])},
                       {'name' : 'boundaries_inflation',
                        'default' : self.boundaries_inflation,
                        'descriptor' : ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=2.0, step=0.1)])}]
        self.declare_dyn_parameters(param_dicts)
        self.add_on_set_parameters_callback(self.dyn_param_cb) # parameter callback

        #--variables--
        self.lock = threading.Lock()
        

        # ego car position
        self.car_pose = carState(x=0.0 , y=0.0 , yaw=0.0 , speed = 0.0 , progress = 0.0)
        
        # raw scans from the lidar 
        self.scans:LaserScan = LaserScan()

        self.current_stamp = None
        self.tracked_obstacles = []
        
        self.tf_buffer = Buffer(rclpy.duration.Duration(seconds = 5) )
        self.tf_listner = TransformListener(self.tf_buffer , self, spin_thread = True)
        self.paths_needs_updated = True
        
        main_timer_cb_group = MutuallyExclusiveCallbackGroup()
        

        self.main_timer = self.create_timer(1/self.rate , self.loop , callback_group = main_timer_cb_group)
        
    def normalize_s(self,x,track_length):
        #self.get_logger().info(f'{self.bold_text}{self.green_color}Track-length:{ track_length}{self.reset_color}{self.reset_text}')
        x = x % ( track_length)
        if x> track_length/2:
            x-=track_length
        return x       
        
    def declare_dyn_parameters(self, param_dicts : List[dict]):
        params = []   
        for param_dict in param_dicts :
            param = self.declare_parameter(param_dict['name'],param_dict['default'] , param_dict['descriptor'])
            params.append(param)
        return 
    
    
    def laserCb(self,msg):
        self.scans = msg
        
    def initialize_converter(self):
        converter = FrenetConverter(self.waypoints[:,0] , self.waypoints[:,1] , self.waypoints[:,2]) # x,y,yaw
        self.get_logger().info("[Opponent Detection]: initialized FrenetConverter object")
        return converter
    
    # def pathCb(self, data):
    #     # Initial calls: initialize the converter
    #     if self.converter is None:
    #         self.waypoints = np.array(
    #             [[wpnt.x_m, wpnt.y_m, wpnt.psi_rad] for wpnt in data.wpnts])
    #         self.converter = self.initialize_converter()

    #     # Second call: create the boundaries arrays
    #     if self.s_array is None or self.path_needs_update:
    #         self.get_logger().info(
    #             '[Opponent Detection]: received global path')
    #         waypoint_array = data.wpnts
    #         points = []
    #         self.s_array = []
    #         self.d_right_array = []
    #         self.d_left_array = []
    #         for waypoint in waypoint_array:
    #             self.s_array.append(waypoint.s_m)
    #             self.d_right_array.append(
    #                 waypoint.d_right-self.boundaries_inflation)
    #             self.d_left_array.append(
    #                 waypoint.d_left-self.boundaries_inflation)
    #             resp = self.converter.get_cartesian(
    #                 waypoint.s_m, -waypoint.d_right+self.boundaries_inflation)
    #             points.append(Point(x=resp[0], y=resp[1], z=0.0))
    #             resp = self.converter.get_cartesian(
    #                 waypoint.s_m, waypoint.d_left-self.boundaries_inflation)
    #             points.append(Point(x=resp[0], y=resp[1], z=0.0))
    #         self.smallest_d = min(self.d_right_array+self.d_left_array)
    #         self.biggest_d = max(self.d_right_array+self.d_left_array)
    #         self.track_length = data.wpnts[-1].s_m

    #         self.detection_boundaries_marker = Marker()
    #         self.detection_boundaries_marker.header.frame_id = "map"
    #         self.detection_boundaries_marker.header.stamp = self.get_clock().now().to_msg()
    #         self.detection_boundaries_marker.id = 0
    #         self.detection_boundaries_marker.type = Marker.SPHERE_LIST
    #         self.detection_boundaries_marker.scale.x = 0.02
    #         self.detection_boundaries_marker.scale.y = 0.02
    #         self.detection_boundaries_marker.scale.z = 0.02
    #         self.detection_boundaries_marker.color.a = 1.
    #         self.detection_boundaries_marker.color.g = 0.
    #         self.detection_boundaries_marker.color.r = 1.
    #         self.detection_boundaries_marker.color.b = 0.
    #         self.detection_boundaries_marker.points = points

    #     # Republish markers
    #     self.boundaries_pub.publish(self.detection_boundaries_marker)
    #     self.path_needs_update = False
    
    def pathCb(self):

        # Initial calls: initialize the converter
        if self.converter is None:
            self.track_object  = racingTrack.racingTrack(0.1,smoothing_distance=0.5, max_width=2.0,circular=True) #racingTrack.racingTrack(1,smoothing_distance=10, max_width=2.0,circular=True)       
            self.Trackwidth = self.track_object.Trackwidth
            self.inner_waypoints,self.wayPointCenter,self.outer_waypoints = self.track_object.get_coordinates_perception()
            #wpnts_boundary_width = load_map()
            self.waypoints = np.array(
                [[wpnt[0], wpnt[1], wpnt[2]] for wpnt in self.wayPointCenter])
            self.converter = self.initialize_converter()
            #self.get_logger().info(f'{self.bold_text}{self.green_color}converter-initialized{self.reset_color}{self.reset_text}')
        # Second call: create the boundaries arrays
        if self.s_array is None or self.path_needs_update:
            self.get_logger().info(
                '[Opponent Detection]: received global path')
            waypoint_array = copy.deepcopy(self.wayPointCenter)
            points = []
            self.s_array = []
            self.d_right_array = []
            self.d_left_array = []
            #(cx,cy,w_right,w_left)
            for waypoint in waypoint_array:
                self.s_array.append(waypoint[3])
                r_dist, l_dist      = self.track_object.get_dist_boundary(waypoint[3])
                self.d_right_array.append(
                    r_dist-self.boundaries_inflation)
                self.d_left_array.append(
                   l_dist -self.boundaries_inflation)
                #self.get_logger().info(f'{self.bold_text}{self.green_color}{ waypoint[3] , r_dist , self.boundaries_inflation }{self.reset_color}{self.reset_text}')
                resp = self.converter.get_cartesian(
                     waypoint[3], -r_dist+self.boundaries_inflation)
                points.append(Point(x=resp[0], y=resp[1], z=0.0))
                resp = self.converter.get_cartesian(
                    waypoint[3], l_dist-self.boundaries_inflation)
                points.append(Point(x=resp[0], y=resp[1], z=0.0))
            
            self.smallest_d = min(self.d_right_array+self.d_left_array)
            self.biggest_d = max(self.d_right_array+self.d_left_array)
            
            self.track_length = waypoint_array[-1][3]

            self.detection_boundaries_marker = Marker()
            self.detection_boundaries_marker.header.frame_id = "map"
            self.detection_boundaries_marker.header.stamp = self.get_clock().now().to_msg()
            self.detection_boundaries_marker.id = 0
            self.detection_boundaries_marker.type = Marker.SPHERE_LIST
            self.detection_boundaries_marker.scale.x = 0.02
            self.detection_boundaries_marker.scale.y = 0.02
            self.detection_boundaries_marker.scale.z = 0.02
            self.detection_boundaries_marker.color.a = 1.
            self.detection_boundaries_marker.color.g = 0.
            self.detection_boundaries_marker.color.r = 1.
            self.detection_boundaries_marker.color.b = 0.
            self.detection_boundaries_marker.points = points

        # Republish markers
        self.boundaries_pub.publish(self.detection_boundaries_marker)
        self.path_needs_update = False
                
    def carStateCb(self,data:Odometry):
        # quaternion to rotation rpy
        r,p,yaw = euler_from_quaternion( [data.pose.pose.orientation.x , data.pose.pose.orientation.y , data.pose.pose.orientation.z , data.pose.pose.orientation.w])
        car_s = carState ( x = data.pose.pose.position.x , y = data.pose.pose.position.y , yaw = yaw)
        
        self.car_s = self.converter.get_frenet([car_s.x] , [car_s.y])
        self.car_s = self.car_s[0][0]
        #self.get_logger().info(f'{self.bold_text}{self.green_color}car_s:{self.car_s}{self.reset_color}{self.reset_text}')
        return 
    
    def dyn_param_cb(self, params: List[Parameter]):
        """
        Notices the change in the parameters and alters the detection params accordingly
        """
        for param in params:
            param_name = param.name
            
            if param_name == 'min_obs_size':
                self.min_obs_size = param.value
            elif param_name == 'max_obs_size':
                self.max_obs_size = param.value
            elif param_name == 'max_viewing_distance':
                self.max_viewing_distance = param.value
            elif param_name == 'boundaries_inflation':
                self.boundaries_inflation = param.value
        
        self.path_needs_update = True
        
        param_list = [self.min_obs_size, self.max_obs_size, self.max_viewing_distance, self.boundaries_inflation]
        self.get_logger().info(
            f'[Opponent Detection]: New dyn reconf values recieved: '
            f'Min size {param_list[0]} [laser points], '
            f'Max size {param_list[1]} [m], '
            f'Max viewing dist {param_list[2]} [m], '
            f'Boundaries Inflation {param_list[3]}')
        
        return SetParametersResult(successful=True)
    
    
    def clearmarkers(Self)->MarkerArray:
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = 3
        marker_array.markers = [marker]
        return marker_array 
    
    def laserPointOnTrack(self,s,d,car_s)->bool:
        # car_s : progress of the car on the track , s: opponent progress 
        if self.normalize_s(s - car_s , self.track_length) > self.max_viewing_distance :
            return False
        if abs(d) >=self.biggest_d:
            return False # outside the inflated track .. the opponent is at the edge of the track
        if abs(d) <= self.smallest_d:
            return True # the opponent is in the middle
        idx = bisect_left(self.s_array , s ) # find the  closest track s-point to s
        if idx:
            idx-=1
        if d<=  - self.d_right_array[idx] or d >= self.d_left_array[idx] :
            return False
        return True

    
    def scans2ObsPointCloud(self, car_s : Float32 , scans : LaserScan , t: TransformStamped):
        " converts the Lidar scans to a 2D pointCloud and segments them into cloud"
        # return empty if transform none else perform algo
        if t is None:
            return []
        
        # -- initialization of some utility  parameters--#
        l = self.lambda_angle
        d_phi  = scans.angle_increment
        sigma  = self.sigma
        
        #--transform the scan ranges to cloud point--
        T = from_vector3_msg(t.transform.translation)
        R = from_quat_msg(t.transform.rotation)
        
        angles = np.linspace(scans.angle_min , scans.angle_max , len(scans.ranges))
        
        x_laser_frame = (scans.ranges*np.cos(angles)).flatten()
        y_laser_frame = (scans.ranges*np.sin(angles)).flatten()
        z_laser_frame = np.zeros(len(scans.ranges))
        
        #4xN matrix
        
        xyz_laser_frame = np.vstack((x_laser_frame,y_laser_frame,z_laser_frame , np.ones(len(scans.ranges))))
        
        
        H_laser2map = np.eye(4)
        H_laser2map[:3,-1] = T
        R_matrix = R.as_matrix()
        H_laser2map[:3,:3] = R_matrix
        
        xyz_map = H_laser2map@xyz_laser_frame #4xN
        
        cloudPoints_list = np.transpose(xyz_map[:2,:]).tolist()  # Nx[2] (only xy coordinates)        
        
        """
        segment the cloud point into smaller point clouds
        that represent potential object using Adaptive methods
        
        """
        first_point:Point2D = (cloudPoints_list[0][0], cloudPoints_list[0][1])
        objects_pointcloud_list : List[List[Point2D]] = [[first_point]]  # rows represent the objects , cloumns represents the associated point clouds
        
        div_const = np.sin(d_phi) / np.sin(l-d_phi)
        for i in range(1,len(cloudPoints_list)):
            curr_range = self.scans.ranges[i]
            d_max = curr_range*div_const + 3*sigma
            
            # distance b/w points does not change in map frame or laser frame
            
            dist_to_next_point = np.linalg.norm(xyz_laser_frame[:2,i]-xyz_laser_frame[:2,i-1])
            
            
            # map frame
            curr_point = ( cloudPoints_list[i][0], cloudPoints_list[i][1] )
            if dist_to_next_point < d_max :
                objects_pointcloud_list[-1].append(curr_point)
            else:
                objects_pointcloud_list.append([curr_point])
        #--------
        """
            removing point clouds that are too small or too big or that have their center point not on the track
        
        """
        # filterig of the point clouds 
        x_points = []
        y_points = []
        for obs in objects_pointcloud_list :
            #self.get_logger().info(f'{self.bold_text}{self.green_color}num-of-detected:{len(current_obstacle_array)}{self.reset_color}{self.reset_text}')
            mean_x_pos = np.mean([point[0] for point in obs])
            mean_y_pos = np.mean([point[1] for point in obs])
            x_points.append(mean_x_pos)
            y_points.append(mean_y_pos)
        #self.get_logger().info(f'{self.bold_text}{self.green_color}before-453{self.reset_color}{self.reset_text}')
        s_points , d_points = self.converter.get_frenet(np.array(x_points) , np.array(y_points))
        
        remove_array = []
        for idx, object in enumerate(objects_pointcloud_list):
            if len(object) < self.min_obs_size:
                remove_array.append(object)
                continue
            if not (self.laserPointOnTrack(s_points[idx], d_points[idx], car_s)):
                remove_array.append(object)
                continue
        
        #self.get_logger().info(f'{self.bold_text}{self.green_color}len{len(remove_array)}{self.reset_color}{self.reset_text}')                
        for object in remove_array:
            objects_pointcloud_list.remove(object)
            
        markers_array = []
        for idx,object in enumerate(objects_pointcloud_list):
            # first element
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.current_stamp
            marker.id = idx*10
            marker.type = Marker.SPHERE
            marker.scale.x = 0.25
            marker.color.a = 0.5
            marker.color.g = 1.
            marker.color.r = 0.
            marker.color.b = idx/len(objects_pointcloud_list)
            marker.pose.position.x = object[0][0]
            marker.pose.position.y = object[0][1]
            marker.pose.orientation.w = 1.
            markers_array.append(marker)
            # last element
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.current_stamp
            marker.id = idx*10+2
            marker.type = Marker.SPHERE
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            marker.color.a = 0.5
            marker.color.g = 1.
            marker.color.r = 0.
            marker.color.b = idx/len(objects_pointcloud_list)
            marker.pose.position.x = object[-1][0]
            marker.pose.position.y = object[-1][1]
            marker.pose.orientation.w = 1.
            markers_array.append(marker)

        # This causes the markers to flicker in RViz, but likely doesn't affect the underlying algo.
        self.breakpoints_markers_pub.publish(self.clearmarkers())
        self.breakpoints_markers_pub.publish(MarkerArray(markers=markers_array))
        
        return objects_pointcloud_list
    
    def obsPointClouds2obsArray(self, objects_pointcloud_list):
        current_obstacle_array = []
        min_dist = self.min_2_points_dist  #"minimum distance b/w two points"
        for obstacle in objects_pointcloud_list : 
            #-- fit a rectangle to the data points ---
            theta = np.linspace(0,np.pi/2 - np.pi/180 , 90 )
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            distance1 = np.dot(obstacle , [cos_theta , sin_theta])
            distance2 = np.dot(obstacle , [-sin_theta , cos_theta])
            D10 = -distance1 + np.amax(distance1, axis=0)
            D11 = distance1 - np.amin(distance1, axis=0)
            D20 = -distance2 + np.amax(distance2, axis=0)
            D21 = distance2 - np.amin(distance2, axis=0)
            min_array = np.argmin(
                [np.linalg.norm(D10, axis=0), np.linalg.norm(D11, axis=0)], axis=0)
            D10 = np.transpose(D10)
            D11 = np.transpose(D11)
            D10[min_array == 1] = D11[min_array == 1]
            D10 = np.transpose(D10)
            min_array = np.argmin(
                [np.linalg.norm(D20, axis=0), np.linalg.norm(D21, axis=0)], axis=0)
            D20 = np.transpose(D20)
            D21 = np.transpose(D21)
            D20[min_array == 1] = D21[min_array == 1]
            D20 = np.transpose(D20)
            D = np.minimum(D10, D20)
            D[D < min_dist] = min_dist
            
            # extract the center of the obstacle assuming that it is actually a square obstacle
            #self.get_logger().info(f'{self.bold_text}{self.green_color}before-326{self.reset_color}{self.reset_text}')
            theta_opt = np.argmax(np.sum(np.reciprocal(D),axis=0))*(np.pi/180)
            distances1 = np.dot(obstacle , [np.cos(theta_opt) , np.sin(theta_opt) ] )
            distances2 = np.dot(obstacle , [-np.sin(theta_opt) , np.cos(theta_opt)] )
            max_dist1 = np.max(distances1)
            min_dist1 = np.min(distances1)
            max_dist2 = np.max(distances2)
            min_dist2 = np.min(distances2)
            
            # corners are detected ANti-CLOCKWISE
            corner1 = None
            corner2 = None
            # the obstacle has more detection in the verticle direction
            if (np.var(distances2) > np.var(distances1)):
                if (np.linalg.norm(-distances1+max_dist1) < np.linalg.norm(distances1-min_dist1)):
                    # the detections are nearer to the right edge
                    # lower_right_corner
                    corner1 = np.array([np.cos(theta_opt)*max_dist1-np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*max_dist1+np.cos(theta_opt)*min_dist2])
                    # upper_right_corner
                    corner2 = np.array([np.cos(theta_opt)*max_dist1-np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*max_dist1+np.cos(theta_opt)*max_dist2])
                else:
                    # the detections are nearer to the left edge
                    # upper_left_corner
                    corner1 = np.array([np.cos(theta_opt)*min_dist1-np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*min_dist1+np.cos(theta_opt)*max_dist2])
                    # lower_left_corner
                    corner2 = np.array([np.cos(theta_opt)*min_dist1-np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*min_dist1+np.cos(theta_opt)*min_dist2])
            else:  # the obstacle has more detection in the horizontal direction
                if (np.linalg.norm(-distances2+max_dist2) < np.linalg.norm(distances2-min_dist2)):
                    # the detections are nearer to the top edge
                    # upper_right_corner
                    corner1 = np.array([np.cos(theta_opt)*max_dist1-np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*max_dist1+np.cos(theta_opt)*max_dist2])
                    # upper_left_corner
                    corner2 = np.array([np.cos(theta_opt)*min_dist1-np.sin(theta_opt)*max_dist2,
                                        np.sin(theta_opt)*min_dist1+np.cos(theta_opt)*max_dist2])
                else:
                    # the detections are nearer to the bottom edge
                    # lower_left_corner
                    corner1 = np.array([np.cos(theta_opt)*min_dist1-np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*min_dist1+np.cos(theta_opt)*min_dist2])
                    # lower_right_corner
                    corner2 = np.array([np.cos(theta_opt)*max_dist1-np.sin(theta_opt)*min_dist2,
                                        np.sin(theta_opt)*max_dist1+np.cos(theta_opt)*min_dist2])
            # vector that goes from corner1 to corner2
            colVec = np.array([corner2[0]-corner1[0], corner2[1]-corner1[1]])
            # orthogonal vector to the one that goes from corner1 to corner2
            orthVec = np.array([-colVec[1], colVec[0]])
            # center position
            center = corner1 + 0.5*colVec + 0.5*orthVec

            current_obstacle_array.append(
                Obstacle(center[0], center[1], np.linalg.norm(colVec), theta_opt))

        # self.get_logger().debug(
        #     f"[Opponent Detection] detected {len(current_obstacle_array)} raw obstacles.")
        
        self.get_logger().info(f'{self.bold_text}{self.green_color}num-of-objects-detected:{len(current_obstacle_array)}{self.reset_color}{self.reset_text}')
        return current_obstacle_array
        
    def checkObstacles(self,current_obstacles):
        "delete obstacles that are too big"
        remove_list = []
        self.tracked_obstacles.clear()
        for obs in current_obstacles:
            if (obs.size > self.max_obs_size) :
                remove_list.append(obs)
        
        self.get_logger().info(f'{self.bold_text}{self.green_color}removed {len(remove_list)} obstacles since they were too big{self.reset_color}{self.reset_text}')
        for obs in remove_list :
            current_obstacles.remove(obs)
        for idx , curr_obs in enumerate(current_obstacles):
            curr_obs.id = idx
            self.tracked_obstacles.append(curr_obs)
        self.get_logger().info(f'{self.bold_text}{self.green_color} current obstackes {len(current_obstacles)}{self.reset_color}{self.reset_text}')
        
    def publishObstaclesMessage(self):
        obstacles_array_message = ObstacleArray()
        obstacles_array_message.header.stamp = self.current_stamp
        obstacles_array_message.header.frame_id = "map"
        
        x_center = []
        y_center = []
        
        for obstacle in self.tracked_obstacles :
            x_center.append(obstacle.center_x)
            y_center.append(obstacle.center_y)
        #self.get_logger().info(f'{self.bold_text}{self.green_color}before-626{self.reset_color}{self.reset_text}')
        s_points , d_points = self.converter.get_frenet( np.array(x_center),np.array(y_center) )
        
        for idx , obstacle in enumerate(self.tracked_obstacles):
            s = s_points[idx]
            d = d_points[idx]
            obsMsg = ObstacleMessage()
            obsMsg.id = obstacle.id
            obsMsg.s_start =  s - obstacle.size / 2
            obsMsg.s_end =    s + obstacle.size /2 
            obsMsg.d_left  =  d  + obstacle.size/2
            obsMsg.d_right =  d - obstacle.size/2
            obsMsg.s_center = s 
            obsMsg.d_center = d
            obsMsg.x_pos = obstacle.center_x
            obsMsg.y_pos = obstacle.center_y
            obsMsg.yaw =  obstacle.theta
            obsMsg.size = obstacle.size
            
            obstacles_array_message.obstacles.append(obsMsg)
        self.obstacles_msg_pub.publish(obstacles_array_message)
        
        
    def publishObstaclesMarkers(self):
        markers_array = []
        for obs in self.tracked_obstacles:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.current_stamp
            marker.id = obs.id
            marker.type = Marker.CUBE
            marker.scale.x = obs.size
            marker.scale.y = obs.size
            marker.scale.z = obs.size
            marker.color.a = 0.5
            marker.color.g = 1.
            marker.color.r = 0.
            marker.color.b = 1.
            marker.pose.position.x = obs.center_x
            marker.pose.position.y = obs.center_y
            q = quaternion_from_euler(0, 0, obs.theta)
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            markers_array.append(marker)

        # This causes the markers to flicker in RViz, but likely doesn't affect the underlying algo.
        self.obstacles_marker_pub.publish(self.clearmarkers())

        self.obstacles_marker_pub.publish(MarkerArray(markers=markers_array))
        Obstacle.current_id = 0
        
    def loop(self):
        if self.converter is None or self.scans is None or self.track_length is None : 
            return
        if self.measuring :
            start_time = time.perf_counter()    
            
        scans = self.scans 
        car_s = self.car_s
        self.current_stamp = self.get_clock().now().to_msg()
        try :
            self.get_logger().info(f'{self.bold_text}{self.green_color}{scans.header.stamp}{self.reset_color}{self.reset_text}') 
            transform = self.tf_buffer.lookup_transform(target_frame = 'map' , source_frame = self.scans.header.frame_id,
                                                        time  = Time(sec=scans.header.stamp.sec, nanosec=scans.header.stamp.nanosec), 
                                                        timeout = rclpy.duration.Duration(seconds = 0.03)
                                                        )
        except Exception as e :
            self.get_logger().warn(f"could not transform b/w 'map' and  '{scans.header.frame_id}:{e}'")
            transform = None
        
        objects_pointcloud_list = self.scans2ObsPointCloud(scans = scans , car_s = car_s , t = transform )
        current_obstacles = self.obsPointClouds2obsArray(objects_pointcloud_list)
        self.checkObstacles(current_obstacles)
        if self.measuring :
            end_time = time.perf_counter()
            latency = end_time - start_time
            self.get_logger().info(f"Latency:{latency}")

        self.publishObstaclesMessage()
        self.publishObstaclesMarkers()
    
def main():
    rclpy.init()
    detect  = Detect()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(detect)
    try:
        executor.spin()
    except KeyboardInterrupt :
        detect.get_logger().info("KeyboardInterrupt")
        
    detect.destroy_node()
    rclpy.shutdown()
    

if __name__=='__main__':
    main()