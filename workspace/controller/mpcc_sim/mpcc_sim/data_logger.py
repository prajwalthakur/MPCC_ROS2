#!/usr/bin/env python3
#  mpcc with delay , using state augmentation as given in MATLAB using scipy to discretize the state
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import ReliabilityPolicy, QoSProfile
from .helper import quat_to_euler
import numpy as np
from .helper import racingTrack
from .helper.findtheta import findTheta
from nav_msgs.msg import Path, Odometry
from rclpy.time import Time




import message_filters


class datalogger(Node):
    def __init__(self,sensor_topic_name ="/ego_racecar/odom",drive_topic="/drive"):
        super().__init__('data_logger')
        self.mpcc_iteration = 0
        self.control_freq = 20
        self.control_Ts = 1/self.control_freq
        self.dum_data_freq = 5
        self.dum_data_Ts =    1/5
        self.xy_error = np.empty((0,11))
        # timer callback for savig data:
        #self.create_timer(self.dum_data_Ts, self.dump_result) 
        self.track_object  = racingTrack.racingTrack(1,smoothing_distance=10, max_width=2.0,circular=True)
        self.inner_waypoints,self.wayPointCenter,self.outer_waypoints = self.track_object.get_coordinates(self.track_object.dict_waypoints)
        _,self.theta_coordinates,_ = self.track_object.get_theta_coordinates()
        self.Trackwidth = self.track_object.Trackwidth
        self._drive_topic = drive_topic
        self._odom_topic_name = sensor_topic_name        
   
        odom_sub = message_filters.Subscriber(self,Odometry,"/ego_racecar/odom") 
        drive_sub = message_filters.Subscriber(self,AckermannDriveStamped,self._drive_topic)         
        self.sync = message_filters.ApproximateTimeSynchronizer([drive_sub,odom_sub],queue_size=10, allow_headerless=False,slop=1e-3)  #add obs3_pose if you have three obstacles      
        self.sync.registerCallback(self.save_data)
    
    
    def listner_callback1(self,msg):
        #self.get_logger().info('IN odom-listner,mpcc-iteration ="%f"'%self.mpcc_iteration)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        roll,pitch,yaw = quat_to_euler.euler_from_quaternion(np.asarray([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])) 
        #self.arc,self.closestIndex,self.last_closestIdx = findTheta(currentPose=np.asarray([x,y]),TrackCenter=self.wayPointCenter,theta_coordinates=self.theta_coordinates,trackWidth=self.Trackwidth, last_closestIdx=self.last_closestIdx)
        theta = -1 #self.arc
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z       
        self.controllerState = np.asarray([x,y,yaw,vx,theta])
    
    def listner_callback2(self,msg):
        #self.get_logger().info('IN odom-listner,mpcc-iteration ="%f"'%self.mpcc_iteration)
        v_x = msg.drive.speed
        steer =  msg.drive.steering_angle
        v_dot = msg.drive.acceleration
        theta_dot = msg.drive.jerk    
        self.u_current = np.asarray([v_dot,steer,theta_dot])



    def save_data(self,drive_sub,odom_sub):
        self.get_logger().info('IN-LOGGER')
        self.listner_callback1(odom_sub)
        self.listner_callback1(drive_sub)
        #np.asarray([x,y,yaw,vx,theta])
        state = self.controllerState
        control = self.u_current
        x = state[0]  
        y = state[1]
        yaw = state[2]
        vx = state[3]
        theta  = state[4]
        v_dot = control[0]
        steer = control[1]
        theta_dot = control[2]
        contour_error = self.track_object.calc_approx_contour_error(theta,np.array([x,y]))
        self.xy_error = np.vstack((self.xy_error,np.asarray([x,y,yaw,vx,theta,v_dot,steer,theta_dot,contour_error,self.mpcc_iteration,1/self.control_Ts])))


    def dump_result(self):
            # # if((MPC_vars.latency!=0) and (LATENCY == MPC_vars.latency)):
            # which_one = "aware"
            # # elif(MPC_vars.latency==0 and LATENCY!=0 ):
            # which_one = "unaware"
            # # elif ( LATENCY == 0 ):
            # which_one = "ideal"
            
            #which_one = "ideal"
            #which_one = "unaware"
            which_one = "mu_1e_3__linear_kinematics_both"
            #which_one = "aware_only_next"
            folder = "result4"
            name_file = f"results/{folder}/onelap/delay_{which_one}_case1"
            xy_error_file = f"{name_file}_xy_error.npy"
            np.save(xy_error_file,self.xy_error )
            print("data Saved ")    

def main():
    rclpy.init()
    data_logger= datalogger()
    #use mltithreadExecutor
    rclpy.spin(data_logger)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
    