

    
    
    
    
"""#!/usr/bin/env python3
#  data logger
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

track_object  = racingTrack.racingTrack(1,smoothing_distance=10, max_width=2.0,circular=True)

xy_error = np.empty((0,11))        

    
    
def listner_callback1(msg):
    #self.get_logger().info('IN odom-listner,mpcc-iteration ="%f"'%self.mpcc_iteration)
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    roll,pitch,yaw = quat_to_euler.euler_from_quaternion(np.asarray([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])) 
    arc,closestIndex,last_closestIdx = findTheta(currentPose=np.asarray([x,y]),TrackCenter=self.wayPointCenter,theta_coordinates=self.theta_coordinates,trackWidth=self.Trackwidth, last_closestIdx=self.last_closestIdx)
    theta = arc
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    vz = msg.twist.twist.linear.z       
    controllerState = np.asarray([x,y,yaw,vx,theta])
    return controllerState
    
def listner_callback2(msg):
    #self.get_logger().info('IN odom-listner,mpcc-iteration ="%f"'%self.mpcc_iteration)
    v_x = msg.drive.speed
    steer =  msg.drive.steering_angle
    v_dot = msg.drive.acceleration
    theta_dot = msg.drive.jerk    
    u_current = np.asarray([v_dot,steer,theta_dot])
    return u_current



def save_data(drive_sub,odom_sub):
    controllerState = listner_callback1(odom_sub)
    u_current = listner_callback1(drive_sub)
    #np.asarray([x,y,yaw,vx,theta])
    state = controllerState
    control = u_current
    x = state[0]  
    y = state[1]
    yaw = state[2]
    vx = state[3]
    theta  = state[4]
    v_dot = control[0]
    steer = control[1]
    theta_dot = control[2]
    contour_error = track_object.calc_approx_contour_error(theta,np.array([x,y]))
    control_Ts  = 0.05 #FIXME:
    mpcc_iteration=-1 #FIXME:
    xy_error = np.vstack((xy_error,np.asarray([x,y,yaw,vx,theta,v_dot,steer,theta_dot,contour_error,mpcc_iteration,1/control_Ts])))


def dump_result():
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
        np.save(xy_error_file,xy_error)
        print("data Saved ")    

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('data_logger')
    _odom_topic_name ="/ego_racecar/odom"
    _drive_topic="/drive"
   
    odom_sub = message_filters.Subscriber(node,Odometry,"/ego_racecar/odom") 
    drive_sub = message_filters.Subscriber(node,AckermannDriveStamped,_drive_topic)        
    

    
    sync = message_filters.ApproximateTimeSynchronizer([drive_sub,odom_sub],queue_size=10, allow_headerless=False,slop=1e-3)  #add obs3_pose if you have three obstacles
    # Register the callback function
    sync.registerCallback(save_data)

    # Register the save_data_on_exit function to be called when exiting the program
    #rclpy.on_shutdown(dump_result)
    rclpy.spin()
    # try:
    #     # Spin to keep the program alive
    #     rclpy.spin(sync)
    # except SystemExit:    
    #     dump_result()  # <--- process the exception 
    #     rclpy.logging.get_logger("Quitting").info('Done')

    # node.destroy_node()
    # rclpy.shutdown()
    
    
    
    
if __name__ == '__main__':
    main()   
    
    
"""
    
    