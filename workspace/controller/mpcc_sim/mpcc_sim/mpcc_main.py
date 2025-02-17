#!/usr/bin/env python3
#  mpcc with delay , using state augmentation and for delay compensation and  discretization of  the state
import os
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import Mpcclogger
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist  # Import for velocity commands
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import ReliabilityPolicy, QoSProfile ,HistoryPolicy , DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, FloatingPointRange, IntegerRange, SetParametersResult  
from scipy.spatial.transform import Rotation
from .helper import aug_state
import numpy as np
from .helper import racingTrack
from .helper.findtheta import findTheta
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA,Header
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.time import Time
from   .helper.params import EgoParams
from f110_msgs.msg import WpntArray, ObstacleArray, Obstacle as ObstacleMessage
from scipy.linalg import block_diag
import scipy.integrate
import scipy.linalg

from hpipm_python import *  #TODO:
from hpipm_python.common import * #TODO: 

from numba import njit,objmode  #TODO:
import numpy as np
import sys
import time

import copy 
LATENCY = 0

from dataclasses import dataclass
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue


class CarParams:
    def __init__(self):
        #params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
        self.model_parameters ={}
        self.model_parameters["sx"] = 4 #"no of state which needs to integrate"
        self.model_parameters["su"]  = 2 
        self.model_parameters["ax"] = 5 #"augmented no of state"
        self.model_parameters["au"]  = 3 
        self.model_parameters["stateindex_x"] = 0
        self.model_parameters["stateindex_y"] = 1 
        self.model_parameters["stateindex_psi"]  = 2 
        self.model_parameters["stateindex_v"]  = 3
        self.model_parameters["stateindex_theta"]  = 4
        
        self.model_parameters["inputindex_vdot"] = 0
        self.model_parameters["inputindex_delta"] = 1    
        self.model_parameters["inputindex_thetadot"] = 2  
            
        self.model_parameters["lf"] = 0.15875
        self.model_parameters["lr"] = 0.17145


        self.model_parameters["L"] = 0.58
        self.model_parameters["W"] = 0.31
                
        self.model_parameters["Scale"] = 1
        
        

@dataclass
class robotState:
    x:np.float32    = 0.0 # x-coordinate of the car wrt to intertial frame
    y:np.float32    = 0.0 # y-coordinate of the car wrt to inertial frame
    z:np.float32    = 0.0 
    yaw:np.float32  = 0.0 # yaw of the car
    vx:np.float32   = 0.0 # longitudinal speed wrt the car reference frame
    vy:np.float32   = 0.0 # lateral Speed wrt the car reference frame
    vz:np.float32   = 0.0 #
    s_:np.float32    = 0.0 # progress along the reference line
    d:np.float32    = 0.0 # lateral displacement from the ref-line
    msgSampled:np.float32 = 0.0 # time stamp 


@dataclass
class robotControl:
    vdot:np.float32 = 0.0
    steerAngle : np.float32 = 0.0
    s_dot : np.float32 = 0.0


@njit
def get_expm(temp_0):
    with objmode(tmp_fwd="float64[:, ::1]"):
        tmp_fwd = scipy.linalg.expm(temp_0)
    return tmp_fwd
def construct_matrix_delay(xbar_k,ubar_k,Ts,lwb,latency_at_i):
        Ts = Ts
        latency = latency_at_i #0.018 #latency_at_i=0.018
        lwb = lwb
        sx = 5
        su = 3
        x0 = xbar_k[0][0]
        y0 = xbar_k[1][0]
        psi0=xbar_k[2][0]
        v0= xbar_k[3][0]
        vdot0 = ubar_k[0][0]
        delta0 = ubar_k[1][0]
        thetadot0 = ubar_k[2][0]
        xk=xbar_k[0:sx]
        uk = ubar_k[0:su]
        xdot = np.asarray([[v0*np.cos(psi0) ],
                [v0*np.sin(psi0)] ,
                [v0*np.tan(delta0)/(lwb)],
                [vdot0],
                [thetadot0]])
        A_c = np.asarray([[0,0,-v0*np.sin(psi0),np.cos(psi0),0],
                          [0,0,v0*np.cos(psi0),np.sin(psi0),0],
                          [0,0,0,np.tan(delta0)/lwb,0],
                          [0,0,0,0,0],
                          [0,0,0,0,0]])
        B_c = np.asarray([[0 ,0,0 ],
                          [0,0,0],
                          [0,v0/(lwb*np.cos(delta0)*np.cos(delta0)),0],
                          [1,0,0],
                          [0,0,1]])   
        
        gc = xdot- A_c@xk.reshape((sx,1)) - B_c@uk.reshape((su,1))
        Bc_aug=np.hstack((B_c,gc))
        A_aug = np.vstack((np.hstack((A_c,Bc_aug)),np.zeros((su+1,sx+su+1))))    
        tmp_fwd =  scipy.linalg.expm(A_aug*Ts)
        
        Phi = tmp_fwd[0:sx,0:sx]    
        Gamma  = tmp_fwd[0:sx,sx:sx+su]        
        Psi_gk = tmp_fwd[0:sx,-1 ]
        Psi_gk = np.reshape(Psi_gk,newshape=(sx,1) )
        
        #print(f'shape:Tx@Gamma_0@invTu ,{tmp_fwd.shape }')
        temp2 = np.vstack((np.hstack((A_c,Bc_aug)),np.zeros((su+1,sx+su+1))))*(Ts-latency)
        temp2 =  get_expm(temp2)
        Gamma_0  = temp2[0:sx,sx:-1]
       
       
        return Phi, Gamma,Gamma_0 , Psi_gk 
def fwd_sim(state:np.ndarray,uk:np.ndarray,dt:np.float32,lwb:np.float32):
    x0 = state[0][0]
    y0 = state[1][0]
    yaw0=state[2][0]
    v0= state[3][0]
    vdot0 = uk[0][0]
    delta0 = uk[1][0]
    sdot0 = uk[2][0]
    xdot = np.asarray([[v0*np.cos(yaw0) ],
            [v0*np.sin(yaw0)] ,
            [v0*np.tan(delta0)/(lwb)],
            [vdot0],
            [sdot0]])
    x_next = state + xdot*dt
    return x_next












class MPCCNode(Node,CarParams,EgoParams):
    def __init__(self,sensor_topic_name ="/ego_racecar/odom",drive_topic="/drive"):
        Node.__init__(self,node_name='mpcc_node',allow_undeclared_parameters=True , automatically_declare_parameters_from_overrides  = True )
        CarParams.__init__(self)
        EgoParams.__init__(self)
        self.ComputationDelay = 0.03 #0.020  + 0.020
        self.Lwb =   self.model_parameters["lf"] +  self.model_parameters["lr"]
        self.fwd_sim_dt = 0.01 #+ 0.03
        self.mpcc_iteration = 0
        
        #pose_topic = "/pf/inferred_pose" if self.real_test else "/ego_racecar/odom"
        
        odom_topic = "/ego_racecar/odom" if self.real_test else "/ego_racecar/odom"
        drive_topic = "/drive"
        waypoint_topic = "/waypoint"
        

        self.odom_sub_ = self.create_subscription(
            msg_type=Odometry,
            topic=odom_topic,
            callback=self.odom_cb,
            qos_profile=QoSProfile( depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST ,durability=DurabilityPolicy.VOLATILE ))
        
        
        self.odom_sub_ # prevent unused variable warning
        self.drive_pub_ = self.create_publisher(msg_type=AckermannDriveStamped  , topic = drive_topic , qos_profile=QoSProfile(depth=1 , reliability=ReliabilityPolicy.RELIABLE))
        self.boundary_pub = self.create_publisher(msg_type =MarkerArray,topic='/boundary_marker', qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        # self.track_pub =  self.create_publisher(msg_type =MarkerArray,topic='/track_marker', qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.mpc_pred = self.create_publisher(msg_type =MarkerArray,topic='/mpc_pred', qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))        
        # self.custom_drive_pub = self.create_publisher(msg_type = AckermannDriveStamped,topic="/ego_racecar/set_custom_pos",qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        # self.custom_msg_pub = self.create_publisher(msg_type = Mpcclogger,topic="/ego_racecar/mpcc_logger",qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    
        self.opp_drive_pub_ = self.create_publisher(msg_type = AckermannDriveStamped , topic = '/opp_drive', qos_profile=QoSProfile(depth=1 , reliability=ReliabilityPolicy.RELIABLE))

        self.opp_drive_pub()

    
        self.attrNames =["_X0","_U0", "_Qk","_qk","_Rk","_Ak","_Bk","_gk","_Ck","_ug","_lg","_lbx","_ubx","_lbu","_ubu"]
        
        self.track_initialization()
        self.last_message_time= time.time_ns()
        self.xy_error = np.empty((0,11))
        # timer callback for savig data:
        #self.create_timer(self.dum_data_Ts, self.dump_result) 
        self.bold_text = '\033[1m'
        self.reset_text = '\033[0m'
        self.red_color = '\033[31m'
        self.reset_color = '\033[0m'
        self.green_color = '\033[32m'
        
        # activate obstacle avoidance moule if the obstacle is within this range 
        self.inf_radius = 0.65
        self.safety_dist  = 0.5
        
    def opp_drive_pub(self):
        # Here you have the callback method
        # create a Twist message
        # opponent is static 
        msg = AckermannDriveStamped()
        msg.drive.speed   = 0.0
        msg.drive.steering_angle = 0.0
        msg.drive.acceleration = 0.0 
        self.opp_drive_pub_.publish(msg)
        return         
        
        
    def do_state_shift(self,X_seqs,U_seqs,X0,Horizon,dT, track_length):
        #shift the current state and input by 1 time step for the next initialization
        aug_states = self.model_parameters["ax"] 
        aug_input = self.model_parameters["au"]
        indexTheta = self.model_parameters["stateindex_theta"]
        indexpsi = self.model_parameters["stateindex_psi"]
        N = Horizon 
        xTemp  = np.zeros((aug_states,N+1))
        uTemp = np.zeros((aug_input,N))
        xTemp[:,0] = X0.reshape((aug_states,))
        uTemp[:,0] = U_seqs[:,1]
        
        xTemp[:,1:N] = X_seqs[:,2:]
        uTemp[:,1:N-1] = U_seqs[:,2:]
        uTemp[:,N-1] = U_seqs[:,N-1]
        #get_simulation_next_state_open(self,x_current,u_current,T,last_closestIdx):
        #fwd_sim(state:np.ndarray,uk:np.ndarray,dt:np.float32,lwb:np.float32)
        xTemp[:self.model_parameters["ax"],N] = fwd_sim(xTemp[:,N-1][...,np.newaxis], U_seqs[:,N-1][...,np.newaxis], dT ,lwb=self.Lwb).reshape(( self.model_parameters["ax"],))
        if xTemp[indexpsi,0] - xTemp[indexpsi,1] > np.pi:
            xTemp[indexpsi,1:] = xTemp[indexpsi,1:]+2*np.pi
        if xTemp[indexpsi,0] - xTemp[indexpsi,1] < - np.pi:
            xTemp[indexpsi,1:] = xTemp[indexpsi,1:]-2*np.pi
        if xTemp[indexTheta,0] - xTemp[indexTheta,1] < -0.75*track_length:
            xTemp[indexTheta,1:] = xTemp[indexTheta,1:]-track_length
        return xTemp,uTemp
    def compute_elapsed_time(self, curr_time: Time, past_time: Time):
        """Compute the duration between two times."""
        curr_time_sec = curr_time.sec + curr_time.nanosec / 1e9
        past_time_sec = past_time.sec + past_time.nanosec / 1e9

        elapsed_sec = curr_time_sec - past_time_sec
        # Create a Duration object for the elapsed time
        #elapsed_duration = rclpy.duration.Duration(seconds=elapsed_sec)

        return elapsed_sec
    def predict_state(self,robot_state:robotState,control:robotControl):
        control_array = np.array([control.vdot , control.steerAngle , control.s_dot])[...,np.newaxis]
        #get current time :
        curr_time = self.get_clock().now().to_msg()
        elapsed_seconds = self.compute_elapsed_time(curr_time , robot_state.msgSampled  )
        #elapsed_seconds = elapsed_time.seconds_nanoseconds()[0] + elapsed_time.seconds_nanoseconds()[1] * 1e-9 # total time elapsed in seconds
        # prediction for how many more seconds?
        pred_time = 0.0  #0.0 #elapsed_seconds + 0.04 #self.ComputationDelay
        state_pred_array = np.array([robot_state.x, robot_state.y,robot_state.yaw , robot_state.vx , robot_state.s_])[...,np.newaxis]
        loop_itr = int(pred_time//self.fwd_sim_dt)
        for i in range(loop_itr):
            state_pred_array = fwd_sim(state_pred_array,uk = control_array , dt=self.fwd_sim_dt ,lwb=self.Lwb)
        #state_pred_array = [fwd_sim(state_pred_array, uk=control_array, dt=self.fwd_sim_dt, lwb=self.Lwb) for _ in range(loop_itr)]
        #self.get_logger().info(f'pred_array = {state_pred_array}')
        state_pred_ = robotState(x=state_pred_array[0][0] , y = state_pred_array[1][0]  , yaw = state_pred_array[2][0] , vx = state_pred_array[3][0] , s_=state_pred_array[4][0])
        #state_pred_array = np.array()
        
        #state_pred_ = copy.deepcopy(robot_state)
        self.get_logger().info(f'{self.bold_text}{self.red_color}Predicted-State:{state_pred_.x, state_pred_.y , state_pred_.yaw , state_pred_.vx}{self.reset_color}{self.reset_text}')
        return state_pred_ , state_pred_array
    def odom_cb(self,msg):
        #self.get_logger().info('IN odom-listner,mpcc-iteration ="%f"'%self.mpcc_iteration)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        rot = Rotation.from_quat([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w]) #quat_to_euler.euler_from_quaternion(np.asarray()) 
        rot_euler = rot.as_euler('xyz',degrees=False)
        yaw = rot_euler[2]
        if  self.ControllerReset == True:
            self.arc,self.closestIndex,self.last_closestIdx = findTheta(currentPose=np.asarray([x,y]),TrackCenter=self.wayPointCenter,theta_coordinates=self.theta_coordinates,trackWidth=self.Trackwidth, last_closestIdx=0 , globalSearch=True) # how to find the lateral position from the reference #TODO:
        else:
            self.arc,self.closestIndex,self.last_closestIdx = findTheta(currentPose=np.asarray([x,y]),TrackCenter=self.wayPointCenter,theta_coordinates=self.theta_coordinates,trackWidth=self.Trackwidth, last_closestIdx=self.last_closestIdx,globalSearch=False) # how to find the lateral position from the reference #TODO:
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z    
        msgSampled = msg.header.stamp
        self.SensorStateSampled:robotState = robotState(x=x,y=y,z=z , yaw=yaw , vx=vx , vy = vy , vz=vz , s_=self.arc , d = -1 ,msgSampled =  msgSampled )
        if  self.ControllerReset == True :
            self.ControllerReset = False
            #timer for control callback
            self.create_timer(timer_period_sec = self.ctrTs, callback=self.control_cb, clock = self.get_clock())
            self.ControllerRbtState:robotState = copy.deepcopy(self.SensorStateSampled)
            self.control_init()
            
            obstacleInfoTopic = "/perception/detection/raw_obstacles"
            self.obstacleInfo_sub_ = self.create_subscription(msg_type=ObstacleArray ,topic = obstacleInfoTopic , callback=self.obsInfo_cb,qos_profile=QoSProfile( depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST ,durability=DurabilityPolicy.VOLATILE ))
        # x_car  = self.ControllerRbtState.x
        # y_car = self.ControllerRbtState.y
        # yaw_car = self.ControllerRbtState.yaw
        # self.Obs_coordinates =  np.array([10.0 ,0.0,0.0])
        # Car_coordinates = np.array([x_car,y_car,yaw_car])
        # isCollisionRange , distance = self.calculateDistance(Car_coordinates , self.Obs_coordinates)
        # if isCollisionRange:
        #     self.OvertakingMode = True
        # else:
        #     self.OvertakingMode = False  
    
    def track_initialization(self):
        #self.waypoints_list = self.track_object.waypoints_list
        self.track_object  = racingTrack.racingTrack(0.1,smoothing_distance=0.5, max_width=2.0,circular=True) #racingTrack.racingTrack(1,smoothing_distance=10, max_width=2.0,circular=True)       
        self.Trackwidth = self.track_object.Trackwidth
        self.inner_waypoints,self.wayPointCenter,self.outer_waypoints = self.track_object.get_coordinates(self.track_object.dict_waypoints)
        self.get_logger().info('TRACKWIDTH ="%f"'%self.Trackwidth)
        _,self.theta_coordinates,_ = self.track_object.get_theta_coordinates()
        #self.car_object =  kinematicsCarModel.carObject(self.x0,self.control_Ts,self.track_object,latency=LATENCY,last_closestIdx=self.last_closestIdx) #TODO:
        self.ax = self.model_parameters["ax"]
        self.au = self.model_parameters["au"]

    def control_init(self):
        #self._publish_control(np.asarray([0.05,0]),speed=0.0)
        self.get_logger().info('IN control-init,mpcc-iteration ="%f"'%self.mpcc_iteration)
        self.ControllerRbtState:robotState = copy.deepcopy(self.SensorStateSampled)
        #self._publish_control( self.ControllerRbtState,  )
        AssumeSpeed = 0.0
        x0 = np.array([self.ControllerRbtState.x , self.ControllerRbtState.y , self.ControllerRbtState.yaw , AssumeSpeed , self.ControllerRbtState.s_])[...,np.newaxis]
        # self.arc,self.closestIndex,self.last_closestIdx = findTheta(currentPose=np.asarray([self.x0[0],self.x0[1]]),TrackCenter=self.wayPointCenter,theta_coordinates=self.theta_coordinates,trackWidth=self.Trackwidth, last_closestIdx=self.last_closestIdx)
        # self.get_logger().info('last_closestIdx ="%f"'%self.last_closestIdx)
        # self.get_logger().info('current_closestIdx ="%f"'%self.closestIndex)
        u_current = np.zeros((self.au,1))
        self.uprev = np.zeros((self.au,1))
        self.X_seqs = np.tile(x0,(1,self.horizonLength+1))
        self.U_seqs = np.zeros((self.au,self.horizonLength))
        self.mpcc_nonfeasible = 0
        self.exit_flag = 0 
        self.carName  = "ego_car"
        self.xy_error = np.empty((0,9))
        #initial run for mpcc controller inspired from SQP
        #self.initial_run()
        self.X_seqs =  self._construct_warm_start_soln(self.U_seqs,self.X_seqs,0)
        self.control_t = robotControl( vdot=0.0 , steerAngle= 0.0 , s_dot = 0.0 )
        self.CurrentTime = self.get_clock().now().to_msg()

    def _construct_warm_start_soln(self ,prev_control_commands,prev_predicted_states,mpcc_iteration):
            #states = [x,y,psi,v,theta]
            #u = [v_dot,delta,thetadot]
            theta_  = np.empty(0)
            xy_= np.empty((0,2))
            v_prev = prev_predicted_states[self.model_parameters["stateindex_v"],0]
            theta = prev_predicted_states[self.model_parameters["stateindex_theta"],0]
            psi_prev= prev_predicted_states[self.model_parameters["stateindex_psi"],0]
            for k in range(1, self.horizonLength+1 ):
                    theta = theta  + v_prev*self.ctrTs      #assuming that my car is running with v velocity , thetadot is also vdot , so next theta would be theta_next = theta_prev + v*Ts
                    psi_next = self.track_object.spline_object_center.calc_yaw(theta)
                    psi_next = self.pi_2_pi(psi_prev-psi_next)
                    psi_prev = psi_next
                    x_next,y_next = self.track_object.spline_object_center.calc_position(theta)
                    prev_predicted_states[:,k]  = np.asarray([x_next.__float__(), y_next.__float__(), psi_next, v_prev,theta])          
            return prev_predicted_states         
    
    



    def control_cb(self):
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.01))
        self.ControllerRbtState:robotState = copy.deepcopy(self.SensorStateSampled)
        #control_pred = robotControl(vdot=0.0 ,steerAngle=0.0 ,  s_dot=0.0 , )
        state_pred , state_pred_array  = self.predict_state(robot_state=self.ControllerRbtState , control=  self.control_t )
        self.mpcc_iteration+=1 
        
        #logger
        #self.get_logger().info('hello')
        #self.publish_boundary_track(self.inner_waypoints,self.wayPointCenter,self.outer_waypoints)
        #self.get_logger().info('entering mpc_control_call mpcc iteration ="%f"'%self.mpcc_iteration)
        
        #augment state and inputs by shifting previus optimal solution
        self.X_seqs,self.U_seqs = self.do_state_shift(self.X_seqs,self.U_seqs,state_pred_array,self.horizonLength, dT=self.ctrTs, track_length=self.track_object.length)
        
        current_time = self.get_clock().now().to_msg()
        self.exit_flag,U_pred_seqs,X_pred_seqs,track_constraints_pts,costValue,costComponents = self.do_ctr_step( prev_predicted_states =self.X_seqs , state_current=state_pred_array, prev_control_commands=self.U_seqs,Uprev= self.uprev ,mpcc_iteration = self.mpcc_iteration)                                    
        #self.last_message_time = current_time
        self.X_seqs=X_pred_seqs
        self.U_seqs=U_pred_seqs  
        u_current = self.U_seqs[:,1]
        time_now =self.get_clock().now().to_msg()
        elapsed_time =  self.compute_elapsed_time( time_now , current_time ) 
        #elapsed_seconds = elapsed_time.seconds_nanoseconds()[0] + elapsed_time.seconds_nanoseconds()[1] * 1e-9 # total time elapsed in seconds
        self.get_logger().info('Computational time in calculating one control Command: "%.6f" seconds' %elapsed_time)
        
        if self.exit_flag == -1:
            car_yaw =   self.ControllerRbtState.yaw
            x_des_virt , y_des_virt , des_yaw  = self.track_object.get_approx_des_pose()
            steering_angle_des = self.pi_2_pi(des_yaw - car_yaw )
            self.control_t = robotControl( vdot=-0.1 , steerAngle= steering_angle_des , s_dot = u_current[2] ) # slow down the car if previous iteration couldnt find the solution 
        else:
            self.control_t = robotControl( vdot=u_current[0] , steerAngle=u_current[1] , s_dot = u_current[2] )
        
        self._publish_control(state_current = state_pred , u_current = self.control_t)
        
        

        self.uprev = u_current 
        
        log_string = f"mpcc iteration:{self.mpcc_iteration},x:{state_pred_array[0]},y:{state_pred_array[1]},yaw:{state_pred_array[2]},vx:{state_pred_array[3],},theta_virt:{state_pred_array[4]},steer:{u_current[1]},velocity:{u_current[0]}"
        self.get_logger().info(log_string) #'before-Publishing vdot: "%f"' %u_current[0])  
        right_points = track_constraints_pts[:,0:3]
        left_points = track_constraints_pts[:,3:6]
        self.publish_boundary_markers(right_points, left_points)
        x_pred_waypoints = X_pred_seqs[:2,0:]
        self.publish_mpc_pred(np.transpose(x_pred_waypoints))
        # print("time=",self.mpcc_iteration*0.2) #,"contour-error=",contour_error,"sum=",sum_contour)        
        #self.get_logger().info('exiting now , mpcc iteratio "%f"'%self.mpcc_iteration)
   
    def _publish_control(self,state_current:robotState , u_current:robotControl):
        # Here you have the callback method
        # create a Twist message
        current_long_speed = state_current.vx
        msg = AckermannDriveStamped()
        ts = self.get_clock().now().to_msg()
        msg.header.stamp = ts
        desired_speed = current_long_speed + u_current.vdot*self.ctrTs   # what is the desired speed at Ts + 1
        currentTime = self.get_clock().now().to_msg()
        timeElapsed = self.compute_elapsed_time(currentTime,self.CurrentTime)
        self.get_logger().info(f'{self.bold_text}{self.red_color}desired-Speed:{current_long_speed , self.ControllerRbtState.vx , timeElapsed}{self.reset_color}{self.reset_text}')
        #self.get_logger().info(f'{self.bold_text}{self.red_color}desired-Speed:{current_long_speed, u_current.vdot*self.ctrTs , desired_speed }{self.reset_color}{self.reset_text}')
        msg.drive.speed   = float(desired_speed)
        
        msg.drive.steering_angle = float(u_current.steerAngle)
        msg.drive.acceleration = u_current.vdot # not relevant to VSEC , VSEC only accepts desired longitudinal speed (and converts to motor speed)
        # msg.drive.jerk  = u_current[2]
        # Publish the message to the Topic
        self.drive_pub_.publish(msg)
        # Display the message on the console
        #self.get_logger().info('Publishing vdot: "%f"' %u_current[0])  
        #self.get_logger().info('Publishing steering : "%f"' %u_current[1]) 
        self.get_logger().info('Publishing Control Msg: "%s"' % msg)   
        #self.pub_custom_data() 
        return 
    
    def do_ctr_step(self, prev_predicted_states,state_current, prev_control_commands,Uprev,mpcc_iteration = 0):
        #self.mpcCont.update( prev_predicted_states =self.X_seqs , state_current=self.X0, prev_control_commands=self.U_seqs,Uprev= self.uprev ,mpcc_iteration = self.mpcc_iteration)
        #self._construct_warm_start_soln(prev_control_commands,prev_predicted_states[:5,:],mpcc_iteration)
        #self._set_approximated_lag_contour_error(prev_control_commands,prev_predicted_states,mpcc_iteration)    
        x0=state_current
        u0=Uprev
        U_seqs = prev_control_commands
        X_seqs = prev_predicted_states
        track_constraints_pts = self._set_track_constraints(prev_predicted_states[self.model_parameters['stateindex_theta'],:],self.track_object)
        #track_constraints_pts =np.zeros((1,6)) #self._set_track_constraints(theta_,xy_)
        cost_list = self.create_matrices(x0,u0,X_seqs,U_seqs)
        X,U,dU,info = self._solve(cost_list)
        if info["exitflag"]==0:
            U_pred = U
            X_pred = X
        elif (info["exitflag"]==1):
            U_pred = prev_control_commands
            X_pred = prev_predicted_states
            self.get_logger().info(f'{self.bold_text}{self.red_color}Make NO sense !{self.reset_color}{self.reset_text}')
            # exit_flag,U_pred_seqs,X_pred_seqs,track_constraints_pts,costValue,costComponents
        costValue=-1
        costComponents  = np.zeros((1,5))
        return info["exitflag"],U_pred,X_pred,track_constraints_pts,costValue,costComponents    

    
 
    def mpc_matrices(self):
        empty_dict = {attr: None for attr in self.attrNames}
        return empty_dict
    
    def create_matrices(self,x0,u0,X_seqs,U_seqs):
        cost_list = []
        for i in range(self.horizonLength):
            cost_list.append(self.mpc_matrices())
            if i==0:
                cost_list[i]["_X0"] = x0.reshape((self.ax,1))
                cost_list[i]["_U0"] = u0.reshape((self.au,1))
            Xk = X_seqs[:,i]
            Uk = U_seqs[:,i]
            cost_list[i]["_Qk"],cost_list[i]["_qk"] =self.generateQkqk(Xk,Uk,i)
            cost_list[i]["_Rk"] = self.costScale*2*self.calcRInput()
            #linearized dynamics
            cost_list[i]["_Ak"],cost_list[i]["_Bk"],cost_list[i]["_gk"] = self.getEqualityConstraints(Xk,Uk)
            #bounds
            cost_list[i]["_lbx"], cost_list[i]["_ubx"],cost_list[i]["_lbu"], cost_list[i]["_ubu"] = self.getBounds()
            #linearized track constraints 
            cost_list[i]["_Ck"], cost_list[i]["_ug"], cost_list[i]["_lg"] = self._getBorderConstraint(i) #self._getInequalityConstraints(self.borders[max(i-1,0),:],self.MPC_vars,car_object,i)
            
        #terminal state
        i = self.horizonLength
        cost_list.append(self.mpc_matrices())
        Xk = X_seqs[:,i]
        Qk,qk= self.generateQkqk(Xk,Uk,i)
        cost_list[i]["_Qk"],cost_list[i]["_qk"] = self.costScale*Qk,self.costScale*qk
        
        cost_list[i]["_Rk"] = self.costScale*2*self.calcRInput()
        #linearized dynamics
        #bounds
        cost_list[i]["_lbx"], cost_list[i]["_ubx"],cost_list[i]["_lbu"], cost_list[i]["_ubu"] = self.getBounds()
        #linearized track constraints 
        cost_list[i]["_Ck"], cost_list[i]["_ug"], cost_list[i]["_lg"] = self._getBorderConstraint(i) #self._getInequalityConstraints(self.borders[max(i-1,0),:],self.MPC_vars,car_object,i)
        
        return cost_list
    
    def generateQkqk(self,Xk,Uk,i):
        QState,qState = self.calcQstate(Xk,Uk,i)
        QInput,qInput = self.calcQInput()
        #add omega regularization not needed
        #make Qtilde symetric (not symetric due to numerical issues)
        Qtilde = 0.5 *(QState+QState.T)
        #Qk = contouring-lag error and real-input cost
        Qk = 2*block_diag(Qtilde,QInput)
        qk = np.vstack((qState,qInput))
        # pdb.set_trace()
        return Qk,qk     
        
    #compute linear contouring and lag errors
    def calcQstate(self,Xk,Uk,i):
        if i==self.horizonLength:
            Q_ = np.diag([self.qCNmult*self.qC,self.qL])
        else:
            Q_ = np.diag([self.qC,self.qL])
        d_contouring_error,e_c,e_l = self.calcdContouringError(Xk,Uk,i,self.track_object) #derivative of contouring error
        Q_contouring_cost = d_contouring_error.T@Q_@d_contouring_error
        QState = np.diag(self.invTx)@Q_contouring_cost@np.diag(self.invTx)  #normalization
        ## q.Tx
        error = np.vstack((e_c, e_l))
        #q = horzcat(2*(error.T*Q_*errorgrad-self.X_prev[k].T*errorgrad.T*Q_*errorgrad),MX.zeros(1,self.au-1),-self.mpc_vars.qVtheta)
        qState = 2*(error.T@Q_@d_contouring_error-Xk.T@d_contouring_error.T@Q_@d_contouring_error).T
        qState =  np.diag(self.invTx)@qState
        return QState,qState
    def calcQInput(self): #Q_input , q_input
        quadCostInput_ = np.asarray([self.rvDot,self.rDelta,self.rVtheta])
        QInput = np.diag(quadCostInput_)  
        QInput = np.diag(self.invTu)@QInput@np.diag(self.invTu)     
        qInput_= np.hstack((np.zeros((self.au-1,)),-self.qVtheta)).reshape((self.au,1))
        qInput = np.diag(self.invTu)@qInput_
        return QInput,qInput 
       
    def calcRInput(self): # rate of change of the  input 
        R_ = np.asarray([self.rdvDot,self.rdDelta,self.rdVtheta])
        R = np.diag(R_)    
        return R          
    
    def calcdContouringError(self,Xk,Uk,k,track_object):
        theta_hat,xtyt,dxdy,cos_phit,sin_phit,trackPoint_dtheta_ref = self._get_track_values(Xk,Uk)
        x_virt =xtyt[0,0]
        y_virt = xtyt[1,0]
        dtheta  = trackPoint_dtheta_ref
        d_contouring_error = np.zeros((2,self.ax))
        dx = dxdy[0]
        dy = dxdy[1]
        #Contouring error
        e_c = -sin_phit*(x_virt - Xk[0]) + cos_phit*(y_virt - Xk[1])  
        dContouringErrorTheta = dtheta*cos_phit*(Xk[0]- x_virt ) + dtheta*sin_phit*(Xk[1]-y_virt )-sin_phit*dx + cos_phit*dy
        dContouringErrorX  = sin_phit
        dContouringErrorY  = -cos_phit
        d_contouring_error[0,self.model_parameters["stateindex_x"]] = dContouringErrorX
        d_contouring_error[0,self.model_parameters["stateindex_y"]] = dContouringErrorY
        d_contouring_error[0,self.model_parameters["stateindex_theta"]] = dContouringErrorTheta
        #Lag error
        e_l =  cos_phit*(x_virt - Xk[0]) + sin_phit*(y_virt - Xk[1])
        dLagErrorTheta = dtheta*sin_phit*(Xk[0]- x_virt ) - dtheta*cos_phit*(Xk[1]-y_virt ) + cos_phit*dx +sin_phit*dy     
        dLagErrorX = -cos_phit
        dLagErrorY = -sin_phit
        d_contouring_error[1,self.model_parameters["stateindex_x"]] = dLagErrorX
        d_contouring_error[1,self.model_parameters["stateindex_y"]] = dLagErrorY
        d_contouring_error[1,self.model_parameters["stateindex_theta"]] = dLagErrorTheta
        #grad_eC,grad_eL = self.getgradError(track_object,self.mpc_vars,self.car_object,theta_hat,Xk[0],Xk[1])
        return d_contouring_error , e_c , e_l
    


    def getEqualityConstraints(self,xk,uk): 
        latency = self.ComputationDelay
        # given x_k+1 = A x_k + B u_k
        # do the following state augmentation
        # s_k = [x_k,u_k-1], v_k = du_k 
        # with the following linear system
        # s_k+1 = [A B;0 I] s_k + [B;I] v_k      
        nx = self.model_parameters["ax"] 
        nu =  self.model_parameters["au"] 
        #Ad, Bd, gd = self.DiscretizedLinearizedModel(xk,uk,car_object,MPC_vars)
        #Phi_,Gamma_0_,Gamma_ ,gdtemp    = self.sym_rk4_lat(xk,uk,latency,u_prev)
        #Phi,Gamma_0,Gamma ,gdtemp    = self.sym_rk4(xk,uk,latency,u_prev)
        lwb = self.Lwb
        xk = xk.reshape((nx,1))
        uk = uk.reshape((nu,1))
        Phi, Gamma,Gamma_0 , Psi_gk     =  construct_matrix_delay(xk,uk,self.ctrTs,lwb,latency)  #ideal latency==0
        Tx = np.diag(self.Tx)
        Tu = np.diag(self.Tu)
        invTx = np.diag(self.invTx)
        invTu = np.diag(self.invTu)
        invTDu = np.diag(self.invTDu)
        Ak_1 = np.hstack((Tx@Phi@invTx ,Tx@Gamma@invTu)) 
        Ak_2 = np.hstack((np.zeros((nu,nx)) ,np.eye(nu)))
        Ak = np.vstack((Ak_1,Ak_2))
        Bk = np.vstack((Tx@Gamma_0,np.eye(nu)))
        #self.get_logger().warn(f'shape:Tx@Gamma_0@invTu ,{Tx.shape ,Psi_gk.shape , invTu.shape}')
        
        gk = np.vstack((Tx@Psi_gk,np.zeros((nu,1)))) 
                        
        return Ak,Bk,gk 

    def getBounds(self):
        lbx = self.lb_boundsStates
        ubx = self.ub_boundsStates
        lbu =  self.lb_boundsInput
        ubu = self.ub_boundsInput
        return lbx,ubx,lbu,ubu
    def pi_2_pi(self,angle):
        while(angle > np.pi):
            angle = angle - 2.0 * np.pi
        while(angle < -np.pi):
            angle = angle + 2.0 * np.pi
        return angle
    def _getInequalityConstraints(self,border):
        nx = self.ax
        nu = self.au
        track_constraints_pts = np.empty((1,6))
        x_inner,y_inner = border[0],border[1] 
        x_outer,y_outer =  border[2],border[3] 
        delta_X = x_inner - x_outer
        delta_Y = y_inner-y_outer
        A = np.zeros((1,nx+nu))
        bmin = min(x_inner*delta_X+y_inner*delta_Y, x_outer*delta_X+y_outer*delta_Y)
        bmax = max(+x_inner*delta_X+y_inner*delta_Y, x_outer*delta_X+y_outer*delta_Y)       
        A[0,0:2] = np.hstack((delta_X,delta_Y)).reshape(2,)
        A = A@block_diag(self.invTx,self.invTu)
        #print("in loop for setting constraints") 
        #xy_ = np.hstack((x_inner,y_inner,np.zeros((x_inner.shape[0],1))))
        #pdb.set_trace()
        #self.mav_view._add_track_constraints(track_constraints_pts,xy_)
        #pdb.set_trace()
        return A,bmax,bmin    


    
   
    def _get_track_values(self,Xk,Uk) :
        track_object = self.track_object
        theta_hat = np.mod(Xk[self.model_parameters['stateindex_theta']],track_object.length) 
        x_virt,y_virt = track_object.spline_object_center.calc_position(theta_hat)
        dy,dx = track_object.spline_object_center.get_dydx(theta_hat)
        t_angle = np.arctan2(dy, dx)
        trackPoint_dtheta_ref = track_object.spline_object_center.calc_curvature(theta_hat)
        cos_phit =np.cos(t_angle)
        sin_phit = np.sin(t_angle)
        xtyt = np.asarray([x_virt,y_virt]).reshape((2,1))
        dxdy =  np.asarray([dx,dy]).reshape((2,1))
        return theta_hat,xtyt,dxdy,cos_phit,sin_phit,trackPoint_dtheta_ref 
    
    def _solve(self,cost_list): 
        X,U,dU,info = self.hpipmSolve(cost_list)
        return  X,U,dU,info
        #delayed_mpc.update( prev_predicted_states =X_seqs , state_current=X0, prev_control_commands=U_seqs,Uprev= U_prev ,weights=weights_mpc,mpcc_iteration=mpcc_iteration)

    def _updateWeights(self, weights):
        #not needed getting controlled by mpc_vars file
        return   
    
    def getgradError(self,track_object,theta_virt,x,y):   
        deC_dtheta, deL_dtheta, cos_phi_virt, sin_phi_virt = self.getderror_dtheta(track_object, theta_virt, x, y)
        grad_eC = np.hstack([ sin_phi_virt, -cos_phi_virt, np.zeros((1, self.model_parameters["ax"]-3)).flatten(), deC_dtheta]) #contour-Error wrt to all states
        grad_eL = np.hstack([-cos_phi_virt, -sin_phi_virt, np.zeros((1, self.model_parameters["ax"]-3)).flatten(), deL_dtheta] ) #lag-Error wrt to all states    
        return grad_eC,grad_eL
    def getderror_dtheta(self,track_object, theta_virt, x, y):
        #calulate d E_c/ d virtual_theta
        dy_dvirt_theta, dx_dvirt_theta = track_object.spline_object_center.get_dydx(theta_virt)
        phi_angle = np.arctan2(dy_dvirt_theta, dx_dvirt_theta)  #tangengt-1 at the virtual theta
        x_virt ,y_virt = track_object.spline_object_center.calc_position(theta_virt)
        #difference in position between virtual and physical
        Diff_x = x - x_virt
        Diff_y = y - y_virt
        dphivirt_dvirttheta = self.getdphivirt_dtheta(theta_virt,track_object)
        cos_phi_theta = np.cos(phi_angle)
        sin_phi_theta = np.sin(phi_angle)
        tmp1=np.hstack([dphivirt_dvirttheta, 1])
        tmp2=np.vstack([cos_phi_theta , sin_phi_theta])
        MC1 = np.hstack((Diff_x ,Diff_y))
        MC2 = np.hstack((dy_dvirt_theta ,-dx_dvirt_theta))
        MC = np.vstack((MC1,MC2))
        ML1 = np.hstack((-Diff_y ,Diff_x))
        ML2 = np.hstack((dx_dvirt_theta,  dy_dvirt_theta))
        ML  = np.vstack((ML1,ML2))
        deC_dtheta = tmp1 @ MC @ tmp2
        deL_dtheta = tmp1 @ ML @ tmp2
        return deC_dtheta, deL_dtheta, cos_phi_theta, sin_phi_theta
    
    def getdphivirt_dtheta(self,theta_virt,track_object):
        #computes {d phi_virt / d theta} evaluated at theta_k   
        dy_dvirt_theta, dx_dvirt_theta = track_object.spline_object_center.get_dydx(theta_virt)  
        denominator = dx_dvirt_theta**2 + dy_dvirt_theta**2
        d2y_dvirt_theta ,d2x_divrt_theta =  track_object.spline_object_center.get_d2yd2x(theta_virt)
        numerator = d2y_dvirt_theta*dx_dvirt_theta - d2x_divrt_theta*dy_dvirt_theta
        dphivirt_dvirttheta = numerator/denominator
        return dphivirt_dvirttheta

    def getErrors(self,track_object, theta_virt,x_phys,y_phys) :
        theta_hat = theta_virt  #1,N+1
        x_virt,y_virt = track_object.spline_object_center.calc_position(theta_hat)
        ryaw = track_object.spline_object_center.calc_yaw(theta_hat)
        dy,dx = track_object.spline_object_center.get_dydx(theta_hat)
        rk = track_object.spline_object_center.calc_curvature(theta_hat)
        t_angle = np.arctan2(dy, dx)
        cos_phit = np.cos(t_angle)
        sin_phit = np.sin(t_angle)
        eC = -sin_phit*(x_virt - x_phys) + cos_phit*(y_virt - y_phys)
        eL =  cos_phit*(x_virt - x_phys) + sin_phit*(y_virt - y_phys)
        return eC,eL    
    def _set_track_constraints(self,theta_,track_object):
        track_constraints_pts = np.empty((0,6))
        theta_virt = theta_.reshape(self.horizonLength+1,)
        # x_inner,y_inner = self.inner_lut_x(theta_last), self.inner_lut_y(theta_last)
        # x_outer,y_outer = self.outer_lut_x(theta_last), self.outer_lut_y(theta_last)
        # delta_X = x_inner - x_outer
        # delta_Y = y_inner-y_outer
        bmin_ = np.empty(0)
        bmax_ = np.empty(0)
        C_ = np.empty((0,self.ax+self.au))
        #pdb.set_trace()
        for k in range(0,self.horizonLength+1):
            x1,y1 = track_object.spline_object_inner.calc_position(np.mod(theta_virt[k],track_object.length-1))
            x2,y2 = track_object.spline_object_outer.calc_position(np.mod(theta_virt[k],track_object.length-1))
            x1 = x1[0]
            y1 = y1[0]
            x2 = x2[0]
            y2 = y2[0]
            #numerator and denominator slope of the border  - (x2-x1) / (y2-y1)
            numer = -(x2-x1)
            denom =(y2-y1)
            bmin = min(numer*x1 - denom*y1,numer*x2-denom*y2)
            bmax = max(numer*x1 - denom*y1,numer*x2-denom*y2)
            #bmin = min(x_inner[k]*delta_X[k]+y_inner[k]*delta_Y[k], x_outer[k]*delta_X[k]+y_outer[k]*delta_Y[k])
            #bmax = max(+x_inner[k]*delta_X[k]+y_inner[k]*delta_Y[k], x_outer[k]*delta_X[k]+y_outer[k]*delta_Y[k])
            A = np.hstack((numer,-denom)).reshape(2,)
            C = np.hstack((A,np.zeros((self.ax+ self.au -2,))))@np.diag(np.hstack((self.invTx,self.invTu)))
            C_ = np.vstack((C_,C))
            bmin_ = np.hstack((bmin_,bmin))
            bmax_ = np.hstack((bmax_,bmax))
            x_virt,y_virt = track_object.spline_object_center.calc_position(theta_virt[k])
            x_center_virt = np.asarray([x_virt,y_virt])
            track_constraints_pts = np.vstack((track_constraints_pts,np.asarray([x1,y1,0,x2,y2,0])))
            #print("theta virt=",theta_virt[k],"b_min",bmin,"b_max",bmax,"A@xy_[k]",A@x_center_virt)
            #assert bmin<=A@x_center_virt <=bmax
        #pdb.set_trace()
        bmin_=bmin_.reshape(self.horizonLength+1,)
        bmax_ = bmax_.reshape(self.horizonLength+1,)
        C_ = C_.reshape(self.horizonLength+1,self.ax+self.au)
        self.bmin_ = bmin_
        self.bmax_ = bmax_
        self.C_ = C_
        return track_constraints_pts

    def _getBorderConstraint(self,i):
        return self.C_[i],self.bmax_[i],self.bmin_[i] 
 

    def pi_2_pi(self,angle):
        while(angle > np.pi):
                angle = angle - 2.0 * np.pi
        while(angle < -np.pi):
                angle = angle + 2.0 * np.pi
        return angle        
    
    
    
    
    
    def hpipmSolve(self,cost_list):
        # check that env.sh has been run
        codegen_data = 0; # export qp data in the file ocp_qcqp_data.c for use from C examples
        env_run = os.getenv('ENV_RUN')
        if env_run!='true':
            print('ERROR: env.sh has not been sourced! Before executing this example, run:')
            print('source env.sh')
            sys.exit(1)
        
        nx = self.model_parameters["ax"]
        nu = self.model_parameters["au"]
        N =  self.horizonLength
        startTime = time.time()
        dim = hpipm_ocp_qp_dim(N)
        dim.set('nx',nx+nu,0,N) #number of States
        dim.set('nu',nu,0,N-1)  #number of inputs
        dim.set('nbx',nx+nu,0,N) #number of state bounds
        dim.set('nbu',nu,0,N-1)  #nuber of Input bounds
        dim.set('ng',0,0)
        dim.set('ng',1,1,N) #general polytopic constraints 
        if codegen_data:
            dim.codegen('mpcc.c', 'w')
        qp = hpipm_ocp_qp(dim)
        x0 = np.diag(np.hstack((self.Tx,self.Tu)))@np.vstack((cost_list[0]["_X0"],cost_list[0]["_U0"]))
        for i in range(N):
            qp.set('A',cost_list[i]["_Ak"],i)
            qp.set('B',cost_list[i]["_Bk"],i)
            qp.set('b',cost_list[i]["_gk"],i)
            qp.set('Q',cost_list[i]["_Qk"],i)
            qp.set('q',cost_list[i]["_qk"],i)
            qp.set('R',cost_list[i]["_Rk"],i)
        i=N
        qp.set('Q',cost_list[i]["_Qk"],i)
        qp.set('q',cost_list[i]["_qk"],i)

            
        #set bounds
        
        for i in range(N+1):
            qp.set('Jbx', np.eye((nx+nu)), i)
            if (i == 0):
                qp.set('lbx', x0, i)
                qp.set('ubx', x0, i)
            else:
                qp.set('lbx', cost_list[i]["_lbx"][0:nx+nu], i)
                qp.set('ubx', cost_list[i]["_ubx"][0:nx+nu], i)
        
            if (i<N):
                qp.set('Jbu', np.eye(nu), i)
                qp.set('lbu', cost_list[i]["_lbu"][0:nu], i)
                qp.set('ubu', cost_list[i]["_ubu"][0:nu], i)
        
        #Constraints
        for i in range(N+1):
            qp.set('C', cost_list[i]["_Ck"], i)
            qp.set('lg', cost_list[i]["_lg"], i)
            qp.set('ug', cost_list[i]["_ug"], i)  
        # print to shell
        #qp.print_C_struct()
        # codegen
        if codegen_data:
            qp.codegen('mpcc.c', 'a')    
        # qp sol
        qp_sol = hpipm_ocp_qp_sol(dim)
        #args
        # set up solver arg
        #mode = 'speed_abs'
        #mode = 'speed'
        #mode = 'balance'
        mode = 'robust'
        # create and set default arg based on mode
        arg = hpipm_ocp_qp_solver_arg(dim, mode)


        # create and set default arg based on mode
        arg.set('mu0', 1e0)
        arg.set('iter_max', 2000)
        arg.set('tol_stat', 1e-6)
        arg.set('tol_eq', 1e-6)
        arg.set('tol_ineq', 1e-6)
        arg.set('tol_comp', 1e-5)
        arg.set('reg_prim', 1e-12)
        if codegen_data:
            arg.codegen('mpcc.c', 'a') 
        ####
        #solver
        solver = hpipm_ocp_qp_solver(dim, arg)
        solver.solve(qp, qp_sol)
        # get solver statistics
        status = solver.get('status')
    
        if status==0:
                    # Log the warning message in red
            self.get_logger().info(f'{self.bold_text}{self.green_color}Solver-Success!{self.reset_color}{self.reset_text}')
        else:
            self.get_logger().info(f'{self.bold_text}{self.red_color}Solver Failed - Not-Success!{self.reset_color}{self.reset_text}')
        
        
        #extract and print sol
        u_opt = np.zeros((nu,N))
        x_opt = np.zeros((nx+nu,N+1))
        for i in range(N):
            u_opt[:,i] = (qp_sol.get('u', i)).squeeze()
        for i in range(N+1):
            x_opt[:,i] = (qp_sol.get('x', i)).squeeze()
        ###############################################################
        endTime = time.time()

        #rescale outputs
        X = (np.array(x_opt[0:nx,:])*self.invTx[..., np.newaxis]).reshape((nx,N+1))
        U = (np.array(x_opt[nx:,1:])*self.invTu[..., np.newaxis]).reshape((nu,N))
        dU = np.array(u_opt)

        info = {"exitflag":0.06,"QPtime":1}
        if status == 0:
            info["exitflag"] = 0
        else:
            info["exitflag"] = 1
        info["QPtime"] = endTime-startTime
        return X,U,dU,info

    def publish_boundary_markers(self, right_points, left_points):
        boundary_array = MarkerArray()
        combined_points = np.row_stack((right_points, left_points))
        #delta = right_points - left_points
        #angles = np.arctan2(delta[:, 0], -delta[:, 1])
        #self.get_logger().info('markers-predicted-size: "%s"' %combined_points.shape[0])  
        for i in range(combined_points.shape[0]):
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.id = i
            path_marker.type = path_marker.SPHERE
            path_marker.action = path_marker.ADD
            path_marker.scale.x = 0.15
            path_marker.scale.y = 0.15 
            path_marker.scale.z = 0.15
            path_marker.color.a = 1.0
            path_marker.color.r =  1.0 
            #path_marker.pose.orientation = self.heading(angles[i % right_points.shape[0]])
            path_marker.pose.position.x = float(combined_points[i, 0]) 
            path_marker.pose.position.y = float(combined_points[i, 1])
            path_marker.pose.position.z = float(combined_points[i, 2])
            boundary_array.markers.append(path_marker)
        self.boundary_pub.publish(boundary_array)       
    
    
    
    def publish_boundary_track(self,inner,center,outer):
        self.get_logger().info('hello')
        boundary_array = MarkerArray()
        combined_points = np.row_stack((inner,center,outer))
        for i in range(combined_points.shape[0]):
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.id = i
            path_marker.type = path_marker.SPHERE
            path_marker.action = path_marker.ADD
            path_marker.scale.x = 0.15
            path_marker.scale.y = 0.15 
            path_marker.scale.z = 0.15
            path_marker.color.a = 1.0
            path_marker.color.g =  1.0 
            #path_marker.pose.orientation = self.heading(angles[i % right_points.shape[0]])
            path_marker.pose.position.x = float(combined_points[i, 0]) 
            path_marker.pose.position.y = float(combined_points[i, 1])
            boundary_array.markers.append(path_marker)
        self.track_pub.publish(boundary_array)  
    
    
    def publish_mpc_pred(self,prediction):
        boundary_array = MarkerArray()
        combined_points = prediction
        for i in range(combined_points.shape[0]):
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.id = i
            path_marker.type = path_marker.SPHERE
            path_marker.action = path_marker.ADD
            path_marker.scale.x = 0.15
            path_marker.scale.y = 0.15 
            path_marker.scale.z = 0.15
            path_marker.color.a = 1.0
            path_marker.color.r =  1.0 
            #path_marker.pose.orientation = self.heading(angles[i % right_points.shape[0]])
            path_marker.pose.position.x = float(combined_points[i, 0]) 
            path_marker.pose.position.y = float(combined_points[i, 1])
            boundary_array.markers.append(path_marker)
        self.mpc_pred.publish(boundary_array) 

    
    def calculateDistance(self,Car_coordinates,Obs_coordinates):
        # assume circle
        inf_radius = self.inf_radius 
        safety_dist  = self.safety_dist
        distance = ((Car_coordinates[0] - Obs_coordinates[0])**2 + (Car_coordinates[1] - Obs_coordinates[1])**2 )**0.5
        if distance < inf_radius + safety_dist :
            isCollisionRange = True
        else:
            isCollisionRange = False
        return isCollisionRange , distance


    def obsInfo_cb(self,msg:ObstacleArray):
        x_obs =None
        y_obs=None
        yaw_obs= None
        #self.get_logger().info(f'{self.bold_text}{self.green_color}{len(msg.obstacles)}{self.reset_color}{self.reset_text}')
        if len(msg.obstacles) == 0 :
             self.OvertakingMode = False
             return
        else :
            for obstacle in msg.obstacles:
                x_obs = obstacle.x_pos
                y_obs = obstacle.y_pos
                yaw_obs = obstacle.yaw
        
            x_car  = self.ControllerRbtState.x
            y_car = self.ControllerRbtState.y
            yaw_car = self.ControllerRbtState.yaw
            
            Car_coordinates = np.array([x_car,y_car,yaw_car])
            self.Obs_coordinates = np.array([x_obs , y_obs , yaw_obs])
            isCollisionRange , distance = self.calculateDistance(Car_coordinates , self.Obs_coordinates)
            if isCollisionRange:
                self.OvertakingMode = True
            else:
                self.OvertakingMode = False 




def main(args=None):
    rclpy.init()
    mpcc_node = MPCCNode()
    #use mltithreadExecutor
    rclpy.spin(mpcc_node)
    
    try:
        rclpy.spin()
    finally:
        mpcc_node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()