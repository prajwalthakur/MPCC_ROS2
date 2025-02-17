import numpy as np
from abc import ABC, abstractclassmethod
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import math
import scipy.integrate
from abc import abstractmethod
import pdb
from .findtheta import *
# Temporal State Vector #
#########################
# Colors
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'
class TemporalState:
    def __init__(self, X0):
        """veh_params = {"v_max": 15.0,
              "length": 0.568,
              "width": 0.296,
              "mass": 3.74,
              "dragcoeff": 0.075, 
              "curvlim": 3.0, 
              "g": 9.81}"""
        """
        Temporal State Vector containing car pose (x, y, psi)
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param psi: yaw angle | [rad]
        :param s : arc-length traversed
        """
        #xdot = X_dot , Y_dot , psi_dot , v_x_dot,v_y_dot , w_dot ,u(3)
        #xvars = ['posx', 'posy', 'phi', 'vx', 'theta']
        #uvars = ['vdot', 'delta']
        self.x = X0[0,0]
        self.y = X0[1,0]
        self.psi = X0[2,0]
        self.theta  = X0[3,0]
        self.members = ['x', 'y', 'psi','theta']
    def __iadd__(self, other):
        """
        Overload Sum-Add operator.
        :param other: numpy array to be added to state vector
        """
        for state_id in range(len(self.members)):
            vars(self)[self.members[state_id]] += other[state_id]
        return self
class car_physical_model(ABC):
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
        
class carObject(car_physical_model):
    def __init__(self, x_init,Ts,reference_path,latency,last_closestIdx=10):
        """
        :param reference_path : reference path object to follow
        : param length length of car in m        
        """
        super(carObject, self).__init__()
        self.Ts = Ts ;          " dt is the simulation time step "
        self.latency = latency
        self.reference_path = reference_path
        _,self.wayPointCenter,_ = self.reference_path.get_coordinates(self.reference_path.dict_waypoints)
        self.Trackwidth = self.reference_path.Trackwidth
        _,self.theta_coordinates,_ = self.reference_path.get_theta_coordinates()
        self.closestIndex = None
        self.arc,self.closestIndex,last_closestIdx = findTheta(currentPose=np.asarray([x_init[0],x_init[1]]),TrackCenter=self.wayPointCenter,theta_coordinates=self.theta_coordinates,trackWidth=self.Trackwidth, last_closestIdx=last_closestIdx)
        self.last_closestIdx = last_closestIdx
        self.temporalState = TemporalState(x_init.reshape((self.model_parameters["ax"],1)))
        self.boundaryTrackConstraint_left = np.zeros((1,3))
        self.boundaryTrackConstraint_right = np.zeros((1,3))
        self.mpcPredictionPoints = np.zeros((1,3))
        self.timeElapsed = 0
        self.mpcCostCOmponents = 0
        self.updatePredictionPoints()
        
    def updatePredictionPoints(self):
        #pdb.set_trace()
        self.predictionPoints = np.vstack((self.mpcPredictionPoints,self.boundaryTrackConstraint_left,self.boundaryTrackConstraint_right))
    def pi_2_pi(self,angle):
        while(angle > np.pi):
            angle = angle - 2.0 * np.pi

        while(angle < -np.pi):
            angle = angle + 2.0 * np.pi

        return angle
    def changelatencyInSimulation(self,delay):
        self.latency = delay
        return 
    def get_simulation_next_state(self,x_current,u_current,T):
        ax= self.model_parameters["ax"]
        au = self.model_parameters["au"]
        i=0
        x_0 = x_current[:ax].reshape(ax,) 
        solution= scipy.integrate.solve_ivp(self.getXdot , t_span=[0,T] ,y0 =x_0,t_eval=[T],args=[np.squeeze(u_current.T)])
        v = solution.t
        s = solution.y
        self.arc,self.closestIndex,self.last_closestIdx = findtheta.findTheta(currentPose=np.hstack((solution.y[0],solution.y[1])),TrackCenter=self.wayPointCenter,theta_coordinates=self.theta_coordinates,trackWidth=self.Trackwidth, last_closestIdx=self.last_closestIdx)
        #x, y, psi,v,theta
        X0 = solution.y.reshape((ax,1))
        X0[self.model_parameters["stateindex_psi"],0] = self.pi_2_pi(X0[self.model_parameters["stateindex_psi"]])
        X0[self.model_parameters["stateindex_theta"],0] = self.arc
        self.temporalState = TemporalState(X0)
        return X0.reshape((ax,1))
    def get_simulation_next_state_open(self,x_current,u_current,T):
        ax= self.model_parameters["ax"]
        au = self.model_parameters["au"]
        i=0
        x_0 = x_current[:ax].reshape(ax,) 
        solution= scipy.integrate.solve_ivp(self.getXdot , t_span=[0,T] ,y0 =x_0,t_eval=[T],args=[np.squeeze(u_current.T)])
        v = solution.t
        s = solution.y
        states = solution.y.reshape((ax,1))
        return states
    # def getXdot(self,t,x,u):
    #     l_f = self.model_parameters["lf"] 
    #     l_r = self.model_parameters["lr"]
    #     lwb = l_f + l_r
    #     xdot = np.asarray([[u[0]*np.cos(x[2]) ],
    #             [u[0]*np.sin(x[2])] ,
    #             [u[0]*np.tan(u[1])/(lwb)],
    #             [u[2]]])    
    #     return xdot.squeeze()
    
    def getXdot(self,t,x,u):
        psi = x[self.model_parameters["stateindex_psi"]]
        current_v =x[self.model_parameters["stateindex_v"]]
        vdot = u[self.model_parameters["inputindex_vdot"]]
        delta = u[self.model_parameters["inputindex_delta"]]
        thetadot = u[self.model_parameters["inputindex_thetadot"]]
        l_f = self.model_parameters["lf"] 
        l_r = self.model_parameters["lr"]
        lwb = l_f + l_r
        #states = [x,y,psi,v,delta,theta,d]
        #u = [d_dot,delta_dot,thetadot]
        xdot = np.asarray([[current_v*np.cos(psi) ],
                [current_v*np.sin(psi)] ,
                [current_v*np.tan(delta)/(lwb)],
                [vdot],
                [thetadot]])
        
        return xdot.squeeze()
    
    

    def get_car_vertices(self):
        l = self.model_parameters["L"]/2  # length of mobile robot
        w = self.model_parameters["W"]/2   # width of mobile robot
        x = self.temporalState.x
        y = self.temporalState.y
        psi = self.temporalState.psi
        # Mobile robot coordinates wrt body frame
        #pdb.set_trace()
        mr_co = np.array([[-l/2, l/2, l/2, -l/2],
                        [-w/2, -w/2, w/2, w/2]])
        R_psi = np.array([[np.cos(psi), -np.sin(psi)],
                            [np.sin(psi), np.cos(psi)]])  # rotation matrix
        v_pos = np.dot(np.squeeze(R_psi), np.squeeze(mr_co))  # orientation w.r.t intertial frame              
        v_pos[0,:] = v_pos[0, :] + x
        v_pos[1,:] = v_pos[1, :] + y
        return v_pos
    def plot_simple_Car(self):
        l = self.model_parameters["L"]  # length of mobile robot
        w = self.model_parameters["W"]   # width of mobile robot
        x = self.temporalState.x[0]
        y = self.temporalState.y[0]
        psi = self.temporalState.psi[0]
        # Mobile robot coordinateself.X0 =s wrt body frame
        #pdb.set_trace()
        mr_co = np.array([[-l/2, l/2, l/2, -l/2],
                        [-w/2, -w/2, w/2, w/2]])
        R_psi = np.array([[np.cos(psi), -np.sin(psi)],
                            [np.sin(psi), np.cos(psi)]])  # rotation matrix
        v_pos = np.dot(R_psi, mr_co)  # orientation w.r.t intertial frame
        ax =plt.gca()
        ax.fill(v_pos[0, :] + x, v_pos[1, :] + y, 'g')  # rotation + translation to get global coordinates  
        #plt.legend(['MR'], fontsize=24)
        # plt.xlabel('x,[m]')
        # plt.ylabel('y,[m]')
        # plt.axis([-1, 3, -1, 3])
        # plt.axis('square')
        # plt.grid(True)
        #plt.show(block=False)
        # plt.pause(0.1)
        # plt.clf()  
        return 
    def show(self):
        """
        Display car on current axis.
        """
        l = self.model_parameters["L"]  # length of mobile robot
        w = self.model_parameters["W"]   # width of mobile robot
        x = self.temporalState.x[0]
        y = self.temporalState.y[0]
        psi = self.temporalState.psi[0]
        # Get car's center of gravity
        cog = (x, y)
        # Get current angle with respect to x-axis
        yaw = np.rad2deg(psi)
        # Draw rectangle
        car = plt_patches.Rectangle(cog, width=w, height=w,
                                    angle=yaw, facecolor=CAR,
                                    edgecolor=CAR_OUTLINE, zorder=20)

        # Shift center rectangle to match center of the car
        car.set_x(car.get_x() - (l / 2 *
                                 np.cos(psi) -
                                 w / 2 *
                                 np.sin(psi)))
        car.set_y(car.get_y() - (w / 2 *
                                 np.cos(psi) +
                                 l / 2 *
                                 np.sin(psi)))

        # Add rectangle to current axis
        ax = plt.gca()
        ax.add_patch(car)
    #self.car.get_next_state(self.u_current,self.uprev,self.X0,self.X_seqs,track_constraints_pts,costComponents)    
    def get_next_state(self,u_current,u_prev,x0,x_open_loop_predictions,track_constraints_pts,costComponents,openLoop=False):
        x_open_loop_predictions = x_open_loop_predictions[0:self.model_parameters["ax"],:]
        #delay =np.random.uniform(0.100,0.199,1)[0] #0.190  #120 ms
        x0 = x0
        if openLoop:
                pdb.set_Trace() #sol,last_closestIndex = car.get_simulation_next_state_open(x0,u_current,T=DiscreteTime_step,last_closestIndex)
        else:
                if self.latency!=0:
                        #print("in delay")
                        sol = self.get_simulation_next_state(x0,u_prev,T=self.latency) #initial state
                        x0 = sol
                sol = self.get_simulation_next_state(x0,u_current, T= abs(self.Ts - self.latency)) #initial state
        
        #update the prediction states and trackBoundary points as well
        self.boundaryTrackConstraint_left = track_constraints_pts[0:,0:3]
        self.boundaryTrackConstraint_right = track_constraints_pts[0:,3:]
        self.mpcPredictionPoints = np.hstack((x_open_loop_predictions.T[0:,0:2],np.zeros((x_open_loop_predictions.T.shape[0],1))))
        self.updatePredictionPoints()
        self.timeElapsed = self.timeElapsed + self.Ts
        self.curretntControl(u_current)
        self.mpcCostCOmponents = costComponents
        return sol
    def curretntControl(self,u_current):
        self.vdot = u_current[self.model_parameters["inputindex_vdot"]]
        self.delta = u_current[self.model_parameters["inputindex_delta"]]
        self.thetadot = u_current[self.model_parameters["inputindex_thetadot"]]
    def returnControl(self):
        return  self.vdot, self.delta,self.thetadot
    def returnCostComponents(self):
        return  self.mpcCostCOmponents
if __name__ == '__main__':
    car = carObject(0.02)
    x0 = np.asarray([-0.8457,1.0979,-0.7854,1.0000,0,0,0])
    #x0 = np.expand_dims(x0,axis=1)
    u_current = np.asarray([0,0,0])
    sol = car.get_simulation_next_state(x0,u_current)
    #plot_simple_Car(2,2,-np.pi/3)