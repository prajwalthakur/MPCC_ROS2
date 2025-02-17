import numpy as np
import math

import matplotlib.pyplot as plt
from scipy import sparse
from .CubicSpline import *
import pdb
from .load_map import load_map3 as  load_map2
from .border_adjustment import *
import dill 
# Colors
DRIVABLE_AREA = '#BDC3C7'
WAYPOINTS = '#D0D3D4'
PATH_CONSTRAINTS = '#F5B041'
OBSTACLE = '#2E4053'


############
# Waypoint #
############
which_object =None
filename_object = f"racetrackStoredInfo/racetrack_{which_object}_spline_object.dill"
which_waypoints = None
filename_list = f"racetrackStoredInfo/racetrack_waypoints_{which_waypoints}.dill"
class Waypoint:
    def __init__(self, x, y, psi, kappa,dist,dy,dx):
        """
        Waypoint object containing x, y location in global coordinate system,
        orientation of waypoint psi and local curvature kappa. Waypoint further
        contains an associated reference velocity computed by the speed profile
        and a path width specified by upper and lower bounds.
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param psi: orientation of waypoint | [rad]
        :param kappa: local curvature | [1 / m]
        """
        self.x = x
        self.y = y
        self.psi = psi
        self.kappa = kappa
        self.dist = dist
        self.dy  = dy
        self.dx = dx
        # Reference velocity at this waypoint according to speed profile
        self.v_ref = None

        # Information about drivable area at waypoint
        # upper and lower bound of drivable area orthogonal to
        # waypoint orientation.
        # Upper bound: free drivable area to the left of center-line in m
        # Lower bound: free drivable area to the right of center-line in m
        self.lb = None
        self.ub = None
        self.static_border_cells = None
        self.dynamic_border_cells = None

    # def __sub__(self, other):
    #     """
    #     Overload subtract operator. Difference of two waypoints is equal to
    #     their euclidean distance.
    #     :param other: subtrahend
    #     :return: euclidean distance between two waypoints
    #     """
    #     return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5


##################
# Reference Path #
##################


class racingTrack:
    def __init__(self, resolution, smoothing_distance,
                 max_width, circular):
        """
        Reference Path object. Create a reference trajectory from specified
        corner points with given resolution. Smoothing around corners can be
        applied. Waypoints represent center-line of the path with specified
        maximum width to both sides.
        :param map: map object on which path will be placed
        :param wp_x: x coordinates of corner points in global coordinates
        :param wp_y: y coordinates of corner points in global coordinates
        :param resolution: resolution of the path in m/wp
        :param smoothing_distance: number of waypoints used for smoothing the
        path by averaging neighborhood of waypoints
        :param max_width: maximum width of path to both sides in m
        :param circular: True if path circular
        """

        # Precision
        self.eps = 1e-12

        # Map
        self.map = map

        # Resolution of the path
        self.resolution = resolution

        # Look ahead distance for path averaging
        self.smoothing_distance = smoothing_distance

        # Circular flag
        self.circular = circular

        # List of waypoint objects
        self.waypoints_list = self._create_map()
        # Number of waypoints
        self.dict_waypoints = {"waypoints_inner": self.waypoints_list[0:,0] , "waypoints_center" : self.waypoints_list[0:,1], "waypoints_outer":self.waypoints_list[0:,2]}
        #self._plot_map()
        self.n_waypoints = len(self.waypoints_list)
        # Length of path
        self.segment_lengths,self.length= self._compute_length()

        # Compute path width (attribute of each waypoint)
        #self._compute_width(max_width=max_width)
        #pdb.set_trace()
        self.Trackwidth =  ((self.dict_waypoints["waypoints_inner"][0].x- self.dict_waypoints["waypoints_outer"][0].x)**2 + (self.dict_waypoints["waypoints_inner"][0].y- self.dict_waypoints["waypoints_outer"][0].y)**2)**0.5
       
        
    def _construct_path(self, wp_x, wp_y,no_points=None):
        # no of interpolated-waypoints considering resolution
        n_wp = [int(np.sqrt((wp_x[i+1] - wp_x[i])**2 + 
                    (wp_y[i+1] - wp_x[i])**2)/self.resolution) for i in range(len(wp_x)-1)]
        
        #splinfy using cubic polynomial-spline (mpcc paper implemented there own spline function , method not mentioned in the paper)
        
        spline_object = Spline2D(wp_x,wp_y)
        if (no_points == None):
            resolution = self.resolution
        else :
            resolution = spline_object.s[-1]/no_points
        sp_new_wp_dist = np.arange(spline_object.s[0],spline_object.s[-1],resolution)
        gp_x, gp_y = wp_x[-1], wp_y[-1]
        waypoints =np.empty((0,1))
        rx, ry, ryaw, rk = [], [], [], []
        thetaCenter = np.empty((0,1))
        xCenter = np.empty((0,1))
        yCenter = np.empty((0,1))
        for wp_dist in sp_new_wp_dist :
            ix,iy = spline_object.calc_position(wp_dist)
            ryaw = spline_object.calc_yaw(wp_dist)
            dy,dx = spline_object.get_dydx(wp_dist)
            rk = spline_object.calc_curvature(wp_dist)
            rx.append(ix)
            ry.append(iy)
            #pdb.set_trace()
            wp = np.asarray([Waypoint(x=ix,y=iy,psi=ryaw,kappa=rk,dist=wp_dist,dy=dy,dx=dx)])
            waypoints = np.vstack((waypoints,wp))
            thetaCenter = np.vstack((thetaCenter,wp_dist))
            xCenter =  np.vstack((xCenter,ix))
            yCenter = np.vstack((yCenter,iy))
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.thetaCenter = thetaCenter
        return waypoints,spline_object
    def _construct_path_border(self,wp_x, wp_y,l,waypoints_center,spline_object_center):
        waypoints_center = waypoints_center.squeeze()
        no_points = l
        refine = 50
        nbreaks = len(waypoints_center)
        spline_object_old = Spline2D(wp_x,wp_y)
        resolution = spline_object_old.s[-1]/nbreaks
        sp_new_wp_dist = np.arange(0,spline_object_old.s[-1],resolution)
        wp_x, wp_y = spline_object_old.calc_positionArray(sp_new_wp_dist)
        t = np.zeros((len(waypoints_center),1))
        t[0]  = waypoints_center[0].dist
        t[-1] = waypoints_center[-1].dist  
        xCenter = self.xCenter
        yCenter = self.yCenter
        thetaCenter = self.thetaCenter
        for i in range(1,nbreaks-1):
            cX = wp_x[i]
            cY = wp_y[i]
            index = np.argmin(np.square((xCenter-cX)) + np.square((yCenter-cY)))
            if(index == nbreaks-1):
                pnext = thetaCenter[index]
            else:
                pnext = thetaCenter[index+1]
            if(index==0):
                pprev = thetaCenter[0]
            else:
                pprev = thetaCenter[index-1]
            Xref,Yref = spline_object_center.calc_positionArray(np.linspace(pprev,pnext,num=refine))
            indexRefined = np.argmin(np.square((Xref-cX)) + np.square((Yref-cY)))
            indexFinal = pprev + indexRefined*(pnext-pprev)/refine
            t[i] = indexFinal
        spline_object = Spline2D(wp_x,wp_y,t.squeeze(),border=True)
        sp_new_wp_dist = np.array(spline_object.s)
        gp_x, gp_y = wp_x[-1], wp_y[-1]
        waypoints =np.empty((0,1))
        rx, ry, ryaw, rk = [], [], [], []
        resolution = spline_object.s[-1]/no_points
        sp_new_wp_dist = np.arange(spline_object.s[0],spline_object.s[-1],resolution)
        for wp_dist in sp_new_wp_dist :
            ix,iy = spline_object.calc_position(wp_dist)
            ryaw = spline_object.calc_yaw(wp_dist)
            dy,dx = spline_object.get_dydx(wp_dist)
            rk = spline_object.calc_curvature(wp_dist)
            rx.append(ix)
            ry.append(iy)
            wp = np.asarray([Waypoint(x=ix.squeeze(),y=iy.squeeze(),psi=ryaw,kappa=rk.squeeze(),dist=wp_dist,dy=dy.squeeze(),dx=dx.squeeze())])
            waypoints = np.vstack((waypoints,wp))
        return waypoints,spline_object    
    
    
    
    def _compute_length(self):
        """
        Compute length of center-line path as sum of euclidean distance between
        waypoints.
        :return: length of center-line path in m
        """
        _,waypoints,_ = self.get_coordinates(self.dict_waypoints)
        segment_lengths = [0.0] + [np.linalg.norm(waypoints[wp_id+1] - waypoints
                    [wp_id]) for wp_id in range(len(waypoints)-1)]
        s = sum(segment_lengths)
        segment_lengths = np.asarray(segment_lengths)
        return segment_lengths,s
    
    def dump_file(self,which_object ,which_waypoints,spline_object,waypoints_list):
        filename_object = f"racetrackStoredInfo/racetrack_{which_object}_spline_object.dill"
        filename_list = f"racetrackStoredInfo/racetrack_waypoints_{which_waypoints}.dill"
        with open(filename_object, "wb") as file:
            dill.dump(spline_object, file)
        with open(filename_list, "wb") as file:
            dill.dump(waypoints_list, file)
        return 
    def load_file(self,which_object ,which_waypoints):
        filename_object = f"racetrackStoredInfo/racetrack_{which_object}_spline_object.dill"
        filename_list = f"racetrackStoredInfo/racetrack_waypoints_{which_waypoints}.dill"
        with open(filename_object, "rb") as file:
            spline_object = dill.load(file)
        with open(filename_list, "rb") as file:
            waypoints_list= dill.load(file)
        return  waypoints_list,spline_object
        
        
    def _create_map(self):
        old_waypoints = load_map2()
        ModelParams = {"W":0.2,"Scale":1} #dont scale anything
        safetyScaling = 0.5
        # ModelParams = {"W":0.34,"Scale":1} #dont scale anything
        # safetyScaling = 0.6
        #waypoints = old_waypoints
        waypoints,_ = border_adjustment(old_waypoints,ModelParams,safetyScaling)
        #waypoints = np.vstack((waypoints,waypoints[0],waypoints[1]))
        #waypoints = np.vstack((waypoints)) # to join some gaps left while creating the map
        waypoints_center,self.spline_object_center = self._construct_path(waypoints[0:,2],waypoints[0:,3])
        #pdb.set_trace()
        #self.dump_file("center" ,"center",self.spline_object_center,waypoints_center)
        #waypoints_center_loaded,self.spline_object_center_loaded  = self.load_file("center" ,"center")
        l=len(waypoints_center)
        waypoints_outer,self.spline_object_outer = self._construct_path_border(waypoints[0:,4],waypoints[0:,5],l,waypoints_center,self.spline_object_center)
        #self.dump_file("outer" ,"outer",self.spline_object_outer,waypoints_outer)
        waypoints_inner,self.spline_object_inner = self._construct_path_border(waypoints[0:,0],waypoints[0:,1],l,waypoints_center,self.spline_object_center)
        #self.dump_file("inner" ,"inner",self.spline_object_inner,waypoints_inner)  
        waypoints_list  = np.hstack((waypoints_inner,waypoints_center,waypoints_outer))
        return waypoints_list
        
    
    def get_coordinates(self,waypoints):
        wp_center = waypoints["waypoints_center"]
        wp_inner = waypoints["waypoints_inner"]
        wp_outer = waypoints["waypoints_outer"]
        xy_inner = np.empty((0,2))
        xy_center = np.empty((0,2))
        xy_outer = np.empty((0,2))
        # pdb.set_trace()
        for i,c,outer in zip(wp_inner,wp_center,wp_outer):
            xy_inner = np.vstack((xy_inner,np.array([i.x,i.y])))
            xy_center = np.vstack((xy_center,np.array([c.x,c.y])))
            xy_outer = np.vstack((xy_outer,np.array([outer.x,outer.y])))
        return xy_inner,xy_center,xy_outer
    
    def calc_pose_contour_error(self,theta,X):
        x_virt,y_virt = self.spline_object_center.calc_position(theta)
        dy,dx = self.spline_object_center.get_dydx(theta)
        t_angle = np.arctan2(dy, dx)
        cos_phit = np.cos(t_angle)
        sin_phit =np.sin(t_angle)
        contouring_Error = -sin_phit*(x_virt - X[0]) + cos_phit*(y_virt - X[1]) 
        return x_virt,y_virt,t_angle,contouring_Error
     
    def get_approx_des_pose(self,theta) :
        x_virt,y_virt = self.spline_object_center.calc_position(theta)
        dy,dx = self.spline_object_center.get_dydx(theta)
        t_angle = np.arctan2(dy, dx)
        #cos_phit = np.cos(t_angle)
        #sin_phit =np.sin(t_angle)
        yaw = math.atan2(dy, dx)
        return x_virt , y_virt , yaw 
    def calc_approx_contour_error(self,theta,X):
        x_virt,y_virt = self.spline_object_center.calc_position(theta)
        dy,dx = self.spline_object_center.get_dydx(theta)
        t_angle = np.arctan2(dy, dx)
        cos_phit = np.cos(t_angle)
        sin_phit =np.sin(t_angle)
        contouring_Error = -sin_phit*(x_virt - X[0]) + cos_phit*(y_virt - X[1]) 
        return contouring_Error
    def get_theta_coordinates(self):
        wp_center = self.dict_waypoints["waypoints_center"]
        wp_inner = self.dict_waypoints["waypoints_inner"]
        wp_outer = self.dict_waypoints["waypoints_outer"]
        xy_inner = np.empty((0,4))
        xy_center = np.empty((0,4))
        xy_outer = np.empty((0,4))
        #pdb.set_trace()
        idx=0
        for i,c,outer in zip(wp_inner,wp_center,wp_outer):
            xy_inner = np.vstack((xy_inner,np.array([idx,i.x,i.y,i.dist])))
            xy_center = np.vstack((xy_center,np.array([idx,c.x,c.y,c.dist])))
            xy_outer = np.vstack((xy_outer,np.array([idx,outer.x,outer.y,outer.dist])))
            idx+=1
        return xy_inner,xy_center,xy_outer
    def get_coordinates_mpcc(self):
        wp_center = self.dict_waypoints["waypoints_center"]
        wp_inner = self.dict_waypoints["waypoints_inner"]
        wp_outer = self.dict_waypoints["waypoints_outer"]
        xy_inner = np.empty((0,6))
        xy_center = np.empty((0,6))
        xy_outer = np.empty((0,6))
        #pdb.set_trace()
        idx=0
        for i,c,outer in zip(wp_inner,wp_center,wp_outer):
            xy_inner = np.vstack((xy_inner,np.array([idx,i.x,i.y,i.dist,i.dy,i.dx])))
            xy_center = np.vstack((xy_center,np.array([idx,c.x,c.y,c.dist,c.dy,c.dx])))
            xy_outer = np.vstack((xy_outer,np.array([idx,outer.x,outer.y,outer.dist,outer.dy,outer.dx])))
            idx+=1
        return xy_inner,xy_center,xy_outer
    
    def get_coordinates_perception(self):
        wp_center = self.dict_waypoints["waypoints_center"]
        wp_inner = self.dict_waypoints["waypoints_inner"]
        wp_outer = self.dict_waypoints["waypoints_outer"]
        xy_inner = np.empty((0,6))
        xy_center = np.empty((0,6))
        xy_outer = np.empty((0,6))
        #pdb.set_trace()
        idx=0
        for i,c,outer in zip(wp_inner,wp_center,wp_outer):
            xy_inner = np.vstack((xy_inner,np.array([i.x,i.y, i.psi ,           i.dist,i.dy,i.dx])))
            xy_center = np.vstack((xy_center,np.array([c.x,c.y,c.psi ,           c.dist,c.dy,c.dx])))
            xy_outer = np.vstack((xy_outer,np.array([outer.x,outer.y,outer.psi , outer.dist,outer.dy,outer.dx])))
            idx+=1
        return xy_inner,xy_center,xy_outer
    
    def calc_euclid_distance(self, coordinate1:np.ndarray , coordinate2:np.ndarray):
        """
        Overload subtract operator. Difference of two waypoints is equal to
        their euclidean distance.
        :param other: subtrahend
        :return: euclidean distance between two waypoints
        """
        return ((coordinate1[0] - coordinate2[0])**2 + (coordinate1[1] - coordinate2[1])**2)**0.5
    
    def get_dist_boundary(self,theta):
            x1,y1 = self.spline_object_inner.calc_position(np.mod(theta,self.length-1))
            right_coordinate = np.array([x1,y1])
            x2,y2 =  self.spline_object_outer.calc_position(np.mod(theta,self.length-1))
            left_coordinate = np.array([x2,y2])
            x_virt,y_virt = self.spline_object_center.calc_position(np.mod(theta,self.length-1))
            center_coordinate = np.array([x_virt , y_virt])
            right_distance = self.calc_euclid_distance(center_coordinate,right_coordinate)
            left_distance = self.calc_euclid_distance(center_coordinate,left_coordinate)
            return right_distance , left_distance
            
    
    def _plot_map(self):
        waypoints = self.dict_waypoints
        waypoints_inner,waypoints_center,waypoints_outer = self.get_coordinates(waypoints)                 
        flg,ax = plt.subplots(1)
        range_waypoints = len(waypoints_center)-1
        plt.plot(waypoints_center[0:range_waypoints,0], waypoints_center[0:range_waypoints,1], "-r", label="center_spline")
        plt.plot(waypoints_outer[0:range_waypoints,0], waypoints_outer[0:range_waypoints,1], "-g", label="outer_spline")
        plt.plot(waypoints_inner[0:range_waypoints,0], waypoints_inner[0:range_waypoints,1], "-b", label="inner_spline")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        #test
        self.plot_simple_Car(waypoints_center[0,0],waypoints_center[0,1],0)
        plt.show() 
        pdb.set_trace()
        return
    def plot_simple_Car(self,x,y,psi):
        l = 5 # length of mobile robot
        w = 2.5   # width of mobile robot

        # Mobile robot coordinates wrt body frame
        mr_co = np.array([[-l/2, l/2, l/2, -l/2],
                        [-w/2, -w/2, w/2, w/2]])
        R_psi = np.array([[np.cos(psi), -np.sin(psi)],
                            [np.sin(psi), np.cos(psi)]])  # rotation matrix
        v_pos = np.dot(R_psi, mr_co)  # orientation w.r.t intertial frame
        plt.fill(v_pos[0, :] + x, v_pos[1, :] + y, 'g')  # rotation + translation to get global coordinates
        # plt.legend(['MR'], fontsize=24)
        # plt.xlabel('x,[m]')
        # plt.ylabel('y,[m]')
        # plt.axis([-1, 3, -1, 3])
        # plt.axis('square')
        # plt.grid(True)
        # plt.show(block=False)
        # pdb.set_trace()
        # plt.pause(0.1)
        # plt.clf()    
        return 
    
if __name__ == '__main__':
    reference_path = racingTrack(1,
                                smoothing_distance=5, max_width=16.0,
                                circular=True)
    reference_path._plot_map()
    # car = bicycle_dynamic_model(0.02)
    # x0 = np.asarray([51.6,  38.69,-0.03241,0.1,0.00])
    # # x0 = np.asarray([-0.8457,1.0979,-0.7854,1.0000])
    # # #x0 = np.expand_dims(x0,axis=1)
    # # u_current = np.asarray([0,0,0])
    # # sol = car.get_simulation_next_state(x0,u_current)
    # car.plot_simple_Car()
    # plt.axis("on")
    # plt.pause(0.001)
    pdb.set_trace()