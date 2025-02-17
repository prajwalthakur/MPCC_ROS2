import numpy as np
# class mpc_params:
#     def __init__(self ):
        
#        #'s_min': -0.4189,'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58} 
#         self.N = 20
#         self.Ts =0.1
#         self.ModelNo = 1
#         self.fullBound = 1;  
#         self.speed_max_limit = 6
#         self.acc_max_limit = 3
#         #x_max,y_max,yaw_max,v_max,theta_max
#         self.Tx = np.asarray([1/1,1/1,1/(2*np.pi),1/10])    #1/551 2276.172566354003
#         self.invTx = np.asarray([1,1,2*np.pi,10])
        
#         #v_max,steering_max,theta_dot_max
#         self.Tu = np.asarray([1/self.speed_max_limit,1/0.4,1/self.speed_max_limit])
#         self.invTu = np.asarray([self.speed_max_limit,0.4,self.speed_max_limit])
        
        

#         self.TDu = np.ones((3,1))
#         self.invTDu = np.ones((3,1))
#         #bounds for nomalized state-inputs [x,y,psi,theta,   v,delta,theta_dot]
        
#         self.lb_boundsStates = np.asarray([-1e4,-1e4,-3, 0,  -1,-1, 0.1]).reshape(7,1)
#         self.ub_boundsStates =  np.asarray([1e4, 1e4, 3, 1e4,   1, 1,1]).reshape(7,1)
        
#         # delta_v , delta_steer , delta_theta_dot
#         self.lb_boundsInput = np.asarray([-0.4 ,-0.4,-10]).reshape(3,1)
#         self.ub_boundsInput =  np.asarray([0.4 ,0.4,10]).reshape(3,1)
        
#         self.qC = 0.01
#         self.qCNmult= 10
#         self.qL= 10000
#         self.qVtheta= 0.02
#         self.qOmega = 5e0
#         self.qOmegaNmult = 1
        
#         self.rVx= 1e-3
#         self.rDelta= 1e-3
#         self.rVtheta=  1e-6

#         self.rdVx= 0.01
#         self.rdDelta= 0.1
#         self.rdVtheta= 0.01


#         self.q_eta = 250

#         self.costScale = 1
#         self.latency = 0.00


# class mpc_params:
#     def __init__(self ):
#         self.N = 20
#         self.Ts = 0.1
#         self.ModelNo = 1
#         self.fullBound = 1; 
#         self.speed_max_limit =9
#         self.acc_max_limit = 5
#         self.max_steering_angle = 0.399
#         self.Tx = np.asarray([1/1,1/1,1/(2*np.pi),1/self.speed_max_limit,1/10])    #1/551 2276.172566354003
#         self.Tu = np.asarray([1/self.acc_max_limit,1/self.max_steering_angle,1/self.speed_max_limit])

#         self.invTx = np.asarray([1,1,2*np.pi,self.speed_max_limit,10])
#         self.invTu = np.asarray([self.acc_max_limit,self.max_steering_angle,self.speed_max_limit])

#         self.TDu = np.ones((3,1))
#         self.invTDu = np.ones((3,1))
#         #bounds for nomalized state-inputs [x,y,psi,v,theta,v_dot,delta_dot,theta_dot]
#         self.lb_boundsStates = np.asarray([-1e4,-1e4,-3, 0.25,   0,-1,-1, 0]).reshape(8,1)
#         self.ub_boundsStates =  np.asarray([1e4, 1e4, 3,  8, 1e4,1, 1,8]).reshape(8,1)
#         self.lb_boundsInput = np.asarray([-0.25 ,-0.1,-8]).reshape(3,1)
#         self.ub_boundsInput =  np.asarray([0.25 , 0.1, 8]).reshape(3,1)
        
        
#         self.qC = 0.01
#         self.qCNmult= 10000
#         self.qL= 1000
#         self.qVtheta= 0.5
#         self.qOmega = 5e0
#         self.qOmegaNmult = 1
        
#         self.rvDot= 1e-4
#         self.rDelta= 1e-4
#         self.rVtheta=  1e-6

#         self.rvdDot= 0.01
#         self.rdDelta= 0.4
#         self.rdVtheta= 0.1


#         self.q_eta = 250

#         self.costScale = 1
#         self.latency = None



class mpc_params:
    def __init__(self ):
        self.N = 12 #20
        self.Ts = 0.05
        self.ModelNo = 1
        self.fullBound = 1;  
        self.accmax = 6 #6
        self.vmax = 8
        self.thetadot_max = 8
        self.Tx = np.asarray([1/1,1/1,1/(2*np.pi),1/self.vmax,1/1])    #1/551 2276.172566354003
        
        self.Tu = np.asarray([1/self.accmax,1/0.40,1/self.thetadot_max])

        self.invTx = np.asarray([1,1,2*np.pi,self.vmax,1])
        self.invTu = np.asarray([self.accmax,0.40,self.thetadot_max])

        self.TDu = np.ones((3,1))
        self.invTDu = np.ones((3,1))
        
        
        #bounds for nomalized state-inputs [x,y,psi,v,theta,v_dot,delta,theta_dot]
        self.lb_boundsStates = np.asarray([-1e4,-1e4,-4, 0.0,   0,-1,-1, 0]).reshape(8,1)
        self.ub_boundsStates =  np.asarray([1e4, 1e4, 4,  1, 1e4,1, 1, 1]).reshape(8,1)
        
        # self.lb_boundsInput = np.asarray([-0.25*4 ,-0.25,-0.25*4]).reshape(3,1)
        # self.ub_boundsInput =  np.asarray([0.25*4, 0.25, 0.25*4]).reshape(3,1)
        
        # [D(vdot),D(delta),D(theta_dot)]
        # self.lb_boundsInput = np.asarray([-self.accmax,-0.40,-self.thetadot_max]).reshape(3,1)
        # self.ub_boundsInput =  np.asarray([self.accmax, 0.40 , self.thetadot_max]).reshape(3,1) 
        self.lb_boundsInput = np.asarray([-6,-0.5,-6]).reshape(3,1)
        self.ub_boundsInput =  np.asarray([6, 0.5 ,6]).reshape(3,1) 
        self.qC = 0.01
        self.qCNmult= 100
        self.qL= 1000
        
        
        self.qVtheta= 0.9#0.0005 wont work
        self.qOmega = 5e0
        self.qOmegaNmult = 1
        
        
        self.rvDot= 1e-6
        self.rDelta= 1e-4
        self.rVtheta=  1e-4

        self.rvdDot = 0.01
        self.rdDelta = 10
        self.rdVtheta= 0.01


        self.q_eta = 250

        self.costScale = 0.5
        self.latency = None