from scipy.linalg import block_diag
import pdb
import numpy as np
import scipy.integrate
import scipy.linalg
from .mpc_vars import mpc_params
from .hpipmInterface import hpipmSolve

# def rw_property(attr_name):
#     """
#     Decorator to create a read-write property with a given attribute name.
#     """
#     def getter(instance):
#         return getattr(instance, "_" + attr_name)

#     def setter(instance, value):
#         setattr(instance, "_" + attr_name, value)

#     return property(getter, setter)

class delayed_mpc:
    def __init__(self, horizon,ts_control,track_object,car_object,Xobstacle):
        self.MPC_vars =  mpc_params()
        self.N = self.MPC_vars.N
        self.car_object = car_object
        self.track_object = track_object
        self.lf = self.car_object.model_parameters["lf"] + self.car_object.model_parameters["lr"]
        inner_lane,center_lane,outer_lane = track_object.get_coordinates_mpcc()
        self.borders = np.hstack((inner_lane[:,1],inner_lane[:,2],outer_lane[:,1],outer_lane[:,2])).reshape((inner_lane.shape[0],4)) 
        self.ax = self.car_object.model_parameters["ax"]
        self.au = self.car_object.model_parameters["au"]
        self.attrNames =["_X0","_U0", "_Qk","_qk","_Rk","_Ak","_Bk","_gk","_Ck","_ug","_lg","_lbx","_ubx","_lbu","_ubu"]
        return
    
    def mpc_matrices(self):
        empty_dict = {attr: None for attr in self.attrNames}
        return empty_dict
    
    def create_matrices(self,x0,u0,X_seqs,U_seqs):
        track_object = self.track_object
        car_object = self.car_object
        costScale = self.MPC_vars.costScale
        cost_list = []
        for i in range(self.MPC_vars.N):
            cost_list.append(self.mpc_matrices())
            if i==0:
                cost_list[i]["_X0"] = x0.reshape((5,1))
                cost_list[i]["_U0"] = u0.reshape((3,1))
            Xk = X_seqs[:,i]
            Uk = U_seqs[:,i]
            cost_list[i]["_Qk"],cost_list[i]["_qk"] =self.generateQkqk(track_object,self.MPC_vars,car_object,Xk,Uk,i)
            cost_list[i]["_Rk"] = costScale*2*self.calcRInput()
            #linearized dynamics
            cost_list[i]["_Ak"],cost_list[i]["_Bk"],cost_list[i]["_gk"] = self.getEqualityConstraints(Xk,Uk,self.MPC_vars,car_object)
            #bounds
            cost_list[i]["_lbx"], cost_list[i]["_ubx"],cost_list[i]["_lbu"], cost_list[i]["_ubu"] = self.getBounds(self.MPC_vars,car_object)
            #linearized track constraints 
            cost_list[i]["_Ck"], cost_list[i]["_ug"], cost_list[i]["_lg"] = self._getBorderConstraint(i) #self._getInequalityConstraints(self.borders[max(i-1,0),:],self.MPC_vars,car_object,i)
            
        #terminal state
        i = self.MPC_vars.N
        cost_list.append(self.mpc_matrices())
        Xk = X_seqs[:,i]
        cost_list[i]["_Qk"],cost_list[i]["_qk"] =self.generateQkqk(track_object,self.MPC_vars,car_object,Xk,Uk,i)
        cost_list[i]["_Rk"] = costScale*2*self.calcRInput()
        #linearized dynamics
        #bounds
        cost_list[i]["_lbx"], cost_list[i]["_ubx"],cost_list[i]["_lbu"], cost_list[i]["_ubu"] = self.getBounds(self.MPC_vars,car_object)
        #linearized track constraints 
        cost_list[i]["_Ck"], cost_list[i]["_ug"], cost_list[i]["_lg"] = self._getBorderConstraint(i) #self._getInequalityConstraints(self.borders[max(i-1,0),:],self.MPC_vars,car_object,i)
        
        return cost_list
    
    def generateQkqk(self,track_object,MPC_vars,car_object,Xk,Uk,i):
        QState,qState = self.calcQstate(MPC_vars,Xk,Uk,i)
        QInput,qInput = self.calcQInput(MPC_vars,Xk,Uk,i)
        #add omega regularization not needed
        #make Qtilde symetric (not symetric due to numerical issues)
        Qtilde = 0.5 *(QState+QState.T)
        #Qk = contouring-lag error and real-input cost
        Qk = MPC_vars.costScale*2*block_diag(Qtilde,QInput)
        qk = MPC_vars.costScale*np.vstack((qState,qInput))
        # fk = self.generatef(track_object,MPC_vars,car_object,Xk,i)
        # pdb.set_trace()
        return Qk,qk     
        #compute linear contouring and lag errors
    def calcQstate(self,MPC_vars,Xk,Uk,i):
        if i==MPC_vars.N:
            Q_ = np.diag([MPC_vars.qCNmult*MPC_vars.qC,MPC_vars.qL])
        else:
            Q_ = np.diag([MPC_vars.qC,MPC_vars.qL])
        d_contouring_error = self.calcdContouringError(Xk,Uk,i,self.track_object)
        Q_contouring_cost = d_contouring_error.T@Q_@d_contouring_error
        QState = np.diag(self.MPC_vars.invTx)@Q_contouring_cost@np.diag(self.MPC_vars.invTx) 
        ## q.Tx
        error = np.vstack((self.e_c, self.e_l))
        #q = horzcat(2*(error.T*Q_*errorgrad-self.X_prev[k].T*errorgrad.T*Q_*errorgrad),MX.zeros(1,self.au-1),-self.MPC_vars.qVtheta)
        qState = 2*(error.T@Q_@d_contouring_error-Xk.T@d_contouring_error.T@Q_@d_contouring_error).T
        qState =  np.diag(self.MPC_vars.invTx)@qState
        return QState,qState
    def calcQInput(self,MPC_vars,Xk,Uk,i):
        quadCostInput_ = np.asarray([self.MPC_vars.rvDot,self.MPC_vars.rDelta,self.MPC_vars.rVtheta])
        QInput = np.diag(quadCostInput_)  
        QInput = np.diag(self.MPC_vars.invTu)@QInput@np.diag(self.MPC_vars.invTu)     
        qInput_  = np.hstack((np.zeros((self.au-1,)),-self.MPC_vars.qVtheta)).reshape((self.au,1))
        qInput = np.diag(self.MPC_vars.invTu)@qInput_
        return QInput,qInput    
    def calcRInput(self):
        R_ = np.asarray([self.MPC_vars.rvdDot,self.MPC_vars.rdDelta,self.MPC_vars.rdVtheta])
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
        self.e_c = -sin_phit*(x_virt - Xk[0]) + cos_phit*(y_virt - Xk[1])  
        dContouringErrorTheta = dtheta*cos_phit*(Xk[0]- x_virt ) + dtheta*sin_phit*(Xk[1]-y_virt )-sin_phit*dx + cos_phit*dy
        dContouringErrorX  = sin_phit
        dContouringErrorY  = -cos_phit
        d_contouring_error[0,self.car_object.model_parameters["stateindex_x"]] = dContouringErrorX
        d_contouring_error[0,self.car_object.model_parameters["stateindex_y"]] = dContouringErrorY
        d_contouring_error[0,self.car_object.model_parameters["stateindex_theta"]] = dContouringErrorTheta
        #Lag error
        self.e_l =  cos_phit*(x_virt - Xk[0]) + sin_phit*(y_virt - Xk[1])
        dLagErrorTheta = dtheta*sin_phit*(Xk[0]- x_virt ) - dtheta*cos_phit*(Xk[1]-y_virt ) + cos_phit*dx +sin_phit*dy     
        dLagErrorX = -cos_phit
        dLagErrorY = -sin_phit
        d_contouring_error[1,self.car_object.model_parameters["stateindex_x"]] = dLagErrorX
        d_contouring_error[1,self.car_object.model_parameters["stateindex_y"]] = dLagErrorY
        d_contouring_error[1,self.car_object.model_parameters["stateindex_theta"]] = dLagErrorTheta
        grad_eC,grad_eL = self.getgradError(track_object,self.MPC_vars,self.car_object,theta_hat,Xk[0],Xk[1])
        return d_contouring_error
    

    def getEqualityConstraints(self,xk,uk,MPC_vars,car_object): 
        # given x_k+1 = A x_k + B u_k
        # do the following state augmentation
        # s_k = [x_k,u_k-1], v_k = du_k 
        # with the following linear system
        # s_k+1 = [A B;0 I] s_k + [B;I] v_k      
        nx = car_object.model_parameters["ax"]
        nu = car_object.model_parameters["au"]
        Ad, Bd, gd = self.DiscretizedLinearizedModel(xk,uk,car_object,MPC_vars)
        Tx = np.diag(MPC_vars.Tx)
        Tu = np.diag(MPC_vars.Tu)
        invTx = np.diag(MPC_vars.invTx)
        invTu = np.diag(MPC_vars.invTu)
        #constructing augmented system with state-input scaling
        Ak_1 = np.hstack((Tx@Ad@invTx ,Tx@Bd@invTu)) 
        Ak_2 = np.hstack((np.zeros((nu,nx)) ,np.eye(nu)))
        Ak = np.vstack((Ak_1,Ak_2))
        Bk = np.vstack((Tx@Bd@invTu,np.eye(nu)))
        gk = np.vstack((Tx@gd,np.zeros((nu,1))))        
        return Ak,Bk,gk
    def getBounds(self,MPC_vars,ModelParams):
        lbx = MPC_vars.lb_boundsStates
        ubx = MPC_vars.ub_boundsStates
        lbu =  MPC_vars.lb_boundsInput
        ubu = MPC_vars.ub_boundsInput
        return lbx,ubx,lbu,ubu
    def pi_2_pi(self,angle):
        while(angle > np.pi):
            angle = angle - 2.0 * np.pi
        while(angle < -np.pi):
            angle = angle + 2.0 * np.pi
        return angle
    def _getInequalityConstraints(self,border,MPC_vars,car_object,i):
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
        A = A@block_diag(MPC_vars.invTx,MPC_vars.invTu)
        #print("in loop for setting constraints") 
        #xy_ = np.hstack((x_inner,y_inner,np.zeros((x_inner.shape[0],1))))
        #pdb.set_trace()
        #self.mav_view._add_track_constraints(track_constraints_pts,xy_)
        #pdb.set_trace()
        return A,bmax,bmin    
    
    
    
    
    def DiscretizedLinearizedModel(self,xk,uk,car_object,MPC_vars):
        sx = car_object.model_parameters["ax"]-1
        su = car_object.model_parameters["au"]-1 
        #f = vertcat(xbar_k[3]*cos(xbar_k[2]), xbar_k[3]*sin(xbar_k[2]), (xbar_k[3] / (self.lf)) * tan(ubar_k[1]), ubar_k[0])
        x0 = xk[0]
        y0 = xk[1]
        psi0=xk[2]
        v0= xk[3]
        vdot0 = uk[0]
        delta0 = uk[1]
        xk=xk[0:sx]
        uk = uk[0:su]
        xdot = np.asarray([[v0*np.cos(psi0) ],
                [v0*np.sin(psi0)] ,
                [v0*np.tan(delta0)/(self.lf)],
                [vdot0]
                ])
        A_c = np.asarray([[0,0,-v0*np.sin(psi0),np.cos(psi0)],
                          [0,0,v0*np.cos(psi0),np.sin(psi0)],
                          [0,0,0,np.tan(delta0)/self.lf],
                          [0,0,0,0]])
        B_c = np.asarray([[0 ,0 ],[0,0],[0,v0/(self.lf*np.cos(delta0)*np.cos(delta0))],[1,0]])
        gc = xdot- A_c@xk.reshape((sx,1)) - B_c@uk.reshape((su,1))
        Bc_aug=np.hstack((B_c,gc))
        temp_0 = np.vstack((np.hstack((A_c,Bc_aug)),np.zeros((su+1,sx+su+1))))*MPC_vars.Ts    
        tmp_fwd = scipy.linalg.expm(temp_0)
        Ad = np.zeros((sx+1,sx+1))
        Bd = np.zeros((sx+1,su+1))
        gd = np.zeros((sx+1,1))
        Ad[0:sx,0:sx] =tmp_fwd[0:sx,0:sx]
        Bd[0:sx,0:su] =tmp_fwd[0:sx,sx:sx+su]
        gd[0:sx] =tmp_fwd[0:sx,sx+su].reshape((sx,1))
        Ad[-1,-1]=1
        Bd[-1,-1]=MPC_vars.Ts   
        return Ad,Bd,gd  
    # def DiscretizedLinearizedModel(self,xk,uk,car_object,MPC_vars,DynamicModel=True):
    #     #(self,xbar_k,ubar_k):
    #     if DynamicModel:
    #         sx = self.car_object.model_parameters["ax"] -1  
    #         su = self.car_object.model_parameters["au"] -1  
    #         #f = vertcat(xbar_k[3]*cos(xbar_k[2]), xbar_k[3]*sin(xbar_k[2]), (xbar_k[3] / (self.lf)) * tan(ubar_k[1]), ubar_k[0])
    #         Cm1 = self.car_object.model_parameters["Cm1"]
    #         Cm2 =  self.car_object.model_parameters["Cm2"]
    #         Cro = self.car_object.model_parameters["Cro"]
    #         Cr2 =  self.car_object.model_parameters["Cr2"]
    #         B_f =  self.car_object.model_parameters["Bf"]
    #         C_f =  self.car_object.model_parameters["Cf"]
    #         D_f =  self.car_object.model_parameters["Df"]

    #         B_r =  self.car_object.model_parameters["Br"]
    #         C_r =  self.car_object.model_parameters["Cr"]
    #         D_r =  self.car_object.model_parameters["Dr"]
    #         phi = xk[self.car_object.model_parameters["stateindex_psi"]]
    #         omega = xk[self.car_object.model_parameters["stateindex_omega"]]
    #         v_x = xk[self.car_object.model_parameters["stateindex_vx"]]
    #         v_y = xk[self.car_object.model_parameters["stateindex_vy"]]

    #         delta = ubar_k[self.car_object.model_parameters["inputindex_delta"]]
    #         D= ubar_k[self.car_object.model_parameters["inputindex_DutyCycle"]]
    #         m = self.car_object.model_parameters["m"]
    #         Iz = self.car_object.model_parameters["Iz"]
    #         l_f = self.car_object.model_parameters["lf"]
    #         l_r = self.car_object.model_parameters["lr"]
    #         if(v_x < 0.5):
    #             v_x = v_x
    #             v_y = 0
    #             omega = 0
    #             delta = 0
    #         elif(v_x < 0.3):
    #             v_x = 0.3   # done this to avoid singularity 
    #         alpha_f = -np.arctan2(omega*l_f + v_y,abs(v_x)) + delta    
    #         alpha_r = np.arctan2(omega*l_r-v_y,abs(v_x))
    #         F_fy = D_f*np.sin(C_f*np.arctan(B_f*alpha_f))
    #         F_ry = D_r*np.sin(C_r*np.arctan(B_r*alpha_r))
    #         F_rx = Cm1*D -Cm2*v_x*D - Cro-Cr2*(v_x**2) 
    #         xdot = np.asarray([[v_x*np.cos(phi) - v_y*np.sin(phi)],
    #                     [v_y*np.cos(phi) + v_x*np.sin(phi)] ,
    #                     [omega],
    #                     [(F_rx-F_fy*np.sin(delta)+ m*v_y*omega)/m],
    #                     [(F_ry + F_fy*np.cos(delta)-m*v_x*omega)/m],
    #                     [(F_fy*l_f*np.cos(delta)-F_ry*l_r)/Iz]])  
    #         dFrx_dvx = -Cm2*D - 2*Cr2*v_x
    #         dFrx_dD  = Cm1 - Cm2*v_x
    #         dFry_dvx = ((B_r*C_r*D_r*np.cos(C_r*np.arctan(B_r*alpha_r)))/(1+B_r**2*alpha_r**2))*(-(l_r*omega - v_y)/((-l_r*omega + v_y)**2+v_x**2))
    #         dFry_dvy = ((B_r*C_r*D_r*np.cos(C_r*np.arctan(B_r*alpha_r)))/(1+B_r**2*alpha_r**2))*((-v_x)/((-l_r*omega + v_y)**2+v_x**2))
    #         dFry_domega = ((B_r*C_r*D_r*np.cos(C_r*np.arctan(B_r*alpha_r)))/(1+B_r**2*alpha_r**2))*((l_r*v_x)/((-l_r*omega + v_y)**2+v_x**2))
    #         dFfy_dvx =     (B_f*C_f*D_f*np.cos(C_f*np.arctan(B_f*alpha_f)))/(1+B_f**2*alpha_f**2)*((l_f*omega + v_y)/((l_f*omega + v_y)**2+v_x**2))
    #         dFfy_dvy =     (B_f*C_f*D_f*np.cos(C_f*np.arctan(B_f*alpha_f)))/(1+B_f**2*alpha_f**2)*(-v_x/((l_f*omega + v_y)**2+v_x**2))
    #         dFfy_domega =    (B_f*C_f*D_f*np.cos(C_f*np.arctan(B_f*alpha_f)))/(1+B_f**2*alpha_f**2)*((-l_f*v_x)/((l_f*omega + v_y)**2+v_x**2))
    #         dFfy_ddelta =  (B_f*C_f*D_f*np.cos(C_f*np.arctan(B_f*alpha_f)))/(1+B_f**2*alpha_f**2)
    #         df1_dphi = -v_x*np.sin(phi) - v_y*np.cos(phi)
    #         df1_dvx  = np.cos(phi)
    #         df1_dvy  = -np.sin(phi)
    #         df2_dphi = -v_y*np.sin(phi) + v_x*np.cos(phi)
    #         df2_dvx  = np.sin(phi)
    #         df2_dvy  = np.cos(phi)
    #         df3_domega = 1
    #         df4_dvx     = 1/m*(dFrx_dvx - dFfy_dvx*np.sin(delta))
    #         df4_dvy     = 1/m*(           - dFfy_dvy*np.sin(delta)     + m*omega)
    #         df4_domega = 1/m*(           - dFfy_domega*np.sin(delta) + m*v_y)
    #         df4_dD      = 1/m*     dFrx_dD
    #         df4_ddelta  = 1/m*(           - dFfy_ddelta*np.sin(delta)  - F_fy*np.cos(delta))
    #         df5_dvx     = 1/m*(dFry_dvx     + dFfy_dvx*np.cos(delta)     - m*omega)
    #         df5_dvy     = 1/m*(dFry_dvy     + dFfy_dvy*np.cos(delta));   
    #         df5_domega = 1/m*(dFry_domega + dFfy_domega*np.cos(delta) - m*v_x)
    #         df5_ddelta  = 1/m*(               dFfy_ddelta*np.cos(delta)  - F_fy*np.sin(delta))
    #         df6_dvx     = 1/Iz*(dFfy_dvx*l_f*np.cos(delta)     - dFry_dvx*l_r)
    #         df6_dvy     = 1/Iz*(dFfy_dvy*l_f*np.cos(delta)     - dFry_dvy*l_r)
    #         df6_domega = 1/Iz*(dFfy_domega*l_f*np.cos(delta) - dFry_domega*l_r)
    #         df6_ddelta  = 1/Iz*(dFfy_ddelta*l_f*np.cos(delta)  - F_fy*l_f*np.sin(delta))
            
    #         A_c=np.asarray([[0, 0 , df1_dphi  ,      df1_dvx     ,    df1_dvy   ,     0 ],
    #             [0 ,0 , df2_dphi    ,    df2_dvx    ,     df2_dvy,        0  ]        ,
    #             [0, 0,  0,               0       ,        0       ,       df3_domega],
    #             [0 ,0 , 0       ,        df4_dvx     ,    df4_dvy   ,     df4_domega],
    #             [0 ,0 , 0         ,      df5_dvx     ,    df5_dvy   ,     df5_domega],
    #             [0, 0,  0         ,      df6_dvx       ,  df6_dvy    ,    df6_domega]])
                
    #         B_c=np.asarray([[0  ,       0  ],
    #             [0  ,       0  ],   
    #             [0   ,      0 ],  
    #             [df4_dD  ,  df4_ddelta  ],
    #             [0  ,       df5_ddelta  ],
    #             [0   ,      df6_ddelta  ]])
    #         gc = xdot- A_c@xbar_k.reshape((sx,1)) - B_c@ubar_k.reshape((su,1))
    #         Bc_aug=np.hstack((B_c,gc))
    #         temp_0 = np.vstack((np.hstack((A_c,Bc_aug)),np.zeros((su+1,sx+su+1))))*self.Ts_control       
    #         tmp_fwd = scipy.linalg.expm(temp_0)
    #         Ad = np.zeros((sx+1,sx+1))
    #         Bd = np.zeros((sx+1,su+1))
    #         gd = np.zeros((sx+1,1))
    #         Ad[0:sx,0:sx] =tmp_fwd[0:sx,0:sx]
    #         Bd[0:sx,0:su] =tmp_fwd[0:sx,sx:sx+su]
    #         gd[0:sx] =tmp_fwd[0:sx,sx+su].reshape((sx,1))
    #         Ad[-1,-1]=1
    #         Bd[-1,-1]=self.Ts_control 
    #     return Ad,Bd,gd   
    def _get_track_values(self,Xk,Uk) :
        track_object = self.track_object
        theta_hat = np.mod(Xk[self.car_object.model_parameters['stateindex_theta']],track_object.length) 
        x_virt,y_virt = track_object.spline_object_center.calc_position(theta_hat)
        dy,dx = track_object.spline_object_center.get_dydx(theta_hat)
        t_angle = np.arctan2(dy, dx)
        trackPoint_dtheta_ref = track_object.spline_object_center.calc_curvature(theta_hat)
        cos_phit =np.cos(t_angle)
        sin_phit = np.sin(t_angle)
        xtyt = np.asarray([x_virt,y_virt]).reshape((2,1))
        dxdy =  np.asarray([dx,dy]).reshape((2,1))
        return theta_hat,xtyt,dxdy,cos_phit,sin_phit,trackPoint_dtheta_ref 
    
    def _solve(self,cost_list,car_object,MPC_vars): 
        X,U,dU,info = hpipmSolve(cost_list,car_object,MPC_vars)
        return  X,U,dU,info
        #delayed_mpc.update( prev_predicted_states =X_seqs , state_current=X0, prev_control_commands=U_seqs,Uprev= U_prev ,weights=weights_mpc,mpcc_iteration=mpcc_iteration)
    def update(self, prev_predicted_states,state_current, prev_control_commands,Uprev,mpcc_iteration = 0):
        #self.mpcCont.update( prev_predicted_states =self.X_seqs , state_current=self.X0, prev_control_commands=self.U_seqs,Uprev= self.uprev ,mpcc_iteration = self.mpcc_iteration)
        #self._construct_warm_start_soln(prev_control_commands,prev_predicted_states[:5,:],mpcc_iteration)
        #self._set_approximated_lag_contour_error(prev_control_commands,prev_predicted_states,mpcc_iteration)    
        x0=state_current
        u0=Uprev
        U_seqs = prev_control_commands
        X_seqs = prev_predicted_states
        track_constraints_pts = self._set_track_constraints(prev_predicted_states[self.car_object.model_parameters['stateindex_theta'],:],self.track_object)
        #track_constraints_pts =np.zeros((1,6)) #self._set_track_constraints(theta_,xy_)
        cost_list = self.create_matrices(x0,u0,X_seqs,U_seqs)
        X,U,dU,info = self._solve(cost_list,self.car_object,self.MPC_vars)
        if info["exitflag"]==0:
            U_pred = U
            X_pred = X
        elif (info["exitflag"]==1):
            U_pred = prev_control_commands
            X_pred = prev_predicted_states
            # exit_flag,U_pred_seqs,X_pred_seqs,track_constraints_pts,costValue,costComponents
        costValue=-1
        costComponents  = np.zeros((1,5))
        return info["exitflag"],U_pred,X_pred,track_constraints_pts,costValue,costComponents
    def _updateWeights(self, weights):
        #not needed getting controlled by mpc_vars file
        return   
    def getgradError(self,track_object,MPC_vars,car_object,theta_virt,x,y):   
        deC_dtheta, deL_dtheta, cos_phi_virt, sin_phi_virt = self.getderror_dtheta(track_object, theta_virt, x, y)
        grad_eC = np.hstack([ sin_phi_virt, -cos_phi_virt, np.zeros((1, car_object.model_parameters["ax"]-3)).flatten(), deC_dtheta]) #contour-Error wrt to all states
        grad_eL = np.hstack([-cos_phi_virt, -sin_phi_virt, np.zeros((1, car_object.model_parameters["ax"]-3)).flatten(), deL_dtheta] ) #lag-Error wrt to all states    
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
    def generatef(self,track_object,MPC_vars,car_object,xk,i):
        x=xk[0]
        y=xk[1]
        theta_virt = np.mod(xk[-1],track_object.length)
        eC, eL = self.getErrors(track_object, theta_virt,x,y)
        error = np.vstack((eC,eL))
        grad_Ec,grad_EL = self.getgradError(track_object,MPC_vars,car_object,theta_virt,xk[0],xk[1])
        grad_E =  np.vstack((grad_Ec, grad_EL ))
        if i==MPC_vars.N:
            Q = np.diag([MPC_vars.qCNmult*MPC_vars.qC,MPC_vars.qL])
        else:
            Q = np.diag([MPC_vars.qC,MPC_vars.qL])        
        theta_virt = np.mod(xk[car_object.model_parameters["stateindex_theta"]],track_object.length)  
        fx=2*(error.T)@Q@grad_E - 2*(xk.reshape((5,1)).T)@(grad_E.T)@Q@grad_E
        f = np.hstack((fx.flatten(), np.zeros((1,car_object.model_parameters["au"]-1)).flatten(), -MPC_vars.qVtheta)).reshape((self.ax+self.au,1))
        f = np.diag(np.hstack((MPC_vars.invTx,MPC_vars.invTu)))@f
        return f
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
        theta_virt = theta_.reshape(self.N+1,)
        # x_inner,y_inner = self.inner_lut_x(theta_last), self.inner_lut_y(theta_last)
        # x_outer,y_outer = self.outer_lut_x(theta_last), self.outer_lut_y(theta_last)
        # delta_X = x_inner - x_outer
        # delta_Y = y_inner-y_outer
        bmin_ = np.empty(0)
        bmax_ = np.empty(0)
        C_ = np.empty((0,self.ax+self.au))
        #pdb.set_trace()
        for k in range(0,self.N+1):
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
            C = np.hstack((A,np.zeros((self.ax+ self.au -2,))))@np.diag(np.hstack((self.MPC_vars.invTx,self.MPC_vars.invTu)))
            C_ = np.vstack((C_,C))
            bmin_ = np.hstack((bmin_,bmin))
            bmax_ = np.hstack((bmax_,bmax))
            x_virt,y_virt = track_object.spline_object_center.calc_position(theta_virt[k])
            x_center_virt = np.asarray([x_virt,y_virt])
            track_constraints_pts = np.vstack((track_constraints_pts,np.asarray([x1,y1,0,x2,y2,0])))
            #print("theta virt=",theta_virt[k],"b_min",bmin,"b_max",bmax,"A@xy_[k]",A@x_center_virt)
            #assert bmin<=A@x_center_virt <=bmax
        #pdb.set_trace()
        bmin_=bmin_.reshape(self.N+1,)
        bmax_ = bmax_.reshape(self.N+1,)
        C_ = C_.reshape(self.N+1,self.ax+self.au)
        self.bmin_ = bmin_
        self.bmax_ = bmax_
        self.C_ = C_
        return track_constraints_pts

    def _getBorderConstraint(self,i):
        return self.C_[i],self.bmax_[i],self.bmin_[i]