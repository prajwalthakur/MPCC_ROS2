import numpy as np
import pdb
def get_open_loop_predictions(car,control_seqs,x0,dT):
        openLoop_states = np.empty((9,0))
        #pdb.set_trace()
        for i in range(control_seqs.shape[1]):
                x_next = car.get_simulation_next_state_open(x0,control_seqs[:,i],T=dT)
                openLoop_states =np.hstack((openLoop_states,x_next))
                x0=x_next
        return openLoop_states
    #aug_state(X_seqs,U_seqs,X0,MPC_vars,ModelParams,tl)


"""function [xTemp,uTemp] = augState(x,u,x0,MPC_vars,ModelParams,tl)
    
    nx = ModelParams.nx;
    nu = ModelParams.nu;
    N = MPC_vars.N;
    Ts = MPC_vars.Ts;
    indexpsi = ModelParams.stateindex_psi;
    indexTheta = ModelParams.stateindex_theta;

    xTemp = zeros(nx,N+1);
    uTemp = zeros(nu,N);
    
    xTemp(:,1) = x0;
    uTemp(:,1) = u(:,2);
    for j=2:N-1
        xTemp(:,j) = x(:,j+1);
        uTemp(:,j) = u(:,j+1);
    end
    j = N;
    xTemp(:,j) = x(:,j+1);
    uTemp(:,j) = u(:,j);
    
    j = N+1;
    xTemp(:,j) = SimTimeStep(x(:,N+1),u(:,N),Ts,ModelParams);
    
    if xTemp(indexpsi,1) - xTemp(indexpsi,2) > pi
        xTemp(indexpsi,2:end) = xTemp(indexpsi,2:end)+2*pi;
    end
    if xTemp(indexpsi,1) - xTemp(indexpsi,2) < -pi
        xTemp(indexpsi,2:end) = xTemp(indexpsi,2:end)-2*pi;
    end
    
    if xTemp(indexTheta,1) - xTemp(indexTheta,2) < -0.75*tl
        xTemp(indexTheta,2:end) = xTemp(indexTheta,2:end)-tl;
    end
    
end"""






def aug_state(car,X_seqs,U_seqs,X0,Horizon,dT, track_length):
    #shift the current state and input by 1 time step for the next initialization
    aug_states = car.model_parameters["ax"] 
    aug_input = car.model_parameters["au"]
    indexTheta = car.model_parameters["stateindex_theta"]
    indexpsi = car.model_parameters["stateindex_psi"]
    N = Horizon 
    xTemp  = np.zeros((aug_states,N+1))
    uTemp = np.zeros((aug_input,N))
    xTemp[:,0] = X0.reshape((aug_states,))
    uTemp[:,0] = U_seqs[:,1]
    
    xTemp[:,1:N] = X_seqs[:,2:]
    uTemp[:,1:N-1] = U_seqs[:,2:]
    uTemp[:,N-1] = U_seqs[:,N-1]
    #get_simulation_next_state_open(self,x_current,u_current,T,last_closestIdx):
    xTemp[:car.model_parameters["ax"],N] = car.get_simulation_next_state_open(xTemp[:,N-1], U_seqs[:,N-1], dT ).reshape(( car.model_parameters["ax"],))
    if xTemp[indexpsi,0] - xTemp[indexpsi,1] > np.pi:
        xTemp[indexpsi,1:] = xTemp[indexpsi,1:]+2*np.pi
    if xTemp[indexpsi,0] - xTemp[indexpsi,1] < - np.pi:
        xTemp[indexpsi,1:] = xTemp[indexpsi,1:]-2*np.pi
    if xTemp[indexTheta,0] - xTemp[indexTheta,1] < -0.75*track_length:
        xTemp[indexTheta,1:] = xTemp[indexTheta,1:]-track_length
    return xTemp,uTemp