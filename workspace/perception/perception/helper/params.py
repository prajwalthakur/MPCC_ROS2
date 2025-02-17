import numpy as np
class EgoParams:
    def __init__(self):
        self.horizonLength = 20
        self.ctrTs = 0.05
        self.ModelNo = 1
        self.fullBound = 1
        self.speed_max_limit = 5
        self.thetadot_max = 5
        self.acc_max_limit = 6
        self.max_steering_angle = 0.40
        self.pi = 3.14159
        self.two_pi = 6.28318
        self.two_pi_inverse = 0.15915494  # Precomputed value for 1/(2*pi)
        
        # Precomputed values for Tx and Tu arrays
        self.Tx = np.array([1.0, 1.0, self.two_pi_inverse, 1/self.speed_max_limit, 0.1])
        self.Tu = np.array([1/self.acc_max_limit, 1/self.max_steering_angle, 1/self.thetadot_max])

        self.invTx = np.array([1.0, 1.0, self.two_pi, self.speed_max_limit, 10.0])
        self.invTu = np.array([self.acc_max_limit, self.max_steering_angle, self.thetadot_max])

        self.TDu = np.ones((3,1)) #[1.0, 1.0, 1.0]  # Assuming np.ones((3,1)) means [1, 1, 1]
        self.invTDu = np.ones((3,1)) #[1.0, 1.0, 1.0]  # Assuming np.ones((3,1)) means [1, 1, 1]
        
        # Bounds for normalized state-inputs
        # [x, y, psi, v, theta, v_dot, delta, theta_dot]
        self.lb_boundsStates = np.array([-10000.0, -10000.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0])
        self.ub_boundsStates = np.array([10000.0, 10000.0, 1.0, 1.0, 10000.0, 1.0, 1.0, 1.0])

        # Bounds for [D(vdot), D(delta), D(theta_dot)]
        self.lb_boundsInput = np.asarray([-0.25 ,-0.25,-5]).reshape(3,1)
        self.ub_boundsInput =  np.asarray([0.25 , 0.25, 5]).reshape(3,1)

        self.qC = 10
        self.qCNmult = 100
        self.qL = 1000

        self.qVtheta = 1.0
        self.qOmega = 5.0
        self.qOmegaNmult = 1

        self.rvDot = 1e-4
        self.rDelta = 1e-4
        self.rVtheta = 1e-6

        self.rdvDot = 0.01
        self.rdDelta = 0.99
        self.rdVtheta = 0.001

        self.q_eta = 250

        self.costScale = 0.8
        self.latency = None  # Use None for null values

        self.real_test = False
        self.ControllerReset = True


class EgoParamsAdvanceKin:
    def __init__(self):
        self.horizonLength = 20
        self.ctrTs = 0.05
        self.ModelNo = 1
        self.fullBound = 1
        self.speed_max_limit = 5
        self.thetadot_max = 5
        self.acc_max_limit = 6
        self.max_steering_angle = 0.40
        self.pi = 3.14159
        self.two_pi = 6.28318
        self.two_pi_inverse = 0.15915494  # Precomputed value for 1/(2*pi)
        
        # Precomputed values for Tx and Tu arrays
        self.Tx = np.array([1.0, 1.0, self.two_pi_inverse, 1/self.speed_max_limit, 0.1])
        self.Tu = np.array([1/self.acc_max_limit, 1/self.max_steering_angle, 1/self.thetadot_max])

        self.invTx = np.array([1.0, 1.0, self.two_pi, self.speed_max_limit, 10.0])
        self.invTu = np.array([self.acc_max_limit, self.max_steering_angle, self.thetadot_max])

        self.TDu = np.ones((3,1)) #[1.0, 1.0, 1.0]  # Assuming np.ones((3,1)) means [1, 1, 1]
        self.invTDu = np.ones((3,1)) #[1.0, 1.0, 1.0]  # Assuming np.ones((3,1)) means [1, 1, 1]
        
        # Bounds for normalized state-inputs
        # [x, y, psi, v, theta, v_dot, delta, theta_dot]
        self.lb_boundsStates = np.array([-10000.0, -10000.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0])
        self.ub_boundsStates = np.array([10000.0, 10000.0, 1.0, 1.0, 10000.0, 1.0, 1.0, 1.0])

        # Bounds for [D(vdot), D(delta), D(theta_dot)]
        self.lb_boundsInput = np.asarray([-0.25 ,-0.25,-5]).reshape(3,1)
        self.ub_boundsInput =  np.asarray([0.25 , 0.25, 5]).reshape(3,1)

        self.qC = 10
        self.qCNmult = 100
        self.qL = 1000

        self.qVtheta = 1.0
        self.qOmega = 5.0
        self.qOmegaNmult = 1

        self.rvDot = 1e-4
        self.rDelta = 1e-4
        self.rVtheta = 1e-6

        self.rdvDot = 0.01
        self.rdDelta = 0.99
        self.rdVtheta = 0.001

        self.q_eta = 250

        self.costScale = 0.8
        self.latency = None  # Use None for null values

        self.real_test = False
        self.ControllerReset = True