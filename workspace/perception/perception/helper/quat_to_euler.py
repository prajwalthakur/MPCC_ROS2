#!/usr/bin/env python3
import numpy as np
import pdb
from transforms3d import euler
def euler_from_quaternion(quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Below should be replaced when porting for ROS2 Python tf_conversions is done.
        """
        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

"""  orientation:
    x: -0.0
    y: 0.0
    z: 0.4763196648159169
    w: -0.8792721859069879
"""
if __name__ == '__main__':
      quat = [-0.0,0.0,0.4763196648159169,-0.8792721859069879]
      roll,pitch,yaw = euler_from_quaternion(quat)
      rqx = -0.0
      rqy = 0.0
      rqw = -0.8792721859069879
      rqz = 0.4763196648159169
      _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
      print(roll,pitch,yaw)
      pdb.set_trace()