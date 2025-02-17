import numpy as np
import csv
import pdb
import scipy.io
import matplotlib.pyplot as plt
def load_map():
    file = open("racetrack-database-master/tracks/Austin.csv", "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    waypoints =np.zeros((0,4))
    for i in range(1,len(data)-1):
        cx = float(data[i][0])
        cy = float(data[i][1])
        w_right = float(data[i][2])
        w_left = float(data[i][3])
        waypoints = np.vstack((waypoints,np.asarray((cx,cy,w_right,w_left))))
    return waypoints   

def load_map3():
    waypoints = np.load("/root/workspace/src/maps/e7_floor5/e7_floor5_square.npy") #TODO:
    waypoints = np.hstack((waypoints[:,0:2],waypoints[:,2:4],waypoints[:,4:6]))  #inner , center , outer
    return waypoints

if __name__ == '__main__':
    waypoints = load_map3()
    flg, ax = plt.subplots(1)
    #pdb.set_trace()
    # plt.plot(x, y, "xb", label="input")
    plt.plot(waypoints[:,0], waypoints[:,1],"-r", label="bd_spline_inner")
    plt.plot(waypoints[:,4],waypoints[:,5], "-g", label="bd_spline_outer")
    plt.plot(waypoints[:,2],waypoints[:,3], "-b", label="spline")
    plt.grid(True)
    #plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.show()
    