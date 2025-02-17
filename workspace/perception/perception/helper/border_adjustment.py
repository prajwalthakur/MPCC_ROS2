import numpy as np
import copy 
import pdb
def get_coordinates(waypoints):
    xy_inner = waypoints[0:,0:2]
    xy_center = waypoints[0:,2:4]
    xy_outer = waypoints[0:,4:6]
    return xy_inner,xy_center,xy_outer
def border_adjustment(track_original,ModelParams,safetyScaling):
    #shrink track on both sides with WidthCar 
    WidthCar = ModelParams["W"]*safetyScaling
    #scale track to car size (right now for simtrack scale is 1)
    track_inner,track_center,track_outer = get_coordinates(track_original)
    new_track_inner,new_track_center,new_track_outer = get_coordinates(track_original)  
        
    track_inner = track_inner*ModelParams["Scale"]
    track_center = track_center*ModelParams["Scale"]
    track_outer = track_outer*ModelParams["Scale"]    
    
    #compute width of track
    widthTrack = np.linalg.norm(track_inner[0,0:] - track_outer[0,0:])
    #creating new shrinked track
    shrinking_ratio = WidthCar
    new_track_outer = track_outer + shrinking_ratio*((track_inner-track_outer)/widthTrack)
    new_track_inner = track_inner - shrinking_ratio*((track_inner-track_outer)/widthTrack)
            
    #pdb.set_trace()
    new_track  = np.hstack((new_track_inner,track_center,new_track_outer))
    return new_track,track_original




if __name__ == '__main__':
    border_adjustment()