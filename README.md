# F1-tenth ros2-humble  simulator
# Model Predictive Contour Control Implementation robustness to Computation Latency
- Implemented MPCC in ROS2 simulator [MPCC](https://onlinelibrary.wiley.com/doi/full/10.1002/oca.2123)
At every iteration, I linearized the dynamics and constraints to formulate the problem as a QP, then used the HPIPM solver to obtain the locally optimal solution. Manual linearization helped reduce the solving time.
- Controller Dynamics Has been modified to consider the computational latency of solver .
# Steps to reproduce the results
- clone this reposittory
- Prerequiste : follow this step to install docker and nvidia-docker toolkit [docker&nvidia-docker](https://github.com/f1tenth/f1tenth_gym_ros?tab=readme-ov-file#with-an-nvidia-gpu)
- from root of directory:
    - image build command : ./scripts/build/f1tenth.sh
    - run the container : ./scripts/build/devel.sh
    - if need to open multiple terminal in already running container do : docker exec -it <name_of_container> bash

- In one terminal run : ```bash ros2 launch f1tenth_gym_ros gym_bridge_launch.py```
- In other terminal run : ```bash ros2 launch pure_pursuit sim_pure_pursuit_launch.py```
This should run the pure pursuit with a deafult map

for obstacle(opponent) detection run the following launch file :
- ```ros2 launch perception perception.launch.py```

To Run the Model Predictive Contouring Control
- ```ros2 launch mpcc_sim mpcc_sim.launch.py```

-Currently MPCC subscribe to the "/perception/detection/raw_obstacles" to get the opponent pose but does not do obstacle avoidance 


# Instructions to Run RRT

To run RRT, follow these steps:

- Publish a drive command:
   ```bash ros2 topic pub /opp_drive ackermann_msgs/msg/AckermannDriveStamped "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, drive: {speed: 0.0, steering_angle: 0.0}}"```

- ``` bash ros2 launch f1tenth_gym_ros gym_bridge_launch.py```
- ```bash ros2 launch rrt sim_rrt_launch.py```
    ( other opponent car will be treated as a static obstacle)

# Nominal MPCC ( not robust to computation latency : 30 msec ) 


https://github.com/user-attachments/assets/5af5b187-505e-4d26-b32e-8fad1be2189b


# Robust MPCC



https://github.com/user-attachments/assets/5c91acf2-f56b-4103-915f-1cf847a61e8b




