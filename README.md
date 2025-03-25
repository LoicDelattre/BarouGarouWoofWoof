# BarouGarouWoofWoof

#Using the power of edits we gonna make a robot bark

#connect to robot via pc
$ ssh ubuntu@<robot IP>

#launch ros master on pc
$ roscore

#on ssh, launch turtlecore
roslaunch turtlebot3_bringup turtlebot3_robot.launch

#on ssh, launch usb_cam
roslaunch usb_cam usb_cam-test.launch

#on ssh, launch go for ball
rosrun dog_behaviors_pkg goForBall.py