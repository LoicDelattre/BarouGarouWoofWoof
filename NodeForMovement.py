import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge, CvBridgeError


def __init__(self):
    rospy.init_node("turtlebot_controller", anonymous=True)

    # Subscriber to the INSERT_NODE topic
    self.subscriber = rospy.Subscriber("/INSERT_NODE_TOPIC", Image, self.callback)
    
    # Publisher for velocity commands
    self.publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    # Define rate (loop frequency)
    self.rate = rospy.Rate(10)  # 10 Hz

def callback(self, msg):
    """
    Callback function for processing incoming messages.
    """
    rospy.loginfo(f"Received data: {msg.data}")
    # Create Twist message
    twist_msg = Twist()

    # Example logic: Move forward if message is "move"
    if msg.data.lower() == "move":
        twist_msg.linear.x = 0.5  # Move forward
        twist_msg.angular.z = 0.0  # No rotation
    elif msg.data.lower() == "stop":
        twist_msg.linear.x = 0.0  # Stop
        twist_msg.angular.z = 0.0
    else:
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.5  # Rotate in place

    # Publish movement command
    self.publisher.publish(twist_msg)

def run(self):
    """
    Keep node running.
    """
    rospy.spin()


if __name__ == "__main__":
    try:
        controller = TurtlebotController()
        controller.run()
    except rospy.ROSInterruptException:
        pass