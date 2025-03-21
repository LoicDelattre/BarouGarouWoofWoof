import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class TurtlebotVisionController:
    def __init__(self):
        rospy.init_node("turtlebot_vision_controller", anonymous=True)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to USB camera topic
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)

        # Publisher for velocity commands
        self.publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Define movement speed
        self.forward_speed = 0.2
        self.turn_speed = 0.5

    def image_callback(self, msg):
        """
        Callback function to process the received image and detect red objects.
        """
        print("image sent")
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Process the image to detect red color
            movement_cmd = self.process_image(cv_image)
            print(movement_cmd)

            print("sending data to robot")
            # Publish movement command
            self.publisher.publish(movement_cmd)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

    def process_image(self, cv_image):
        """
        Detect red color in the image and determine movement.
        """
        twist_msg = Twist()

        # Convert image to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define the red color range (OpenCV uses HSV for better color segmentation)
        lower_red1 = np.array([0, 120, 70])   # Lower boundary for red
        upper_red1 = np.array([10, 255, 255]) # Upper boundary for red
        lower_red2 = np.array([170, 120, 70]) # Second lower boundary (red appears at both ends of HSV spectrum)
        upper_red2 = np.array([180, 255, 255])# Second upper boundary

        # Create masks to detect red color in the image
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2  # Combine both masks

        # Find contours of detected red regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest red object
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Get bounding box around the red object
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2  # Find center of red object
            img_center_x = cv_image.shape[1] // 2  # Center of the image

            # Define movement logic based on object position
            if area > 50:  # Ignore small objects (filter out noise)
                rospy.loginfo("Red object detected! Skibidiing toward it.")

                if abs(center_x - img_center_x) < 50:
                    twist_msg.linear.x = self.forward_speed  # Move forward
                    twist_msg.angular.z = 0.0
                elif center_x < img_center_x:
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = self.turn_speed  # Turn left
                else:
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = -self.turn_speed  # Turn right
            else:
                rospy.loginfo("Red object too small, ignoring.")
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = self.turn_speed  # Rotate to search

        else:
            rospy.loginfo("No red object detected. Searching...")
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = self.turn_speed  # Rotate to search

        return twist_msg

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        controller = TurtlebotVisionController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
