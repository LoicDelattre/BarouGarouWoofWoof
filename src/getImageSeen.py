import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class GetImageSeen():
    def __init__(self):
        rospy.init_node("turtlebot_vision_controller", anonymous=True)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to image topic
        self.subscriber = rospy.Subscriber("/ball_image", Image, self.image_callback)
        pass

    def image_callback(self, msg):
        """
        Callback function to process the received image and detect red objects.
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv_image = self.addFrameToImage(cv_image)
        cv2.imwrite("saved_image.png", cv_image)
        print("image saved !")

    def addFrameToImage(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        controller = GetImageSeen()
        controller.run()
    except rospy.ROSInterruptException:
        pass

                        