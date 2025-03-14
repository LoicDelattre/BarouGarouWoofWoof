import rospy
from geometry_msgs.msg import Twist
import hid

#!/usr/bin/env python


# Initialize HID device
device = hid.device()
device.open(0x80EE, 0x21)  # HID device's ID

def cmd_vel_callback(msg):
    # Define predefined hexadecimal values for actions
    forward = [0x00, 0x01, 0x00, 0x00]
    backward = [0x00, 0xFF, 0x00, 0x00]
    left = [0x00, 0x00, 0x01, 0x00]
    right = [0x00, 0x00, 0xFF, 0x00]

    # Determine action based on Twist message
    if msg.linear.x > 0:
        report = forward
    elif msg.linear.x < 0:
        report = backward
    elif msg.angular.z > 0:
        report = left
    elif msg.angular.z < 0:
        report = right
    else:
        report = [0x00, 0x00, 0x00, 0x00]  # Stop

    # Send HID report
    device.write(report)

def main():
    rospy.init_node('cmd_vel_subscriber', anonymous=True)
    rospy.Subscriber('cmd_vel', Twist, cmd_vel_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        device.close()