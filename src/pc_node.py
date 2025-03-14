import rospy
from geometry_msgs.msg import Twist


def cmd_vel_callback(msg):
    rospy.logdebug("feur")

def main():
    rospy.init_node('cmd_vel_subscriber', anonymous=True)
    rospy.Subscriber('cmd_vel', Twist, cmd_vel_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass