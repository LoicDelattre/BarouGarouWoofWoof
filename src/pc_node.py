import rospy
from geometry_msgs.msg import Twist

def moveForward(speed : float, publisher : rospy.Publisher):
    print("send data")
    rospy.loginfo("Moving TurtleBot...")
    twistMessage = Twist()
    twistMessage.linear.x = speed   
    publisher.publish(twistMessage) 
    pass

def main():
    print("node started")
    rospy.init_node('data_to_cmd_vel')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    speed = 1
    rate = rospy.Rate(10) # 10 Hz

    while not rospy.is_shutdown():
        moveForward(speed, pub)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass