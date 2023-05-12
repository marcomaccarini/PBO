#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray


def callback(data):
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    index = 0
    for a in data.data:
        print(str(index) + " th param: " + str(a))
        index += 1


def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('chatter2', Float64MultiArray, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
