#!/usr/bin/env python
# Used to publish

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
def talkerString(stringa):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    # rate = rospy.Rate(10) # 10hz
    if not rospy.is_shutdown():
        hello_str = "hello world %s" % stringa
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        # rate.sleep()
def talkerParam(array):
    # pub = rospy.Publisher('chatter', String, queue_size=10)
    pub = rospy.Publisher('chatter2', Float64MultiArray, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    if not rospy.is_shutdown():
        data_to_send = Float64MultiArray()
        data_to_send.data = array
        pub.publish(data_to_send)
        print("done")
def sendIt(array):
    try:
        talkerParam(array)
    except rospy.ROSInterruptException:
        pass
if __name__ == '__main__':
    ciao = "ciao!"
    array = [4, 8, 2.2, 4, 23, 5, 23, 1, 1, 1, 2, 4, 6]
    sendIt(array)



