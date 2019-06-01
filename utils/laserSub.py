import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt

class LaserSubs:
    point_cloud_raw = np.empty((0,2))

    def __init__(self):
        # initialize susbscriber
        rospy.Subscriber('/scan', LaserScan, self.callback)


    def callback(self, msg):
        self.point_cloud_raw = []
        num_points = len(msg.ranges)
        angle_increment = msg.angle_increment
        i=0
        # only points in the front
        for point in msg.ranges[int(-num_points/2):int(num_points/2)]:
            if point!=inf:
                self.point_cloud_raw = np.append(self.point_cloud_raw,[[point*np.cos(i),point*np.sin(i)]],axis=0)
            else:
                pass
            i+=angle_increment

        print('call backed')
        ###############################
        #
        #   Debug only
        # if len(self.point_cloud_raw)!=0:
        #     f = plt.figure(figsize=(18, 14))
        #     ax = f.add_subplot(111)
        #     ax.scatter(self.point_cloud_raw[:,1],self.point_cloud_raw[:,0],
        #             s=0.1, c=[(0,1,0)]*len(point_cloud_raw))
        #     ax.set_title('BEV before frustum extraction')
        #     ax.set_facecolor((0,0,0))
        #     plt.show
        ###############################
