#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

# Global variables
accumulated_points = []
frame_count = 0  # To count the number of frames

def callback(data):
    global accumulated_points, frame_count
    try:
        # Convert PointCloud2 message to a list of points
        points = []
        for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        # Append new points to the accumulated points list
        accumulated_points.extend(points)
        # Increase the frame count
        frame_count += 1
        rospy.loginfo("Frame {}: Accumulated {} points".format(frame_count, len(accumulated_points)))

        # Save the point cloud after receiving 3 frames
        if frame_count >= 6:
            save_point_cloud()

    except Exception as e:
        rospy.logerr(f"Error in callback: {e}")

def save_point_cloud():
    global accumulated_points
    # Convert to numpy array with float32 type
    np_points = np.array(accumulated_points, dtype=np.float32)

    # Write the .ply file manually with float32
    with open('./points_6circle.ply', 'w') as ply_file:
        # Write the header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(np_points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # Write the points
        for point in np_points:
            ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")

    rospy.loginfo("Saved point cloud to accumulated_pointcloud_3.ply")
    rospy.signal_shutdown("Point cloud saved")  # Shutdown the node after saving the data

def listener():
    rospy.init_node('save_filtered_pointcloud', anonymous=True)
    rospy.Subscriber("/ouster/points", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

