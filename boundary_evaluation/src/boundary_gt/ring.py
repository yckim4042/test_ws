import rospy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def cloud_callback(msg):
    # 포인트 클라우드 데이터를 읽어와서 numpy 배열로 변환
    point_cloud = []
    colors = []
    
    for point in pc2.read_points(msg, field_names=("x", "y", "z", "ring"), skip_nans=True):
        x, y, z, ring = point

        # 모든 포인트를 저장
        point_cloud.append([x, y, z])

        # ring 값이 4의 배수이면 빨간색, 아니면 흰색으로 표시
        if ring % 4 == 0:
            colors.append([1, 0, 0])  # 빨간색
        else:
            colors.append([1, 1, 1])  # 흰색

    # 포인트 클라우드를 open3d의 PointCloud로 변환
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(point_cloud))
    pc.colors = o3d.utility.Vector3dVector(np.array(colors))

    # PLY 파일로 저장
    o3d.io.write_point_cloud("filtered_cloud_with_colors.ply", pc)
    rospy.loginfo("Filtered point cloud saved to filtered_cloud_with_colors.ply")

def main():
    # ROS 노드 초기화
    rospy.init_node('ouster_channel_filter', anonymous=True)

    # Ouster 포인트 클라우드 토픽 구독
    rospy.Subscriber("/ouster/points", PointCloud2, cloud_callback)

    # ROS 이벤트 루프 실행
    rospy.spin()

if __name__ == '__main__':
    main()

