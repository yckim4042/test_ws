import rospy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def calculate_range(point):
    """Calculate the range (distance from origin) of a point."""
    return np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)

def process_ring_points(points, ring_indices):
    """For each point in the ring, compare depth with its left and right neighbors."""
    colors = np.ones((len(points), 3))  # 기본적으로 모든 포인트는 흰색

    # 각 점에 대해 왼쪽과 오른쪽의 깊이 차이를 비교
    for i in range(1, len(points) - 1):
        current_range = calculate_range(points[i])
        left_range = calculate_range(points[i - 1])
        right_range = calculate_range(points[i + 1])

        # 깊이 차이 계산
        max_difference = max(abs(current_range - left_range), abs(current_range - right_range), 0)

        # 차이가 0.5 이상인 경우, 빨간색으로 표시
        if max_difference >= 0.05:
            colors[i] = [1, 0, 0]  # 빨간색

    return colors

def cloud_callback(msg):
    # 포인트 클라우드를 numpy 배열로 변환
    point_cloud = []
    ring_data = []

    for point in pc2.read_points(msg, field_names=("x", "y", "z", "ring"), skip_nans=True):
        x, y, z, ring = point
        point_cloud.append([x, y, z])
        ring_data.append(ring)

    point_cloud = np.array(point_cloud)
    ring_data = np.array(ring_data)

    # 결과 포인트 클라우드 색상 배열
    colors = np.ones((len(point_cloud), 3))  # 기본 흰색 (RGB: [1, 1, 1])

    # 각 ring 값에 대해 처리
    unique_rings = np.unique(ring_data)
    for ring in unique_rings:
        # 해당 ring에 속하는 인덱스 추출
        ring_indices = np.where(ring_data == ring)[0]
        
        # 해당 ring에 있는 포인트들을 추출
        ring_points = point_cloud[ring_indices]
        
        # 각 포인트의 왼쪽, 오른쪽과 깊이 차이 비교
        ring_colors = process_ring_points(ring_points, ring_indices)

        # 색상 적용
        colors[ring_indices] = ring_colors

    # 포인트 클라우드를 open3d의 PointCloud로 변환
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    pc.colors = o3d.utility.Vector3dVector(colors)

    # PLY 파일로 저장
    o3d.io.write_point_cloud("filtered_cloud_with_depth_comparison.ply", pc)
    rospy.loginfo("Filtered point cloud saved to filtered_cloud_with_depth_comparison.ply")

def main():
    # ROS 노드 초기화
    rospy.init_node('ouster_depth_comparison', anonymous=True)

    # Ouster 포인트 클라우드 토픽 구독
    rospy.Subscriber("/ouster/points", PointCloud2, cloud_callback)

    # ROS 이벤트 루프 실행
    rospy.spin()

if __name__ == '__main__':
    main()

