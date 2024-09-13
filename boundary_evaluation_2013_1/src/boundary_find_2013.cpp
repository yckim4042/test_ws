#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <cmath>
#include <unordered_map>

typedef pcl::PointXYZI PointT;  // XYZ + intensity, ring 정보를 저장하기 위해 intensity를 사용

// 포인트의 깊이 계산 (원점으로부터의 거리)
float calculate_range(const PointT& point) {
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

// ring별로 깊이 차이 계산 및 필터링
void process_ring_points(pcl::PointCloud<PointT>::Ptr cloud, std::vector<int>& indices, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output_cloud) {
    for (size_t i = 1; i < indices.size() - 1; ++i) {
        int current_idx = indices[i];
        int left_idx = indices[i - 1];
        int right_idx = indices[i + 1];

        float current_range = calculate_range(cloud->points[current_idx]);
        float left_range = calculate_range(cloud->points[left_idx]);
        float right_range = calculate_range(cloud->points[right_idx]);

        // 왼쪽, 오른쪽과의 깊이 차이 중 가장 큰 값
        float max_difference = std::max({ std::abs(current_range - left_range), std::abs(current_range - right_range), 0.0f });

        // 차이가 0.5 이상인 경우에만 해당 포인트를 저장
        if (max_difference >= 0.05) {
            pcl::PointXYZRGBA significant_point;
            significant_point.x = cloud->points[current_idx].x;
            significant_point.y = cloud->points[current_idx].y;
            significant_point.z = cloud->points[current_idx].z;
            significant_point.r = 255;  // 색상은 따로 변경할 필요가 없지만, 기본값을 빨간색으로 설정
            significant_point.g = 0;
            significant_point.b = 0;
            output_cloud->points.push_back(significant_point);
        }
    }
}

void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // ROS 포인트 클라우드 메시지를 PCL 포인트 클라우드로 변환
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // ring 값을 기반으로 포인트들을 그룹화
    std::unordered_map<int, std::vector<int>> ring_map;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        int ring = static_cast<int>(cloud->points[i].ring);  // ring 값을 intensity로 저장
        ring_map[ring].push_back(i);
    }

    // 결과 포인트 클라우드 (깊이 차이가 큰 포인트들만 저장)
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr significant_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

    // 각 ring에 대해 깊이 차이 계산 및 필터링 수행
    for (const auto& ring_entry : ring_map) {
        const std::vector<int>& ring_indices = ring_entry.second;
        process_ring_points(cloud, ring_indices, significant_cloud);
    }

    // 필터링된 포인트 클라우드를 PLY 파일로 저장 (깊이 차이가 큰 점들만)
    pcl::io::savePLYFileASCII("boundary_points.ply", *significant_cloud);
    ROS_INFO("boundary points saved to boundary_points.ply");
}

int main(int argc, char** argv) {
    // ROS 노드 초기화
    ros::init(argc, argv, "ouster_depth_comparison");
    ros::NodeHandle nh;

    // Ouster 포인트 클라우드 토픽 구독
    ros::Subscriber sub = nh.subscribe("/ouster/points", 1, cloud_callback);

    // ROS 이벤트 루프 실행
    ros::spin();

    return 0;
}

