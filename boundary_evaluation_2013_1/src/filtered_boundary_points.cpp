#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl_ros/point_cloud.h>
#include <unordered_map>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <cmath>  // To calculate sqrt for distance

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


// Variables to store cloud and publishers
ros::Publisher filtered_pub;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
bool save_plane_to_ply = true;  // Set true to save plane points to PLY file

// Function to visualize both the filtered points and the plane points with different colors
void visualizePointClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud) {
    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
    viewer.setBackgroundColor(0, 0, 0);

    // 1. Visualize the filtered points (remaining points after distance filtering) in white
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white_color(filtered_cloud, 255, 255, 255);
    viewer.addPointCloud<pcl::PointXYZ>(filtered_cloud, white_color, "filtered cloud");

    // 2. Visualize the plane points (points belonging to the plane) in red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color(plane_cloud, 255, 0, 0);
    viewer.addPointCloud<pcl::PointXYZ>(plane_cloud, red_color, "plane cloud");

    // Set point size and properties
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "filtered cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "plane cloud");

    viewer.addCoordinateSystem(1.0);  // Add coordinate system for reference
    viewer.initCameraParameters();
    viewer.spin();  // Keep viewer open
}

// Function to save the plane points to a PLY file
void savePlaneToPLY(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& filename) {
    if (pcl::io::savePLYFile(filename, *cloud) == -1) {
        ROS_ERROR("Failed to save PLY file");
    } else {
        ROS_INFO("Successfully saved plane points to %s", filename.c_str());
    }
}

// Function to remove walls based on the given range (equivalent to remove_walls in Python)
pcl::PointCloud<pcl::PointXYZ>::Ptr removeWalls(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                float x_min = 1, float x_max = 1.5,
                                                float y_min = -0.7, float y_max = 0.7,
                                                float z_min = -1, float z_max = 1) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : cloud->points) {
        if (point.x > x_min && point.x < x_max &&
            point.y > y_min && point.y < y_max &&
            point.z > z_min && point.z < z_max) {
            filtered_cloud->points.push_back(point);
        }
    }
    return filtered_cloud;
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
    
    // 1. removeWalls 함수를 사용하여 포인트 필터링
    pcl::PointCloud<pcl::PointXYZ>::Ptr distance_filtered;
    distance_filtered = removeWalls(cloud);  // 원하는 범위 설정
    distance_filtered->header = cloud->header;  // Keep the header information

    // 2. Segment the plane using RANSAC with specific axis constraint
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);  // Use PERPENDICULAR_PLANE model
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.04);  // Set the distance threshold for inliers

    // Set the axis for finding a plane perpendicular to this vector
    seg.setAxis(Eigen::Vector3f(0.9080245425960971, -0.14744341656355536, -0.39211206173135554));  // Set the axis constraint
    seg.setEpsAngle(0.20);  // Set the angle tolerance (in radians)

    seg.setInputCloud(distance_filtered);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        ROS_WARN("No plane found in the point cloud.");
        return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr pattern_cloud;
    // Get points belonging to plane in pattern pointcloud
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr dit(
      new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(significant_cloud));
    std::vector<int> inliers2;
    dit->selectWithinDistance(coefficients, 0.04, inliers2);
    pcl::copyPointCloud<pcl::PointXYZ>(*significant_cloud, inliers2, *pattern_cloud);

    // 필터링된 포인트 클라우드를 PLY 파일로 저장 (깊이 차이가 큰 점들만)
    pcl::io::savePLYFileASCII("filtered_boundary_points.ply", *pattern_cloud);
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

