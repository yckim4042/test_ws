#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl_ros/point_cloud.h>
#include <unordered_map>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/common/pca.h>
#include <cmath>  // To calculate sqrt for distance

typedef pcl::PointXYZI PointT;  // XYZ + intensity, ring 정보를 저장하기 위해 intensity를 사용
double plane_distance_inliers_1 = 0.03;
double plane_distance_inliers_2 = 0.1;

// 포인트의 깊이 계산 (원점으로부터의 거리)
float calculate_range(const PointT& point) {
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

// ring 정보를 포함한 포인트 클라우드를 추출하는 함수
void extract_points_and_ring(const sensor_msgs::PointCloud2ConstPtr& cloud_msg, 
                             pcl::PointCloud<PointT>::Ptr& cloud, 
                             std::unordered_map<int, std::vector<int>>& ring_map) {
    // PointCloud2Iterator를 사용하여 'x', 'y', 'z', 'ring' 필드를 추출
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
    sensor_msgs::PointCloud2ConstIterator<uint16_t> iter_ring(*cloud_msg, "ring");

    int point_idx = 0;

    // 포인트 클라우드를 순회하며 ring 정보를 추출
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_ring, ++point_idx) {
        PointT point;
        point.x = *iter_x;
        point.y = *iter_y;
        point.z = *iter_z;
        point.intensity = static_cast<float>(*iter_ring);  // ring 정보를 intensity에 저장

        cloud->points.push_back(point);
        int ring = static_cast<int>(*iter_ring);
        ring_map[ring].push_back(point_idx);  // 포인트를 ring별로 그룹화
    }
}


// ring별로 깊이 차이 계산 및 필터링
void process_ring_points(pcl::PointCloud<PointT>::Ptr cloud, const std::vector<int>& indices, pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud) {
    for (size_t i = 1; i < indices.size() - 1; ++i) {
        int current_idx = indices[i];
        int left_idx = indices[i - 1];
        int right_idx = indices[i + 1];

        float current_range = calculate_range(cloud->points[current_idx]);
        float left_range = calculate_range(cloud->points[left_idx]);
        float right_range = calculate_range(cloud->points[right_idx]);

        // 왼쪽, 오른쪽과의 깊이 차이 중 가장 큰 값
        float max_difference = std::max({ std::abs(current_range - left_range), std::abs(current_range - right_range), 0.0f });

        // 차이가 0.05 이상인 경우에만 해당 포인트를 저장
        if (max_difference >= 0.1) {
            pcl::PointXYZI significant_point;
            significant_point.x = cloud->points[current_idx].x;
            significant_point.y = cloud->points[current_idx].y;
            significant_point.z = cloud->points[current_idx].z;
            output_cloud->points.push_back(significant_point);
        }
    }
}

// Function to remove walls based on the given range (equivalent to remove_walls in Python)
pcl::PointCloud<pcl::PointXYZI>::Ptr removeWalls(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                                float x_min = 1, float x_max = 1.5,
                                                float y_min = -0.5, float y_max = 0.2,
                                                float z_min = -0.1, float z_max = 0.35) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto &point : cloud->points) {
        if (point.x > x_min && point.x < x_max &&
            point.y > y_min && point.y < y_max &&
            point.z > z_min && point.z < z_max) {
            filtered_cloud->points.push_back(point);
        }
    }
    return filtered_cloud;
}

Eigen::Affine3f getRotationMatrix(Eigen::Vector3f source,
                                  Eigen::Vector3f target) {
  Eigen::Vector3f rotation_vector = target.cross(source);
  rotation_vector.normalize();
  double theta = acos(source[2] / sqrt(pow(source[0], 2) + pow(source[1], 2) +
                                       pow(source[2], 2)));

  
  //  cout << "Rot. vector: " << rotation_vector << " / Angle: " << theta << endl;

  Eigen::Matrix3f rotation =
      Eigen::AngleAxis<float>(theta, rotation_vector) * Eigen::Scaling(1.0f);
  Eigen::Affine3f rot(rotation);
  return rot;
}
Eigen::Vector3f pcaFindComponent(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
    pcl::PCA<pcl::PointXYZI> pca;
    pca.setInputCloud(cloud);

    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors().topLeftCorner<3, 3>();
    return eigen_vectors.col(0);
}
double distanceFromPointToLine(const Eigen::Vector3f &point, const Eigen::Vector3f &line_point, const Eigen::Vector3f &direction_vector) {
    Eigen::Vector3f direction = direction_vector.normalized();
    Eigen::Vector3f point_to_line = point - line_point;
    double projection_length = point_to_line.dot(direction);
    Eigen::Vector3f projection = projection_length * direction;
    Eigen::Vector3f distance_vector = point_to_line - projection;
    return distance_vector.norm();
}


// 포인트 클라우드의 평균점을 계산하는 함수
Eigen::Vector3f computeMeanPoint(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
    Eigen::Vector3f mean_point(0, 0, 0);
    for (const auto &point : cloud->points) {
        mean_point += point.getVector3fMap();
    }
    mean_point /= cloud->points.size();
    return mean_point;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr findAndRemoveLines(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                                       const Eigen::Vector3f &first_principal_component,
                                                       const Eigen::Vector3f &second_principal_component,
                                                       const pcl::PointCloud<pcl::PointXYZI>::Ptr &mean_cloud) {
    // 입력된 포인트 클라우드에서 평균점 계산
    Eigen::Vector3f mean_point = computeMeanPoint(mean_cloud);

    std::set<int> indices_to_remove;

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        Eigen::Vector3f p = cloud->points[i].getVector3fMap();

        // 평균점에서 첫 번째 축까지의 거리 계산
        double distance_to_first_axis = distanceFromPointToLine(p, mean_point, first_principal_component);
        // 평균점에서 두 번째 축까지의 거리 계산
        double distance_to_second_axis = distanceFromPointToLine(p, mean_point, second_principal_component);

        // 조건에 따라 포인트 제거
        if (distance_to_first_axis > 0.13 || distance_to_second_axis > 0.23) {
            indices_to_remove.insert(i);
        }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        if (indices_to_remove.find(i) == indices_to_remove.end()) {
            remaining_cloud->points.push_back(cloud->points[i]);
        }
    }

    return remaining_cloud;
}

void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // ROS 포인트 클라우드 메시지를 PCL 포인트 클라우드로 변환
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    // ring 정보를 담을 맵 생성
    std::unordered_map<int, std::vector<int>> ring_map;

    // ring 정보 및 포인트 클라우드 추출 함수 호출
    extract_points_and_ring(cloud_msg, cloud, ring_map);

    // 결과 포인트 클라우드 (깊이 차이가 큰 포인트들만 저장)
    pcl::PointCloud<pcl::PointXYZI>::Ptr significant_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    // 각 ring에 대해 깊이 차이 계산 및 필터링 수행
    for (const auto& ring_entry : ring_map) {
        const std::vector<int>& ring_indices = ring_entry.second;
        process_ring_points(cloud, ring_indices, significant_cloud);
    }
    significant_cloud = removeWalls(significant_cloud);
    // 1. removeWalls 함수를 사용하여 포인트 필터링
    pcl::PointCloud<pcl::PointXYZI>::Ptr distance_filtered = removeWalls(cloud);  // 원하는 범위 설정
    distance_filtered->header = cloud->header;  // Keep the header information

    // 2. Segment the plane using RANSAC with specific axis constraint
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);  // Use PERPENDICULAR_PLANE model
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(plane_distance_inliers_1);  // Set the distance threshold for inliers
    // Set the axis for finding a plane perpendicular to this vector
    seg.setAxis(Eigen::Vector3f(0.9080245425960971, -0.14744341656355536, -0.39211206173135554));  // Set the axis constraint
    seg.setEpsAngle(0.20);  // Set the angle tolerance (in radians)

    seg.setInputCloud(distance_filtered);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        ROS_WARN("No plane found in the point cloud.");
        return;
    }
    
    // Copy coefficients to proper object for further filtering
    Eigen::VectorXf coefficients_v(4);
    coefficients_v(0) = coefficients->values[0];
    coefficients_v(1) = coefficients->values[1];
    coefficients_v(2) = coefficients->values[2];
    coefficients_v(3) = coefficients->values[3];
    cout<<coefficients_v;
      // Get points belonging to plane in pattern pointcloud
    pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr dit(
        new pcl::SampleConsensusModelPlane<pcl::PointXYZI>(significant_cloud));
    std::vector<int> inliers2;
    dit->selectWithinDistance(coefficients_v, plane_distance_inliers_2, inliers2);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr pattern_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr auxrotated_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    
    pcl::copyPointCloud<pcl::PointXYZI>(*significant_cloud, inliers2, *pattern_cloud);
    //라인제거
    pcl::PointCloud<pcl::PointXYZI>::Ptr ransac_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr ditt(
        new pcl::SampleConsensusModelPlane<pcl::PointXYZI>(distance_filtered));
    std::vector<int> inliers22;
    ditt->selectWithinDistance(coefficients_v, plane_distance_inliers_2, inliers22);
    pcl::copyPointCloud<pcl::PointXYZI>(*distance_filtered, inliers22, *ransac_cloud);
    Eigen::Vector3f first_principal_component = pcaFindComponent(ransac_cloud);
    Eigen::Vector3f second_principal_component = first_principal_component.cross(Eigen::Vector3f(coefficients_v(0), coefficients_v(1), coefficients_v(2)));
    pattern_cloud = findAndRemoveLines(pattern_cloud, first_principal_component, second_principal_component, ransac_cloud);
    //평면에 사영
    Eigen::Vector3f xy_plane_normal_vector, floor_plane_normal_vector;
    xy_plane_normal_vector[0] = 0.0;
    xy_plane_normal_vector[1] = 0.0;
    xy_plane_normal_vector[2] = -1.0;

    floor_plane_normal_vector[0] = coefficients->values[0];
    floor_plane_normal_vector[1] = coefficients->values[1];
    floor_plane_normal_vector[2] = coefficients->values[2];
    
    Eigen::Affine3f rotation =
        getRotationMatrix(floor_plane_normal_vector, xy_plane_normal_vector);
    pcl::transformPointCloud(*pattern_cloud, *xy_cloud, rotation);
    
    pcl::PointXYZI aux_point;
    aux_point.x = 0;
    aux_point.y = 0;
    aux_point.z = (-coefficients_v(3) / coefficients_v(2));
    aux_cloud->push_back(aux_point);
    pcl::transformPointCloud(*aux_cloud, *auxrotated_cloud, rotation);
    double zcoord_xyplane = auxrotated_cloud->at(0).z;

    for (pcl::PointCloud<pcl::PointXYZI>::iterator pt = xy_cloud->points.begin();
        pt < xy_cloud->points.end(); ++pt) {
        pt->z = zcoord_xyplane;
    }
    
    
    // 필터링된 포인트 클라우드를 PLY 파일로 저장 (깊이 차이가 큰 점들만)
    pcl::io::savePLYFileASCII("manual_remove_wall.ply", *distance_filtered);
    pcl::io::savePLYFileASCII("filtered_boundary_points.ply", *xy_cloud);
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

