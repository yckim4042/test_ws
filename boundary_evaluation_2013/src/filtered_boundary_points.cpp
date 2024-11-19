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
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

typedef pcl::PointXYZI PointT;  // XYZ + intensity, ring 정보를 저장하기 위해 intensity를 사용
double plane_distance_inliers_1 = 0.03;
double plane_distance_inliers_2 = 0.05;

// 포인트의 깊이 계산 (원점으로부터의 거리)
float calculate_range(const PointT& point) {
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}
// Function to perform Euclidean Cluster Extraction
std::vector<pcl::PointIndices> clusterPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
                                             float eps = 0.03, int min_samples = 10) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(eps);
    ec.setMinClusterSize(min_samples);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    return cluster_indices;
}
// Function to perform PCA and return explained variance
Eigen::Vector3f pcaAnalysis(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);

    Eigen::Vector3f eigen_values = pca.getEigenValues().head<3>();
    return eigen_values;
}
// Function to find the most planar cluster
pcl::PointCloud<pcl::PointXYZ>::Ptr findMostPlanarCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
                                                          const std::vector<pcl::PointIndices> &cluster_indices) {
    float min_variance_ratio = std::numeric_limits<float>::max();
    pcl::PointCloud<pcl::PointXYZ>::Ptr most_planar_cluster(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto &indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &index : indices.indices) {
            cluster->points.push_back(cloud->points[index]);
        }

        if (cluster->points.size() < 3) {
            continue;
        }

        Eigen::Vector3f eigen_values = pcaAnalysis(cluster);
        float variance_ratio = (eigen_values[2] / eigen_values[0]) / eigen_values[1];

        if (variance_ratio < min_variance_ratio) {
        float max_z = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        pcl::PointXYZ max_z_point, min_z_point;

        for (const auto &point : cluster->points) {
            if (point.z > max_z) {
                max_z = point.z;
                max_z_point = point;  // max_z에 해당하는 포인트 저장
           }
            if (point.z < min_z) {
                min_z = point.z;
                min_z_point = point;  // min_z에 해당하는 포인트 저장
            }
        }
    
        // max_z_point와 min_z_point 간의 유클리드 거리 계산
        float distance = std::sqrt(std::pow(max_z_point.x - min_z_point.x, 2) + 
                                   std::pow(max_z_point.y - min_z_point.y, 2) + 
                                   std::pow(max_z_point.z - min_z_point.z, 2));

        if (distance < 0.6) {
            min_variance_ratio = variance_ratio;
            most_planar_cluster = cluster;
        }
    }
    }

    return most_planar_cluster;
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
void process_ring_points(pcl::PointCloud<PointT>::Ptr cloud, 
                         const std::vector<int>& indices, 
                         pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud, 
                         const pcl::PointCloud<pcl::PointXYZ>::Ptr& most_planar_cluster) {

    // most_planar_cluster에 있는 점들을 비교하기 위해 좌표를 저장해둠
    std::set<std::tuple<float, float, float>> planar_points_set;
    for (const auto& point : most_planar_cluster->points) {
        planar_points_set.insert(std::make_tuple(point.x, point.y, point.z));
    }

    for (size_t i = 1; i < indices.size() - 1; ++i) {
        int current_idx = indices[i];
        int left_idx = indices[i - 1];
        int right_idx = indices[i + 1];

        float current_range = calculate_range(cloud->points[current_idx]);
        float left_range = calculate_range(cloud->points[left_idx]);
        float right_range = calculate_range(cloud->points[right_idx]);

        float max_difference = std::max({ std::abs(current_range - left_range), 
                                          std::abs(current_range - right_range), 
                                          0.0f });

        // 차이가 0.05 이상인 경우에만 처리
        if (max_difference >= 0.05) {
            const auto& current_point = cloud->points[current_idx];
            auto point_tuple = std::make_tuple(current_point.x, current_point.y, current_point.z);

            // current_point가 most_planar_cluster에 속하는지 확인
            if (planar_points_set.find(point_tuple) != planar_points_set.end()) {
                pcl::PointXYZI significant_point;
                significant_point.x = current_point.x;
                significant_point.y = current_point.y;
                significant_point.z = current_point.z;
                output_cloud->points.push_back(significant_point);
            }
        }
    }
}

// Function to remove walls based on the given range (equivalent to remove_walls in Python)
pcl::PointCloud<pcl::PointXYZI>::Ptr removeWalls(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                                float x_min = 0, float x_max = 2,
                                                float y_min = -1.3, float y_max =1,
                                                float z_min = -1, float z_max = 1) {
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
Eigen::Vector3f pcaFindComponent(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::PCA<pcl::PointXYZ> pca;
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
Eigen::Vector3f computeMeanPoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
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
                                                       const pcl::PointCloud<pcl::PointXYZ>::Ptr &mean_cloud) {
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
        if (distance_to_first_axis > 0.13 || distance_to_second_axis > 0.20) {
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

// Function to compute distance from a point to a line
// 두 축을 찾는 함수 (Python의 find_axes 함수와 동일)
std::pair<Eigen::Vector3f, Eigen::Vector3f> findAxes(const Eigen::Vector4f &plane_coeffs) {
    float a = plane_coeffs[0], b = plane_coeffs[1], c = plane_coeffs[2];
    Eigen::Vector3f u1, u2;

    if (a != 0 || b != 0) {
        u1 = Eigen::Vector3f(-b, a, 0);
    } else {
        u1 = Eigen::Vector3f(1, 0, 0);
    }

    u2 = Eigen::Vector3f(a, b, c).cross(u1);

    u1.normalize();
    u2.normalize();

    return {u1, u2};
}

// 2D 평면에서 원의 중심을 구하는 함수 (Python의 circle_fit_constrained_2d 함수와 동일)
std::pair<Eigen::Vector3f, float> circleFitConstrained2D(const pcl::PointCloud<pcl::PointXYZ>::Ptr &points, const Eigen::Vector4f &plane_coeffs) {
    auto [u1, u2] = findAxes(plane_coeffs);
    Eigen::Vector3f center_point(0, 0, 0);

    for (const auto &point : points->points) {
        center_point += point.getVector3fMap();
    }
    center_point /= points->points.size();

    Eigen::Matrix4f Rot;
    Rot << u1[0], u2[0], plane_coeffs[0], center_point[0],
           u1[1], u2[1], plane_coeffs[1], center_point[1],
           u1[2], u2[2], plane_coeffs[2], center_point[2],
           0, 0, 0, 1;

    Eigen::Matrix4f R_inv = Rot.inverse();
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_2d(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto &point : points->points) {
        Eigen::Vector4f point_homogeneous(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f point_2d_homogeneous = R_inv * point_homogeneous;
        points_2d->points.emplace_back(point_2d_homogeneous[0], point_2d_homogeneous[1], 0.0f);
    }

    Eigen::MatrixXf A(points_2d->points.size(), 3);
    Eigen::VectorXf B(points_2d->points.size());

    for (size_t i = 0; i < points_2d->points.size(); ++i) {
        A(i, 0) = 2 * points_2d->points[i].x;
        A(i, 1) = 2 * points_2d->points[i].y;
        A(i, 2) = 1;
        B(i) = points_2d->points[i].x * points_2d->points[i].x + points_2d->points[i].y * points_2d->points[i].y;
    }

    Eigen::Vector3f circle_params = A.colPivHouseholderQr().solve(B);
    float x0 = circle_params[0], y0 = circle_params[1];
    float radius = std::sqrt(x0 * x0 + y0 * y0 + circle_params[2]);

    Eigen::Vector4f circle_center_2d(x0, y0, 0, 1);
    Eigen::Vector4f circle_center_3d_homogeneous = Rot * circle_center_2d;
    Eigen::Vector3f circle_center_3d(circle_center_3d_homogeneous[0], circle_center_3d_homogeneous[1], circle_center_3d_homogeneous[2]);

    return {circle_center_3d, radius};
}

// RANSAC을 이용해 원의 중심과 반지름을 찾는 함수 (Python의 ransac_circle_fit 함수와 동일)
std::tuple<Eigen::Vector3f, float, pcl::PointCloud<pcl::PointXYZ>::Ptr> ransacCircleFit(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster_points,
                                                                                        const Eigen::Vector4f &plane_coeffs,
                                                                                        int ransac_iterations = 200,
                                                                                        float distance_threshold = 0.005) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr max_inliers(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Vector3f best_circle(0, 0, 0);
    float best_radius = 0;

    // Boost 라이브러리를 사용하여 랜덤 장치와 분포 생성
    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());
    boost::random::uniform_int_distribution<> dis(0, cluster_points->points.size() - 1);

    for (int i = 0; i < ransac_iterations; ++i) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr sample_points(new pcl::PointCloud<pcl::PointXYZ>);
        std::set<int> sample_indices;

        while (sample_indices.size() < 5) {
            sample_indices.insert(dis(gen)); // Boost를 사용하여 랜덤 샘플링
        }

        for (const auto &index : sample_indices) {
            sample_points->points.push_back(cluster_points->points[index]);
        }

        auto [circle_candidate, radius_candidate] = circleFitConstrained2D(sample_points, plane_coeffs);

        pcl::PointCloud<pcl::PointXYZ>::Ptr inliers(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &point : cluster_points->points) {
            Eigen::Vector3f p(point.x, point.y, point.z);
            float distance = (p.head<2>() - circle_candidate.head<2>()).norm();
            if (std::abs(distance - radius_candidate) < distance_threshold) {
                inliers->points.push_back(point);
            }
        }

        if (inliers->points.size() > max_inliers->points.size()) {
            max_inliers = inliers;
            best_circle = circle_candidate;
            best_radius = radius_candidate;
        }
    }

    return {best_circle, best_radius, max_inliers};
}

// 3D 공간에서 세 점으로 원의 중심과 반지름을 찾는 함수 (Python의 circle_fit_from_3_points 함수와 동일)
std::pair<Eigen::Vector3f, float> circleFitFrom3Points(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const Eigen::Vector3f &p3) {
    Eigen::Vector3f A = p2 - p1;
    Eigen::Vector3f B = p3 - p1;
    Eigen::Vector3f normal = A.cross(B);

    float A_sq = A.squaredNorm();
    float B_sq = B.squaredNorm();
    Eigen::Vector3f cross_AB = A.cross(B);
    float cross_AB_norm = cross_AB.squaredNorm();

    if (cross_AB_norm == 0) {
        return {Eigen::Vector3f(0, 0, 0), -1};
    }

    Eigen::Vector3f center_proj = p1 + (B_sq * normal.cross(A) + A_sq * B.cross(normal)) / (2 * cross_AB_norm);
    float radius = (center_proj - p1).norm();

    return {center_proj, radius};
}

// RANSAC을 사용하여 원을 찾는 클러스터링 함수 (Python의 ransac_circle_clustering 함수와 동일)
std::vector<int> ransacCircleClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &points, const Eigen::Vector4f &plane_coeffs,
                                        float min_radius = 0.024, float max_radius = 0.036,
                                        float radius_threshold = 0.01, int iterations = 300) {
    std::vector<int> labels(points->points.size(), -1);
    int current_label = 0;

    // Boost 라이브러리를 사용하여 랜덤 장치와 분포 생성
    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());
    boost::random::uniform_int_distribution<> dis(0, points->points.size() - 1);

    for (int cluster_count = 0; cluster_count < 6; ++cluster_count) {
        int max_inliers = 0;
        std::vector<int> best_inliers;

        for (int i = 0; i < iterations; ++i) {
            int p1_idx = dis(gen); // Boost를 사용하여 랜덤 샘플링
            Eigen::Vector3f p1 = points->points[p1_idx].getVector3fMap();

            std::vector<int> close_points;
            for (size_t j = 0; j < points->points.size(); ++j) {
                if (labels[j] == -1 && (points->points[j].getVector3fMap() - p1).norm() < 0.07) {
                    close_points.push_back(j);
                }
            }

            if (close_points.size() < 2) continue;

            int p2_idx = close_points[dis(gen) % close_points.size()];
            int p3_idx = close_points[dis(gen) % close_points.size()];
            Eigen::Vector3f p2 = points->points[p2_idx].getVector3fMap();
            Eigen::Vector3f p3 = points->points[p3_idx].getVector3fMap();

            auto [center, radius] = circleFitFrom3Points(p1, p2, p3);
            if (radius < min_radius || radius > max_radius) continue;

            std::vector<int> inliers;
            for (size_t j = 0; j < points->points.size(); ++j) {
                if (labels[j] == -1 && std::abs((points->points[j].getVector3fMap() - center).norm() - radius) < radius_threshold) {
                    inliers.push_back(j);
                }
            }

            if (inliers.size() > max_inliers) {
                max_inliers = inliers.size();
                best_inliers = inliers;
            }
        }

        if (!best_inliers.empty()) {
            for (const auto &index : best_inliers) {
                labels[index] = current_label;
            }
            ++current_label;
        }
    }

    return labels;
}
// RANSAC 기반의 클러스터 라벨링을 사용해 원의 중심을 찾는 함수 (Python의 analyze_and_fit_circles_with_ransac_clustering 함수와 동일)
std::vector<Eigen::Vector3f> analyzeAndFitCirclesWithRansacClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &noise_free_points,
                                                                      const Eigen::Vector4f &plane_coeffs,
                                                                      float min_radius = 0.024, float max_radius = 0.036,
                                                                      float radius_threshold = 0.005, int iterations = 100) {
    std::vector<int> labels = ransacCircleClustering(noise_free_points, plane_coeffs, min_radius, max_radius, radius_threshold, iterations);

    std::vector<Eigen::Vector3f> circle_centers;
    std::set<int> unique_labels(labels.begin(), labels.end());

    for (int label : unique_labels) {
        if (label == -1) continue;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == label) {
                cluster_points->points.push_back(noise_free_points->points[i]);
            }
        }

        if (cluster_points->points.size() >= 5) {
            auto [best_circle, best_radius, inliers] = ransacCircleFit(cluster_points, plane_coeffs);
            auto [final_circle_center, final_radius] = circleFitConstrained2D(inliers, plane_coeffs);
            circle_centers.push_back(final_circle_center);
            std::cout << "Circle center for cluster " << label << ": " << final_circle_center.transpose() << " with radius " << final_radius << std::endl;

            if (final_radius >= 0.042 || final_radius <= 0.024) {
                std::cout << "원이 잘못 피팅됨" << std::endl;
                return {};
            }
        } else {
            auto [center, radius] = circleFitConstrained2D(cluster_points, plane_coeffs);
            circle_centers.push_back(center);
            std::cout << "Circle center for cluster " << label << ": " << center.transpose() << " with radius " << radius << std::endl;
        }
    }

    return circle_centers;
}

// 원 중심 정렬 함수
std::vector<Eigen::Vector3f> sortCircleCenters(const std::vector<Eigen::Vector3f> &circle_centers) {
    std::vector<Eigen::Vector3f> sorted_centers = circle_centers;

    // y 좌표 기준 내림차순 정렬
    std::sort(sorted_centers.begin(), sorted_centers.end(), [](const Eigen::Vector3f &a, const Eigen::Vector3f &b) {
        return a[1] > b[1];
    });

    Eigen::Vector3f point_1, point_2, point_3, point_4, point_5, point_6;

    // y 좌표가 가장 큰 두 점 중 z 좌표가 더 큰 점을 1번, 작은 점을 5번
    if (sorted_centers[0][2] > sorted_centers[1][2]) {
        point_1 = sorted_centers[0];
        point_5 = sorted_centers[1];
    } else {
        point_1 = sorted_centers[1];
        point_5 = sorted_centers[0];
    }

    // y 좌표가 세 번째로 큰 점은 3번
    point_3 = sorted_centers[2];

    // y 좌표가 네 번째와 다섯 번째로 큰 두 점 중 z 좌표가 더 큰 점을 2번, 작은 점을 6번
    if (sorted_centers[3][2] > sorted_centers[4][2]) {
        point_2 = sorted_centers[3];
        point_6 = sorted_centers[4];
    } else {
        point_2 = sorted_centers[4];
        point_6 = sorted_centers[3];
    }

    // y 좌표가 가장 작은 점은 4번
    point_4 = sorted_centers[5];

    // 최종 정렬된 순서로 리스트 반환
    return {point_1, point_2, point_3, point_4, point_5, point_6};
}
void saveCircleCentersToFile(const std::vector<Eigen::Vector3f>& circle_centers, const std::string& file_name) {
    std::ofstream file(file_name, std::ios_base::app);  // 파일을 append 모드로 엽니다.
    if (file.is_open()) {
        for (const auto& center : circle_centers) {
            file << center.x() << " " << center.y() << " " << center.z() << std::endl;
        }
        file.close();
    } else {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
    }
}

void savePoseToFile(float x, float y, float z, float rot1, float rot2, float rot3, const std::string& file_name) {
    std::ofstream file(file_name, std::ios_base::app);  // 파일을 추가 모드로 엽니다.
    if (file.is_open()) {
        file << x << " " << y << " " << z << " " << rot1 << " " << rot2 << " " << rot3 << std::endl;
        file.close();
    } else {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
    }
}
Eigen::MatrixXf board_points(3, 6);


// float 타입의 Matrix4f를 반환하는 함수
Eigen::Matrix4f compute_R(Eigen::Vector3f w, Eigen::Vector3f v, float a, float b, float c, Eigen::Vector3f center_5) {
    Eigen::Matrix4f R;
    R << w(0), v(0), a, center_5(0),
         w(1), v(1), b, center_5(1),
         w(2), v(2), c, center_5(2),
         0, 0, 0, 1;
    return R;
}

// Ceres Solver의 비용 함수 구조체
struct Residuals {
    Residuals(Eigen::Matrix4f R_inv, Eigen::MatrixXf transformed_world_points)
        : R_inv(R_inv), transformed_world_points(transformed_world_points) {}

    template <typename T>
    bool operator()(const T* const theta, const T* const x_offset, const T* const y_offset, T* residual) const {
        T cos_theta = ceres::cos(theta[0]);
        T sin_theta = ceres::sin(theta[0]);

        Eigen::Matrix<T, 3, 3> R_board;
        R_board << cos_theta, -sin_theta, x_offset[0],
                   sin_theta,  cos_theta, y_offset[0],
                   T(0), T(0), T(1);

        Eigen::Matrix<T, 3, 6> transformed_points = R_board.template cast<T>() * board_points.template cast<T>();
        Eigen::Matrix<T, 2, 6> transformed_points_2d = transformed_points.topRows(2);

        Eigen::Matrix<T, 2, 6> transformed_world_points_2d = transformed_world_points.topRows(2).template cast<T>();

        Eigen::Matrix<T, 2, 6> diff = transformed_points_2d - transformed_world_points_2d;

        for (int i = 0; i < 6; ++i) {
            residual[2 * i] = diff(0, i);
            residual[2 * i + 1] = diff(1, i);
        }

        return true;
    }

    Eigen::Matrix4f R_inv;
    Eigen::MatrixXf transformed_world_points;
};
//새로 추가한 변환법
Eigen::Vector3d to_so3(const Eigen::Matrix3d &R) {
    double trace_R = R.trace();
    double theta = acos((trace_R - 1.0) / 2.0);

    if (theta == 0.0) return Eigen::Vector3d::Zero();

    Eigen::Matrix3d skew_m = (theta / (2.0 * sin(theta))) * (R - R.transpose());
    return Eigen::Vector3d(skew_m(2, 1), skew_m(0, 2), skew_m(1, 0));
}
// 최적화 함수 정의, float 타입 사용
std::tuple<Eigen::Matrix4f, float, float, float, float, float, float> optimize_rotation_transform(
    std::vector<Eigen::Vector3f> cluster_centers, Eigen::Vector4f plane_coeffs) {
    
    Eigen::Vector3f center_5 = cluster_centers[4];
    Eigen::Vector3f center_6 = cluster_centers[5];
    float a = plane_coeffs(0);
    float b = plane_coeffs(1);
    float c = plane_coeffs(2);
    float d = plane_coeffs(3);
    if(plane_coeffs(3)<0){
        a=-a;
        b=-b;
        c=-c;
        d=-d;
    }
    Eigen::Vector3f w = center_6 - center_5;
    w.normalize();  // 벡터 정규화

    Eigen::Vector3f v = Eigen::Vector3f(a, b, c).cross(w);
    Eigen::Matrix4f R = compute_R(w, v, a, b, c, center_5);
    //std::cout << R<<std::endl;
    Eigen::Matrix4f R_inv = R.inverse();
    //std::cout << R_inv <<std::endl;
    Eigen::MatrixXf world_points(4, cluster_centers.size());
    for (size_t i = 0; i < cluster_centers.size(); ++i) {
        world_points.col(i) << cluster_centers[i], 1.0f;
    }

    Eigen::MatrixXf transformed_world_points = R_inv * world_points;
    //std::cout << world_points <<std::endl;
    //std::cout << transformed_world_points <<std::endl;
    transformed_world_points = transformed_world_points.topRows(3).eval();
    
    
    // 초기 파라미터
    double theta_init = 0.0;
    double x_offset_init = -0.09;
    double y_offset_init = -0.09;

    // Ceres 문제 정의
    ceres::Problem problem;
    Residuals* residuals = new Residuals(R_inv, transformed_world_points);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Residuals, 12, 1, 1, 1>(residuals),
        nullptr,
        &theta_init,
        &x_offset_init,
        &y_offset_init
    );

    // Solver 옵션 설정
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // Solver 수행
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    // 최적화된 파라미터
    float optimized_theta = theta_init;
    float optimized_x_offset = x_offset_init;
    float optimized_y_offset = y_offset_init;

    Eigen::Matrix4f optimized_R_board;
    optimized_R_board << std::cos(optimized_theta), -std::sin(optimized_theta), 0.0f, optimized_x_offset,
                         std::sin(optimized_theta),  std::cos(optimized_theta), 0.0f, optimized_y_offset,
                         0.0f, 0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::Matrix4f final_transform = R * optimized_R_board;
    Eigen::Matrix3d R_3x3 = final_transform.block<3,3>(0,0).cast<double>();
    
    Eigen::Vector3d rot_v= to_so3(R_3x3);

    return std::make_tuple(final_transform, rot_v[0], rot_v[1], rot_v[2], final_transform(0, 3), final_transform(1, 3), final_transform(2, 3));
}

void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
board_points << 0.09f, 0.27f, 0.18f, 0.36f, 0.09f, 0.27f,
                0.27f, 0.27f, 0.18f, 0.18f, 0.09f, 0.09f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f;
    // ROS 포인트 클라우드 메시지를 PCL 포인트 클라우드로 변환
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr removed_cloud_I=removeWalls(cloud);
    if (removed_cloud_I->points.empty()) {
    std::cerr << "Error: Input cloud is empty!" << std::endl;
    return;
}
    pcl::PointCloud<pcl::PointXYZ>::Ptr removed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : removed_cloud_I->points) {
        pcl::PointXYZ xyz_point;
        xyz_point.x = point.x;
        xyz_point.y = point.y;
        xyz_point.z = point.z;
        removed_cloud->points.push_back(xyz_point);
    }
    std::vector<pcl::PointIndices> cluster_indices = clusterPoints(removed_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr most_planar_cluster = findMostPlanarCluster(removed_cloud, cluster_indices);
    // ring 정보를 담을 맵 생성
    std::unordered_map<int, std::vector<int>> ring_map;
    // ring 정보 및 포인트 클라우드 추출 함수 호출
    extract_points_and_ring(cloud_msg, cloud, ring_map);
    // 결과 포인트 클라우드 (깊이 차이가 큰 포인트들만 저장)
    pcl::PointCloud<pcl::PointXYZI>::Ptr significant_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // 각 ring에 대해 깊이 차이 계산 및 필터링 수행
    for (const auto& ring_entry : ring_map) {
        const std::vector<int>& ring_indices = ring_entry.second;
        process_ring_points(cloud, ring_indices, significant_cloud,most_planar_cluster);
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
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(most_planar_cluster);
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors().topLeftCorner<3, 3>();
    seg.setAxis(Eigen::Vector3f(eigen_vectors.col(2)[0], eigen_vectors.col(2)[1], eigen_vectors.col(2)[2]));  // Set the axis constraint
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
    Eigen::Vector3f first_principal_component = pcaFindComponent(most_planar_cluster);
    Eigen::Vector3f second_principal_component = first_principal_component.cross(Eigen::Vector3f(coefficients_v(0), coefficients_v(1), coefficients_v(2)));
    pattern_cloud = findAndRemoveLines(pattern_cloud, first_principal_component, second_principal_component, most_planar_cluster);
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
    pcl::PointCloud<pcl::PointXYZI>::Ptr noise_free_points1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*xy_cloud, *noise_free_points1, rotation.inverse());
    pcl::PointCloud<pcl::PointXYZ>::Ptr noise_free_points(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto& point : noise_free_points1->points) {
        pcl::PointXYZ xyz_point;
        xyz_point.x = point.x;
        xyz_point.y = point.y;
        xyz_point.z = point.z;
        noise_free_points->points.push_back(xyz_point);
    }
    Eigen::Vector4f pca_plane = coefficients_v;
    std::vector<Eigen::Vector3f> circle_centers = analyzeAndFitCirclesWithRansacClustering(noise_free_points, pca_plane);
    for (const auto& center : circle_centers) {
        pcl::PointXYZ point;
        point.x = center.x();
        point.y = center.y();
        point.z = center.z();
        noise_free_points->points.push_back(point);
    }
    std::vector<Eigen::Vector3f> sorted_centers = sortCircleCenters(circle_centers);
    //saveCircleCentersToFile(sorted_centers, "circle_centers.txt");
    // Saving the combined point cloud to a PLY file
    //pcl::io::savePLYFile("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/with_circle_center.ply", *noise_free_points);
    auto [final_transform, roll, pitch, yaw, x, y, z] = optimize_rotation_transform(sorted_centers, pca_plane);
    std::cout << "Final Transform Matrix:\n" << final_transform << std::endl;
    //이름만 roll pitch yaw지 실제로는 rot벡터임
    std::cout << "Roll: " << roll << ", Pitch: " << pitch << ", Yaw: " << yaw << std::endl;
    std::cout << "X offset: " << x << ", Y offset: " << y << ", Z offset: " << z << std::endl;
    savePoseToFile(x, y, z, roll, pitch, yaw, "theirs_pose.txt");
    // 필터링된 포인트 클라우드를 PLY 파일로 저장 (깊이 차이가 큰 점들만)
    pcl::io::savePLYFileASCII("manual_remove_wall.ply", *distance_filtered);
    pcl::io::savePLYFileASCII("boundary_points.ply", *significant_cloud);
    
    pcl::io::savePLYFileASCII("filtered_boundary_points.ply", *noise_free_points);
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

