#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
using namespace Eigen;
// Function to remove walls based on the given range (equivalent to remove_walls in Python)
pcl::PointCloud<pcl::PointXYZ>::Ptr removeWalls(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                float x_min = 0, float x_max = 2.2,
                                                float y_min = -1.5, float y_max = 1.5,
                                                float z_min = -1.0, float z_max = 1.5) {
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

// Function to perform Euclidean Cluster Extraction
std::vector<pcl::PointIndices> clusterPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
                                             float eps = 0.04, int min_samples = 30) {
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

// Function to remove small clusters
std::vector<pcl::PointIndices> removeSmallClusters(const std::vector<pcl::PointIndices> &cluster_indices, int min_size) {
    std::vector<pcl::PointIndices> filtered_indices;

    for (const auto &indices : cluster_indices) {
        if (indices.indices.size() >= min_size) {
            filtered_indices.push_back(indices);
        }
    }

    return filtered_indices;
}

// Function to perform PCA and return explained variance
Eigen::Vector3f pcaAnalysis(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);

    Eigen::Vector3f eigen_values = pca.getEigenValues().head<3>();
    return eigen_values;
}

// Function to find the first principal component
Eigen::Vector3f pcaFindComponent(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);

    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors().topLeftCorner<3, 3>();
    return eigen_vectors.col(0);
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

std::pair<Eigen::Vector4f, pcl::PointIndices::Ptr> detectPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                                double distance_threshold = 0.025,
                                                                int ransac_n = 3, int num_iterations = 1000) {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(num_iterations);
    seg.setDistanceThreshold(distance_threshold);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    Eigen::Vector4f plane_model(coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3]);
    return std::make_pair(plane_model, inliers);
}

// Function to fit a plane using PCA
Eigen::Vector4f pcaPlaneFitting(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    Eigen::Matrix3f covariance_matrix;
    pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance_matrix);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
    Eigen::Vector3f normal = eigen_solver.eigenvectors().col(0);

    float d = -centroid.head<3>().dot(normal);

    return Eigen::Vector4f(normal[0], normal[1], normal[2], d);
}

// Function to project points to a plane using the plane's normal and point on the plane
pcl::PointCloud<pcl::PointXYZ>::Ptr projectPointsToPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
                                                         const Eigen::Vector4f &plane_model) {
    Eigen::Vector3f normal = plane_model.head<3>();
    float d = plane_model[3];
    normal.normalize();

    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : cloud->points) {
        Eigen::Vector3f p(point.x, point.y, point.z);
        float t = -d / normal.dot(p);
        Eigen::Vector3f projected_point = t * p;
        projected_cloud->points.emplace_back(projected_point[0], projected_point[1], projected_point[2]);
    }
    return projected_cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr findBoundaryPointsImproved(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                               double radius = 0.014, double min_distance = 0.003) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<bool> boundary_mask(cloud->points.size(), false);

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        std::vector<int> point_idx_radius_search;
        std::vector<float> point_radius_squared_distance;
        if (kdtree.radiusSearch(cloud->points[i], radius, point_idx_radius_search, point_radius_squared_distance) > 0) {
            if (point_idx_radius_search.size() <= 1)
                continue;

            const auto &point = cloud->points[i];

            std::vector<Eigen::Vector3f> valid_neighbors;
            for (const auto &neighbor_idx : point_idx_radius_search) {
                const auto &neighbor = cloud->points[neighbor_idx];
                Eigen::Vector3f neighbor_vec(neighbor.x - point.x, neighbor.y - point.y, neighbor.z - point.z);
                if (neighbor_vec.norm() > min_distance) {
                    valid_neighbors.push_back(neighbor_vec);
                }
            }

            if (valid_neighbors.size() <= 1)
                continue;

            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
            for (const auto &vec : valid_neighbors) {
                centroid += vec;
            }
            centroid /= valid_neighbors.size();

            double mean_distance = 0.0;
            for (const auto &vec : valid_neighbors) {
                mean_distance += vec.norm();
            }
            mean_distance /= valid_neighbors.size();

            double centroid_distance = centroid.norm();
            double cdr = centroid_distance / mean_distance;

            if (cdr <= 0.5)
                continue;

            // Calculate variance manually
            Eigen::Vector3f mean_direction = Eigen::Vector3f::Zero();
            for (const auto &vec : valid_neighbors) {
                mean_direction += vec.normalized();
            }
            mean_direction /= valid_neighbors.size();

            Eigen::Vector3f direction_variance = Eigen::Vector3f::Zero();
            for (const auto &vec : valid_neighbors) {
                direction_variance += (vec.normalized() - mean_direction).cwiseAbs2();
            }
            direction_variance /= valid_neighbors.size();

            if (direction_variance.sum() < 0.8) {
                boundary_mask[i] = true;
            }
        }
    }

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        if (boundary_mask[i]) {
            boundary_cloud->points.push_back(cloud->points[i]);
        }
    }

    return boundary_cloud;
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
// Function to find and remove edges of rectangle (alternative of the find_and_remove_lines in Python)
// 주어진 조건에 따라 포인트를 제거하는 함수
// Function to find and remove points that lie on lines (equivalent to find_and_remove_lines in Python)
pcl::PointCloud<pcl::PointXYZ>::Ptr findAndRemoveLines(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
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
        if (distance_to_first_axis > 0.13 || distance_to_second_axis > 0.2) {
            indices_to_remove.insert(i);
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
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

            if (final_radius >= 0.042 || final_radius <= 0.026) {
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
void savePoseToFile(float roll, float pitch, float yaw, float x, float y, float z, const std::string& file_name) {
    std::ofstream file(file_name, std::ios_base::app);  // 파일을 추가 모드로 엽니다.
    if (file.is_open()) {
        file << "Roll: " << roll << " Pitch: " << pitch << " Yaw: " << yaw << " "
             << "X: " << x << " Y: " << y << " Z: " << z << std::endl;
        file.close();
    } else {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
    }
}
int main(int argc, char **argv) {
    board_points << 0.09f, 0.27f, 0.18f, 0.36f, 0.09f, 0.27f,
                0.27f, 0.27f, 0.18f, 0.18f, 0.09f, 0.09f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f;  // 동차 좌표계로 확장
    // Load the point cloud from a PLY file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/points_6circle.ply", *cloud) == -1) {
        PCL_ERROR("Couldn't read the PLY file \n");
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = removeWalls(cloud);
    std::vector<pcl::PointIndices> cluster_indices = clusterPoints(filtered_cloud);
    cluster_indices = removeSmallClusters(cluster_indices, 50); // Adjust as needed
    pcl::PointCloud<pcl::PointXYZ>::Ptr most_planar_cluster = findMostPlanarCluster(filtered_cloud, cluster_indices);


    if (most_planar_cluster->empty()) {
        std::cerr << "No planar cluster found." << std::endl;
        return -1;
    }
    pcl::io::savePLYFile("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/debug/most_planer_cluster.ply", *most_planar_cluster);
    std::cout<<1;
     auto [plane_model, inliers] = detectPlanes(most_planar_cluster);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(most_planar_cluster);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*inlier_cloud);

    // Fit a plane using PCA
    Eigen::Vector4f pca_plane = pcaPlaneFitting(inlier_cloud);
    std::cout << "Plane equation: " << pca_plane[0] << "x + " << pca_plane[1] << "y + " << pca_plane[2] << "z + " << pca_plane[3] << " = 0" << std::endl;

    // Project points to the plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud = projectPointsToPlane(inlier_cloud, pca_plane);

    pcl::io::savePLYFile("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/debug/projected_cloud.ply", *projected_cloud);
std::cout<<2;
    // Find boundary points and remove lines
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_pcd = findBoundaryPointsImproved(projected_cloud);
    pcl::io::savePLYFile("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/debug/boundary_points.ply", *boundary_pcd);
    std::cout<<3;
    Eigen::Vector3f first_principal_component = pcaFindComponent(projected_cloud);
    Eigen::Vector3f second_principal_component = first_principal_component.cross(Eigen::Vector3f(pca_plane[0], pca_plane[1], pca_plane[2]));
    pcl::PointCloud<pcl::PointXYZ>::Ptr noise_free_points = findAndRemoveLines(boundary_pcd, first_principal_component, second_principal_component, projected_cloud);

    pcl::io::savePLYFile("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/debug/removed_line.ply", *noise_free_points);
    // Save the noise-free points to a PLY file
    
    std::vector<Eigen::Vector3f> circle_centers = analyzeAndFitCirclesWithRansacClustering(noise_free_points, pca_plane);
    for (const auto& center : circle_centers) {
        pcl::PointXYZ point;
        point.x = center.x();
        point.y = center.y();
        point.z = center.z();
        noise_free_points->points.push_back(point);
    }

    // Saving the combined point cloud to a PLY file
    pcl::io::savePLYFile("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/debug/with_circle_center.ply", *noise_free_points);
    std::cout<<1;
    // Do something with circle_centers
    for (const auto &center : circle_centers) {
        std::cout << "Circle center: " << center.transpose() << std::endl;
    }
    if (!circle_centers.empty()) {
        std::vector<Eigen::Vector3f> sorted_centers = sortCircleCenters(circle_centers);
        auto [final_transform, roll, pitch, yaw, x, y, z] = optimize_rotation_transform(sorted_centers, pca_plane);
    std::cout << "Final Transform Matrix:\n" << final_transform << std::endl;
    std::cout << "Roll: " << roll << ", Pitch: " << pitch << ", Yaw: " << yaw << std::endl;
    std::cout << "X offset: " << x << ", Y offset: " << y << ", Z offset: " << z << std::endl;
    savePoseToFile(x, y, z, roll, pitch, yaw, "/home/rp/urop/test_ws/src/boundary_evaluation/src/ours_pose.txt");
    }
    //std::cout << "Noise-free point cloud saved to 'noise_free_points.ply'." << std::endl;


    return 0;
}

