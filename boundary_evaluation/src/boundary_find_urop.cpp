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
                                                float x_min = 0, float x_max = 2,
                                                float y_min = -1, float y_max = 1,
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

// Function to perform Euclidean Cluster Extraction
std::vector<pcl::PointIndices> clusterPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
                                             float eps = 0.02, int min_samples = 10) {
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
            // Z축 최대/최소 값 가진 포인트 찾기
            pcl::PointXYZ max_z_point = cluster->points[0];
            pcl::PointXYZ min_z_point = cluster->points[0];
            
            for (const auto &point : cluster->points) {
                if (point.z > max_z_point.z) max_z_point = point;
                if (point.z < min_z_point.z) min_z_point = point;
            }

            // Z축 최대/최소 값 포인트 간 거리 계산
            float distance = std::sqrt(std::pow(max_z_point.x - min_z_point.x, 2) +
                                       std::pow(max_z_point.y - min_z_point.y, 2) +
                                       std::pow(max_z_point.z - min_z_point.z, 2));
                                       
            if (distance < 0.6) { // 포인트 간 거리가 0.6 미만인 경우
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
                                                               double radius = 0.0135, double min_distance = 0.003) {
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

            if (cdr <= 0.55)
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

            if (direction_variance.sum() < 0.7) {
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

// 포인트 클라우드를 2D 평면에 투영하는 함수
pcl::PointCloud<pcl::PointXYZ>::Ptr projectTo2DPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &noise_free_points, 
                                                     const Eigen::Vector4f &pca_plane) {
    // 8번의 결과에서 평균점 계산
    Eigen::Vector3f mean_point = Eigen::Vector3f::Zero();
    for (const auto& point : noise_free_points->points) {
        mean_point += point.getVector3fMap();
    }
    mean_point /= noise_free_points->points.size();

    // PCA를 사용해 첫 번째 주성분(첫 번째 축)과 두 번째 주성분(두 번째 축)을 계산
    Eigen::Vector3f first_principal_component = pcaFindComponent(noise_free_points);
    Eigen::Vector3f second_principal_component = first_principal_component.cross(Eigen::Vector3f(pca_plane[0], pca_plane[1], pca_plane[2]));

    // 모든 포인트를 2D 평면으로 투영
    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_2d_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : noise_free_points->points) {
        // 각 포인트에서 평균점을 빼서 상대 좌표로 변환
        Eigen::Vector3f relative_point = point.getVector3fMap() - mean_point;

        // 새로운 좌표계를 사용하여 2D 투영 (Z 좌표는 0)
        float x_2d = relative_point.dot(first_principal_component);  // x축 방향 성분
        float y_2d = relative_point.dot(second_principal_component); // y축 방향 성분

        // 투영된 2D 포인트 클라우드에 추가
        projected_2d_cloud->points.emplace_back(x_2d, y_2d, 0.0f);
    }
    return projected_2d_cloud;
}


int main(int argc, char **argv) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/accumulated_pointcloud_3.ply", *cloud) == -1) {
        PCL_ERROR("Couldn't read the PLY file \n");
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = removeWalls(cloud);
    std::vector<pcl::PointIndices> cluster_indices = clusterPoints(filtered_cloud);
    cluster_indices = removeSmallClusters(cluster_indices, 50); // 필요에 따라 조정
    pcl::PointCloud<pcl::PointXYZ>::Ptr most_planar_cluster = findMostPlanarCluster(filtered_cloud, cluster_indices);

    if (most_planar_cluster->empty()) {
        std::cerr << "No planar cluster found." << std::endl;
        return -1;
    }
    
    auto [plane_model, inliers] = detectPlanes(most_planar_cluster);
    
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(most_planar_cluster);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*inlier_cloud);

    Eigen::Vector4f pca_plane = pcaPlaneFitting(inlier_cloud);
    std::cout << "Plane equation: " << pca_plane[0] << "x + " << pca_plane[1] << "y + " << pca_plane[2] << "z + " << pca_plane[3] << " = 0" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud = projectPointsToPlane(inlier_cloud, pca_plane);

    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_pcd = findBoundaryPointsImproved(projected_cloud);
    
    Eigen::Vector3f first_principal_component = pcaFindComponent(projected_cloud);
    Eigen::Vector3f second_principal_component = first_principal_component.cross(Eigen::Vector3f(pca_plane[0], pca_plane[1], pca_plane[2]));
    pcl::PointCloud<pcl::PointXYZ>::Ptr noise_free_points = findAndRemoveLines(boundary_pcd, first_principal_component, second_principal_component, projected_cloud);

    // 2D 평면으로 투영 및 결과 저장
    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_2d_cloud = projectTo2DPlane(noise_free_points, pca_plane);
    pcl::io::savePLYFile("/home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/boundary.ply", *projected_2d_cloud);
    std::cout << "file saved to: /home/rp/urop/test_ws/src/boundary_evaluation/src/boundary_gt/boundary.ply" << std::endl;
    return 0;

}

