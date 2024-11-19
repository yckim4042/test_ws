#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <cmath>  // To calculate sqrt for distance
#include <cmath> // for sqrt

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
                                                float x_min = 0, float x_max = 2,
                                                float y_min = -0.7, float y_max = 1,
                                                float z_min = 0, float z_max = 1) {
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


// Callback function to process the incoming PointCloud2 data
void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& input) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*input, *cloud);  // Convert ROS PointCloud2 to PCL PointCloud

    
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
    seg.setAxis(Eigen::Vector3f(-0.698514, 0.709046, -0.0966029));  // Set the axis constraint
    seg.setEpsAngle(0.20);  // Set the angle tolerance (in radians)

    seg.setInputCloud(distance_filtered);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        ROS_WARN("No plane found in the point cloud.");
        return;
    }

    // Extract the inlier points (points belonging to the plane)
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(distance_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane_cloud);

    // 3. Save plane points to a PLY file
    if (save_plane_to_ply) {
        savePlaneToPLY(plane_cloud, "plane_points.ply");
    }

    // 4. Visualize both filtered points (in white) and plane points (in red)
    visualizePointClouds(distance_filtered, plane_cloud);

    // Publish the filtered point cloud (optional)
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*distance_filtered, output);
    output.header = input->header;
    filtered_pub.publish(output);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "ouster_plane_extraction");
    ros::NodeHandle nh;

    // Subscriber for Ouster LiDAR point cloud topic
    ros::Subscriber sub = nh.subscribe("/ouster/points", 1, pointCloudCallback);

    // Publisher for filtered point cloud
    filtered_pub = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);

    ros::spin();
    return 0;
}

