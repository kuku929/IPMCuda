#include <functional>
#include <math.h>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "cv_bridge/cv_bridge.h"
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

using namespace std;
using namespace Eigen;
using std::placeholders::_1;

class IPM : public rclcpp::Node
{
  public:
    IPM()
    : Node("ipm")
    {
       this->set_parameter(rclcpp::Parameter("use_sim_time", true));
       left_caminfo_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>(
			   "/camera1/camera_info", 10, std::bind(&IPM::left_info_callback, this, _1));

       left_img_subscription = this->create_subscription<sensor_msgs::msg::Image>(
			   "/camera1/image_raw", 10, std::bind(&IPM::left_img_callback, this, _1));

       right_caminfo_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>(
			   "/short_1_camera/camera_info", 10, std::bind(&IPM::right_info_callback, this, _1));

       right_img_subscription = this->create_subscription<sensor_msgs::msg::Image>(
			   "/short_1_camera/image_raw", 10, std::bind(&IPM::right_img_callback, this, _1));

       publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/igvc/ipm", 10);
    }

  private:
    void right_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        this->right_camera_info= *msg;
    }
    void left_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        this->left_camera_info= *msg;
    }
    void right_img_callback(const sensor_msgs::msg::Image::SharedPtr msg){
		process_img(msg, "camera_short_link", this->right_camera_info);
    }

    void left_img_callback(const sensor_msgs::msg::Image::SharedPtr msg){
		process_img(msg, "camera_link", this->left_camera_info);
    }
    void process_img(const sensor_msgs::msg::Image::SharedPtr msg, std::string &&frame, sensor_msgs::msg::CameraInfo &camera_info)
    {
		// processing recieved image
		sensor_msgs::msg::PointCloud2 pub_pointcloud;
		unique_ptr<PointCloud> cloud_msg  = std::make_unique<PointCloud>();
		cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
		cv::Mat gray_image;
		cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_RGB2GRAY);
		cv::inRange(gray_image, cv::Scalar(245), cv::Scalar(255), gray_image);
		cv::Mat nonZeroCoordinates;
		cv::findNonZero(gray_image, nonZeroCoordinates);

		/*
		 * ipm code: refer to https://thomasfermi.github.io/Algorithms-for-Automated-Driving/LaneDetection/InversePerspectiveMapping.html
		 */

		// camera parameters
		float roll = 0;
		float pitch = 0;
		float yaw = 0;
		float h = 1.02f;

		// k is the rotation matrix, Rcr( see link ).
		// describes how camera is oriented wrt to road frame.
		Eigen::Matrix<double, 3, 3> k;
		double cy, cr, sy, sr, sp, cp;
		cy = cos(yaw);
		sy = sin(yaw);
		cp = cos(pitch);
		sp = sin(pitch);
		cr = cos(roll);
		sr = sin(roll);
		k(0,0) = cr*cy+sp*sr+sy;
		k(0,1) = cr*sp*sy-cy*sr;
		k(0,2) = -cp*sy;
		k(1,0) = cp*sr;
		k(1,1) = cp*cr;
		k(1,2) = sp;
		k(2,0) = cr*sy-cy*sp*sr;
		k(2,1) = -cr*cy*sp -sr*sy;
		k(2,2) = cp*cy;

		// normal vector to road
		Eigen::Matrix<double, 3, 1> nor;
		nor(0,0) = 0;
		nor(1,0) = 1.0;
		nor(2,0) = 0;

		// transformed normal vector( ncT )
		Eigen::Matrix<double, 1, 3> nT = (k*nor).transpose(); //will this work?

		// no of points to map
		cv::Size s = nonZeroCoordinates.size();
		unsigned int cols = s.height;
		std::cout << "cols: " << cols << '\n';
		assert(cols > 0);

		// K matrix is camera's intrinsic parameters 
		auto caminfo = camera_info.k;
		Eigen::Map<Matrix<double,3,3,RowMajor> > camera_params(caminfo.data());
		auto inv_caminfo = camera_params.inverse();

		// uv_hom contains coordinates in the image
		// why is this named _hom?
		Eigen::Matrix<double, 3, Dynamic> uv_hom(3, cols);
		for(int i=0;i < cols; ++i){
			int x = nonZeroCoordinates.at<cv::Point>(i).x;
			int y = nonZeroCoordinates.at<cv::Point>(i).y;
			uv_hom(0, i) = x;
			uv_hom(1, i) = y;
			uv_hom(2, i) = 1;
		}

		//kin_uv  = K^(-1) * uv
		Eigen::Matrix<double, 3, Dynamic> kin_uv(3, cols);
		kin_uv = inv_caminfo * uv_hom;

		// denominator to divide by when mapping points 
		Eigen::Matrix<double, 1, Dynamic> denom(1, cols);
		denom = nT * kin_uv;

		for(int i=0; i < cols; ++i){
			pcl::PointXYZ vec;
			// vec is points in road frame
			vec.x = h * kin_uv(2, i) / denom(0,i);
			// NOTE: idk if this is supposed to -h
			// but it works
			vec.y =  -h * kin_uv(0, i) / denom(0,i);
			vec.z =  -h * kin_uv(1, i) / denom(0,i);
			cloud_msg->points.push_back(vec);
		}

		cloud_msg->height   = 1;
		cloud_msg->width    = cloud_msg->points.size();
		cloud_msg->is_dense = false;
		pcl::toROSMsg(*cloud_msg, pub_pointcloud);
		pub_pointcloud.header.frame_id = frame;
		pub_pointcloud.header.stamp = this->now();
		publisher_->publish(pub_pointcloud);

		cloud_msg->points.clear();
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_caminfo_subscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_img_subscription;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_caminfo_subscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_img_subscription;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    sensor_msgs::msg::CameraInfo right_camera_info;
    sensor_msgs::msg::CameraInfo left_camera_info;
};

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<IPM>());
	rclcpp::shutdown();
	return 0;
}
