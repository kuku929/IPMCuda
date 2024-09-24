#include <math.h>
#include <functional>
#include <memory>
#include <string>
#include "cuda_runtime.h"
#include "rclcpp/rclcpp.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "std_msgs/msg/header.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "cv_bridge/cv_bridge.h"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
#define BLOCKS 64
using namespace std;
using namespace Eigen;
using std::placeholders::_1;
template <typename T>
__global__ void dev_matmul(const T *a, const T *b, T *output, int rows)
{
    // a is 3x3 matrix
    // b is 3x1 set of matrices
    // output is 3x1 set of matrices
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    int offset = block_id * (rows + BLOCKS - 1) / BLOCKS + thread_id;

    if (offset < rows)
    {
#pragma unroll 3
        for (int i = 0; i < 3; ++i)
        {
            double temp = 0;
#pragma unroll 3
            for (int k = 0; k < 3; ++k)
            {
                temp += a[i * 3 + k] * b[offset * 3 + k];
            }
            output[offset * 3 + i] = temp;
        }
    }
}

void matmul(double *a, double *b, double *c)
{
// a is 3x3
// b is 3x1
#pragma unroll 3
    for (int i = 0; i < 3; ++i)
    {
        double temp = 0;
#pragma unroll 3
        for (int k = 0; k < 3; ++k)
        {
            temp += a[i * 3 + k] * b[k];
        }
        c[i] = temp;
    }
}

__global__ void dot(double *a, double *b, double *c, int rows)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;

    int offset = block_id * (rows + BLOCKS - 1) / BLOCKS + thread_id;

    if (offset < rows)
    {
        double temp = 0;
#pragma unroll 3
        for (int i = 0; i < 3; ++i)
        {
            temp += a[i] * b[offset * 3 + i];
        }
        c[offset] = temp;
    }
}

void log(cudaError_t &&error, int line = 0)
{
    cout << cudaGetErrorString(error) << "line : " << line << '\n' << flush;
}

#define log(x) log(x, __LINE__)

class IPM : public rclcpp::Node
{
  public:
    IPM() : Node("ipm")
    {
	   this->set_parameter(rclcpp::Parameter("use_sim_time", true));
       left_caminfo_subscription = this->create_subscription<sensor_msgs::msg::CameraInfo>(
			   "/camera1/camera_info", 10, std::bind(&IPM::left_info_callback, this, _1));
        // NOTE: this is important for simulation, else rviz cups
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        left_caminfo_subscription =
            this->create_subscription<sensor_msgs::msg::CameraInfo>(
                "/camera1/camera_info", 10,
                std::bind(&IPM::left_info_callback, this, _1));

        left_img_subscription =
            this->create_subscription<sensor_msgs::msg::Image>(
                "/model_lanes", 10,
                std::bind(&IPM::left_img_callback, this, _1));
		

        right_caminfo_subscription =
            this->create_subscription<sensor_msgs::msg::CameraInfo>(
                "/short_1_camera/camera_info", 10,
                std::bind(&IPM::right_info_callback, this, _1));

        right_img_subscription =
            this->create_subscription<sensor_msgs::msg::Image>(
                "/model_lanes2", 10,
                std::bind(&IPM::right_img_callback, this, _1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/igvc/ipm", 10);
    }

  private:
    void right_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        this->right_camera_info = *msg;
    }
    void left_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        this->left_camera_info = *msg;
    }
    void right_img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
	    process_img(msg, "camera_short_link", right_camera_info);
    }
    void left_img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
	    process_img(msg, "camera_link", left_camera_info);
    }
    void process_img(const sensor_msgs::msg::Image::SharedPtr msg,
		    std::string &&frame,
		    sensor_msgs::msg::CameraInfo &camera_info)
    {
	// processing recieved image
	sensor_msgs::msg::PointCloud2 pub_pointcloud;
	unique_ptr<PointCloud> cloud_msg = std::make_unique<PointCloud>();
	cv_bridge::CvImagePtr cv_ptr =
	    cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
	cv::Mat gray_image;
	cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_RGB2GRAY);
	cv::inRange(gray_image, cv::Scalar(250), cv::Scalar(255), gray_image);
	cv::Mat nonZeroCoordinates;
	cv::findNonZero(gray_image, nonZeroCoordinates);
	// these are the camera parameters,
	// they are HARD CODED!
	// NOTE: rpy is not set correctly
	float roll =0.0f;
	float pitch =0.0f;
	float yaw =0.785f;
	float h =1.02f;

	vector<double> k(9), nor(3), uv(3);
	//processing recieved image

	double cy, cr, sy, sr, sp, cp;
	cy = cos(yaw);
	sy = sin(yaw);
	cp = cos(pitch);
	sp = sin(pitch);
	cr = cos(roll);
	sr = sin(roll);
	k[0] = cr * cy + sp * sr + sy;
	k[1] = cr * sp * sy - cy * sr;
	k[2] = -cp * sy;
	k[3] = cp * sr;
	k[4] = cp * cr;
	k[5] = sp;
	k[6] = cr * sy - cy * sp * sr;
	k[7] = -cr * cy * sp - sr * sy;
	k[8] = cp * cy;

	nor[0] = 0;
	nor[1] = 1.0;
	nor[2] = 0;

	matmul(k.data(), nor.data(), uv.data());
	// no of points to map
	cv::Size s = nonZeroCoordinates.size();
	int rows = s.height;
	std::cout << "rows : " << rows << '\n';
	auto caminfo = camera_info.k;
	Eigen::Map<Matrix<double, 3, 3, RowMajor>> mat(caminfo.data());
	mat = mat.inverse();
	double *inv_caminfo = mat.data();
	vector<double> kin_uv(3 * rows), uv_hom(3 * rows), denom(rows);

	for (int i = 0; i < rows; ++i)
	{
	    int x = nonZeroCoordinates.at<cv::Point>(i).x;
	    int y = nonZeroCoordinates.at<cv::Point>(i).y;
	    uv_hom[i * 3] = x;
	    uv_hom[i * 3 + 1] = y;
	    uv_hom[i * 3 + 2] = 1;
	}

        // device
        double *d_uv_hom, *d_kin_uv, *d_caminfo, *d_denom, *d_uv;
        log(cudaMalloc((void **)&d_uv_hom, sizeof(double) * 3 * rows));
        log(cudaMalloc((void **)&d_kin_uv, sizeof(double) * 3 * rows));
        log(cudaMalloc((void **)&d_caminfo, sizeof(double) * 9));
        log(cudaMalloc((void **)&d_denom, sizeof(double) * rows));
        log(cudaMalloc((void **)&d_uv, sizeof(double) * 3));

        // copying to device
        log(cudaMemcpy(d_caminfo, inv_caminfo, sizeof(double) * 9,
                   cudaMemcpyHostToDevice));
        log(cudaMemcpy(d_uv_hom, uv_hom.data(), sizeof(double) * 3 * rows,
                   cudaMemcpyHostToDevice));
        log(cudaMemcpy(d_uv, uv.data(), sizeof(double) * 3, cudaMemcpyHostToDevice));
        // batch multiplication
        dim3 dim_grid(BLOCKS, 1);
        dim3 dim_block((rows + BLOCKS - 1) / BLOCKS, 1);
        void *args[4] = {&d_caminfo, &d_uv_hom, &d_kin_uv, &rows};
        log(cudaLaunchKernel((void *)dev_matmul<float>, dim_grid, dim_block, args, 0, nullptr));
        // dev_matmul<<<dim_grid, dim_block>>>(d_caminfo, d_uv_hom, d_kin_uv,
                                            // rows);
        cudaDeviceSynchronize();
        void *args_dot[4] = {&d_uv, &d_kin_uv, &d_denom, &rows};
        log(cudaLaunchKernel((void *)dot, BLOCKS, (rows+BLOCKS-1)/BLOCKS, args_dot, 0, nullptr));
        // dot<<<BLOCKS, (rows + BLOCKS - 1) / BLOCKS>>>(d_uv, d_kin_uv, d_denom,
                                                    //   rows);
        cudaDeviceSynchronize();

        log(cudaMemcpy(kin_uv.data(), d_kin_uv, sizeof(double) * 3 * rows,
                   cudaMemcpyDeviceToHost));
        log(cudaMemcpy(denom.data(), d_denom, sizeof(double) * rows,
                   cudaMemcpyDeviceToHost));

        for (int i = 0; i < rows; ++i)
        {
            pcl::PointXYZ vec;
            vec.x = h * kin_uv[i * 3 + 2] / denom[i];
            vec.y = -h * kin_uv[i * 3] / denom[i];
            vec.z = -h * kin_uv[i * 3 + 1] / denom[i];
            cloud_msg->points.push_back(vec);
        }

        cudaFree(d_uv_hom);
        cudaFree(d_uv);
        cudaFree(d_kin_uv);
        cudaFree(d_caminfo);
        cudaFree(d_denom);
        cloud_msg->height = 1;
        cloud_msg->width = cloud_msg->points.size();
        cloud_msg->is_dense = false;
        pcl::toROSMsg(*cloud_msg, pub_pointcloud);
        pub_pointcloud.header.frame_id = frame;
        // use the internal clock, will use gazebo's clock
        // when use_sim_time set true
        pub_pointcloud.header.stamp = this->get_clock()->now();
	
        // Publishing our cloud image
        publisher_->publish(pub_pointcloud);
		
        cloud_msg->points.clear();
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
        right_caminfo_subscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
        right_img_subscription;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
        left_caminfo_subscription;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
        left_img_subscription;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    sensor_msgs::msg::CameraInfo right_camera_info;
    sensor_msgs::msg::CameraInfo left_camera_info;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<IPM>());
    rclcpp::shutdown();
    return 0;
}
