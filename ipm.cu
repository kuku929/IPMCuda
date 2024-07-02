#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include "cuda_runtime.h"
#include "rclcpp/rclcpp.hpp"
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include <math.h>

#include "cv_bridge/cv_bridge.h"
#define BLOCK_SIZE 16
#define imin(a,b) (a<b?a:b)

using namespace std::chrono_literals;
using namespace std;
using std::placeholders::_1;
const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);
template<typename T>
__global__ void dev_matmul(const T *a, const T *b, T *output, int rows){
	//a is 3x3 set of matrices
	//b is 3x1 set of matrices
	//output is 3x1 set of matrices
	int thread_id= threadIdx.x;
	int block_id = blockIdx.x;

	int offset = block_id*threadsPerBlock + thread_id;
	if(offset < rows){
		for(int i=0; i < 3; ++i){
			double temp=0;
			for(int k=0; k < 3; ++k){
				temp += a[offset*9 + i*3 + k]*b[offset*3 + k];
			}
			output[offset*3 + i] = temp;
		}
	}
}


void matmul(double *a, doubel *b, double *c){
	//a is 3x3
	//b is 3x1
	for(int i=0; i < 3; ++i){
		double temp=0;
		for(int k=0; k < 3; ++k){
			temp += a[i*3 + k]*b[k]; 
		}
		c[i] = temp;
		}
	}
}

__global__ void dot(double* a, double* b, double* c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	
	float temp = 0;
	while (tid < N){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	
	// set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	__syncthreads();
	
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

class IPM : public rclcpp::Node
{
  public:
    IPM()
    : Node("ipm")
    {
       subscription_caminfo = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera_forward/camera_info", 10, std::bind(&IPM::call, this, _1));
       subscription_img = this->create_subscription<sensor_msgs::msg::Image>("/camera_forward/image_raw", 10, std::bind(&IPM::process_img, this, _1));
       publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/igvc/ipm", 10);
    }

  private:
    void call(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        this->camera_info = msg;
    }
    void process_img(const sensor_msgs::msg::Image::SharedPtr msg)
    {
	//processing recieved image
        sensor_msgs::msg::PointCloud2 pub_pointcloud;
        PointCloud::Ptr cloud_msg (new PointCloud);
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);

        cv::Mat gray_image;
        cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_RGB2GRAY);

        cv::inRange(gray_image, cv::Scalar(125), cv::Scalar(140), gray_image); 
        cv::Mat nonZeroCoordinates;
        cv::findNonZero(gray_image, nonZeroCoordinates);


	//some calculations
	float roll = 0;
	float pitch = -17 * M_PI / 180;
	float yaw = 0;
	float h = 1.18;
	int m = 3;
	int n = 3;
	double *k, *nor, *uv;
	cudaMallocHost((void **) &nor, sizeof(double)*3);
	cudaMallocHost((void **) &uv, sizeof(double)*3);
	cudaMallocHost((void **) &k, sizeof(double)*n*m);
	double cy, cr, sy, sr, sp, cp;
	cy = cos(yaw);
	sy = sin(yaw);
	cp = cos(pitch);
	sp = sin(pitch);
	cr = cos(roll);
	sr = sin(roll);
	k[0] = cr*cy+sp*sr+sy;
	k[1] = cr*sp*sy-cy*sr;
	k[2] = -cp*sy;
	k[3] = cp*sr;
	k[4] = cp*cr;
	k[5] = sp;
	k[6] = cr*sy-cy*sp*sr;
	k[7] = -cr*cy*sp -sr*sy;
	k[8] = cp*cy;

	nor[0] = 0;
	nor[1] = 1.0;
	nor[2] = 0;

	//what does this do?
	matmul(k, nor, uv);

	//no of points to map
	cv::Size s = nonZeroCoordinates.size();
	int rows = s.height;

	double *uv_hom, *kin_uv, *denom;
	auto caminfo = this->camera_info->k;
	cudaMallocHost((void **) &uv_hom, sizeof(double)*m*1*rows);
	cudaMallocHost((void **) &kin_uv, sizeof(double)*m*1*rows);
	cudaMallocHost((void **) &denom, sizeof(double)*rows);

	//device
	double *d_uv_hom, *d_kin_uv, *d_caminfo, *d_denom, *d_uv;
	cudaMalloc((void **) &d_uv_hom, sizeof(double)*m*1*rows);
	cudaMalloc((void **) &d_kin_uv, sizeof(double)*m*1*rows);
	cudaMalloc((void **) &d_caminfo, sizeof(double)*m*n*rows);
	cudaMalloc((void **) &d_denom, sizeof(double)*rows);
	cudaMalloc((void **) &d_uv, sizeof(double)*m*1*rows);
 
	//gathering data for all points
	 for (int i = 0; i < rows; i++)
	 {
	     int x = nonZeroCoordinates.at<cv::Point>(i).x;
	     int y = nonZeroCoordinates.at<cv::Point>(i).y;
	     uv_hom[i] = x;
	     uv_hom[i+1] = y; 
	     uv_hom[i+2] = 1;
	 }
	 
	//copying to device
	cudaMemcpy(d_caminfo, caminfo, sizeof(double)*m*n*rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uv_hom, uv_hom, sizeof(double)*m*1*rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uv, uv, sizeof(double)*m*1*rows, cudaMemcpyHostToDevice);

	//batch multiplication
	//launching rows no of threads and one block
	dev_matmul<<<2, (rows+1)/2>>(d_caminfo, d_uv_hom, d_kin_uv, rows);
	dot<<<2, (rows+1)/2>>>(d_uv, d_kin_uv, d_denom);
	
	cudaMemcpy(kin_uv, d_kin_uv, sizeof(double)*m*rows, cudaMemcpyDeviceToHost);
	cudaMemcpy(denom, d_denom, sizeof(double)*rows, cudaMemcpyDeviceToHost);

	double h = 1.18;
	for(int i=0; i < rows; ++i){
		pcl::PointXYZ vec;
		vec.x = h * kin_uv[0] / denom;
		vec.y =  h * kin_uv[1] / denom;
		vec.z =  h * kin_uv[2] / denom;
		cloud_msg->points.push_back(vec);
	}

	 cudaFree(d_uv_hom);
	 cudaFree(d_uv_hom);
	 cudaFree(d_kin_uv);
	 cudaFree(d_caminfo);
	 cudaFree(d_denom);
	 cudaFreeHost(uv_hom);
	 cudaFreeHost(kin_uv);
	 cudaFreeHost(caminfo);    
	 cloud_msg->height   = 1;
	 cloud_msg->width    = cloud_msg->points.size();
	 cloud_msg->is_dense = false;
	 pcl::toROSMsg(*cloud_msg, pub_pointcloud);
	 pub_pointcloud.header.frame_id = "base_link";
	 pub_pointcloud.header.stamp = rclcpp::Clock().now();

	 // Publishing our cloud image
	 publisher_->publish(pub_pointcloud);

	 cloud_msg->points.clear();
	 cudaFreeHost(k);
	 cudaFreeHost(nor);
	 cudaFreeHost(uv);
    }
    
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr subscription_caminfo;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_img;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IPM>());
  rclcpp::shutdown();
  return 0;
}
