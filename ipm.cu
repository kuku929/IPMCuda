#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <Eigen/Dense>
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
#define BLOCKS 64
#define imin(a,b) (a<b?a:b)

using namespace std::chrono_literals;
using namespace std;
using namespace Eigen;
using std::placeholders::_1;
template<typename T>
__global__ void dev_matmul(const T *a, const T *b, T *output, int rows){
	//a is 3x3 matrix
	//b is 3x1 set of matrices
	//output is 3x1 set of matrices
	int thread_id= threadIdx.x;
	int block_id = blockIdx.x;
	int offset = block_id*(rows+BLOCKS-1)/BLOCKS + thread_id;

	if(offset < rows){
 		#pragma unroll
		for(int i=0; i < 3; ++i){
			double temp=0;
			for(int k=0; k < 3; ++k){
				temp += a[i*3+k]*b[offset*3 + k];
			}
			output[offset*3 + i] = temp;
		}
	}
}

void matmul(double *a, double *b, double *c){
	//a is 3x3
	//b is 3x1
	#pragma unroll
	for(int i=0; i < 3; ++i){
		double temp=0;
		for(int k=0; k < 3; ++k){
			temp += a[i*3 + k]*b[k]; 
		}
		c[i] = temp;
	}
}

__global__ void dot(double* a, double* b, double* c, int rows) {
	int thread_id = threadIdx.x;
	int block_id = blockIdx.x;

	int offset = block_id*(rows+BLOCKS-1)/BLOCKS + thread_id;
	if(offset < rows){
		double temp=0;
 		#pragma unroll
		for(int i=0; i < 3; ++i){
		    temp += a[i]*b[offset*3+i];    
		}
		c[offset] = temp;
	}
	//if(offset == 0){
		//for(int i=0;i < 3; ++i){
			//printf("gpu : %f ", c[0]);
		//}
		//printf("\n");
	//}
}


void log(cudaError_t &&error, int line=0){
	//std::cout << cudaGetErrorString(error) << "line : " << line << '\n' << std::flush;
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
        this->camera_info = *msg;
    }
    void process_img(const sensor_msgs::msg::Image::SharedPtr msg)
    {
	//processing recieved image
	sensor_msgs::msg::PointCloud2 pub_pointcloud;
	unique_ptr<PointCloud> cloud_msg  = std::make_unique<PointCloud>();
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);

	cv::Mat gray_image;
	cv::cvtColor(cv_ptr->image, gray_image, cv::COLOR_RGB2GRAY);

	cv::inRange(gray_image, cv::Scalar(245), cv::Scalar(255), gray_image); 
	cv::Mat nonZeroCoordinates;
	cv::findNonZero(gray_image, nonZeroCoordinates);


	//some calculations
	float roll = 0;
	float pitch = 0;//-17 * M_PI / 180;
	float yaw = 0;
	float h = 0.8;
	int m = 3;
	int n = 3;
	vector<double> k(9), nor(3), uv(3);

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
	matmul(k.data(), nor.data(), uv.data());

	// no of points to map
	cv::Size s = nonZeroCoordinates.size();
	int rows = s.height;
	std::cout << "rows : " << rows << '\n';
	auto caminfo = this->camera_info.k;
	Eigen::Map<Matrix<double,3,3,RowMajor> > mat(caminfo.data());
	mat = mat.inverse();
	//std::cout << mat;
	double *inv_caminfo = mat.data();

	//for(int i=0;i < 9; ++i){
		//std::cout << inv_caminfo[i] << ' ';
	//}
	//std::cout << '\n';
	//std::cout << mat;
	vector<double> kin_uv(3*rows), uv_hom(3*rows), denom(rows);


	//device
	double *d_uv_hom, *d_kin_uv, *d_caminfo, *d_denom, *d_uv;
	log(cudaMalloc((void **) &d_uv_hom, sizeof(double)*3*rows));
	log(cudaMalloc((void **) &d_kin_uv, sizeof(double)*3*rows));
	log(cudaMalloc((void **) &d_caminfo, sizeof(double)*9));
	log(cudaMalloc((void **) &d_denom, sizeof(double)*rows));
	log(cudaMalloc((void **) &d_uv, sizeof(double)*3));
 
	 
	//copying to device
	log(cudaMemcpy(d_caminfo, inv_caminfo, sizeof(double)*9, cudaMemcpyHostToDevice));
	cudaMemcpy(d_uv_hom, uv_hom.data(), sizeof(double)*3*rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uv, uv.data(), sizeof(double)*3, cudaMemcpyHostToDevice);

	//batch multiplication
	//launching rows no of threads and one block
	//std::cout << "cpu : " << caminfo[8] << '\n';
	dev_matmul<<<BLOCKS, (rows+BLOCKS-1)/BLOCKS>>>(d_caminfo, d_uv_hom, d_kin_uv, rows);
	dot<<<BLOCKS, (rows+BLOCKS-1)/BLOCKS>>>(d_uv, d_kin_uv, d_denom, rows);
	
	cudaMemcpy(kin_uv.data(), d_kin_uv, sizeof(double)*3*rows, cudaMemcpyDeviceToHost);
	cudaMemcpy(denom.data(), d_denom, sizeof(double)*rows, cudaMemcpyDeviceToHost);
	

	for(int i=0; i < rows; ++i){
		pcl::PointXYZ vec;
		//fix, make it work im not doing it
		vec.x = h * kin_uv[i*3+2] / denom[i];
		vec.y =  -h * kin_uv[i*3] / denom[i];
		vec.z =  0;//h * kin_uv[i*3+1] / denom[i];
		//std::cout << "value : " << denom[i] << ' ';
		cloud_msg->points.push_back(vec);
	}
	//std::cout << std::endl;

	 cudaFree(d_uv_hom);
	 cudaFree(d_uv);
	 cudaFree(d_kin_uv);
	 cudaFree(d_caminfo);
	 cudaFree(d_denom);   
	 cloud_msg->height   = 1;
	 cloud_msg->width    = cloud_msg->points.size();
	 cloud_msg->is_dense = false;
	 pcl::toROSMsg(*cloud_msg, pub_pointcloud);
	 pub_pointcloud.header.frame_id = "camera_forward_frame";
	 pub_pointcloud.header.stamp = rclcpp::Clock().now();

	 // Publishing our cloud image
	 publisher_->publish(pub_pointcloud);

	 cloud_msg->points.clear();
    }
    
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr subscription_caminfo;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_img;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    sensor_msgs::msg::CameraInfo camera_info;
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IPM>());
  rclcpp::shutdown();
  return 0;
}
