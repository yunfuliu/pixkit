#include <ctime>
#include "../../include/pixkit-image.hpp"
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
	/*
	 * @param		src: input image
	 * @param		dst: output image
	 * @param		BlockSize: the height and width of block
	 *
	 */
bool pixkit::comp::IBTC_KQ2014(cv::Mat &src,cv::Mat &dst, int BlockSize){

	if(src.cols%BlockSize!=0 || src.rows%BlockSize!=0){
		printf("input size error\n");
		system("pause");
		exit(0);
	}

	Mat HSV;
	//Tool::RGBtransHSV(src,HSV);
	Mat tsrc;
	src.convertTo(tsrc,CV_32FC3);
	cv::cvtColor(tsrc,HSV,CV_BGR2HSV);

	//k-means set
	Mat Data(BlockSize*BlockSize,1,CV_32FC1), Cluster, Centers;
	int ClusterK=4;
	int Attempts=10;

	for(int i=0; i<src.rows; i+=BlockSize){
		for(int j=0; j<src.cols; j+=BlockSize){

			for(int channel=0; channel<3; channel++){
				
				for(int m=i; m<i+BlockSize; m++)
					for(int n=j; n<j+BlockSize; n++)
						Data.ptr<float>((m%BlockSize)*BlockSize+(n%BlockSize))[0]=HSV.at<Vec3f>(m,n)[channel];

				cv::kmeans(Data, ClusterK, Cluster, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.1), Attempts, KMEANS_PP_CENTERS, Centers);

				for(int m=i; m<i+BlockSize; m++){
					for(int n=j; n<j+BlockSize; n++){
						int ClusterIdx = Cluster.at<int>((m%BlockSize)*BlockSize+(n%BlockSize));
						HSV.at<Vec3f>(m,n)[channel]=Centers.ptr<float>(ClusterIdx)[0];
					}
				}
			}

		}
	}

	Mat tdst;
	cv::cvtColor(HSV,tdst,CV_HSV2BGR);
	cv::normalize(tdst,tdst,0,255,32);
	tdst.convertTo(tdst,CV_8UC3);
	dst=tdst.clone();
	return true;

}