#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "..\include\pixkit-image.hpp"


using namespace std;
using namespace cv;


double calculate_MSE(const cv::Mat X,const cv::Mat Y){
	double MSE=0;
	int Nj=0;//Np is the number of pixels in X	
	for(int i=0;i<X.rows;i++){
		for(int j=0;j<X.cols;j++){
			Nj++;
			MSE+=pow((X.ptr<uchar>(i)[j]-Y.ptr<uchar>(i)[j]),2);
		}
	}
	MSE/=Nj;
	return MSE;
}
//caculate IW_PSNR value



//caculate eace value of IW_PSNR in jth scale
double * IW_MSEj(const cv::Mat X,const cv::Mat Y,double * MSEj){
	int scale_j=4;
	double MSE;
	//build 5 scale of original 
	vector<Mat> ScaleX,ScaleY;
	buildPyramid(X,ScaleX,scale_j);
	buildPyramid(Y,ScaleY,scale_j);


	for(int jth=0;jth<5;jth++){
		MSE=calculate_MSE(ScaleX[jth],ScaleY[jth]);
		MSEj[jth]=MSE;	
	}

	return MSEj;
}

double pixkit::qualityassessment::IW_PSNR(const cv::Mat src1,const cv::Mat src2){
	if(src1.type()!=CV_8UC1 || src2.type()!=CV_8UC1 ){CV_Assert(false);}
	
	int L=255;
	double MSEj[5]={0,0,0,0,0};
	double IW_MSE=1;//initialize the value of IW_MSE
	double IW_PSNR=1;//initialize the value of IW_PSNR
	double betaj[5]={0.0448,0.2856,0.3001,0.2363,0.1333};

	IW_MSEj(src1,src2,MSEj);
	for (int j = 0; j < 5; j++)
	{
		IW_MSE*=pow(MSEj[j],betaj[j]);
	}
	//cout<<"IW_MSE value= "<<IW_MSE<<endl;  test the value of IW-MSE
	IW_PSNR=10*log10(L*L/IW_MSE);

	return IW_PSNR;
}