#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//opencv headers
#include <iostream>
#include <cmath>
#include "..\..\include\pixkit-image.hpp"
///////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace cv;

//calculate mean value of each patch
void patchmean(const cv::Mat &src,cv::Mat &mean){	

	for(int i=1;i<src.rows-1;i++){
		for(int j=1;j<src.cols-1;j++){
			//patch size(3,3)
			int Np=0;//Np is the number of pixels in a patch
			for(int fr=-1;fr<2;fr++){
				for(int fc=-1;fc<2;fc++)
				{
					mean.ptr<double>(i)[j]+=src.ptr<uchar>(i+fr)[j+fc];
					Np++;
				}
			}
			mean.ptr<double>(i)[j]/=Np;
		}
	}
}

//funtion to calculate patchsigama
void patchsigma(const cv::Mat &src,const cv::Mat &mean,cv::Mat &sigama){

	cv::Mat tmp=cv::Mat::zeros(src.rows,src.cols,CV_64FC1);
	for(int i=1;i<src.rows-1;i++){
		for(int j=1;j<src.cols-1;j++){
			//Np is the number of pixels in each patch
			int Np=0;
			//patch size(3,3) 
			for(int fr=-1;fr<2;fr++){
				for(int fc=-1;fc<2;fc++){
					tmp.ptr<double>(i)[j]+=pow(src.ptr<uchar>(i+fc)[j+fr],2);
					Np++;
				}
			}
			tmp.ptr<double>(i)[j]/=Np;//9 is the pixel numbers of each patch
			sigama.ptr<double>(i)[j]=sqrt(tmp.ptr<double>(i)[j]-pow(mean.ptr<double>(i)[j],2));
		}
	}
}

//funtion to calculate  the covariance of X and Y
void covarianceXY(const cv::Mat &X,const cv::Mat &Y,const cv::Mat &meanX,const cv::Mat &meanY,cv::Mat &covariance){

	for(int i=1;i<covariance.rows-1;i++){
		for(int j=1;j<covariance.cols-1;j++){
			int Np=0;//Np is the number of pixel in a patch
			//patch size (3,3)
			for(int fr=-1;fr<2;fr++){
				for(int fc=-1;fc<2;fc++){
					covariance.ptr<double>(i)[j]+=(X.ptr<uchar>(i+fr)[j+fc]*Y.ptr<uchar>(i+fr)[j+fc]);
					Np++;
				}
			}
			covariance.ptr<double>(i)[j]=covariance.ptr<double>(i)[j]/Np-meanX.ptr<double>(i)[j]*meanY.ptr<double>(i)[j];
		}
	}
}

//function to calculate l Matrix
void calculate_l(const cv::Mat &meanX,const cv::Mat &meanY,cv::Mat &l){
	int L=255;
	double K1=0.01;
	double C1=pow(K1*L,2);
	
	for(int i=1;i<l.rows-1;i++){
		for(int j=1;j<l.cols-1;j++){
			l.ptr<double>(i)[j]=(2*meanX.ptr<double>(i)[j]*meanY.ptr<double>(i)[j]+C1)/(pow(meanX.ptr<double>(i)[j],2)+pow(meanY.ptr<double>(i)[j],2)+C1);
		}
	}
}

//function to calculate c Matrix
void calculate_c(const cv::Mat &sigmaX,const cv::Mat &sigmaY,cv::Mat &c){
	int L=255;
	double K2=0.03;
	double C2=pow(K2*L,2);
	for(int i=1;i<c.rows-1;i++){
		for(int j=1;j<c.cols-1;j++){
			c.ptr<double>(i)[j]=(2*sigmaX.ptr<double>(i)[j]*sigmaY.ptr<double>(i)[j]+C2)/(pow(sigmaX.ptr<double>(i)[j],2)+pow(sigmaY.ptr<double>(i)[j],2)+C2);
		}
	}
}

//function to calculate s Matrix
void calculate_s(const cv::Mat &sigmaX,const cv::Mat &sigmaY,const cv::Mat &covariance,cv::Mat &s){
	int L=255;
	double K2=0.03;
	double C2=pow(K2*L,2);
	double C3=C2/2.;

	for(int i=1;i<s.rows-1;i++){
		for(int j=1;j<s.cols-1;j++){
			s.ptr<double>(i)[j]=(covariance.ptr<double>(i)[j]+C3)/(sigmaX.ptr<double>(i)[j]*sigmaY.ptr<double>(i)[j]+C3);
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////
float *IW_SSIMj(const cv::Mat &X,const cv::Mat &Y,float *SSIMj){

	int scale_j=4;
	//build 5 scale of original signal X and distorted signal Y
	cv::vector<cv::Mat> ScaleX,ScaleY;
	cv::buildPyramid(X,ScaleX,scale_j);
	cv::buildPyramid(Y,ScaleY,scale_j);

	for(int jth=0;jth<scale_j+1;jth++){
		//create the mean of signal X and Y
		cv::Mat meanX = cv::Mat::zeros(ScaleX[jth].rows,ScaleX[jth].cols,CV_64FC1);
		cv::Mat meanY = cv::Mat::zeros(ScaleY[jth].rows,ScaleY[jth].cols,CV_64FC1);
		//create the sigama of signal X and Y
		cv::Mat sigmaX= cv::Mat::zeros(ScaleX[jth].rows,ScaleX[jth].cols,CV_64FC1);
		cv::Mat sigmaY= cv::Mat::zeros(ScaleY[jth].rows,ScaleY[jth].cols,CV_64FC1);
		//create the covariance of X and Y
		cv::Mat covariance= cv::Mat::zeros(ScaleX[jth].rows,ScaleX[jth].cols,CV_64FC1);//ScaleX.rows=DcaleY.rows=covarience.rows
		cv::Mat l= cv::Mat::zeros(ScaleX[jth].rows,ScaleX[jth].cols,CV_64FC1);
		cv::Mat c= cv::Mat::zeros(ScaleX[jth].rows,ScaleX[jth].cols,CV_64FC1);
		cv::Mat s= cv::Mat::zeros(ScaleX[jth].rows,ScaleX[jth].cols,CV_64FC1);
		//announce need Matrix to store mean,sigma and covariance///////////////////////////////////////////////////////////////////

		//calculate the  mean,sigma,and covariance of X and Y and the value of l(x,y),c(x,y) and s(x,y)
		patchmean(ScaleX[jth],meanX);
		patchmean(ScaleY[jth],meanY);
		patchsigma(ScaleX[jth],meanX,sigmaX);
		patchsigma(ScaleY[jth],meanY,sigmaY);
		covarianceXY(ScaleX[jth],ScaleY[jth],meanX,meanY,covariance);
		calculate_l(meanX,meanY,l);
		calculate_c(sigmaX,sigmaY,c);
		calculate_s(sigmaX,sigmaY,covariance,s);

		int Nj=0;
		//Sum up all l(Xi,Yi)*c(Xi,Yi)*s(Xi,Yi)
		for(int i=1;i<ScaleX[jth].rows-1;i++){
			for(int j=1;j<ScaleX[jth].cols-1;j++){
				SSIMj[jth]+=(l.ptr<double>(i)[j]*c.ptr<double>(i)[j]*s.ptr<double>(i)[j]);
				Nj++;
			}
		}
		SSIMj[jth]/=Nj;
	}
	return SSIMj;
}

float pixkit::qualityassessment::IW_SSIM(const cv::Mat &src1,const cv::Mat &src2){

	if(src1.type()!=CV_8UC1 || src2.type()!=CV_8UC1 ){CV_Assert(false);}
	//initialize the value of each scale
	float SSIMj[5]={0,0,0,0,0};
	//given constant of beta
	float betaj[5]={0.0448,0.2856,0.3001,0.2363,0.1333};
	// calculate the SSIM value of each scale
	IW_SSIMj(src1,src2,SSIMj);
	
	// get IW_SSIM
	float IW_SSIM=1.;
	for(int i=0;i<5;i++){
		IW_SSIM*= pow(SSIMj[i],betaj[i]);
	}
	return IW_SSIM;
}

