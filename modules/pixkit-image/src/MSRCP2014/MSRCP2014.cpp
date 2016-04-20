/*
MSRCP
*/
# include <stdlib.h>   
# include <stdio.h>   
# include <math.h>   
# include <string.h> 
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv\cv.h>
#include "../../include/pixkit-image.hpp"


bool SimplestColorBalance(cv::Mat ori,float upperthresh,float lowerthresh){

	int totalarea=ori.rows*ori.cols;
	upperthresh=upperthresh*totalarea;
	lowerthresh=lowerthresh*totalarea;
	cv::Mat sorted_ori;
	cv::Mat reshapeOri;
	reshapeOri=ori.reshape(0,1);
	cv::sort(reshapeOri,sorted_ori,CV_SORT_ASCENDING  );

	int Vmin=(sorted_ori.at<float>(lowerthresh));
	int Vmax=sorted_ori.at<float>((ori.rows*ori.cols-1)-upperthresh);
	for (int i=0; i<ori.rows; i++ ){
		for (int j=0; j<ori.cols; j++ ){
			if(ori.at<float>(i,j)<Vmin)ori.at<float>(i,j)=0;
			else if(ori.at<float>(i,j)>Vmax)ori.at<float>(i,j)=255;
			else ori.at<float>(i,j)= (ori.at<float>(i,j)-Vmin)*255./(Vmax-Vmin);

		}
	}

	return 1;

}
bool pixkit::enhancement::local::MSRCP2014(const cv::Mat &src,cv::Mat &dst){
	
	///// exceptions
	if(src.type()!=CV_8UC3){
		CV_Assert(false);
	}
	
	int Nscale=3;
	int RetinexScales[3]={7,81,241};
	float weight = 1.0f / Nscale;     
	int NumChannel=src.channels();
	cv::Mat GaussianOut;
	
	//zeros
	dst=cv::Mat::zeros(src.rows,src.cols,CV_8UC3);
	cv::Mat Intensity=cv::Mat::zeros(src.rows,src.cols,CV_8UC1);
	cv::Mat MSR_Nor;
	cv::Mat RetinexOut=cv::Mat::zeros(src.rows,src.cols,CV_32FC1);
	
	

	/*  
	compute intensity channel
	*/
	for (int i=0; i<src.rows; i++ ){
		for (int j=0; j<src.cols; j++ ){
			Intensity.ptr<uchar>(i)[j]= (src.at<cv::Vec3b>(i,j)[0]+src.at<cv::Vec3b>(i,j)[1]+src.at<cv::Vec3b>(i,j)[2])/3;
		}
	}

	//compute 3-scale retinex output, log domain
	for (int scale = 0; scale <3; scale++ ){
		cv::Size a(RetinexScales[scale],RetinexScales[scale]);
		cv::GaussianBlur(Intensity,GaussianOut,a,(int)RetinexScales[scale]);
		float zero=0;
		for (int i = 0; i < src.rows; i++ ){ 
			for (int j = 0; j < src.cols; j++ ){
				RetinexOut.ptr<float>(i)[j] += 1/3. *( log((float)Intensity.ptr<uchar>(i)[j]+1)-log((float)GaussianOut.ptr<uchar>(i)[j]+1.)); 
			}   
		}
	}


	//SimplestColorBalance, see IPOL for more information
	cv::normalize(RetinexOut,RetinexOut,0,255,32);
	SimplestColorBalance(RetinexOut,0.01,0.01);
	RetinexOut.convertTo(RetinexOut,CV_8UC1);
	MSR_Nor=RetinexOut.clone();


	//According to the reference paper(page 79,Algorithm 2), output the final image.
	 double  factor, max;
	 cv::Mat tempshow=cv::Mat::zeros(src.rows,src.cols,CV_32FC3);
	for (int i = 0; i < src.rows; i ++ ){
		for (int j = 0; j < src.cols; j ++ ){
		int B=0;
		float A=0;
		B=0;
		A=0;
		if(src.at<cv::Vec3b>(i,j)[0]>B)B=src.at<cv::Vec3b>(i,j)[0];
		if(src.at<cv::Vec3b>(i,j)[1]>B)B=src.at<cv::Vec3b>(i,j)[1];
		if(src.at<cv::Vec3b>(i,j)[2]>B)B=src.at<cv::Vec3b>(i,j)[2];

		float temp=(src.at<cv::Vec3b>(i,j)[0]+src.at<cv::Vec3b>(i,j)[1]+src.at<cv::Vec3b>(i,j)[2])/3.;	//intensity channel(before /3)
			
			if(temp <= 1.) temp=1.;
			factor=MSR_Nor.at<uchar>(i,j)/ temp;
			if( factor > 3.) factor=3.;
			tempshow.at<cv::Vec3f>(i,j)[2]=factor* src.at<cv::Vec3b>(i,j)[2];
			tempshow.at<cv::Vec3f>(i,j)[1]=factor*src.at<cv::Vec3b>(i,j)[1];
			tempshow.at<cv::Vec3f>(i,j)[0]=factor*src.at<cv::Vec3b>(i,j)[0];
			if( tempshow.at<cv::Vec3f>(i,j)[0] > 255. || tempshow.at<cv::Vec3f>(i,j)[1] > 255. ||tempshow.at<cv::Vec3f>(i,j)[0] > 255.)
			{
				max=src.at<cv::Vec3b>(i,j)[2];
				if(src.at<cv::Vec3b>(i,j)[1]> max) max=src.at<cv::Vec3b>(i,j)[1];
				if( src.at<cv::Vec3b>(i,j)[0] > max) max=src.at<cv::Vec3b>(i,j)[0];
				factor= 255. /max;
				tempshow.at<cv::Vec3f>(i,j)[2]=factor*src.at<cv::Vec3b>(i,j)[2];
				tempshow.at<cv::Vec3f>(i,j)[1]=factor*src.at<cv::Vec3b>(i,j)[1];
				tempshow.at<cv::Vec3f>(i,j)[0]=factor*src.at<cv::Vec3b>(i,j)[0];
			}

		}
	}

		std::vector<cv::Mat> tt(3);
		cv::Mat finalshow;
		cv::split(tempshow,tt);
		tt[0].convertTo(tt[0],CV_8UC1);
		tt[1].convertTo(tt[1],CV_8UC1);
		tt[2].convertTo(tt[2],CV_8UC1);
		cv::merge(tt,finalshow);
		dst=finalshow.clone();

	return true;
}   
