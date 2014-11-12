#define _USE_MATH_DEFINES // for C
#include<math.h>
#include <float.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
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
			if(ori.ptr<float>(i)[j]<Vmin)ori.ptr<float>(i)[j]=0;
			else if(ori.ptr<float>(i)[j]>Vmax)ori.ptr<float>(i)[j]=255;
			else ori.ptr<float>(i)[j]= (ori.ptr<float>(i)[j]-Vmin)*255./(Vmax-Vmin);
		}
	}

	return 1;

}
bool pixkit::enhancement::local::NRCIR2009( cv::Mat ori,cv::Mat &ret){

	if(ori.type()!=CV_8UC3){
		CV_Assert(false);
	}

	//calculate key value
	cv::Mat Lab=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC3);
	cv::Mat GL_mapped=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC1);
	cv::Mat ori_32FC1=ori.clone();
	std::vector<cv::Mat> RGB_Indepen(3);
	std::vector<cv::Mat> RGB_GLmapped(3);
	std::vector<cv::Mat> GLmapped_LabIndepen(3);
	std::vector<cv::Mat> Final_Lab_In(3);
	std::vector<cv::Mat> Lab_Indepen(3);
	for(int i=0;i<3;i++){
		RGB_GLmapped[i]=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC1);
		RGB_Indepen[i]=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC1);
		GLmapped_LabIndepen[i]=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC1);
		Final_Lab_In[i]=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC1);
		Lab_Indepen[i]=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC1);
	}

	ori.convertTo(ori_32FC1,CV_32FC3);
	ori.convertTo(Lab,CV_32FC3);
	Lab=Lab/255.;
	cv::cvtColor(Lab,Lab,CV_BGR2Lab);
	cv::split(ori_32FC1,RGB_Indepen);
	cv::split(Lab,Lab_Indepen);

	//calculate keys, equation (1)
	double key_value=0;
	double radius=0;
	double Igm=0;
	double CenterPrecision=0.001;
	cv::Point2d MCcenter;	//mapping circle center
	float Fzero=0;
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			if(Lab.at<cv::Vec3f>(i,j)[0]==Fzero)continue;
			key_value	=	key_value+log(((float)Lab.at<cv::Vec3f>(i,j)[0]));
		}
	}	
	key_value=key_value/(ori.rows*ori.cols);
	key_value=exp(key_value);

	//calculate radius and perform global mapping for each RGB channel
	if(key_value<=50){
		radius=3.*log10(key_value/10+1);
		if(radius<1.4)radius=1.4;
		for(MCcenter.x=0;;MCcenter.x=MCcenter.x+CenterPrecision){
			MCcenter.y=1-MCcenter.x;
			if(sqrt(pow(MCcenter.y,2)+pow(MCcenter.x,2))>radius)break;
		}

		for(int z=0;z<3;z++){
			for(int i=0;i<ori.rows;i++){
				for(int j=0;j<ori.cols;j++){
					double RGBscaled=RGB_Indepen[z].ptr<float>(i)[j]/255.;
					Igm=MCcenter.y+sqrt(pow(radius,2)-pow(RGBscaled-MCcenter.x,2));
					RGB_GLmapped[z].ptr<float>(i)[j]=Igm;

				}
			}
		}
	}
	else if (key_value>=60){
		radius=3.*log10(10-key_value/10+1);
		if(radius<1.4)radius=1.4;
		for(MCcenter.x=0;;MCcenter.x=MCcenter.x-CenterPrecision){
			MCcenter.y=1-MCcenter.x;
			if(sqrt(pow(MCcenter.y,2)+pow(MCcenter.x,2))>radius)break;
		}
		for(int z=0;z<ori.channels();z++){
			for(int i=0;i<ori.rows;i++){
				for(int j=0;j<ori.cols;j++){

					double RGBscaled=RGB_Indepen[z].ptr<float>(i)[j]/255;
					Igm=MCcenter.y-sqrt(pow(radius,2)-pow(RGBscaled-MCcenter.x,2));
					RGB_GLmapped[z].ptr<float>(i)[j]=Igm;
				}
			}
		}
	}//until here, global tone mapping is done

	cv::merge(RGB_GLmapped,GL_mapped);
	cv::Mat GL_mapped_LAB;
	cv::cvtColor(GL_mapped,GL_mapped_LAB,CV_BGR2Lab);
	cv::split(GL_mapped_LAB,GLmapped_LabIndepen);
	cv::Mat Im=cv::Mat::zeros(GL_mapped.size(),CV_32FC1);
	cv::Mat E4_Result;//equation 4

	//acquire the proposed filter, equation (6) (7)
	float K=(ori.cols*ori.rows)/8;
	int Path_K=((int)sqrt(K)&-2)+1;
	cv::Mat Filter_mask=cv::Mat::zeros(Path_K,Path_K,CV_32FC1);
	for(int i=0;i<log((float)Path_K)/log((float)2);i++){
		for(int y=-Path_K/2;y<=Path_K/2;y++){
			for(int x=-Path_K/2;x<=Path_K/2;x++){
				Filter_mask.ptr<float>(y+Path_K/2)[x+Path_K/2]=exp(-(y*y+x*x)/pow((float)2,2*i));
			}
		}
	}

	//acquire Im, equation (5)
	cv::filter2D(GLmapped_LabIndepen[0],Im,GLmapped_LabIndepen[0].depth(),Filter_mask);

	//Retinex luminance, equation (4)
	cv::Mat E4_Domi=cv::Mat::zeros(ori.size(),CV_32FC1);
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			E4_Domi.ptr<float>(i)[j]=log10(Im.ptr<float>(i)[j]+1);
		}
	}
	cv::divide(GLmapped_LabIndepen[0],E4_Domi,E4_Result);

	//according to the flowchart(Fig5), perform histogram rescaling after Modified Retinex
	cv::normalize(E4_Result,E4_Result,0,255,32);
	SimplestColorBalance(E4_Result,0.01,0);
	

	//obtain reference map, equation(8)
	cv::Mat M_ref=cv::Mat::zeros(ori.size(),CV_32FC1);
	cv::Mat rescaled_L;
	cv::normalize(Lab_Indepen[0],rescaled_L,0,255,32);
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			double log2Value=  log(E4_Result.ptr<float>(i)[j]/rescaled_L.ptr<float>(i)[j]+1)/log(2.);
			M_ref.ptr<float>(i)[j]=(log2Value);
		}
	}

	//enhance the globally mapped RGB image, equation(9)
	cv::Mat_<cv::Vec3f> enhancedRGB=cv::Mat::zeros(ori.size(),CV_32FC3);

	for(int z=0;z<ori.channels();z++){
		for(int i=0;i<ori.rows;i++){
			for(int j=0;j<ori.cols;j++){
				enhancedRGB.at<cv::Vec3f>(i,j)[z]=RGB_Indepen[z].ptr<float>(i)[j]/255.*M_ref.ptr<float>(i)[j];
				if(enhancedRGB.at<cv::Vec3f>(i,j)[z]>=1)enhancedRGB.at<cv::Vec3f>(i,j)[z]=1;
			}
		}
	}

	//according to the flowchart(Fig5), perform final histogram rescaling after the step 'Luminance channel'
	cv::Mat Final_Lab;
	cv::Mat FinalResult;
	cv::cvtColor(enhancedRGB,Final_Lab,CV_BGR2Lab);
	cv::split(Final_Lab,Final_Lab_In);
	SimplestColorBalance(Final_Lab_In[0],0.01,0);
	cv::normalize(Final_Lab_In[0],Final_Lab_In[0],0,100,32);
	cv::merge(Final_Lab_In,Final_Lab);
	cv::cvtColor(Final_Lab,FinalResult,CV_Lab2BGR);
	FinalResult=FinalResult*255;
	FinalResult.convertTo(FinalResult,CV_8UC3);
	ret=FinalResult.clone();
	return 1;
}

