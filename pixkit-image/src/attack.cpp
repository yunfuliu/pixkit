#include "../pixkit-image.hpp"

//////////////////////////////////////////////////////////////////////////
bool	pixkit::attack::addGaussianNoise(const cv::Mat &src,cv::Mat &dst,const double sd){
	
	//////////////////////////////////////////////////////////////////////////
	if(src.empty()){
		return false;
	}
	if(src.type()!=CV_8U){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const	double	PI			=	3.1415926;
	const	int		MAXVALUE	=	255;

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst;
	tdst	=	src.clone();

	//////////////////////////////////////////////////////////////////////////
	///// get cdf [output] dis
	double	cdf[257]={0};
	double	fm=0.;	// 分母
	for(int i=-128;i<=128;i++){
		if(sd==0.){
			if(i!=0){	// 使得以下計算error最小值為i==0時
				cdf[i+128]	=	0.;
			}else{
				cdf[i+128]	=	1.;
			}
		}else{
			cdf[i+128]=1./sqrt((double)2.*PI*sd*sd)*exp((double)-0.5*i*i/sd/sd);	// get pdf
		}
		fm+=cdf[i+128];	// get fm
	}
	for(int i=0;i<257;i++){
		cdf[i]/=fm;	// normalize
	}
	for(int i=1;i<257;i++){
		cdf[i]+=cdf[i-1];	// cdf
	}

	//////////////////////////////////////////////////////////////////////////
	// add noise
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			///// obtain noise, 垂直為white noise, 以找到對應的水平雜訊強度, 故以尋找最小值方式進行, [output] noise
			double	rand_value=(double)rand()/RAND_MAX;	// white noise from 0 to 1
			double	minv=9999999.;
			double	noise_position=0;
			for(int k=0;k<257;k++){
				if(cdf[k]!=0){
					double	temp=fabs(rand_value-cdf[k]);
					if(temp<minv){
						minv=temp;
						noise_position=k;
					}
				}
			}
			double	noise_mag=noise_position-128.;

			///// add noise
			tdst.data[i*tdst.cols+j]+=noise_mag;
			if(tdst.data[i*tdst.cols+j]>MAXVALUE){
				tdst.data[i*tdst.cols+j]=MAXVALUE;
			}
			if(tdst.data[i*tdst.cols+j]<0){
				tdst.data[i*tdst.cols+j]=0;
			}
		}
	}

	dst	=	tdst.clone();
	return true;
}
bool	pixkit::attack::addWhiteNoise(const cv::Mat &src,cv::Mat &dst,const double maxMag){


	//////////////////////////////////////////////////////////////////////////
	if(src.empty()){
		return false;
	}
	if(src.type()!=CV_8U){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const	int		MAXVALUE	=	255;

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst;
	tdst	=	src.clone();

	//////////////////////////////////////////////////////////////////////////
	///// add noise
	for(int i=0;i<tdst.rows;i++){
		for(int j=0;j<tdst.cols;j++){

			// get noise
			double	noise_mag=(double)rand()/RAND_MAX;	// white noise from 0 to 1
			noise_mag	*=maxMag*2.-maxMag;	// 上下兩倍

			// add noise
			double	temp_output	=	(double)tdst.data[i*tdst.cols+j]	+	noise_mag;
			if(temp_output>=MAXVALUE){
				tdst.data[i*tdst.cols+j]=MAXVALUE;
			}else if(temp_output<=0){
				tdst.data[i*tdst.cols+j]=0;
			}else{
				tdst.data[i*tdst.cols+j]=(int)(temp_output+0.5);
			}
		}
	}

	dst	=	tdst.clone();
	return true;
}
