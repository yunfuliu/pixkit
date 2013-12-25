#include "../pixkit-image.hpp"

//////////////////////////////////////////////////////////////////////////
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
			noise_mag	*=maxMag*2.-maxMag;	// ¤W¤U¨â­¿

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
