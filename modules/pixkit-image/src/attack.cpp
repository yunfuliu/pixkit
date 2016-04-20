#include "../include/pixkit-image.hpp"

//////////////////////////////////////////////////////////////////////////
bool	pixkit::attack::addGaussianNoise(const cv::Mat &src,cv::Mat &dst,const double sd){
	
	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src.empty()){
		return false;
	}
	if(src.type()!=CV_8U){
		return false;
	}
	if(sd<0.){
		CV_Error(CV_StsBadArg,"[pixkit::attack::addGaussianNoise] sd should bigger than 0.");
	}else if(sd==0.){
		dst=src.clone();
		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const	double	PI			=	3.1415926;
	const	int		MAXVALUE	=	255;
	const	int		CENTER		=	128;

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst;
	tdst	=	src.clone();

	//////////////////////////////////////////////////////////////////////////
	///// get cdf [output] dis
	double	cdf[257]={0};
	double	fm=0.;	// denominator
	for(int i=-128;i<=128;i++){
		cdf[i+CENTER]=1./sqrt((double)2.*PI*sd*sd)*exp((double)-0.5*i*i/sd/sd);	// get pdf
		fm+=cdf[i+CENTER];	// get fm
	}
	cdf[0]/=fm;
	for(int i=1;i<257;i++){
		cdf[i]/=fm;	// normalize	
		cdf[i]+=cdf[i-1];	// get cdf
	}

	//////////////////////////////////////////////////////////////////////////
	///// add noise
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			// obtain noise
			double	rand_value=(double)rand()/RAND_MAX;	// white noise from 0 to 1
			double	minv=fabs(rand_value-cdf[CENTER]);	// start from center
			double	noise_position=CENTER;
			if(rand_value<0.5){	// to speed up, it is separated into two parts
				for(int k=127;k>=0;k--){	// find out the minimum to be the noise mag 
					if(cdf[k]!=0){
						double	temp=fabs(rand_value-cdf[k]);
						if(temp<minv){
							minv=temp;
							noise_position=k;
						}
					}else{
						break;
					}
				}
			}else{
				for(int k=129;k<257;k++){	// find out the minimum to be the noise mag 
					if(cdf[k]!=1.){
						double	temp=fabs(rand_value-cdf[k]);
						if(temp<minv){
							minv=temp;
							noise_position=k;
						}
					}else{
						break;
					}
				}
			}

			// get mag
			double	noise_mag=noise_position-CENTER;

			// add noise
			tdst.data[i*tdst.cols+j]+=static_cast<uchar>(noise_mag);			
			if(tdst.data[i*tdst.cols+j]>MAXVALUE){
				tdst.data[i*tdst.cols+j]=MAXVALUE;
			}else if(tdst.data[i*tdst.cols+j]<0){
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
			noise_mag	=	(noise_mag*2.	-1.)	*	maxMag;	// ¤W¤U¨â­¿

			// add noise
			double	temp_output	=	((double)tdst.data[i*tdst.cols+j])	+	noise_mag;
			if(temp_output>=MAXVALUE){
				tdst.data[i*tdst.cols+j]=MAXVALUE;
			}else if(temp_output<=0){
				tdst.data[i*tdst.cols+j]=0;
			}else{
				tdst.data[i*tdst.cols+j]=cvRound(temp_output);
			}
		}
	}

	dst	=	tdst.clone();
	return true;
}
