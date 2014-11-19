//========================================================================
//
// pohe.cpp
// Authors: Yun-Fu Liu (1), Jing-Ming Guo (2), Bo-Syun Lai (3), Jiann-Der Lee (4)
// Institutions: National Taiwan University of Science and Technology
// Date: May 26, 2013
// Email: yunfuliu@gmail.com, jmguo@seed.net.tw
// Paper: Yun-Fu Liu, Jing-Ming Guo, Bo-Syun Lai, and Jiann-Der Lee, "High efficient 
//        contrast enhancement using parametric approximation," IEEE Trans. 
//        Image Processing, pp. 2444-2448, 26-31 May 2013.
//
// POHE Image Contrast Enhancement Copyright (c) 2013, Yun-Fu Liu, Jing-Ming Guo, 
// Bo-Syun Lai, and Jiann-Der Lee, all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * The name of the copyright holders may not be used to endorse or promote products
//   derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF 
// THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//========================================================================

#include "../../include/pixkit-image.hpp"
#include <opencv2/imgproc/imgproc.hpp>

// calc cdf of Gaussian
inline double calcCDF_Gaussian(double &src,double &mean,double &sd){
	// exception when sd=0
	if(sd==0&&mean==src){
		return 1.;
	}
	double x=(src-mean)/(sqrt(2.)*sd);
	double t=1./(1.+0.3275911*fabs(x));
	double erf=		0.254829592	*t
				-	0.284496736	*t*t
				+	1.421413741	*t*t*t
				-	1.453152027	*t*t*t*t
				+	1.061405429	*t*t*t*t*t;
	erf=1.-erf*exp(-(x*x));
	return 0.5*(1+(x<0?-erf:erf));	
}

// calc the sum of an area 
inline bool calcAreaMean(const cv::Mat &src,
	cv::Point currpos,
	cv::Size blockSize,
	const cv::Mat &sum=cv::Mat(),
	double *mean=NULL,
	const cv::Mat &sqsum=cv::Mat(),
	double *sd=NULL){

		//////////////////////////////////////////////////////////////////////////
		///// exceptions
		if(blockSize.width%2==0||blockSize.height%2==0){
			CV_Error(CV_StsBadArg,"[calcAreaMean] either block's height or width is 0.");
		}


		//////////////////////////////////////////////////////////////////////////
		///// initialization
		int SSFilterSize_h	=	blockSize.height/2;	// SSFilterSize: Single Side Filter Size
		int SSFilterSize_w	=	blockSize.width/2;	// SSFilterSize: Single Side Filter Size
		const	int	&height	=	src.rows;
		const	int	&width	=	src.cols;
		const	int	&i		=	currpos.y;
		const	int	&j		=	currpos.x;



		//////////////////////////////////////////////////////////////////////////
		bool	A=true,	// bottom, top, left, right; thus br: bottom-right
			B=true,
			C=true,
			D=true;
		if((i+SSFilterSize_h)>=height){
			A=false;
		}
		if((j+SSFilterSize_w)>=width){
			B=false;
		}
		if((i-SSFilterSize_h-1)<0){
			C=false;
		}
		if((j-SSFilterSize_w-1)<0){
			D=false;
		}


		////////////////////////////////////////////////////////////////////////// ok
		///// get area size
		// width
		double	areaWidth	=	blockSize.width,
				areaHeight	=	blockSize.height;
		if(!D){
			areaWidth	=	j+SSFilterSize_w+1;
		}else if(!B){
			areaWidth	=	width-j+SSFilterSize_w;
		}
		// height
		if(!C){
			areaHeight	=	i+SSFilterSize_h+1;
		}else if(!A){
			areaHeight	=	height-i+SSFilterSize_h;
		}


		//////////////////////////////////////////////////////////////////////////
		///// get value
		// get positions
		int	y_up	=	i+SSFilterSize_h		+1,	// '+1' is the bias term of the integral image 
			y_dn	=	i-SSFilterSize_h-1		+1;
		if(!A){
			y_up	=	height-1				+1;
		}
		int	x_left	=	j-SSFilterSize_w-1		+1,
			x_right	=	j+SSFilterSize_w		+1;
		if(!B){
			x_right	=	width-1					+1;
		}
		// get values
		double	cTR_sum=0,cTR_sqsum=0,
				cTL_sum=0,cTL_sqsum=0,
				cBR_sum=0,cBR_sqsum=0,
				cBL_sum=0,cBL_sqsum=0;
		cTR_sum			=	sum.ptr<double>(y_up)[x_right];
		cTR_sqsum		=	sqsum.ptr<double>(y_up)[x_right];
		if(C&&D){
			cBL_sum		=	sum.ptr<double>(y_dn)[x_left];		
			cBL_sqsum	=	sqsum.ptr<double>(y_dn)[x_left];	
		}
		if(D){
			cTL_sum		=	-sum.ptr<double>(y_up)[x_left];
			cTL_sqsum	=	-sqsum.ptr<double>(y_up)[x_left];
		}
		if(C){
			cBR_sum		=	-sum.ptr<double>(y_dn)[x_right];
			cBR_sqsum	=	-sqsum.ptr<double>(y_dn)[x_right];
		}


		////////////////////////////////////////////////////////////////////////// ok
		///// get mean and sd
		*mean	=cTR_sum		+cTL_sum	+cBR_sum	+cBL_sum;
		*mean	/=areaHeight*areaWidth;
		CV_DbgAssert((*mean)>=0.&&(*mean)<=255.);
		*sd		=cTR_sqsum		+cTL_sqsum	+cBR_sqsum	+cBL_sqsum;	
		*sd		/=areaHeight*areaWidth;
		// compensate error
		if(fabs((*sd)-(*mean)*(*mean))<0.00001){
			*sd=0;
		}else{
			CV_DbgAssert((*sd)>=(*mean)*(*mean));
			*sd		=sqrt(*sd-(*mean)*(*mean));
		}		
		CV_DbgAssert((*sd)>=0.&&(*sd)<=255.);

	return true;
}

 bool pixkit::enhancement::local::POHE2013(const cv::Mat &src,cv::Mat &dst,const cv::Size blockSize,const cv::Mat &sum,const cv::Mat &sqsum){

	//////////////////////////////////////////////////////////////////////////
	///// Exceptions
	// blockSize
	if(blockSize.height>=src.rows || blockSize.width>=src.cols){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] block size should be smaller than image size. ");
	}
	if(blockSize.height%2==0 || blockSize.width%2==0){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] both block's width and height should be odd.");
	}
	if(blockSize.height==1&&blockSize.width==1){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] blockSize=(1,1) will turn entire image to dead white.");
	}
	// src
	if(src.type()!=CV_8UC1&&src.type()!=CV_8UC3&&src.type()!=CV_32FC1){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] src should be one of CV_8UC1, CV_8UC3, and CV_32FC1.");
	}
	// sum and sqsum
	if(!sum.empty()){
		if(sum.type()!=CV_64FC1){
			CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] both sum and sqsum should be CV_64FC1");
		}
	}
	if(!sqsum.empty()){
		if(sqsum.type()!=CV_64FC1){
			CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] both sum and sqsum should be CV_64FC1");
		}
	}
	
	//////////////////////////////////////////////////////////////////////////
	///// color image
	cv::Mat	_src1x,_src3x;
	if(src.channels()==1){
		_src1x=src;
	}else if(src.channels()==3){
		// create space for _src1x
		if(src.type()==CV_8UC3){
			_src1x.create(src.size(),CV_8UC1);
		}else if(src.type()==CV_32FC3){
			_src1x.create(src.size(),CV_32FC1);
		}else{
			CV_Assert(false);
		}
		// convert color
		cvtColor(src,_src3x,CV_BGR2Lab);
		int	from_to[]={0,0};
		mixChannels(&_src3x,1,&_src1x,1,from_to,1);
	}else{
		CV_Error(CV_StsUnsupportedFormat,"[pixkit::enhancement::local::POHE2013] images should be grayscale or color.");
	}

	//////////////////////////////////////////////////////////////////////////
	// initialization
	cv::Mat	tdst;	// temp dst. To avoid that when src == dst occur.
	tdst.create(_src1x.size(),_src1x.type());
	const int &height=_src1x.rows;
	const int &width=_src1x.cols;

	//////////////////////////////////////////////////////////////////////////
	///// create integral images
	cv::Mat	tsum,tsqsum;
	if(sum.empty()||sqsum.empty()){
		cv::integral(_src1x,tsum,tsqsum,CV_64F);
	}else{
		tsum	=	sum;
		tsqsum	=	sqsum;
	}

	//////////////////////////////////////////////////////////////////////////
	///// process
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			
			//////////////////////////////////////////////////////////////////////////
			///// get mean and sd
			double	mean,sd;
			calcAreaMean(_src1x,cv::Point(j,i),blockSize,tsum,&mean,tsqsum,&sd);

			//////////////////////////////////////////////////////////////////////////
			// get current src value
			double	current_src_value=0.;
			if(_src1x.type()==CV_8UC1){
				current_src_value	=	_src1x.ptr<uchar>(i)[j];
			}else if(_src1x.type()==CV_32FC1){
				current_src_value	=	_src1x.ptr<float>(i)[j];
			}else{
				assert(false);
			}

			//////////////////////////////////////////////////////////////////////////
			// calc Gaussian's cdf
			double	cdf	=	calcCDF_Gaussian(current_src_value,mean,sd);

			//////////////////////////////////////////////////////////////////////////
			// get output
			if(tdst.type()==CV_8UC1){
				if(current_src_value >= mean-1.885*sd){
					tdst.data[i*width+j]	=	cvRound(cdf*255.);
				}else{
					tdst.data[i*width+j]	=	0;
				}
				CV_DbgAssert(((uchar*)tdst.data)[i*width+j]>=0.&&((uchar*)tdst.data)[i*width+j]<=255.);
			}else if(tdst.type()==CV_32FC1){
				if(current_src_value >= mean-1.885*sd){
					((float*)tdst.data)[i*width+j]	=	cvRound(cdf*255.);
				}else{
					((float*)tdst.data)[i*width+j]	=	0;
				}
				CV_DbgAssert(((float*)tdst.data)[i*width+j]>=0.&&((float*)tdst.data)[i*width+j]<=255.);
			}else{
				CV_Assert(false);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// copy to dst
	if(src.channels()==1){
		dst	=	tdst.clone();
	}else if(src.channels()==3){
		int	from_to[]={0,0};
		mixChannels(&tdst,1,&_src3x,1,from_to,1);
		cvtColor(_src3x,dst,CV_Lab2BGR);
	}else{
		CV_Error(CV_StsUnsupportedFormat,"[pixkit::enhancement::local::POHE2013] images should be grayscale or color.");
	}
	
	return true;
}
