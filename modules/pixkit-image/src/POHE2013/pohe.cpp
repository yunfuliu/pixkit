//========================================================================
//
// pohe.cpp
// Authors: Yun-Fu Liu (1), Jing-Ming Guo (2), Bo-Syun Lai (3), Jiann-Der Lee (4)
// Institutions: National Taiwan University of Science and Technology
// Date: May 26, 2013
// Email: yunfuliu@gmail.com
// Paper: Yun-Fu Liu, Jing-Ming Guo, Bo-Syun Lai, and Jiann-Der Lee, "High efficient 
//        contrast enhancement using parametric approximation," IEEE Trans. 
//        Image Processing, pp. 2444-2448, 26-31 May 2013.
//
// POHE Image Contrast Enhancement Copyright (c) 2013, Yun-Fu Liu, all rights reserved.
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

// calc cdf of uniform
double calcCDF_Uniform(double src,double &maxv,double &minv){
	if(src==maxv){
		return 1.0f;
	}else if(src==minv){
		return 0.0f;
	}else{
		return (src-minv)/(maxv-minv);
	}
}
// calc cdf of Gaussian
double calcCDF_Gaussian(double src,double &mean,double &sd){
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
// calc integral image
void integrate(const cv::Mat &src,double *dst,float order){

	const int &mHeight=src.rows;
	const int &mWidth=src.cols;

	if(src.type()==CV_8UC1){
		for(int i=0;i<mHeight;i++){
			for(int j=0;j<mWidth;j++){
				dst[i*mWidth+j]=pow((double)((uchar*)src.data)[i*src.cols+j],(double)order);
			}
		}
	}else if(src.type()==CV_32FC1){
		for(int i=0;i<mHeight;i++){
			for(int j=0;j<mWidth;j++){
				dst[i*mWidth+j]=pow((double)((float*)src.data)[i*src.cols+j],(double)order);
			}
		}
	}else{
		CV_Error(CV_StsBadArg,"[integrate] src should be either CV_8UC1 or CV_32FC1");
	}

	//////////////////////////////////////////////////////////////////////////
	for(int i=1;i<mHeight;i++){
		dst[i*mWidth]=dst[i*mWidth]+dst[(i-1)*mWidth];
	}
	for(int j=1;j<mWidth;j++){
		dst[j]=dst[j]+dst[j-1];
	}
	for(int i=1;i<mHeight;i++){
		for(int j=1;j<mWidth;j++){
			dst[i*mWidth+j]=dst[i*mWidth+j]+dst[(i-1)*mWidth+j]+dst[i*mWidth+j-1]-dst[(i-1)*mWidth+j-1];
		}
	}
}
bool pixkit::enhancement::local::POHE2013(const cv::Mat &src,cv::Mat &dst,const cv::Size blockSize){

	//////////////////////////////////////////////////////////////////////////
	// Exceptions
	//////////////////////////////////////////////////////////////////////////
	// larger problem
	if(blockSize.height>=src.rows || blockSize.width>=src.cols){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] block size should be smaller than image size. ");
	}
	// even/odd problem
	if(blockSize.height%2==0 || blockSize.width%2==0){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] either block's height or width is 0.");
	}
	if(blockSize.height==1&&blockSize.width==1){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] blockSize=(1,1) will turn entire image to dead white.");
	}
	if(src.type()!=CV_8UC1&&src.type()!=CV_32FC1){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::POHE2013] src should be either CV_8UC1 or CV_32FC1");
	}


	//////////////////////////////////////////////////////////////////////////
	// initialization
	cv::Mat	tdst;	// temp dst. To avoid that when src == dst occur.
	tdst.create(src.size(),src.type());
	const int &mHeight=src.rows;
	const int &mWidth=src.cols;
	double	*Sum=new double [mHeight*mWidth];
	double	*sqsum=new double [mHeight*mWidth];	


	//////////////////////////////////////////////////////////////////////////
	///// create integral images
	integrate(src,Sum,1);
	integrate(src,sqsum,2);


	//////////////////////////////////////////////////////////////////////////
	///// process
	int SSFilterSize_h=blockSize.height/2;	// SSFilterSize: Single Side Filter Size
	int SSFilterSize_w=blockSize.width/2;	// SSFilterSize: Single Side Filter Size
	for(int i=0;i<mHeight;i++){
		for(int j=0;j<mWidth;j++){

			bool	A=true,	// bottom, top, left, right; thus br: bottom-right
				B=true,
				C=true,
				D=true;
			double	mean=0;
			double	var=0;

			if(i+SSFilterSize_h>=mHeight){
				A=false;
			}
			if(j+SSFilterSize_w>=mWidth){
				B=false;
			}
			if(i-SSFilterSize_h-1<0){
				C=false;
			}
			if(j-SSFilterSize_w-1<0){
				D=false;
			}

			if(A==true && B==true && C==false && D==false)
			{
				mean=Sum[(i+SSFilterSize_h)*mWidth+(j+SSFilterSize_w)]/(double)((i+SSFilterSize_h+1)*(j+SSFilterSize_w+1));
				var=sqrt((sqsum[(i+SSFilterSize_h)*mWidth+(j+SSFilterSize_w)]/(double)((i+SSFilterSize_h+1)*(j+SSFilterSize_w+1)))-mean*mean);
			}
			else if(A==true && B==true && C==false && D==true)
			{
				mean=(Sum[(i+SSFilterSize_h)*mWidth+(j+SSFilterSize_w)]-Sum[(i+SSFilterSize_h)*mWidth+(j-SSFilterSize_w-1)])/(double)((i+SSFilterSize_h+1)*blockSize.width);
				var=sqrt(((sqsum[(i+SSFilterSize_h)*mWidth+(j+SSFilterSize_w)]-sqsum[(i+SSFilterSize_h)*mWidth+(j-SSFilterSize_w-1)])/(double)((i+SSFilterSize_h+1)*blockSize.width))-mean*mean);
			}
			else if(A==true && B==false && C==false && D==true)
			{
				mean=(Sum[(i+SSFilterSize_h)*mWidth+mWidth-1]-Sum[(i+SSFilterSize_h)*mWidth+(j-SSFilterSize_w-1)])/(double)((i+SSFilterSize_h+1)*(mWidth-j+SSFilterSize_w));
				var=sqrt(((sqsum[(i+SSFilterSize_h)*mWidth+mWidth-1]-sqsum[(i+SSFilterSize_h)*mWidth+(j-SSFilterSize_w-1)])/(double)((i+SSFilterSize_h+1)*(mWidth-j+SSFilterSize_w)))-mean*mean);
			}
			else if(A==true && B==true && C==true && D==false)
			{
				mean=(Sum[(i+SSFilterSize_h)*mWidth+(j+SSFilterSize_w)]-Sum[(i-SSFilterSize_h-1)*mWidth+(j+SSFilterSize_w)])/(double)(blockSize.height*(j+SSFilterSize_w+1));
				var=sqrt(((sqsum[(i+SSFilterSize_h)*mWidth+(j+SSFilterSize_w)]-sqsum[(i-SSFilterSize_h-1)*mWidth+(j+SSFilterSize_w)])/(double)(blockSize.height*(j+SSFilterSize_w+1)))-mean*mean);
			}
			else if(A==true && B==false && C==true && D==true)
			{
				mean=(Sum[(i+SSFilterSize_h)*mWidth+(mWidth-1)]-Sum[(i-SSFilterSize_h-1)*mWidth+(mWidth-1)]-Sum[(i+SSFilterSize_h)*mWidth+(j-SSFilterSize_w-1)]+Sum[(i-SSFilterSize_h-1)*mWidth+(j-SSFilterSize_w-1)])/(double)(blockSize.height*(mWidth-j+SSFilterSize_w));
				var=sqrt(((sqsum[(i+SSFilterSize_h)*mWidth+(mWidth-1)]-sqsum[(i-SSFilterSize_h-1)*mWidth+(mWidth-1)]-sqsum[(i+SSFilterSize_h)*mWidth+(j-SSFilterSize_w-1)]+sqsum[(i-SSFilterSize_h-1)*mWidth+(j-SSFilterSize_w-1)])/(double)(blockSize.height*(mWidth-j+SSFilterSize_w)))-mean*mean);
			}
			else if(A==false && B==true && C==true && D==false){

				mean=(Sum[(mHeight-1)*mWidth+(j+SSFilterSize_w)]-Sum[(i-SSFilterSize_h-1)*mWidth+(j+SSFilterSize_w)])/(double)((mHeight-i+SSFilterSize_h)*(j+SSFilterSize_w+1));
				var=sqrt(((sqsum[(mHeight-1)*mWidth+(j+SSFilterSize_w)]-sqsum[(i-SSFilterSize_h-1)*mWidth+(j+SSFilterSize_w)])/(double)((mHeight-i+SSFilterSize_h)*(j+SSFilterSize_w+1)))-mean*mean);

			}else if(A==false && B==true && C==true && D==true)
			{
				mean=(Sum[(mHeight-1)*mWidth+(j+SSFilterSize_w)]-Sum[(i-SSFilterSize_h-1)*mWidth+(j+SSFilterSize_w)]-Sum[(mHeight-1)*mWidth+(j-SSFilterSize_w-1)]+Sum[(i-SSFilterSize_h-1)*mWidth+(j-SSFilterSize_w-1)])/(double)((mHeight-i+SSFilterSize_h)*blockSize.width);
				var=sqrt(((sqsum[(mHeight-1)*mWidth+(j+SSFilterSize_w)]-sqsum[(i-SSFilterSize_h-1)*mWidth+(j+SSFilterSize_w)]-sqsum[(mHeight-1)*mWidth+(j-SSFilterSize_w-1)]+sqsum[(i-SSFilterSize_h-1)*mWidth+(j-SSFilterSize_w-1)])/(double)((mHeight-i+SSFilterSize_h)*blockSize.width))-mean*mean);
			}
			else if(A==false && B==false && C==true && D==true)
			{
				mean=(Sum[(mHeight-1)*mWidth+(mWidth-1)]-Sum[(i-SSFilterSize_h-1)*mWidth+(mWidth-1)]-Sum[(mHeight-1)*mWidth+(j-SSFilterSize_w-1)]+Sum[(i-SSFilterSize_h-1)*mWidth+(j-SSFilterSize_w-1)])/(double)((mHeight-i+SSFilterSize_h)*(mWidth-j+SSFilterSize_w));
				var=sqrt(((sqsum[(mHeight-1)*mWidth+(mWidth-1)]-sqsum[(i-SSFilterSize_h-1)*mWidth+(mWidth-1)]-sqsum[(mHeight-1)*mWidth+(j-SSFilterSize_w-1)]+sqsum[(i-SSFilterSize_h-1)*mWidth+(j-SSFilterSize_w-1)])/(double)((mHeight-i+SSFilterSize_h)*(mWidth-j+SSFilterSize_w)))-mean*mean);
			}
			else{
				mean=(Sum[(i+SSFilterSize_h)*mWidth+(j+SSFilterSize_w)]-Sum[(i-SSFilterSize_h-1)*mWidth+(j+SSFilterSize_w)]-Sum[(i+SSFilterSize_h)*mWidth+(j-SSFilterSize_w-1)]+Sum[(i-SSFilterSize_h-1)*mWidth+(j-SSFilterSize_w-1)])/(double)(blockSize.height*blockSize.width);
				var=sqrt(((sqsum[(i+SSFilterSize_h)*mWidth+(j+SSFilterSize_w)]-sqsum[(i-SSFilterSize_h-1)*mWidth+(j+SSFilterSize_w)]-sqsum[(i+SSFilterSize_h)*mWidth+(j-SSFilterSize_w-1)]+sqsum[(i-SSFilterSize_h-1)*mWidth+(j-SSFilterSize_w-1)])/(double)(blockSize.height*blockSize.width))-mean*mean);
			}


			// get current src value
			double	current_src_value=0.;
			if(src.type()==CV_8UC1){
				current_src_value	=	((uchar*)src.data)[i*mWidth+j];
			}else if(src.type()==CV_32FC1){
				current_src_value	=	((float*)src.data)[i*mWidth+j];
			}else{
				assert(false);
			}

			// calc Gaussian's cdf
			double	input;
			if(var==0){	// means all the value in this block are the same
				input=0.;
			}else{
				input=(current_src_value-mean)/(sqrt(2.0)*var);
			}
			double t=1/(1+0.3275911*input);
			double erf=0.25482929592*t-0.284496736*t*t+1.421413741*t*t*t-1.453152027*t*t*t*t+1.061405429*t*t*t*t*t;
			erf=1-(erf*exp(-(input*input)));
			erf=0.5*(1+erf);

			// get output
			if(tdst.type()==CV_8UC1){
				if(current_src_value >= mean-1.885*var){
					((uchar*)tdst.data)[i*mWidth+j]= erf*255;
				}else{
					((uchar*)tdst.data)[i*mWidth+j]=0;
				}
				CV_DbgAssert(((uchar*)tdst.data)[i*mWidth+j]>=0.&&((uchar*)tdst.data)[i*mWidth+j]<=255.);
			}else if(tdst.type()==CV_32FC1){
				if(current_src_value >= mean-1.885*var){
					((float*)tdst.data)[i*mWidth+j]= erf*255;
				}else{
					((float*)tdst.data)[i*mWidth+j]=0;
				}
				CV_DbgAssert(((float*)tdst.data)[i*mWidth+j]>=0.&&((float*)tdst.data)[i*mWidth+j]<=255.);
			}else{
				assert(false);
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////
	///// copy
	dst	=	tdst.clone();
	// delete
	delete [] Sum;
	delete [] sqsum;

	return true;
}