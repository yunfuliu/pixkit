#include "../include/pixkit-image.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//////////////////////////////////////////////////////////////////////////
bool pixkit::filtering::medianfilter(const cv::Mat &src,cv::Mat &dst,cv::Size blocksize){

	//////////////////////////////////////////////////////////////////////////
	if(blocksize.width>src.cols||blocksize.height>src.rows){
		return false;	}
	if(blocksize.width%2==0||blocksize.height%2==0){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	const	int	half_block_height	=	blocksize.height/2;
	const	int	half_block_width	=	blocksize.width/2;
	std::vector<uchar>	temp_img(blocksize.height*blocksize.width,0);
	// process
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			for(int m=-half_block_height;m<=half_block_height;m++){
				for(int n=-half_block_width;n<=half_block_width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						temp_img[(m+half_block_height)*blocksize.width+(n+half_block_width)]=src.data[(i+m)*src.cols+(j+n)];
					}else{
						temp_img[(m+half_block_height)*blocksize.width+(n+half_block_width)]=0;
					}
					
				}
			}
			// ordering
			for(int m=0;m<blocksize.height*blocksize.width;m++){
				for(int n=0;n<blocksize.height*blocksize.width-1;n++){
					if(temp_img[n]>temp_img[n+1]){
						double temp=temp_img[n+1];
						temp_img[n+1]=temp_img[n];
						temp_img[n]=temp;
					}	//將value陣列內數值由大至小排列
				}
			}
			tdst.data[i*tdst.cols+j]	=	temp_img[(blocksize.height*blocksize.width-1)/2];
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}

//////////////////////////////////////////////////////////////////////////
// fast box filtering
bool pixkit::filtering::FBF(const cv::Mat &src,cv::Mat &dst,cv::Size blockSize,cv::Mat &sum){

	////////////////////////////////////////////////////////////////////////// ok
	///// exceptions
	if(src.type()!=CV_32FC1&&src.type()!=CV_8UC1){
		CV_Error(CV_StsBadArg,"[pixkit::filtering::FBF] src type should be either CV_8UC1 or CV_32FC1.");
	}
	if(blockSize.height%2==0||blockSize.width%2==0){
		CV_Error(CV_StsBadArg,"[pixkit::filtering::FBF] block size should be odd.");
	}


	////////////////////////////////////////////////////////////////////////// ok
	///// initialization
	cv::Mat	tdst;	// temp dst. To avoid that when src == dst occur.
	tdst.create(src.size(),src.type());
	int SSFilterSize_h	=	blockSize.height/2;	// SSFilterSize: Single Side Filter Size
	int SSFilterSize_w	=	blockSize.width/2;	// SSFilterSize: Single Side Filter Size

	////////////////////////////////////////////////////////////////////////// ok
	///// get integral image
	if(sum.empty()){
		cv::integral(src,sum,CV_64FC1);
	}else{
		if(sum.type()!=CV_64FC1){
			CV_Error(CV_StsBadArg,"[pixkit::filtering::FBF] sum type should be CV_64FC1.");
		}
	}


	////////////////////////////////////////////////////////////////////////// ok
	///// process
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			////////////////////////////////////////////////////////////////////////// ok
			bool	A=true,	// bottom, top, left, right; thus br: bottom-right
				B=true,
				C=true,
				D=true;
			if((i+SSFilterSize_h)>=src.rows){
				A=false;
			}
			if((j+SSFilterSize_w)>=src.cols){
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
				areaWidth	=	src.cols-j+SSFilterSize_w;
			}
			// height
			if(!C){
				areaHeight	=	i+SSFilterSize_h+1;
			}else if(!A){
				areaHeight	=	src.rows-i+SSFilterSize_h;
			}


			//////////////////////////////////////////////////////////////////////////
			///// get value
			// get positions
			int	y_up	=	i+SSFilterSize_h		+1,	// '+1' is the bias term of the integral image 
				y_dn	=	i-SSFilterSize_h-1		+1;
			if(!A){
				y_up	=	src.rows-1				+1;
			}
			int	x_left	=	j-SSFilterSize_w-1		+1,
				x_right	=	j+SSFilterSize_w		+1;
			if(!B){
				x_right	=	src.cols-1				+1;
			}
			// get values
			double	cTR_sum=0,cTR_sqsum=0,
				cTL_sum=0,cTL_sqsum=0,
				cBR_sum=0,cBR_sqsum=0,
				cBL_sum=0,cBL_sqsum=0;
			cTR_sum			=	sum.ptr<double>(y_up)[x_right];
			if(C&&D){
				cBL_sum		=	sum.ptr<double>(y_dn)[x_left];
			}
			if(D){
				cTL_sum		=	-sum.ptr<double>(y_up)[x_left];
			}
			if(C){
				cBR_sum		=	-sum.ptr<double>(y_dn)[x_right];
			}

			////////////////////////////////////////////////////////////////////////// ok
			///// get output
			if(tdst.type()==CV_32FC1){
				tdst.ptr<float>(i)[j]	=	(cTR_sum		+cTL_sum	+cBR_sum	+cBL_sum)/(areaHeight*areaWidth);
				tdst.ptr<float>(i)[j]	=	fabs(tdst.ptr<float>(i)[j]);
				CV_DbgAssert(tdst.ptr<float>(i)[j]>=0.&&tdst.ptr<float>(i)[j]<=255.);
			}else if(tdst.type()==CV_8UC1){
				tdst.ptr<uchar>(i)[j]	=	(cTR_sum		+cTL_sum	+cBR_sum	+cBL_sum)/(areaHeight*areaWidth);
				CV_DbgAssert(tdst.ptr<uchar>(i)[j]>=0.&&tdst.ptr<uchar>(i)[j]<=255.);
			}
		}
	}

	dst	=	tdst.clone();
	return true;
}
