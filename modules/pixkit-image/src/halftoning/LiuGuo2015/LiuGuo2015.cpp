//========================================================================
//
// LiuGuo2015.cpp
// Authors: Yun-Fu Liu (1), Jing-Ming Guo (2)
// Institutions: National Taiwan University of Science and Technology
// Date: Sept. 14, 2015.
// Email: yunfuliu@gmail.com, jmguo@seed.net.tw
//
// Dot-diffused halftoning with improved homogeneity, Copyright (c) 2015,  
// Yun-Fu Liu and Jing-Ming Guo, all rights reserved.
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

#include "../../../include/pixkit-image.hpp"

using namespace cv;
using namespace std;

pixkit::halftoning::dotdiffusion::CLiuGuo2015::CLiuGuo2015(std::string pthfname_resources){

	cmsize = cv::Size(8, 8);	// defined in paper.
	const	int	num_coe = 3;

	// read resources
	cv::FileStorage file(pthfname_resources, cv::FileStorage::READ);
	vector<Mat> buff(num_coe);
	vector<cv::SparseMat>	spmat(num_coe);
	Mat	ct_src;
	file["table_256"] >> ct_src;
	file["paramsmap"] >> spmat;
	file.release();

	// sparse matrix convert to mat
	for (int k = 0; k < num_coe; k++){
		spmat[k].convertTo(buff[k], CV_32FC3);
	}

	// read class tiling (CT)
	ctread(ct_src, cmsize, cct);
	ctread(ct_src, cv::Size(16, 16), cct_ori);

	// read parameters
	read_paramsmap(buff, paramsmap);

}
pixkit::halftoning::dotdiffusion::CLiuGuo2015::~CLiuGuo2015(){}

/************************************************************************/
/* INADD                                                                */
/************************************************************************/
bool pixkit::halftoning::dotdiffusion::CLiuGuo2015::ctread(const Mat &src, const Size cmsize, Mat &cct1b){

	cct1b.create(src.size(), src.type());
	int	fz	=	cmsize.area();
	static	float	L	=	255;
	// 
	float	tv;
	for (int i = 0; i<src.rows; i++){
		for (int j = 0; j<src.cols; j++){
			tv = ((float)src.ptr<uchar>(i)[j])	*	fz / (L + 1.);
			cct1b.ptr<uchar>(i)[j]	=	cvFloor(tv);
		}
	}

	return true;
}

bool pixkit::halftoning::dotdiffusion::CLiuGuo2015::read_paramsmap(std::vector<cv::Mat> &vec_src, vector<vector<CPARAMS>> &paramsmap){

	// get space
	paramsmap.resize(128);	// g
	for(int i=0;i<paramsmap.size();i++){
		paramsmap[i].resize(256);	// s
	}

	// get values
	for (int k = 0; k < 3; k++){
		double	value;
		for (int j = 0; j < paramsmap[0].size(); j++){		// s
			for (int i = 0; i < paramsmap.size(); i++){	// g			
				paramsmap[i][j].coe[k] = vec_src[k].ptr<float>(i)[j];
			}
		}
	}
	
	return true;
}

void pixkit::halftoning::dotdiffusion::CLiuGuo2015::getPointList(const Size imgSize){
	pointlist.resize(cmsize.area());
	for (int i = 0; i < imgSize.height; i++){
		for (int j = 0; j < imgSize.width; j++){
			pointlist[cct.ptr<uchar>(i%cct.rows)[j%cct.cols]].push_back(Point(j, i));
		}
	}
}

bool pixkit::halftoning::dotdiffusion::CLiuGuo2015::process(const cv::Mat &src1b, cv::Mat &dst1b){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src1b.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"");
	}
	if(src1b.empty()){
		CV_Error(CV_StsBadArg,"src is empty.");
	}
	if (pointlist.empty()){
		CV_Error(CV_StsInternal, "`pointlist` is empty. Please run `getPointList()` with the source image size to get the `pointlist`.");
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const float MAX_GRAYSCALE	=	255;
	const float MIN_GRAYSCALE	=	0;
	const int	&height	=	src1b.rows;
	const int	&width	=	src1b.cols;
	Mat	tdst1f=src1b.clone();	//	temporary space for the dst
	tdst1f.convertTo(tdst1f,CV_32FC1);

	//////////////////////////////////////////////////////////////////////////
	///// get ct
	// create CT for image processing
	Mat	order(src1b.size(),CV_8UC1);
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			order.ptr<uchar>(i)[j]=cct.ptr<uchar>(i%cct.rows)[j%cct.cols];
		}
	}
	Mat	order_ori(src1b.size(),CV_8UC1);
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			order_ori.ptr<uchar>(i)[j]=cct_ori.ptr<uchar>(i%cct_ori.rows)[j%cct_ori.cols];
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// dot diffusion process
	// setting
	int	DW_Size=3;	// size of diffused matrix
	int	hDW_Size=DW_Size/2;		
	// perform dot diffusion
	float	threshold,afa,beta;	
	for(int k=0;k<cmsize.area();k++){ // processing order
		for(int p=0;p<pointlist[k].size();p++){
			// get i and j;
			int i=pointlist[k][p].y;
			int j=pointlist[k][p].x;

			//////////////////////////////////////////////////////////////////////////
			///// get parameters
			const uchar	&srcv	=	src1b.ptr<uchar>(i)[j];
			uchar		&ordv	=	order_ori.ptr<uchar>(i)[j];
			if(srcv<128){
				threshold	=	paramsmap[srcv][ordv].coe[0];
				afa			=	paramsmap[srcv][ordv].coe[1];
				beta		=	paramsmap[srcv][ordv].coe[2];
			}else{
				threshold	=	paramsmap[MAX_GRAYSCALE-srcv][ordv].coe[0];
				afa			=	paramsmap[MAX_GRAYSCALE-srcv][ordv].coe[1];
				beta		=	paramsmap[MAX_GRAYSCALE-srcv][ordv].coe[2];
			}

			//////////////////////////////////////////////////////////////////////////
			// get error
			double	error;
			if(tdst1f.ptr<float>(i)[j]<threshold){
				error=tdst1f.ptr<float>(i)[j];
				tdst1f.ptr<float>(i)[j]=MIN_GRAYSCALE;
			}else{
				error=tdst1f.ptr<float>(i)[j]-MAX_GRAYSCALE;
				tdst1f.ptr<float>(i)[j]=MAX_GRAYSCALE;
			}

			//////////////////////////////////////////////////////////////////////////
			///// check whether it needs to diffuse error, for the sake of saving time
			if(afa==0.&&beta==0.){
				continue;
			}else{

				//////////////////////////////////////////////////////////////////////////
				///// get DW
				double	DW[9]={afa,beta,afa,beta,0.,beta,afa,beta,afa};

				// get fm
				double	fm=0.;					
				for(int m=-hDW_Size;m<=hDW_Size;m++){
					for(int n=-hDW_Size;n<=hDW_Size;n++){
						if(i+m>=0&&i+m<height&&j+n>=0&&j+n<width){	// in the image region
							if(order.ptr<uchar>(i+m)[j+n]>k){	// diffusible region
								fm+=DW[(m+hDW_Size)*DW_Size+(n+hDW_Size)];
							}
						}
					}
				}
				// diffuse
				if(fm!=0){
					for(int m=-hDW_Size;m<=hDW_Size;m++){
						for(int n=-hDW_Size;n<=hDW_Size;n++){
							if(i+m>=0&&i+m<height&&j+n>=0&&j+n<width){	// in the image region
								if(order.ptr<uchar>(i+m)[j+n]>k){	// diffusible region						
									tdst1f.ptr<float>(i+m)[j+n]+=error*DW[(m+hDW_Size)*DW_Size+(n+hDW_Size)]/fm;
								}
							}
						}
					}
				}
			}
		}
	}
	
	// copy from tdst1f to dst1b
	dst1b.create(tdst1f.size(),CV_8UC1);
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			dst1b.ptr<uchar>(i)[j]=cvRound(tdst1f.ptr<float>(i)[j]);
		}
	}
	
	return true;
}
