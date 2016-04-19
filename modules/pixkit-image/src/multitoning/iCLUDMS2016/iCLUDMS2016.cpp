//========================================================================
//
// iCLUDMS2016.cpp
// Authors: Yun-Fu Liu (1), Jing-Ming Guo (2)
// Institutions: National Taiwan University of Science and Technology
// Date: April 19, 2016.
// Email: yunfuliu@gmail.com, jmguo@seed.net.tw
//
// Clustered-Dot Screen Design for Digital Multitoning, Copyright (c) 2016,  
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

using namespace	cv;
using namespace std;

Mat	get_representable_colors(const int nTones){
	// get representable colors.
	// input:
	//		nTones: number of colors.
	// return: 
	//		a 32FC1 Mat of absorptance; from the smallest value to largest value, alpha_1 to alpha_S.

	Mat	vec1f(Size(nTones,1),CV_32FC1);
	for(int i=0;i<nTones;i++){
		vec1f.ptr<float>(0)[i]	=	(float)i/((float)(nTones-1.));
	}
	return	vec1f.clone();
}
map<float,int> get_map_absorptance_to_idxDA(const int nTones){
	// get map between absorptance and dither array index.
	// input: 
	//		nTones: number of tones. 
	// return:
	//		the map.

	map<float,int>	mapfi;
	Mat	tones1f	=	get_representable_colors(nTones);
	for(int i=0;i<nTones-1;i++){ // the number of dither array
		mapfi.insert(pair<float,int>(tones1f.ptr<float>(0)[i],i));
	}	
	return	mapfi;
}
inline int	cirCC(int cor,int limitation){
	// coordinate correction.
	// input:
	//		cor: coordinate.
	//		limitation: image boundary.
	// return:
	//		corrected coordinate. 

	int	tempv	=	cor%limitation;
	if(tempv>=0){
		return	tempv;
	}else{	// if(tempv<0)
		return	limitation+tempv;
	}
}
bool get_mode_param(bool &is_swap,const int mode,int &m,int &n,float &a0,float &a1,const Mat &tones1f,const Mat &dst1f,const Mat &init1f,const int &i,const int &j,float &new_cur_v,float &new_nei_v,float polarity,bool is_first_gray){
	// get parameters as per the given mode. 

	//////////////////////////////////////////////////////////////////////////
	///// position
	// swap, mode=0~7
	if(mode==0){
		m=1;	n=-1;
	}else if(mode==1){
		m=1;	n=0;
	}else if(mode==2){
		m=1;	n=1;
	}else if(mode==3){
		m=0;	n=1;
	}else if(mode==4){
		m=-1;	n=1;
	}else if(mode==5){
		m=-1;	n=0;
	}else if(mode==6){
		m=-1;	n=-1;
	}else if(mode==7){
		m=0;	n=-1;
	}else{
		m=0;	n=0;
	}

	//////////////////////////////////////////////////////////////////////////
	///// get a0 and a1
	if(mode>=0&&mode<=7){	// swap
		is_swap=true;
		new_cur_v	=	dst1f.ptr<float>(cirCC(i+m,dst1f.rows))[cirCC(j+n,dst1f.cols)];
		new_nei_v	=	dst1f.ptr<float>(i)[j];
		if((polarity*new_cur_v)>=(polarity*init1f.ptr<float>(i)[j])&&(polarity*new_nei_v)>=(polarity*init1f.ptr<float>(cirCC(i+m,init1f.rows))[cirCC(j+n,init1f.cols)])){
			a0	=	new_cur_v	-	new_nei_v;
		}else{
			a0	=	0.;	// to inherit
		}
		a1	=	-a0;

	}else{	// toggle
		is_swap=false;
		new_cur_v	=	tones1f.ptr<float>(0)[mode-8];
		new_nei_v	=	0.;
		if((polarity*new_cur_v)>=(polarity*init1f.ptr<float>(i)[j])){
			a0	=	new_cur_v	-	dst1f.ptr<float>(i)[j];
		}else{
			a0	=	0.;	// to inherit
		}
		a1=0.;
	}

	return true;
}
bool process(const cv::Mat &src1b,const cv::Mat &init1f,cv::Mat &dst1f,const Mat &c_pp_update,const Mat &c_pp_init,const Mat &tones,float polarity,bool is_first_gray){
	// intra- and inter-iterations introduced in paper. 

	//////////////////////////////////////////////////////////////////////////
	/// exceptions
	if(src1b.type()!=CV_8UC1){
		CV_Assert(false);
	}
	if(c_pp_update.total()!=c_pp_init.total()){
		CV_Assert(false);
	}
	if(c_pp_update.empty()||c_pp_init.empty()){
		CV_Assert(false);
	}
	int FilterSize	=	c_pp_init.rows;
	if(FilterSize==1){
		CV_Assert(false);
	}else if(FilterSize%2==0){
		CV_Assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	/// initialization
	const	int	&height	=	src1b.rows;
	const	int	&width	=	src1b.cols;
	if(dst1f.empty()){	// offered by outside
		CV_Assert(false);
	}
	int		cnt_K_intra=0,cnt_K_inter=0;
	dst1f.create(src1b.size(),CV_32FC1);
	int	exFS=FilterSize;
	int	halfFS=cvFloor((float)FilterSize/2.);

	//////////////////////////////////////////////////////////////////////////
	/// get source image
	Mat	src1f(src1b.size(),CV_32FC1);
	src1b.convertTo(src1f,CV_32FC1);
	/// Change grayscale to absorptance
	src1f	=	1.-src1f/255.;

	//////////////////////////////////////////////////////////////////////////
	///// iterative process
	bool is_alter_any_halftone_pixel = false;
	while(true){

		is_alter_any_halftone_pixel = false;

		//////////////////////////////////////////////////////////////////////////
		/// get error matrix
		Mat	em1f(src1b.size(),CV_32FC1);
		em1f	=	dst1f	-	src1f;
		/// get cross correlation
		Mat	crosscoe1d(Size(width,height),CV_64FC1);
		crosscoe1d.setTo(0);
		Mat	delta_cpe01d(Size(width,height),CV_64FC1);
		delta_cpe01d.setTo(0);
		for(int i=0;i<crosscoe1d.rows;i++){
			for(int j=0;j<crosscoe1d.cols;j++){
				double	cpe0_init=0.,cpe0_update=0.;
				for(int m=i-halfFS;m<=i+halfFS;m++){
					for(int n=j-halfFS;n<=j+halfFS;n++){
						cpe0_init+=em1f.ptr<float>(cirCC(m,height))[cirCC(n,width)]*c_pp_init.ptr<double>(halfFS+m-i)[halfFS+n-j];
						cpe0_update+=em1f.ptr<float>(cirCC(m,height))[cirCC(n,width)]*c_pp_update.ptr<double>(halfFS+m-i)[halfFS+n-j];
					}
				}
				crosscoe1d.ptr<double>(i)[j]	=	cpe0_update;
				delta_cpe01d.ptr<double>(i)[j]	=	cpe0_init	-	cpe0_update;
			}
		}

		//////////////////////////////////////////////////////////////////////////
		///// process
		int		BenefitPixelNumber;
		int		nModes	=	8+tones.cols;	// number of modes, to all possibilities, thus 8 (for swap) + different tones
		Mat		dE1d(Size(nModes,1),CV_64FC1);
		while(1){

			BenefitPixelNumber=0;
			for(int i=0;i<height;i++){	// entire image
				for(int j=0;j<width;j++){

					//////////////////////////////////////////////////////////////////////////
					// = = = = = trial part = = = = = //
					// initialize err		0: original err, 0~7: Swap, >=8: toggle.
					// 0 1 2
					// 7 x 3
					// 6 5 4
					dE1d.setTo(0.);	// original error =0
					// change the delta error as per different replacement methods
					for(int mode=0;mode<nModes;mode++){

						// get parameters 
						int		m,n;
						float	a0=0.,a1=0.;
						bool is_swap=false;
						float	new_cur_v,new_nei_v;
						get_mode_param(is_swap,mode,m,n,a0,a1,tones,dst1f,init1f,i,j,new_cur_v,new_nei_v,polarity,is_first_gray);	// set position

						// get error
						double	theta_homog	=	(a0*a0+a1*a1)	*c_pp_update.ptr<double>(halfFS)[halfFS]	
						+2.*a0		*	crosscoe1d.ptr<double>(i)[j]
						+2.*a0*a1	*	c_pp_update.ptr<double>(halfFS+m)[halfFS+n]
						+2.*a1		*	crosscoe1d.ptr<double>(cirCC(i+m,height))[cirCC(j+n,width)];
						double	theta_clust	=	2.*a0*delta_cpe01d.ptr<double>(i)[j]	+	2.*a1*delta_cpe01d.ptr<double>(cirCC(i+m,height))[cirCC(j+n,width)];
						dE1d.ptr<double>(0)[mode]	=	theta_homog	-	theta_clust;
					}

					//////////////////////////////////////////////////////////////////////////
					///// get minimum delta error and its position
					int		tempMinNumber	=0;
					double	tempMindE		=dE1d.ptr<double>(0)[0];	// original error =0
					for(int x=1;x<nModes;x++){
						if(dE1d.ptr<double>(0)[x]<tempMindE){	// get smaller error only
							tempMindE		=dE1d.ptr<double>(0)[x];
							tempMinNumber	=x;
						}
					}

					//////////////////////////////////////////////////////////////////////////
					// = = = = = update part = = = = = //
					if(tempMindE<0.){	// error is reduced

						is_alter_any_halftone_pixel	=	true;

						// get position, and check swap position
						int nm,nn;
						float	a0=0.,a1=0.;
						bool is_swap=false;
						float	new_cur_v,new_nei_v;
						get_mode_param(is_swap,tempMinNumber,nm,nn,a0,a1,tones,dst1f,init1f,i,j,new_cur_v,new_nei_v,polarity,is_first_gray);

						// update current hft position
						dst1f.ptr<float>(i)[j]	=	new_cur_v;

						// update
						for(int m=-halfFS;m<=halfFS;m++){
							for(int n=-halfFS;n<=halfFS;n++){
								crosscoe1d.ptr<double>(cirCC(i+m,height))[cirCC(j+n,width)]+=a0*c_pp_update.ptr<double>(halfFS+m)[halfFS+n];
							}
						}

						// update
						if(is_swap){	// swap case
							// update swapped hft position
							dst1f.ptr<float>(cirCC(i+nm,height))[cirCC(j+nn,width)]	=	new_nei_v;
							// update cross correlation
							for(int m=-halfFS;m<=halfFS;m++){
								for(int n=-halfFS;n<=halfFS;n++){
									crosscoe1d.ptr<double>(cirCC(i+m+nm,height))[cirCC(j+n+nn,width)]+=a1*c_pp_update.ptr<double>(halfFS+m)[halfFS+n];
								}
							}
						}
						BenefitPixelNumber++;
					} // end of entire image
				}
			}
			cnt_K_intra+=1;

			if(BenefitPixelNumber==0){
				break;
			}
		}
		cnt_K_inter+=1;

		if(!is_alter_any_halftone_pixel){
			break;
		}
	}

	return true;
}
bool gen_mask_at_each_grayscale(vector<Mat> &vec_dst1f,const Mat &dst1281f,const bool is_going_lighter,const int daSize,const Mat &hvs_model_cpp_update,const Mat &hvs_model_cpp_init,const Mat &tones){

	int	extreme_tone;
	float	polarity;
	if(is_going_lighter){
		extreme_tone	=	255;
		polarity		=	-1.;
	}else{
		extreme_tone	=	0;
		polarity		=	1.;
	}

	//////////////////////////////////////////////////////////////////////////
	///// process for masks of from 0 to 128
	cout<<"Generating screens..."<<endl;
	bool	is_first_gray	=	true;
	Mat	src1b(Size(daSize,daSize),CV_8UC1),pre_dst1f,dst1f(Size(daSize,daSize),CV_32FC1);
	if(is_going_lighter){
		dst1f.setTo(1.);
	}else{
		dst1f.setTo(0.);
	}
	
	for (int eta = 128; is_going_lighter?(eta <= extreme_tone):(eta >= extreme_tone); eta=eta-(polarity*1.)){		// eta is defined as the grayscale in paper
		cout << "\tgrayscale = " << (int)eta << endl;

		//////////////////////////////////////////////////////////////////////////
		///// init
		src1b.setTo(eta);
		pre_dst1f = dst1f.clone(); // recode the result of dst(g-1)

		//////////////////////////////////////////////////////////////////////////
		///// process
		// get initial halftone pattern
		if(is_first_gray){
			dst1f	=	dst1281f.clone();
		}
		// get previous halftone image for stacking constraint
		process(src1b,pre_dst1f,dst1f,hvs_model_cpp_update,hvs_model_cpp_init,tones,polarity,is_first_gray);
		is_first_gray	=	false;

		// push to vec_dst1f
		vec_dst1f[eta]	=	dst1f.clone();
	}

	return true;
}
bool pixkit::multitoning::ordereddithering::iCLUDMS2016_genDitherArray(std::vector<cv::Mat> &vec_DA1b, int daSize, int nTones,float sd_init,float sd_update){
	// generate screens.
	// input:
	//		daSize: screen size.
	//		nTones: representable number of tones.
	//		sd_init: standard deviation_init.
	//		sd_update: standard deviation_update.
	// output:
	//		vec_DA1b: screens. 

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(nTones<2){
		CV_Error(CV_StsBadArg,"nColors should >= 2.");
	}
	if(daSize<1){
		CV_Error(CV_StsBadArg,"daSize should >= 1.");
	}
	if(sd_update<sd_init){
		CV_Assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	///// const parameters
	const	uchar	MAX_GRAYSCALE	=	255;
	const	uchar	MIN_GRAYSCALE	=	0;

	//////////////////////////////////////////////////////////////////////////
	// tones
	Mat	tones	=	get_representable_colors(nTones);
	// dither arrays
	vec_DA1b.resize(nTones-1);
	for(int di=0;di<vec_DA1b.size();di++){
		vec_DA1b[di].create(Size(daSize, daSize), CV_8UC1);
		vec_DA1b[di].setTo((float)MAX_GRAYSCALE);
	}

	//////////////////////////////////////////////////////////////////////////
	///// get cpp
	Mat	hvs_model_cpp_init,hvs_model_cpp_update;
	int	ksize	=	cvRound(sd_update*6.)+3;
	if(ksize%2==0){
		ksize+=1;
	}
	hvs_model_cpp_init	=	getGaussianKernel(ksize,sd_init,CV_64FC1);
	hvs_model_cpp_update	=	getGaussianKernel(ksize,sd_update,CV_64FC1);
	mulTransposed(hvs_model_cpp_init,hvs_model_cpp_init,false);
	mulTransposed(hvs_model_cpp_update,hvs_model_cpp_update,false);

	//////////////////////////////////////////////////////////////////////////
	///// generate the mask at gray scale 128. It will be treated as the initial pattern for the coming masks. 
	cv::Mat src1b, dst1f;
	src1b.create(Size(daSize, daSize), CV_8UC1);
	dst1f.create(Size(daSize, daSize), CV_32FC1);
	cout<<"Generate the first mask at gray scale 128 ..."<<endl;
	srand(0);
	for(int i=0;i<dst1f.rows;i++){
		for(int j=0;j<dst1f.cols;j++){
			float	prob	=	(float)rand()/(float)RAND_MAX;
			if(prob<0.5){
				dst1f.ptr<float>(i)[j]=1.;
			}else{
				dst1f.ptr<float>(i)[j]=0.;
			}
		}
	}
	src1b.setTo(128);
	Mat	pre_dst1f	=	dst1f.clone();
	pre_dst1f.setTo(0);
	process(src1b,pre_dst1f,dst1f,hvs_model_cpp_update,hvs_model_cpp_init,tones,1.,true);
	Mat	dst1281f	=	dst1f.clone();	// result at 128.

	//////////////////////////////////////////////////////////////////////////
	// vec_dst1f
	vector<Mat>	vec_dst1f(256);	// total number of masks. 
	for(int i=0;i<256;i++){
		vec_dst1f[i].create(Size(daSize,daSize),CV_32FC1);
		vec_dst1f[i].setTo(0.);
	}
	// generation
	gen_mask_at_each_grayscale(vec_dst1f,dst1281f,false,daSize,hvs_model_cpp_update,hvs_model_cpp_init,tones);
	gen_mask_at_each_grayscale(vec_dst1f,dst1281f,true,daSize,hvs_model_cpp_update,hvs_model_cpp_init,tones);

	//////////////////////////////////////////////////////////////////////////
	///// record position that the white point is firstly toggled
	map<float,int>	mapfi	=	get_map_absorptance_to_idxDA(nTones);
	for (int i = 0; i < vec_dst1f[0].rows; i++){
		for (int j = 0; j < vec_dst1f[0].cols; j++){
			vec_dst1f[0].ptr<float>(i)[j]	=	1.;
			vec_dst1f[255].ptr<float>(i)[j]	=	0.;
		}
	}
	// yield screen
	for(int gray=1;gray<=255;gray++){
		for (int i = 0; i < vec_dst1f[gray].rows; i++){
			for (int j = 0; j < vec_dst1f[gray].cols; j++){
				if(vec_dst1f[gray].ptr<float>(i)[j]!=vec_dst1f[gray-1].ptr<float>(i)[j]){ // changed
					// get index
					int index	= mapfi[vec_dst1f[gray].ptr<float>(i)[j]]; // absorptance to index of screen
					for(int iidx=index;iidx>=0;iidx--){
						vec_DA1b[iidx].ptr<uchar>(i)[j]	=	gray;
					}
				}
			}
		}
	}

	return true;
}
bool pixkit::multitoning::ordereddithering::iCLUDMS2016(const cv::Mat &src1b, const std::vector<cv::Mat> &vec_DA1b,cv::Mat &dst1b){	
	// do screening with the generated screen. 
	// input:
	//		src1b: source image. 
	//		vec_DA1b: screens. 
	// output:
	//		dst1b: output image. 

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const	int	MAX_GRAYSCALE	=	255;
	int nTones	=	vec_DA1b.size()+1;	// representable number of tones
	Mat	tones1f	=	get_representable_colors(nTones);	// from range of 0 to 1
	tones1f	=	(1.-tones1f)	* ((float) MAX_GRAYSCALE);	// absorptance to gray scale
	//////////////////////////////////////////////////////////////////////////
	dst1b.create(src1b.size(),src1b.type());
	for(int i=0;i<src1b.rows;i++){
		for(int j=0;j<src1b.cols;j++){
			const	uchar	&curr_tone	=	src1b.ptr<uchar>(i)[j];
			bool	ishalftoned	=	false;
			for(int g=0;g<nTones-1;g++){ // try every dither array
				const	uchar	&thres	=	vec_DA1b[g].ptr<uchar>(i%vec_DA1b[g].rows)[j%vec_DA1b[g].cols];
				if(curr_tone>=thres){
					dst1b.ptr<uchar>(i)[j]	=	cvRound(tones1f.ptr<float>(0)[g]);
					ishalftoned	=	true;
					break;
				}
			}
			if(!ishalftoned){
				dst1b.ptr<uchar>(i)[j]	=	cvRound(tones1f.ptr<float>(0)[nTones-1]);
			}
		}
	}
	return true;
}