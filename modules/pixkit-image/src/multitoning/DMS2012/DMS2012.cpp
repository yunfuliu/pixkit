#include "../../../include/pixkit-image.hpp"

using namespace	cv;
using namespace std;

Mat	get_presentable_color(const int nColors){
	// get presentable colors
	// input: number of colors
	// output: 32FC1 Mat of absorption, from small to large value, afa_1 to afa_S
	Mat	vec1f(Size(nColors,1),CV_32FC1);
	for(int i=0;i<nColors;i++){
		vec1f.ptr<float>(0)[i]	=	(float)i/((float)(nColors-1.));
	}
	return	vec1f.clone();
}
map<float,int> get_map_of_absorption_to_daindex(const int nColors){
	map<float,int>	mapfi;
	Mat	tones1f	=	get_presentable_color(nColors);
	for(int i=0;i<nColors-1;i++){ // the number of dither array
		mapfi.insert(pair<float,int>(tones1f.ptr<float>(0)[i+1],i));
	}
	return	mapfi;
}
// coordination correction
inline int	cirCC(int cor,int limitation){
	int	tempv	=	cor%limitation;
	if(tempv>=0){
		return	tempv;
	}else{	// if(tempv<0)
		return	limitation+tempv;
	}
}
bool get_mode_param(bool &is_swap,const int mode,int &m,int &n,float &a0,float &a1,const Mat &tones1f,const Mat &dst1f,const Mat &init1f,const int &i,const int &j,float &new_cur_v,float &new_nei_v){

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
		if(new_cur_v>=init1f.ptr<float>(i)[j]&&new_nei_v>=init1f.ptr<float>(cirCC(i+m,dst1f.rows))[cirCC(j+n,dst1f.cols)]){
			a0	=	new_cur_v	-	new_nei_v;
		}else{
			a0	=	0.;	// to inherit
		}
		a1			=	-a0;
	}else{	// toggle
		is_swap=false;
		new_cur_v	=	tones1f.ptr<float>(0)[mode-8];
		new_nei_v	=	0.;
		if(new_cur_v>=init1f.ptr<float>(i)[j]){
			a0	=	new_cur_v	-	dst1f.ptr<float>(i)[j];
		}else{
			a0	=	0.;	// to inherit
		}
		a1=0.;
	}

	return true;
}

// SCDBS
bool SCDBS(const cv::Mat &src1b, const cv::Mat &init1f,const bool is_first_grayscale,cv::Mat &dst1f,double *c_ppData,int FilterSize,const Mat &tones){

	//////////////////////////////////////////////////////////////////////////
	/// exceptions
	if(src1b.type()!=CV_8UC1){
		assert(false);
	}
	if(FilterSize==1){
		assert(false);
	}else if(FilterSize%2==0){
		assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	/// initialization
	const	int	&height	=	src1b.rows;
	const	int	&width	=	src1b.cols;
	dst1f.create(src1b.size(),CV_32FC1);

	//////////////////////////////////////////////////////////////////////////
	/// get autocorrelation.
	int	exFS=FilterSize;
	int	halfFS=cvFloor((float)FilterSize/2.);
	double	**	c_pp		=	new	double	*	[exFS];
	for(int i=0;i<exFS;i++){
		c_pp[i]=&c_ppData[i*exFS];
	}

	//////////////////////////////////////////////////////////////////////////
	/// load original image
	Mat	src1f(src1b.size(),CV_32FC1);
	src1b.convertTo(src1f,CV_32FC1);
	// get initial image
	dst1f	=	init1f.clone();

	//////////////////////////////////////////////////////////////////////////
	/// Change grayscale to absorb
	src1f	=	1.-src1f/255.;
	/// get error matrix
	Mat	em1f(src1b.size(),CV_32FC1);
	em1f	=	dst1f	-	src1f;
	/// get cross correlation
	Mat	crosscoe1d(Size(width,height),CV_64FC1);
	crosscoe1d.setTo(0);
	for(int i=0;i<crosscoe1d.rows;i++){
		for(int j=0;j<crosscoe1d.cols;j++){
			for(int m=i-halfFS;m<=i+halfFS;m++){
				for(int n=j-halfFS;n<=j+halfFS;n++){
					crosscoe1d.ptr<double>(i)[j]+=em1f.ptr<float>(cirCC(m,height))[cirCC(n,width)]*c_pp[halfFS+m-i][halfFS+n-j];
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// DBS process
	int		BenefitPixelNumber;
	int		nModes	=	8+tones.cols;	// number of modes, to all possibilities, thus 8 (swap) + different tones
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
					get_mode_param(is_swap,mode,m,n,a0,a1,tones,dst1f,init1f,i,j,new_cur_v,new_nei_v);	// set position

					// get error
					dE1d.ptr<double>(0)[mode]=(a0*a0+a1*a1)	*c_pp[halfFS][halfFS]	
					+2.*a0	*crosscoe1d.ptr<double>(i)[j]
					+2.*a0*a1	*	c_pp[halfFS+m][halfFS+n]
					+2.*a1			*	crosscoe1d.ptr<double>(cirCC(i+m,height))[cirCC(j+n,width)];
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

					// get position, and check swap position
					int nm,nn;
					float	a0=0.,a1=0.;
					bool is_swap=false;
					float	new_cur_v,new_nei_v;
					get_mode_param(is_swap,tempMinNumber,nm,nn,a0,a1,tones,dst1f,init1f,i,j,new_cur_v,new_nei_v);

					// update current hft position
					dst1f.ptr<float>(i)[j]	=	new_cur_v;

					// update
					for(int m=-halfFS;m<=halfFS;m++){
						for(int n=-halfFS;n<=halfFS;n++){
							crosscoe1d.ptr<double>(cirCC(i+m,height))[cirCC(j+n,width)]+=a0*c_pp[halfFS+m][halfFS+n];
						}
					}

					// update
					if(is_swap){	// swap case
						// update swapped hft position
						dst1f.ptr<float>(cirCC(i+nm,height))[cirCC(j+nn,width)]	=	new_nei_v;
						// update cross correlation
						for(int m=-halfFS;m<=halfFS;m++){
							for(int n=-halfFS;n<=halfFS;n++){
								crosscoe1d.ptr<double>(cirCC(i+m+nm,height))[cirCC(j+n+nn,width)]+=a1*c_pp[halfFS+m][halfFS+n];
							}
						}
					}
					BenefitPixelNumber++;
				} // end of entire image
			}
		}
		if(BenefitPixelNumber==0){
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// release space
	delete	[]	c_pp;

	return true;
}

//////////////////////////////////////////////////////////////////////////
bool pixkit::multitoning::ordereddithering::DMS2012_genDitherArray(std::vector<cv::Mat> &vec_DA1b, int daSize, int nColors){
	// the N defined in paper is supposed as daSize in this program.

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(nColors<2){
		CV_Error(CV_StsBadArg,"nColors should >= 2.");
	}
	if(daSize<1){
		CV_Error(CV_StsBadArg,"daSize should >= 1.");
	}

	//////////////////////////////////////////////////////////////////////////
	///// init
	const	uchar	MAX_GRAYSCALE	=	255;	// defined R in paper
	const	uchar	MIN_GRAYSCALE	=	0;
	const	float	afa_1			=	0;		// parameter defined in paper

	//////////////////////////////////////////////////////////////////////////
	// tones
	Mat	tones	=	get_presentable_color(nColors);
	// dither arrays
	vec_DA1b.resize(nColors-1);
	for(int di=0;di<vec_DA1b.size();di++){
		vec_DA1b[di].create(Size(daSize, daSize), CV_8UC1);
		vec_DA1b[di].setTo(MIN_GRAYSCALE);
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization 
	cv::Mat src1b, dst1f;
	src1b.create(Size(daSize, daSize), CV_8UC1);
	dst1f.create(Size(daSize, daSize), CV_32FC1);
	src1b.setTo(MIN_GRAYSCALE);
	dst1f.setTo(MIN_GRAYSCALE);
	// get coe
	Mat	hvs_model_cpp;
	pixkit::halftoning::ungrouped::generateTwoComponentGaussianModel(hvs_model_cpp,43.2,38.7,0.02,0.06); // defined in their paper

	//////////////////////////////////////////////////////////////////////////
	///// process for masks of 0 to 255
	Mat	pre_dst1f;
	Mat	init1f(Size(daSize, daSize), CV_32FC1);
	init1f.setTo(afa_1);	// as described in paper, this value should be zero.
	cout<<"Generating screens..."<<endl;
	for (int eta = MAX_GRAYSCALE; eta >= MIN_GRAYSCALE; eta--){		// eta is defined as a grayscale in paper
		cout << "\tgrayscale = " << (int)eta << endl;

		//////////////////////////////////////////////////////////////////////////
		///// init
		src1b.setTo(eta);
		pre_dst1f = dst1f.clone(); // recode the result of dst(g-1)

		//////////////////////////////////////////////////////////////////////////
		///// process
		// check whether it's the first grayscale
		bool	first_grayscale	=false;
		if(cv::sum(init1f)[0]==0){
			first_grayscale	=	true;
		}	
		// process
		SCDBS(src1b,init1f,first_grayscale,dst1f,&(double&)hvs_model_cpp.data[0],hvs_model_cpp.rows,tones);

		//////////////////////////////////////////////////////////////////////////
		///// get init
		init1f	=	dst1f.clone();

		//////////////////////////////////////////////////////////////////////////
		///// record position that the white point is firstly toggled
		map<float,int>	mapfi	=	get_map_of_absorption_to_daindex(nColors);
		for (int i = 0; i < dst1f.rows; i++){
			for (int j = 0; j < dst1f.cols; j++){
				if(dst1f.ptr<float>(i)[j]!=pre_dst1f.ptr<float>(i)[j]){ // changed
					int index = mapfi[dst1f.ptr<float>(i)[j]]; // absorption to index of da
					for(int iidx=index;iidx<=nColors-2;iidx++){
						vec_DA1b[iidx].ptr<uchar>(i)[j]	=	eta;
					}
				}
			}
		}
	}

	return true;
}
bool pixkit::multitoning::ordereddithering::DMS2012(const cv::Mat &src1b, const std::vector<cv::Mat> &vec_DA1b,cv::Mat &dst1b){	
	// perform DMS screening process
	// output: dst1b

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const	int	MAX_GRAYSCALE	=	255;
	int nColors	=	vec_DA1b.size()+1;	// presentable number of colors
	Mat	tones1f	=	get_presentable_color(nColors);	// from range of 0 to 1
	tones1f	=	(1.-tones1f)	* ((float) MAX_GRAYSCALE);	// absorption to grayscale

	//////////////////////////////////////////////////////////////////////////
	dst1b.create(src1b.size(),src1b.type());
	for(int i=0;i<src1b.rows;i++){
		for(int j=0;j<src1b.cols;j++){
			const	uchar	&curr_tone	=	src1b.ptr<uchar>(i)[j];
			bool	ishalftoned	=	false;
			for(int g=0;g<nColors-1;g++){ // try every dither array
				const	uchar	&thres	=	vec_DA1b[g].ptr<uchar>(i%vec_DA1b[g].rows)[j%vec_DA1b[g].cols];
				if(curr_tone>=thres){
					dst1b.ptr<uchar>(i)[j]	=	cvRound(tones1f.ptr<float>(0)[g]);
					ishalftoned	=	true;
					break;
				}
			}
			if(!ishalftoned){
				dst1b.ptr<uchar>(i)[j]	=	cvRound(tones1f.ptr<float>(0)[nColors-1]);
			}
		}
	}
	return true;
}