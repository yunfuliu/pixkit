#include "../../../include/pixkit-image.hpp"

using namespace	cv;
using namespace std;

// coordination correction
inline int	cirCC(int cor,int limitation){
	int	tempv	=	cor%limitation;
	if(tempv>=0){
		return	tempv;
	}else{	// if(tempv<0)
		return	limitation+tempv;
	}
}

// SCDBS
bool SCDBS(const cv::Mat &src1b, const cv::Mat &init1b,const bool first_grayscale,cv::Mat &dst1b,double *c_ppData,int FilterSize){

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
	Mat	dst1f;
	dst1f.create(src1b.size(),CV_32FC1);

	//////////////////////////////////////////////////////////////////////////
	/// get autocorrelation.
	int	exFS=FilterSize;
	int	tempFS=FilterSize/2;
	double	**	c_pp		=	new	double	*	[exFS];
	for(int i=0;i<exFS;i++){
		c_pp[i]=&c_ppData[i*exFS];
	}

	//////////////////////////////////////////////////////////////////////////
	/// load original image
	Mat	src1f(src1b.size(),CV_32FC1);
	src1b.convertTo(src1f,CV_32FC1);
	// get initial image
	init1b.convertTo(dst1f,CV_32FC1);

	//////////////////////////////////////////////////////////////////////////
	/// Change grayscale to absorb
	src1f	=	1.-src1f/255.;
	dst1f	=	1.-dst1f/255.;

	/// get error matrix
	Mat	em1f(src1b.size(),CV_32FC1);
	em1f	=	dst1f	-	src1f;

	/// get cross correlation
	Mat	crosscoe1d(Size(width,height),CV_64FC1);
	crosscoe1d.setTo(0);
	for(int i=0;i<crosscoe1d.rows;i++){
		for(int j=0;j<crosscoe1d.cols;j++){
			for(int m=i-tempFS;m<=i+tempFS;m++){
				for(int n=j-tempFS;n<=j+tempFS;n++){
					crosscoe1d.ptr<double>(i)[j]+=em1f.ptr<float>(cirCC(m,height))[cirCC(n,width)]*c_pp[tempFS+m-i][tempFS+n-j];
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// DBS process
	int		BenefitPixelNumber;
	double	dE[10],a0[10],a1[10];
	while(1){
		BenefitPixelNumber=0;
		for(int i=0;i<height;i++){	// entire image
			for(int j=0;j<width;j++){

				//////////////////////////////////////////////////////////////////////////
				// check whether the current position is needed to be processed or not
				if(!first_grayscale){
					if(init1b.ptr<uchar>(i)[j]==255){	// inherit
						float	initv	=	1 - (float)init1b.ptr<uchar>(i)[j]/255,	// change to absorb for comparison
								dstv	=	dst1f.ptr<float>(i)[j];
						CV_DbgAssert(initv==dstv);	// these two should be the same
						continue;	// ignore.
					}
				}

				//////////////////////////////////////////////////////////////////////////
				// = = = = = trial part = = = = = //
				// initialize psnr		0: original psnr, 1~8: Swap, 9: Toggel.
				// 8 1 2
				// 7 0 3
				// 6 5 4	
				for(int m=0;m<10;m++){
					dE[m]=0.;	// original error =0
					a0[m]=0.;
					a1[m]=0.;
				}
				// change the delta error as per different replacement methods
				for(int mode=1;mode<10;mode++){
					int		m,n;
					if(mode>=1&&mode<=8){
						// set position
						if(mode==1){
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
						}else if(mode==8){
							m=1;	n=-1;
						}

						//////////////////////////////////////////////////////////////////////////
						// get dE
						if(!first_grayscale){
							if(init1b.ptr<uchar>(cirCC(i+m,height))[cirCC(j+n,width)]==255){	// inherit
								continue;	// ignore 
							}
						}
						// get a0 and a1
						if(dst1f.ptr<float>(i)[j]==1){
							a0[mode]=-1;
						}else{
							a0[mode]=1;
						}
						if(dst1f.ptr<float>(cirCC(i+m,height))[cirCC(j+n,width)]==1){
							a1[mode]=-1;
						}else{
							a1[mode]=1;
						}
						// get error
						if(dst1f.ptr<float>(i)[j]!=dst1f.ptr<float>(cirCC(i+m,height))[cirCC(j+n,width)]){
							dE[mode]=(a0[mode]*a0[mode]+a1[mode]*a1[mode])	*c_pp[tempFS][tempFS]	+	2.*a0[mode]	*crosscoe1d.ptr<double>(i)[j]		
							+2.*a0[mode]*a1[mode]	*	c_pp[tempFS+m][tempFS+n]	
							+2.*a1[mode]			*	crosscoe1d.ptr<double>(cirCC(i+m,height))[cirCC(j+n,width)];
						}
					}else if(mode==9){
						if(dst1f.ptr<float>(i)[j]==1){
							a0[mode]=-1;
						}else{
							a0[mode]=1;
						}
						dE[mode]=	c_pp[tempFS][tempFS]	+	2.*a0[mode]*crosscoe1d.ptr<double>(i)[j];
					}
				}

				//////////////////////////////////////////////////////////////////////////
				///// get minimum delta error and its position
				int		tempMinNumber	=0;
				double	tempMindE		=dE[0];	// original error =0
				for(int x=1;x<10;x++){
					if(dE[x]<tempMindE){	// get smaller error only
						tempMindE		=dE[x];
						tempMinNumber	=x;
					}
				}

				//////////////////////////////////////////////////////////////////////////
				// = = = = = update part = = = = = //
				if(tempMindE<0.){	// error is reduced
					// update current hft position
					dst1f.ptr<float>(i)[j]	=	saturate_cast<float>(1.-dst1f.ptr<float>(i)[j]);
					if(tempMinNumber>=1&&tempMinNumber<=8){	// swap case
						// get position, and check swap position
						int nm,nn;
						if(tempMinNumber==1){
							nm=1;	nn=0;
						}else if(tempMinNumber==2){
							nm=1;	nn=1;
						}else if(tempMinNumber==3){
							nm=0;	nn=1;
						}else if(tempMinNumber==4){
							nm=-1;	nn=1;
						}else if(tempMinNumber==5){
							nm=-1;	nn=0;
						}else if(tempMinNumber==6){
							nm=-1;	nn=-1;
						}else if(tempMinNumber==7){
							nm=0;	nn=-1;
						}else if(tempMinNumber==8){
							nm=1;	nn=-1;
						}
						// update swapped hft position
						dst1f.ptr<float>(cirCC(i+nm,height))[cirCC(j+nn,width)]	=	saturate_cast<float>(1.	-	dst1f.ptr<float>(cirCC(i+nm,height))[cirCC(j+nn,width)]);
						
						// update cross correlation
						for(int m=-tempFS;m<=tempFS;m++){
							for(int n=-tempFS;n<=tempFS;n++){
								crosscoe1d.ptr<double>(cirCC(i+m,height))[cirCC(j+n,width)]+=a0[tempMinNumber]*c_pp[tempFS+m][tempFS+n];
								crosscoe1d.ptr<double>(cirCC(i+m+nm,height))[cirCC(j+n+nn,width)]+=a1[tempMinNumber]*c_pp[tempFS+m][tempFS+n];
							}
						}
					}else if(tempMinNumber==9){	// toggle case
						// update cross correlation
						for(int m=-tempFS;m<=tempFS;m++){
							for(int n=-tempFS;n<=tempFS;n++){
								crosscoe1d.ptr<double>(cirCC(i+m,height))[cirCC(j+n,width)]+=a0[tempMinNumber]*c_pp[tempFS+m][tempFS+n];
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
	/// Change absorb to grayscale
	dst1b.create(src1b.size(),CV_8UC1);
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			dst1b.ptr<uchar>(i)[j]=cvRound((1.-dst1f.ptr<float>(i)[j])*255.);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// release space
	delete	[]	c_pp;

	return true;
}

//////////////////////////////////////////////////////////////////////////
bool pixkit::halftoning::ordereddithering::DMS_2Level2012_genDitherArray(cv::Mat &DA, int daSize){
// the N defined in paper is supposed as daSize in this program.
	
	//////////////////////////////////////////////////////////////////////////
	///// init
	const	uchar	MAX_GRAYSCALE	=	255;	// defined R in paper
	const	uchar	MIN_GRAYSCALE	=	0;
	const	uchar	afa_1			=	0;		// parameter defined in paper

	//////////////////////////////////////////////////////////////////////////
	///// initialization 
	cv::Mat compressedTable1b, src1b, dst1b;
	compressedTable1b.create(Size(daSize, daSize), CV_8UC1);
	src1b.create(Size(daSize, daSize), CV_8UC1);
	dst1b.create(Size(daSize, daSize), CV_8UC1);
	src1b.setTo(MIN_GRAYSCALE);
	dst1b.setTo(MIN_GRAYSCALE);
	compressedTable1b.setTo(MIN_GRAYSCALE);
	// get coe
	Mat	hvs_model_cpp;
	pixkit::halftoning::ungrouped::generateTwoComponentGaussianModel(hvs_model_cpp,43.2,38.7,0.02,0.06);	// checked

	//////////////////////////////////////////////////////////////////////////
	///// process for masks of 0 to 255
	Mat	pre_dst1b;
	Mat	init1b(Size(daSize, daSize), CV_8UC1);
	init1b.setTo(afa_1);	// as described in paper, this value should be zero.
	for (int eta = MIN_GRAYSCALE; eta <= MAX_GRAYSCALE; eta++){		// eta is defined as a grayscale in paper
		cout << "grayscale = " << (int)eta << endl;

		//////////////////////////////////////////////////////////////////////////
		///// init
		src1b.setTo(eta);
		pre_dst1b = dst1b.clone(); // recode the result of dst(g-1)

		//////////////////////////////////////////////////////////////////////////
		///// process
		// check whether it's the first grayscale
		bool	first_grayscale	=false;
		if(cv::sum(init1b)[0]==0){
			first_grayscale	=	true;
		}	
		// process
		SCDBS(src1b,init1b,first_grayscale,dst1b,&(double&)hvs_model_cpp.data[0],hvs_model_cpp.rows);

		//////////////////////////////////////////////////////////////////////////
		///// get init
		init1b	=	dst1b.clone();

		//////////////////////////////////////////////////////////////////////////
		///// record position that the white point is firstly toggled
		for (int i = 0; i < dst1b.rows; i++){
			for (int j = 0; j < dst1b.cols; j++){
				if (dst1b.ptr<uchar>(i)[j] == MAX_GRAYSCALE &&  pre_dst1b.ptr<uchar>(i)[j] == MIN_GRAYSCALE){
					compressedTable1b.ptr<uchar>(i)[j]	=	eta;
				}
			}
		}
	}

	DA	=	compressedTable1b.clone();

	return true;
}

bool pixkit::halftoning::ordereddithering::DMS_2Level2012(const cv::Mat &src1b, const cv::Mat &ditherarray1b,cv::Mat &dst1b){	
	dst1b.create(src1b.size(),src1b.type());
	for(int i=0;i<src1b.rows;i++){
		for(int j=0;j<src1b.cols;j++){
			if(src1b.ptr<uchar>(i)[j]>=ditherarray1b.ptr<uchar>(i%ditherarray1b.rows)[j%ditherarray1b.cols]){
				dst1b.ptr<uchar>(i)[j]	=	255;
			}else{
				dst1b.ptr<uchar>(i)[j]	=	0;
			}
		}
	}
	return true;
}