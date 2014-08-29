#include "edgedetection.cpp"
#include "../include/pixkit-image.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////
float pixkit::qualityassessment::EME(const cv::Mat &src,const cv::Size nBlocks,const short mode){

	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		CV_Assert(false);
	}
	if(nBlocks.width>src.cols||nBlocks.height>src.rows){
		CV_Assert(false);
	}
	if(mode!=1&&mode!=2){
		CV_Assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	// param
	const	float	c	=	0.0001;

	//////////////////////////////////////////////////////////////////////////
	// define the stepsize
	float	tempv1	=	(float)src.cols/nBlocks.width,
			tempv2	=	(float)src.rows/nBlocks.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	stepsize((int)tempv1,(int)tempv2);

	//////////////////////////////////////////////////////////////////////////
	// estimate
	int		count	=	0;
	float	eme		=	0.;
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){

			// get local max and min
			float	local_maxv	=	src.data[i*src.cols+j],
					local_minv	=	src.data[i*src.cols+j];		
			if(mode==1){	// standard mode

				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){

						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							if(src.data[(i+m)*src.cols+(j+n)]>local_maxv){
								local_maxv	=	src.data[(i+m)*src.cols+(j+n)];
							}
							if(src.data[(i+m)*src.cols+(j+n)]<local_minv){
								local_minv	=	src.data[(i+m)*src.cols+(j+n)];
							}
						}
					}
				}

			}else if(mode==2){	// BTC's mode

				// find first moment and second moment
				double	moment1=0.,moment2=0.;
				int		count_mom=0;
				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){
						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							moment1+=src.data[(i+m)*src.cols+(j+n)];
							moment2+=src.data[(i+m)*src.cols+(j+n)]*src.data[(i+m)*src.cols+(j+n)];
							count_mom++;
						}						
					}
				}
				moment1/=(double)count_mom;
				moment2/=(double)count_mom;

				// find variance
				double	sd=sqrt(moment2-moment1*moment1);

				// find num of higher than moment1
				int	q=0;
				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){
						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							if(src.data[(i+m)*src.cols+(j+n)]>=moment1){
								q++;
							}
						}
					}
				}
				int		m_q=count_mom-q;
				local_minv=moment1-sd*sqrt((double)q/m_q),
				local_maxv=moment1+sd*sqrt((double)m_q/q);
				if(local_minv>255){
					local_minv=255;
				}
				if(local_minv<0){
					local_minv=0;
				}
				if(local_maxv>255){
					local_maxv=255;
				}
				if(local_maxv<0){
					local_maxv=0;
				}
			}else{
				assert(false);
			}

			// calc EME (Eq. 2) -totally same
			if(local_maxv!=local_minv){
				eme	+=	log((double)local_maxv/(local_minv+c));
			}
			count++;

		}
	}

	return (float)20.*eme/count;
}
float pixkit::qualityassessment::TEN(const cv::Mat &src){

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	est;	
	// process
	edgedetection::Sobel(src,est);

	//////////////////////////////////////////////////////////////////////////
	// estimation
	double	ten	=	0.;
	for(int i=0;i<est.rows;i++){
		for(int j=0;j<est.cols;j++){
			ten	+=	est.data[i*est.cols+j]	*	est.data[i*est.cols+j];	// eq. 6
		}
	}

	return (double)ten/(est.rows*est.cols);
}
float pixkit::qualityassessment::AMBE(const cv::Mat &src1,const cv::Mat &src2){

	//////////////////////////////////////////////////////////////////////////
	if((src1.rows!=src2.rows)||(src2.cols!=src2.cols)){
		CV_Assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	double	mean1=0.,mean2=0.;
	for(int i=0;i<src1.rows;i++){
		for(int j=0;j<src1.cols;j++){
			mean1	+=	(double)src1.data[i*src1.cols+j];
			mean2	+=	(double)src2.data[i*src1.cols+j];
		}
	}
	mean1	/=	(double)(src1.cols*src1.rows);
	mean2	/=	(double)(src2.cols*src2.rows);

	return abs((double)(mean1-mean2));
}
float pixkit::qualityassessment::CII(const cv::Mat &ori1b,const cv::Mat &pro1b){

	//////////////////////////////////////////////////////////////////////////
	if((ori1b.rows!=pro1b.rows)||(pro1b.cols!=pro1b.cols)){
		CV_Assert(false);
	}
	if(ori1b.type()!=CV_8UC1||pro1b.type()!=CV_8UC1){
		CV_Assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	double	c_proposed=0.,c_original=0.;
	double	minv,maxv;
	// cal
	for(int i=0;i<ori1b.rows-3;i++){
		for(int j=0;j<ori1b.cols-3;j++){
			Rect	roi(j,i,3,3);
			Mat	tmat_ori1b(ori1b,roi),	tmat_pro1b(pro1b,roi);			
			minMaxLoc(tmat_ori1b,&minv,&maxv);
			c_original+=(maxv-minv)/(maxv+minv);
			minMaxLoc(tmat_pro1b,&minv,&maxv);
			c_proposed+=(maxv-minv)/(maxv+minv);
		}
	}
	return	c_proposed/c_original;

	return 0.;
}
float pixkit::qualityassessment::PSNR(const cv::Mat &src1,const cv::Mat &src2){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src1.empty()||src2.empty()){
		CV_Error(CV_HeaderIsNull,"[qualityassessment::PSNR] image is empty");
	}
	if(src1.type()!=src2.type()){
		CV_Error(CV_StsBadArg,"[qualityassessment::PSNR] both types of image do not match");
	}
	if(src1.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[qualityassessment::PSNR] image should be grayscale");
	}

	//////////////////////////////////////////////////////////////////////////
	///// derive psnr
	double	total_err=0.;
	for(int i=0;i<src1.rows;i++){
		for(int j=0;j<src1.cols;j++){
			total_err+=(src1.data[i*src1.cols+j]-src2.data[i*src1.cols+j])*(src1.data[i*src1.cols+j]-src2.data[i*src1.cols+j]);
		}
	}

	// = = = = = Return PSNR = = = = = //
	return 10*log10((double)(src1.cols)*(src1.rows)*(255.)*(255.)/total_err);
}
float pixkit::qualityassessment::HPSNR(const cv::Mat &src1, const cv::Mat &src2)
{
	
	double  mse = 0;
	const int height = 15;
	const int width = 15;
	int  wd_size = static_cast<int>(height/2*2);;
	
	if(src1.empty()||src2.empty()){
		CV_Error(CV_HeaderIsNull,"[qualityassessment::HPSNR] image is empty");
	}
	if(src1.type()!=src2.type()){
		CV_Error(CV_StsBadArg,"[qualityassessment::HPSNR] both types of image do not match");
	}
	if(src1.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[qualityassessment::HPSNR] image should be grayscale");
	}

	double gaussianFilter[15][15] = {
	0,	0,	0,	0,	0.000001,	0.000002,	0.000003,	0.000004,	0.000003,	0.000002,	0.000001,	0,	0,	0,	0,
	0,	0,	0,	0.000002,	0.000008,	0.000021,	0.000039,	0.000048,	0.000039,	0.000021,	0.000008,	0.000002,	0,	0,	0,
	0,	0,	0.000003,	0.000017,	0.000071,	0.000193,	0.000351,	0.000429,	0.000351,	0.000193,	0.000071,	0.000017,	0.000003,	0,	0,
	0,	0.000002,	0.000017,	0.000106,	0.000429,	0.001166,	0.002125,	0.002595,	0.002125,	0.001166,	0.000429,	0.000106,	0.000017,	0.000002,	0,
	0.000001,	0.000008,	0.000071,	0.000429,	0.001739,	0.004728,	0.008616,	0.010523,	0.008616,	0.004728,	0.001739,	0.000429,	0.000071,	0.000008,	0.000001,
	0.000002,	0.000021,	0.000193,	0.001166,	0.004728,	0.012853,	0.02342,	0.028605,	0.02342,	0.012853,	0.004728,	0.001166,	0.000193,	0.000021,	0.000002,
	0.000003,	0.000039,	0.000351,	0.002125,	0.008616,	0.02342,	0.042674,	0.052122,	0.042674,	0.02342,	0.008616,	0.002125,	0.000351,	0.000039,	0.000003,
	0.000004,	0.000048,	0.000429,	0.002595,	0.010523,	0.028605,	0.052122,	0.063662,	0.052122,	0.028605,	0.010523,	0.002595,	0.000429,	0.000048,	0.000004,
	0.000003,	0.000039,	0.000351,	0.002125,	0.008616,	0.02342,	0.042674,	0.052122,	0.042674,	0.02342,	0.008616,	0.002125,	0.000351,	0.000039,	0.000003,
	0.000002,	0.000021,	0.000193,	0.001166,	0.004728,	0.012853,	0.02342,	0.028605,	0.02342,	0.012853,	0.004728,	0.001166,	0.000193,	0.000021,	0.000002,
	0.000001,	0.000008,	0.000071,	0.000429,	0.001739,	0.004728,	0.008616,	0.010523,	0.008616,	0.004728,	0.001739,	0.000429,	0.000071,	0.000008,	0.000001,
	0,	0.000002,	0.000017,	0.000106,	0.000429,	0.001166,	0.002125,	0.002595,	0.002125,	0.001166,	0.000429,	0.000106,	0.000017,	0.000002,	0,
	0,	0,	0.000003,	0.000017,	0.000071,	0.000193,	0.000351,	0.000429,	0.000351,	0.000193,	0.000071,	0.000017,	0.000003,	0,	0,
	0,	0,	0,	0.000002,	0.000008,	0.000021,	0.000039,	0.000048,	0.000039,	0.000021,	0.000008,	0.000002,	0,	0,	0,
	0,	0,	0,	0,	0.000001,	0.000002,	0.000003,	0.000004,	0.000003,	0.000002,	0.000001,	0,	0,	0,	0
	};

	cv::Mat OriImageWd, ResImageWd;

	//boundary extension ==========================
	//wd_reg memory((IW+wd_size)*(IL+wd_size))
	OriImageWd.create(src1.rows+wd_size, src1.cols+wd_size, 0);
	ResImageWd.create(src2.rows+wd_size, src2.cols+wd_size, 0);
	
	//WdImage((Height+wd_size)x(Width+wd_size)) <- Input(HeightxWidth)
	for(int i=0; i<src1.rows; i++){
		for(int j=0; j<src1.cols; j++){
			OriImageWd.data[(i+wd_size/2)*(OriImageWd.cols) + (j+wd_size/2)] = src1.data[i*src1.cols +j];
			ResImageWd.data[(i+wd_size/2)*(ResImageWd.cols) + (j+wd_size/2)] = src2.data[i*src2.cols +j];
		}
	}

	//copy(:, wd_size/2 ~wd_size/2+Width-1)
	for(int j=0; j<src1.cols ;j++){
		for(int k=0; k<wd_size/2; k++){
			OriImageWd.data[(wd_size/2-k-1)*(OriImageWd.cols) + (j+wd_size/2)] = OriImageWd.data[(wd_size/2+k)*(OriImageWd.cols) + (j+wd_size/2)];
			ResImageWd.data[(wd_size/2-k-1)*(ResImageWd.cols) + (j+wd_size/2)] = ResImageWd.data[(wd_size/2+k)*(ResImageWd.cols) + (j+wd_size/2)];
			OriImageWd.data[(src1.rows+wd_size/2+k)*(OriImageWd.cols) + (j+wd_size/2)] = OriImageWd.data[(src1.rows+wd_size/2-k-1)*(OriImageWd.cols) + (j+wd_size/2)];
			ResImageWd.data[(src2.rows+wd_size/2+k)*(ResImageWd.cols) + (j+wd_size/2)] = ResImageWd.data[(src2.rows+wd_size/2-k-1)*(ResImageWd.cols) + (j+wd_size/2)];
		}
	}

	//copy(wd_size/2~wd_size/2+Width-1, : )
	for(int i=0;i<OriImageWd.rows;i++){
		for(int k=0;k<wd_size/2;k++){
			OriImageWd.data[i*(OriImageWd.cols) + (wd_size/2-k-1)] = OriImageWd.data[i*(OriImageWd.cols) + (wd_size/2+k)];
			ResImageWd.data[i*(ResImageWd.cols) + (wd_size/2-k-1)] = ResImageWd.data[i*(ResImageWd.cols) + (wd_size/2+k)];
			OriImageWd.data[i*(OriImageWd.cols) + (src1.cols+wd_size/2+k)] = OriImageWd.data[i*(OriImageWd.cols) + (src1.cols+wd_size/2-k-1)];
			ResImageWd.data[i*(ResImageWd.cols) + (src2.cols+wd_size/2+k)] = ResImageWd.data[i*(ResImageWd.cols) + (src2.cols+wd_size/2-k-1)];
		}
	}

	//PSNR calculation =========================
	for(int i=0; i<src1.rows; i++){
		for(int j=0; j<src1.cols; j++){
			double temp = 0.0;
			for(int x=0; x<height; x++){
				for(int y=0; y<width; y++){
					temp += (ResImageWd.data[(i+x)*ResImageWd.cols + (j+y)] - OriImageWd.data[(i+x)*OriImageWd.cols + (j+y)]) * gaussianFilter[x][y];
				}
			}
			mse += (temp*temp);
		}
	}	
	mse /= (src1.rows * src1.cols);
	return static_cast<float>(20*log10(255/sqrt(mse)));
}
bool pixkit::qualityassessment::GaussianDiff(InputArray &_src1,InputArray &_src2,double sd){

	cv::Mat	src1	=	_src1.getMat();
	cv::Mat	src2	=	_src2.getMat();

	// Gaussian blur
	cv::Mat	dst1,dst2;
	cv::GaussianBlur(src1,dst1,cv::Size(0,0),sd);
	cv::GaussianBlur(src2,dst2,cv::Size(0,0),sd);

	// get difference
	dst1.convertTo(dst1,CV_32FC1);
	dst2.convertTo(dst2,CV_32FC1);
	Mat	diff=dst1-dst2;
	diff=diff*10+128;

	// show images
	dst1.convertTo(dst1,CV_8UC1);
	dst2.convertTo(dst2,CV_8UC1);
	diff.convertTo(diff,CV_8UC1);
	imshow("Blurred original image",dst1);
	imshow("Blurred halftone image",dst2);
	imshow("Difference image (G(a)-G(b))*10+128",diff);
	waitKey(0);

	imwrite("a.bmp",diff);

	return true;
}
bool pixkit::qualityassessment::PowerSpectrumDensity(cv::InputArray &_src,cv::OutputArray &_dst){
// Original code: http://docs.opencv.org/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
	
	cv::Mat	src=_src.getMat();

	//////////////////////////////////////////////////////////////////////////
	///// exception 
	if(src.type()!=CV_8UC1){
		CV_Assert(false);
	}
	
	//////////////////////////////////////////////////////////////////////////
	///// calculation
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize( src.rows );
	int n = getOptimalDFTSize( src.cols ); // on the border add zero values
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat tdst1f = planes[0];
	// power spectrum
	tdst1f	=	abs(tdst1f);
	tdst1f	=	tdst1f.mul(tdst1f)/	((float)tdst1f.total());
	// eliminate the DC value
	tdst1f.ptr<float>(0)[0]	=	(tdst1f.ptr<float>(0)[1]	+	tdst1f.ptr<float>(1)[0]	+	tdst1f.ptr<float>(1)[1])	/	3.;	
	// scale 
 	tdst1f += Scalar::all(1);       // switch to logarithmic scale
 	log(tdst1f, tdst1f);

	// crop	the spectrum, if it has an odd number of rows or columns
	tdst1f = tdst1f(Rect(0, 0, tdst1f.cols & -2, tdst1f.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = tdst1f.cols/2;
	int cy = tdst1f.rows/2;
	Mat q0(tdst1f, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(tdst1f, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(tdst1f, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(tdst1f, Rect(cx, cy, cx, cy)); // Bottom-Right
	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(tdst1f, tdst1f, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).

	// copy
	_dst.create(src.size(),tdst1f.type());
	cv::Mat	dst	=	_dst.getMat();
	tdst1f.copyTo(dst);

	return true;
}
bool pixkit::qualityassessment::spectralAnalysis_Bartlett(cv::InputArray &_src,cv::OutputArray &_dst){

	Mat	src	=	_src.getMat();
	const	int	unitsize	=	256;

	if(src.size()!=cv::Size(unitsize,unitsize*10)){
		CV_Error(CV_StsBadSize,"[pixkit::qualityassessment::spectralAnalysis_Bartlett] src's image size should be 256x2560");
	}
	
	const	int	rounds		=	10;

	Mat	tdst1f(cv::Size(unitsize,unitsize),CV_32FC1);
	tdst1f.setTo(0);
	for(int k=0;k<rounds;k++){

		Rect	roi(0,unitsize*k,unitsize,unitsize);
		Mat	tsrc	=	src(roi);

		Mat	tps;	// temp power spectrum
		PowerSpectrumDensity(tsrc,tps);	// get power spectrum

		tdst1f	=	tdst1f	+	tps;

	}

	tdst1f	=	tdst1f	/	static_cast<float>(rounds);

	_dst.create(tdst1f.size(),tdst1f.type());
	Mat	dst	=	_dst.getMat();
	tdst1f.copyTo(dst);

	return true;
}