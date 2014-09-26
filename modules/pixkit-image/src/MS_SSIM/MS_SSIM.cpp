#include "../../include/pixkit-image.hpp"

bool downSample(cv::Mat &src, cv::Mat &dst)
{
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.empty()||dst.empty()){
		CV_Error(CV_HeaderIsNull,"[downSample] image is empty");
		return false;
	}
	if((src.rows/2) != dst.rows || (src.cols/2) != dst.cols){
		CV_Error(CV_BadNumChannels,"[downSample] sizes of two images are not multiples of 2");
		return false;
	}
	if(src.type()!=CV_8U || dst.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[downSample] image should be grayscale");
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	double LL_mask[5][5] = {
		0.015625,	-0.03125,	-0.09375,	-0.03125,	0.015625,
		-0.03125,	0.0625,		0.1875,		0.0625,		-0.03125,
		-0.09375,	0.1875,		0.5625,		0.1875,		-0.09375,
		-0.03125,	0.0625,		0.1875,		0.0625,		-0.03125,
		0.015625,	-0.03125,	-0.09375,	-0.03125,	0.015625
	};
	int filterHeight=5, filterWidth=5, wd_size=filterHeight/2*2;

	cv::Mat ext_src;
	ext_src.create(src.rows+filterHeight/2*2, src.cols+filterWidth/2*2, 0);

	// Boundary Extension //////////////////////////////
	//WdImage((Height+wd_size)x(Width+wd_size)) <- Input(HeightxWidth)
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			ext_src.data[(i+wd_size/2)*ext_src.cols + (j+wd_size/2)] = src.data[i*src.cols + j];
		}
	}

	//copy(:, wd_size/2 ~wd_size/2+Width-1)
	for(int j=0; j<src.cols; j++){
		for(int k=1; k<=wd_size/2; k++){
			ext_src.data[(wd_size/2-k)*ext_src.cols + (j+wd_size/2)] = ext_src.data[(wd_size/2+k)*ext_src.cols + (j+wd_size/2)];
			ext_src.data[(src.rows+wd_size/2+k-1)*ext_src.cols + (j+wd_size/2)] = ext_src.data[(src.rows+wd_size/2-k-1)*ext_src.cols + (j+wd_size/2)];
		}
	}

	//copy(wd_size/2~wd_size/2+Width-1, : )
	for(int i=0; i<src.rows+wd_size; i++){
		for(int k=1; k<=wd_size/2; k++){
			ext_src.data[i*ext_src.cols + (wd_size/2-k)] = ext_src.data[i*ext_src.cols + (wd_size/2+k)];
			ext_src.data[i*ext_src.cols + (src.cols+wd_size/2+k-1)] = ext_src.data[i*ext_src.cols + (src.cols+wd_size/2-k-1)];
		}
	}

	// convolution 
	for (int i=0; i<src.rows/2; i++){
		for (int j=0; j<src.cols/2; j++){ 
			double g = 0;
			for (int x=-filterHeight/2; x<=filterHeight/2; x++){
				for (int y=-filterWidth/2; y<=filterWidth/2; y++){
					g +=  ext_src.data[(i*2 + x+filterHeight/2)*ext_src.cols + (j*2 + y+filterWidth/2)] * LL_mask[x + filterHeight/2][y + filterWidth/2];		
				}
			}
			dst.data[i*dst.cols + j] = cv::saturate_cast<uchar>(g);
		}
	}
	return true;
}

float pixkit::qualityassessment::MS_SSIM(const cv::Mat &src1, const cv::Mat &src2, int HVSsize)
{
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src1.empty()||src2.empty()){
		CV_Error(CV_HeaderIsNull,"[qualityassessment::MS_SSIM] image is empty");
	}
	if(src1.cols != src2.cols || src1.rows != src2.rows){
		CV_Error(CV_StsBadArg,"[qualityassessment::MS_SSIM] sizes of two images are not equal");
	}
	if(src1.type()!=CV_8U || src2.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[qualityassessment::MS_SSIM] image should be grayscale");
	}
	if(src1.cols < 32 || src1.rows < 32 || src2.cols < 32 || src2.rows < 32){
		CV_Error(CV_BadNumChannels,"[qualityassessment::MS_SSIM] sizes of two images should be greater than 32x32");
	}

	//////////////////////////////////////////////////////////////////////////
	// initialize the ssim weights of each scales
	int HalfSize = static_cast<int>(HVSsize/2);
	double alpha = 0.1333;
	std::vector< double >beta(5);
	std::vector< double >gamma(5);
	beta[0] = gamma[0] = 0.0448;
	beta[1] = gamma[1] = 0.2856;
	beta[2] = gamma[2] = 0.3001;
	beta[3] = gamma[3] = 0.2363;
	beta[4] = gamma[4] = alpha;
	cv::Mat tsrc1, tsrc2;
	tsrc1 = src1.clone();
	tsrc2 = src2.clone();

	///////////////////////////////////////////////////
	// HVS filter
	std::vector< std::vector<double> > gaussianFilter( HVSsize, std::vector<double>(HVSsize) );
	double sum = 0, STD = 1.5 ;

	for (int i=-HalfSize; i<=HalfSize; i++){
		for (int j=-HalfSize; j<=HalfSize; j++){	
			gaussianFilter[i+HalfSize][j+HalfSize] = exp( -1 * (i*i+j*j) / (2*STD*STD) );
			sum += gaussianFilter[i+HalfSize][j+HalfSize];
		}
	}
	 
	// Normalize to 0~1
	for (int i=-HalfSize; i<=HalfSize; i++){
		for (int j=-HalfSize; j<=HalfSize; j++){	
			gaussianFilter[i+HalfSize][j+HalfSize] /= sum;
		}
	}
	/////////////////////////////////////////////////////
	double SSIMresult=1, luminance=0, contrast=0, structure=0;
	double *lu_co_st = new double [3];

	for (int m=0; m<5; m++){
		
		// mssim calculation
		pixkit::qualityassessment::MSSIM(tsrc1, tsrc2, HVSsize, &lu_co_st[0]);

		contrast = pow(lu_co_st[1], beta[m]);
		structure = pow(lu_co_st[2], gamma[m]);

		if (m==4){
			luminance = pow(lu_co_st[0], alpha);
			SSIMresult *=  (luminance * contrast * structure);
		}else{
			SSIMresult *=  (contrast * structure);
		}
		//////////////////////////////////////////////////////////////////////////
			
		// down sample
		cv::Mat rsrc2, rsrc1;
		rsrc1 = tsrc1.clone();
		rsrc2 = tsrc2.clone();

		tsrc1.create(rsrc1.rows/2, rsrc1.cols/2, rsrc1.type());
		tsrc2.create(rsrc2.rows/2, rsrc2.cols/2, rsrc2.type());

		downSample(rsrc1, tsrc1);
		downSample(rsrc2, tsrc2);
	}
	delete [] lu_co_st;
	return SSIMresult;
}

 

 