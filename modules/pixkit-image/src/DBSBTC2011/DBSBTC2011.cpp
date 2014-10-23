#include <time.h>
#include "../../include/pixkit-image.hpp"

// iteration-based converting for local min & max (quantization level version)
bool DirectBinarySearch_QLver(const cv::Mat &src, cv::Mat &dst, double highValue, double lowValue){

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[DirectBinarySearch_QLver] image is empty");
		return false;
	}

	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[DirectBinarySearch_QLver] image should be grayscale");
		return false;
	}

	dst.create(src.size(), src.type());
	srand( static_cast< unsigned int >(time(NULL)) );
	// =========================================
	// initialize the dst_reg
	for (int i=0; i<dst.rows; i++) {
		for (int j=0; j<dst.cols; j++) {
			double pixel = static_cast< double >(rand()) / static_cast<double>(RAND_MAX);
			if (pixel < 0.5) {
				dst.data[i*dst.cols + j] = static_cast< unsigned char >(lowValue);
			} 
			else {
				dst.data[i*dst.cols + j] = static_cast< unsigned char >(highValue);
			}
		}
	}

	// ========================================================
	// gauss filter initial
	const int FilterHeight = 7;
	const int FilterWidth = 7;
	std::vector< std::vector< double > > G_filter(FilterHeight, std::vector<double>(FilterWidth));
	int FilterHalfSize = (static_cast<int>(FilterHeight) - 1) / 2;
	double std = static_cast<double>(FilterHeight-1)/6, sum = 0;

	for (int i=-FilterHalfSize; i<=FilterHalfSize; i++){
		for (int j=-FilterHalfSize; j<=FilterHalfSize; j++){	
			G_filter[i+FilterHalfSize][j+FilterHalfSize] = exp( -1 * (i*i+j*j) / (2*std*std) );
			sum += G_filter[i+FilterHalfSize][j+FilterHalfSize];
		}
	}

	// Normalize to 0~1
	for (int i=-FilterHalfSize; i<=FilterHalfSize; i++){
		for (int j=-FilterHalfSize; j<=FilterHalfSize; j++){	
			G_filter[i+FilterHalfSize][j+FilterHalfSize] /= sum;
		}
	}
	// ========================================================
	// CPP, E initial
	int HalfSize = (FilterHeight-1)/2;
	int CPPHeight = FilterHeight + 2*HalfSize;
	int CPPWidth = FilterWidth + 2*HalfSize;
	int ErHeight = src.rows;
	int ErWidth = src.cols;
	std::vector< std::vector< double > > CPP_Image(CPPHeight, std::vector< double >(CPPWidth));
	std::vector< std::vector< double > > Er_Image(ErHeight, std::vector< double >(ErWidth));

	for (int i = 0 ; i < FilterHeight ; i++){
		for (int j = 0 ; j < FilterWidth ; j++){
			for (int y = 0 ; y < FilterHeight; y++){
				for (int x = 0 ; x < FilterWidth ; x++){
					CPP_Image[i+y][j+x] += G_filter[i][j]*G_filter[y][x];
				}
			}
		}
	}

	for (int i = 0 ; i < ErHeight ; i++){
		for (int j = 0 ; j < ErWidth ; j++){
			dst.data[i*ErWidth + j] = static_cast<uchar>((dst.data[i*ErWidth + j] - lowValue) / (highValue - lowValue));
			Er_Image[i][j] = (dst.data[i*ErWidth + j]) - ((src.data[i*ErWidth + j] - lowValue) / (highValue - lowValue));
		}
	}

	// ========================================================
	// CEP initial
	int HalfCPPSize = (CPPHeight-1)/2;
	int CEPHeight = ErHeight + 2*HalfCPPSize;
	int CEPWidth = ErWidth + 2*HalfCPPSize;
	std::vector< std::vector< double > > CEP_Image(CEPHeight, std::vector< double >(CEPWidth));

	for (int i = 0 ; i < ErHeight ; i++){
		for (int j = 0 ; j < ErWidth ; j++){
			double sum = 0.;
			for(int y = -HalfCPPSize ; y <= HalfCPPSize ; y++){
				for(int x = -HalfCPPSize ; x <= HalfCPPSize ; x++){			
					if (i+y < 0 || i+y >= ErHeight || j+x < 0 || j+x >= ErWidth){
						continue;
					}
					else{
						sum += Er_Image[i+y][j+x] * CPP_Image[y+HalfCPPSize][x+HalfCPPSize];
					}	
				}
			}
			CEP_Image[i+HalfCPPSize][j+HalfCPPSize] = sum;
		}
	}

	// ========================================================
	// DBS process
	double CurrentError = 0;
	for (int i = 0 ; i < CEPHeight ; i++){
		for (int j = 0 ; j < CEPWidth ; j++){
			CurrentError += CEP_Image[i][j];
		}
	}
	double PastError = 0.;
	double EPS = 0.;
	double EPS_MIN = 0.;
	while (fabs(CurrentError - PastError)>=1)
	{
		PastError = CurrentError;

		double a0 = 0., a1 = 0., Current_a0 = 0., Current_a1 = 0.;
		CurrentError = 0;
		int CurrentPx = 0;
		int CurrentPy = 0;

		for (int i = 0 ; i < dst.rows ; i++){
			for (int j = 0 ; j < dst.cols ; j++){

				Current_a0 = 0.;
				Current_a1 = 0.;
				CurrentPx = 0;
				CurrentPy = 0;
				EPS_MIN = 0.;
				for (int y = -1 ; y <= 1 ; y++){
					if (i+y < 0 || i+y >= dst.rows){
						continue;
					}
					for (int x = -1 ; x <= 1 ; x++){
						if (j+x < 0 || j+x >= dst.cols){
							continue;
						}
						if (y == 0 && x == 0){	// toggle
							if (dst.data[i*dst.cols + j] == 1){			//dst.data[i*dst.cols + j] == highValue (1 -> 0)
								a0 = -1;	//a0 = -(highValue - lowValue);	
								a1 = 0;		
							}else{	// 0 -> 1
								a0 = 1;	//a0 = (highValue - lowValue);
								a1 = 0;
							}
						}else{	// Swap
							if (dst.data[(i+y)*dst.cols + (j+x)] != dst.data[i*dst.cols + j]){
								if (dst.data[i*dst.cols + j] == 1){		//dst.data[i*dst.cols + j] == highValue (1 -> 0)
									a0 = -1;	//a0 = -(h1ighValue - lowValue);	
									a1 = -a0;
								}else{	// 0 -> 1
									a0 = 1;	//a0 = (highValue - lowValue);
									a1 = -a0;
								}
							}else{
								a0 = 0;
								a1 = 0;
							}
						}
						EPS =	(a0*a0+a1*a1)*CPP_Image[HalfCPPSize][HalfCPPSize] +
							2*a0*a1*CPP_Image[HalfCPPSize + y][HalfCPPSize + x] +
							2*a0*CEP_Image[i + HalfCPPSize][j + HalfCPPSize] +
							2*a1*CEP_Image[i + y + HalfCPPSize][j + x + HalfCPPSize];

						if (EPS_MIN > EPS){
							EPS_MIN = EPS;
							Current_a0 = a0;
							Current_a1 = a1;
							CurrentPx = x;
							CurrentPy = y;
						}
					}
				}
				if(EPS_MIN < 0){
					for(int y = -HalfCPPSize ; y <= HalfCPPSize ; y++){
						for(int x = -HalfCPPSize ; x <= HalfCPPSize ; x++){
							CEP_Image[i+y+HalfCPPSize][j+x+HalfCPPSize] += Current_a0*CPP_Image[y+HalfCPPSize][x+HalfCPPSize];	
						}
					}
					for(int y = -HalfCPPSize ; y <= HalfCPPSize ; y++){
						for(int x = -HalfCPPSize ; x <= HalfCPPSize ; x++){	
							CEP_Image[i+y+CurrentPy+HalfCPPSize][j+x+CurrentPx+HalfCPPSize] += Current_a1*CPP_Image[y+HalfCPPSize][x+HalfCPPSize];											
						}
					}
					dst.data[i*dst.cols + j] += static_cast < int > (Current_a0);
					dst.data[(i+CurrentPy)*dst.cols + (j+CurrentPx)] += static_cast < int > (Current_a1);
				}
			}
		}

		for (int y = 0 ; y < CEPHeight ; y++){
			for (int x = 0 ; x < CEPWidth ; x++){
				CurrentError += CEP_Image[y][x];
			}
		}
	}
	// DBS process

	for (int i = 0 ; i < dst.rows ; i++){
		for (int j = 0 ; j < dst.cols ; j++){
			if (dst.data[i*ErWidth + j] == 1){
				dst.data[i*ErWidth + j] = static_cast< unsigned char >(highValue);		
			}
			else{
				dst.data[i*ErWidth + j] = static_cast< unsigned char >(lowValue);	
			}
		}
	}
	return true;
}

// BTC iteration-based converting by using "alpha data"
bool pixkit::comp::DBSBTC2011(cv::Mat src, cv::Mat &dst, int blockSize)
{
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[comp::DBSBTC2011] image is empty");
		return false;
	}
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[comp::DBSBTC2011] image type should be gray scale");
		return false;
	}
	if(blockSize!=8 && blockSize!=16){
		CV_Error(CV_StsBadArg,"[comp::DBSBTC2011] BlockSize should be 8 or 16.");
		return false;
	}

	// ========================================
	std::vector < int > histo(256);
	dst.create(src.size(), src.type());
	int alphaSize = 0, segSize = 0;

	// Read training alpha data
	if(blockSize == 8){
		alphaSize = 2;
		segSize = 3;
	}else if(blockSize == 16){
		alphaSize = 3;
		segSize = 4;
	}else{
		CV_Error(CV_StsBadArg,"[comp::DBSBTC2011] BlockSize should be 8 or 16.");
		return false;
	}

	std::vector < double > alphaData(alphaSize);
	std::vector < double > segData(segSize);

	if(blockSize == 8){
		const double alpha[2][3] = {
			{0.180000, 0.200000}, 
			{0.000000, 0.903090, 1.806180} };

			for (int i=0; i!=alphaData.size(); i++){
				alphaData[i] = alpha[0][i];
			}
			for (int i=0; i!=segData.size(); i++){
				segData[i] = alpha[1][i];
			}

	}else if(blockSize == 16){
		const double alpha[3][4] = {
			{0.080000, 0.170000, 0.170000},
			{0.000000, 1.204120, 1.806180, 2.408240} };

			for (int i=0; i!=alphaData.size(); i++){
				alphaData[i] = alpha[0][i];
			}
			for (int i=0; i!=segData.size(); i++){
				segData[i] = alpha[1][i];
			}
	}else{
		CV_Error(CV_StsBadArg,"[comp::DBSBTC2011] BlockSize should be 8 or 16.");
		return false;
	}

	// ========================================
	for (int i=0; i<src.rows; i+=blockSize){
		for (int j=0; j<src.cols; j+=blockSize){

			// Entropy calculation 
			double entropy = 0;

			for (int k=0; k!=histo.size(); k++)
				histo[k] = 0;

			for (int m=0; m<blockSize; m++){
				for (int n=0; n<blockSize; n++){
					int pixelIndex = static_cast<int>( src.data[(i+m)*src.cols + (j+n)] );
					histo[pixelIndex]++;
				}
			}

			for (int k=0; k!=histo.size(); k++){
				if (histo[k]!=0){
					double prob = histo[k] / static_cast<double>(blockSize*blockSize);
					entropy += (-log10(prob) * prob);
				}
			}

			// =========================================
			// max & min & mean
			double alpha;
			double max = 0, min = 255, mean = 0;

			for (int m=0; m<blockSize; m++){
				for (int n=0; n<blockSize; n++){
					if (src.data[(i+m)*src.cols +(j+n)] > max){
						max = src.data[(i+m)*src.cols +(j+n)];
					}
					if (src.data[(i+m)*src.cols +(j+n)] < min){
						min = src.data[(i+m)*src.cols +(j+n)];
					}
					mean += src.data[(i+m)*src.cols +(j+n)];
				}
			}
			mean /= static_cast<double>(blockSize*blockSize);

			for (int k=0; k!=alphaData.size(); k++){
				if (segData[k]<=entropy && segData[k+1]>entropy){
					alpha = alphaData[k];
					max = max-(alpha*(max-mean));
					min = min+(alpha*(mean-min));
					break;
				}
			}

			cv::Mat src_reg, dst_reg;
			src_reg.create(blockSize, blockSize, 0);

			// copy block information
			for (int m=0; m<blockSize; m++){
				for (int n=0; n<blockSize; n++){
					src_reg.data[m*src_reg.cols + n] = src.data[(i+m)*src.cols + (j+n)];
				}
			}		

			// DirectBinarySearch for block
			DirectBinarySearch_QLver(src_reg, dst_reg, max, min);

			// renew
			for (int m=0; m<blockSize; m++){
				for (int n=0; n<blockSize; n++){
					dst.data[(i+m)*dst.cols + (j+n)] = dst_reg.data[m*dst_reg.cols + n];
				}
			}

		}
	}
	return true;
}