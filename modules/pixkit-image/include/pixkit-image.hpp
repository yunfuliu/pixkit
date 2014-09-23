//////////////////////////////////////////////////////////////////////////
// 
// SOURCE CODE: https://github.com/yunfuliu/pixkit
// 
// BEIRF: pixkit-image contains image processing related methods which have been published (on articles, e.g., journal or conference papers). 
//	In addition, some frequently used related tools are also involved.
// 
//////////////////////////////////////////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>


#ifndef __PIXKIT_IMAGE_HPP__
#define __PIXKIT_IMAGE_HPP__

namespace pixkit{

	//////////////////////////////////////////////////////////////////////////
	/// Thresholding
	namespace thresholding{

		bool LAT2011(const cv::Mat &src,cv::Mat &dst,int windowSize, double k);

	}

	//////////////////////////////////////////////////////////////////////////
	/// Attack
	namespace attack{

		/**
		* @brief		add Gaussian noise to each pixel
		*
		* @param		sd:	standard deviation, unit: grayscale. range: 0~255
		*/
		bool	addGaussianNoise(const cv::Mat &src,cv::Mat &dst,const double sd);

		/**
		* @brief		add white noise to each pixel
		*
		* @param		maxMag: the biggest mag of the noise
		*/
		bool	addWhiteNoise(const cv::Mat &src,cv::Mat &dst,const double maxMag);

	}

	//////////////////////////////////////////////////////////////////////////
	/// Filtering related
	namespace filtering{

		/**
		* @brief		filtering with median filter
		* @brief		paper: no
		*
		* @author		Yunfu Liu (yunfuliu@gmail.com)
		* @date			Sept. 6, 2013
		* @version		1.0
		*
		* @param		src: input image (grayscale only)
		* @param		dst: output image
		* @param		blocksize: the used block size
		*
		* @return		bool: true: successful, false: failure
		*/
		bool medianfilter(const cv::Mat &src,cv::Mat &dst,cv::Size blocksize);

		// fast box filtering
		bool FBF(const cv::Mat &src,cv::Mat &dst,cv::Size blockSize,cv::Mat &sum=cv::Mat());

	}

	//////////////////////////////////////////////////////////////////////////
	/// Edge detection related
	namespace edgedetection{

		/**
		* @brief		Sobel edge detection
		* @brief		paper: digital image processing textbook
		*
		* @author		Yunfu Liu (yunfuliu@gmail.com)
		* @date			Sept. 4, 2013
		* @version		1.0
		*
		* @param		src: input image (grayscale only)
		* @param		dst: output image
		*
		* @return		bool: true: successful, false: failure
		*/
		bool Sobel(const cv::Mat &src, cv::Mat &dst);
	}

	//////////////////////////////////////////////////////////////////////////
	/// Halftoning related
	namespace halftoning{

		/// Error Diffusion related
		namespace errordiffusion{ 
			bool			Ostromoukhov2001(const cv::Mat &src, cv::Mat &dst);
			bool			ZhouFang2003(const cv::Mat &src, cv::Mat &dst);
			bool			FloydSteinberg1976(const cv::Mat &src,cv::Mat &dst);
		}

		/// Direct binary search
		namespace directbinarysearch{
			// efficient DBS
			bool			LiebermanAllebach1997(const cv::Mat &src1b, cv::Mat &dst1b,double *coeData=NULL,int FilterSize=7);
		}

		/// Ordered Dither related
		namespace ordereddithering{
			bool			KackerAllebach1998(const cv::Mat &src, cv::Mat &dst);
		}

		/// Dot diffusion related
		namespace dotdiffusion{

			/**
			* @brief		paper: Yun-Fu Liu and Jing-Ming Guo, "New class tiling design for dot-diffused halftoning," IEEE Trans. Image Processing, vol. 22, no. 3, pp. 1199-1208, March 2013.
			* 
			* @param		ctSize:	CT size
			*/
			class CNADDCT{
			public:
				int				m_CT_height;	// CT's height and width
				int				m_CT_width;
				unsigned char	**m_ct;
				CNADDCT();
				~CNADDCT();
				bool			generation(cv::Size ctSize);	// ct generation
				bool			save(char name[]);
				bool			load(char name[]);
			private:
				int				m_CTmap_height;	// ct map
				int				m_CTmap_width;
				int				m_CM_size;		// class matrix
				int				m_numberOfCM;	// number of cm in a ct
				int				**m_cm;
				int				*m_cmData;
				unsigned char	*m_ctmap;
				unsigned char	*m_ctData;
			};
			bool NADD2013(cv::Mat &src,cv::Mat &dst,pixkit::halftoning::dotdiffusion::CNADDCT &cct);

			/**
			* @brief		paper: J. M. Guo and Y. F. Liu"Improved dot diffusion by diffused matrix and class matrix co-optimization," IEEE Trans. Image Processing, vol. 18, no. 8, pp. 1804-1816, 2009.
			*
			* @author		Yunfu Liu (yunfuliu@gmail.com)
			* @date			May 17, 2013
			* @version		1.0
			* 
			* @param		src: input image (grayscale only)
			* @param		dst: output image
			* @param		ClassMatrixSize: allow only 8x8 and 16x16
			*
			* @return		bool: true: successful, false: failure
			*/ 
			bool GuoLiu2009(const cv::Mat &src, cv::Mat &dst,const int ClassMatrixSize);
			
			/**
			* @brief		paper: S. Lippens and W. Philips, ��Green-noise halftoning with dot diffusion,�� in Proc. SPIE/IS&T - The International Society for Optical Engineering, vol. 6497, no. 64970V, 2007.
			*
			* @author		Yunfu Liu (yunfuliu@gmail.com)
			* @date			Feb 25, 2014
			* @version		1.0
			* 
			* @param		src: input image (grayscale only)
			* @param		dst: output image
			*
			* @return		bool: true: successful, false: failure
			*/ 
			bool LippensPhilips2007(const cv::Mat &src, cv::Mat &dst);

		}

		namespace ungrouped{

			/*
			* @brief	This function generates white points with CVT method. The details please refer to the following website.
			*			
			* @website	http://people.sc.fsu.edu/~jburkardt/cpp_src/cvt/cvt.html
			*
			* @param	dst: dst image. It will be a image with CV_8UC1 format. 
			* @param	imageSize: The size of "dst," and this size will be the width and height of this image. 
			* @param	definitions of other parameters please refer to the following file.
			*			pixkit\modules\pixkit-image\src\halftoning\CVT\cvt.cpp
			*
			* @example	
			*			int seed	= 123456789;
			*			double		r[DIM_NUM*N];	// where DIM_NUM=2 and N=500 in this case.
			*			double it_diff;	int it_num;	double energy;			
			*			pixkit::halftoning::ungrouped::cvt_ (dst, imgsize,2, 500, 1000, 1, 0, 10000, 40, 1, &seed, r, &it_num, &it_diff, &energy );
			*
			*			// more examples can be found at http://people.sc.fsu.edu/~jburkardt/cpp_src/cvt/cvt_prb.cpp
			*/
			void cvt_(cv::Mat &dst, const int imageSize, int dim_num, int n, int batch, int init, int sample, int sample_num, 
				int it_max, int it_fixed, int *seed, double *r, int *it_num, double *it_diff, double *energy);

		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Image compression
	namespace comp{

		bool	DDBTC2014(const cv::Mat &src,cv::Mat &dst,int blockSize);
		enum	ODBTC_TYPE{ODBTC_TYPE_ClusteredDot,ODBTC_TYPE_DispersedDot};
		bool	ODBTC(const cv::Mat &src,cv::Mat &dst,int blockSize,ODBTC_TYPE type);
		enum	EDBTC_TYPE{EDBTC_TYPE_Floyd,EDBTC_TYPE_Jarvis,EDBTC_TYPE_Stucki};
		bool	EDBTC(const cv::Mat &src,cv::Mat &dst,int blockSize,EDBTC_TYPE type);
		bool	BTC(const cv::Mat &src,cv::Mat &dst,int blockSize);
		bool	YangTsai1998(const cv::Mat &src3b, cv::Mat &dst3b, const int K = 256);

	}

	//////////////////////////////////////////////////////////////////////////
	/// Image enhancement related
	namespace enhancement{
		
		/// Local methods
		namespace local{
			bool	LCE_BSESCS2014(const cv::Mat &src,cv::Mat &dst,cv::Size blockSize);
			bool	Lal2014(const cv::Mat &src,cv::Mat &dst, cv::Size title, float L = 0.03,float K1 = 10,float K2 =0.5);
			bool	MSRCP2014(const cv::Mat &src,cv::Mat &dst,int Nscale);
			bool	Kimori2013(cv::Mat &src,cv::Mat &dst,cv::Size B, int N = 8);
			bool	POHE2013(const cv::Mat &src,cv::Mat &dst,const cv::Size blockSize,const cv::Mat &sum=cv::Mat(),const cv::Mat &sqsum=cv::Mat());
			bool	Sundarami2011(const cv::Mat &src,cv::Mat &dst, cv::Size N, float L = 0.03, float phi = 0.5);
			bool	LiuJinChenLiuLi2011(const cv::Mat &src,cv::Mat &dst,const cv::Size N);
			bool	LambertiMontrucchioSanna2006(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S);
			bool	FAHE2006(const cv::Mat &src1b,cv::Mat &dst1b,cv::Size blockSize);
			bool	YuBajaj2004(const cv::Mat &src,cv::Mat &dst,const int blockheight,const int blockwidth,const float C=0.85f,bool anisotropicMode=false,const float R=0.09f);
			bool	KimKimHwang2001(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S);
			bool	Stark2000(const cv::Mat &src,cv::Mat &dst,const int blockheight,const int blockwidth,const float alpha=0.5f,const float beta=0.5f);
			bool	MSRCR1997(const cv::Mat &src,cv::Mat &dst,int Nscale);
			bool	CLAHEnon1987(const cv::Mat &src,cv::Mat &dst, cv::Size nBlock, float L = 0.03);
			bool	CLAHE1987(const cv::Mat &src,cv::Mat &dst, cv::Size blockSize, float L = 0.03);
			bool	AHE1974(const cv::Mat &src1b,cv::Mat &dst1b,const cv::Size blockSize);
		}

		/// Global methods
		namespace global{
			bool	RajuNair2014(const cv::Mat &src,cv::Mat &dst);
			bool	CelikTjahjadi2012(cv::Mat &src,cv::Mat &dst,int N);
			bool	MaryKim2008(const cv::Mat &src, cv::Mat &dst,int MorD , int r=2);
			bool	WadudKabirDewanChae2007(const cv::Mat &src, cv::Mat &dst, const int x);
			bool	GlobalHistogramEqualization1992(const cv::Mat &src,cv::Mat &dst);
		}

	}

	//////////////////////////////////////////////////////////////////////////
	/// IQA related 
	namespace qualityassessment{

		// for contrast evaluation
		float EME(const cv::Mat &src,const cv::Size nBlocks,const short mode=1);
		float TEN(const cv::Mat &src);
		float AMBE(const cv::Mat &src1,const cv::Mat &src2);
		float CII(const cv::Mat &src1,const cv::Mat &src2);	// contrast improvement index (CII)

		// for evaluating noises
		float SNS(const cv::Mat &src1b,int ksize=25);	// speckle noise strength (SNS)

		// signal similarity 
  		float PSNR(const cv::Mat &src1,const cv::Mat &src2);

		// for halftone images
		float HPSNR(const cv::Mat &src1, const cv::Mat &src2);

		/**
		*	@brief		Display the difference of two Gaussian blurred images.
		*
		*	@paper		C. Schmaltz, P. Gwosdek, A. Bruhn, and J. Weickert, “Electrostatic halftoning,” Computer Graphics Forum, vol. 29, no. 8, pp. 2313-2327, 2010.
		*/
		bool GaussianDiff(cv::InputArray &_src1,cv::InputArray &_src2,double sd=1.);

		/**
		*	@brief		Get the power spectrum density by DFT.
		*
		*/
		bool PowerSpectrumDensity(cv::InputArray &_src,cv::OutputArray &_dst);

		/**
		*	@brief		Get averaged 
		*
		*	@paper		M. S. Bartlett, "The spectral analysis of two-dimensional point processes," Biometrika, Dec. 1964.
		*	
		*	@Note		1. Input should be generated from a constant grayscale.
		*				2. _src should be 256x(256x10), and output (_dst) will be 256x256.
		*/
		bool spectralAnalysis_Bartlett(cv::InputArray &_src,cv::OutputArray &_dst);

	}

}
#endif
