//////////////////////////////////////////////////////////////////////////
// 
// pixkit-image.hpp
//
// SOURCE CODE: https://github.com/yunfuliu/pixkit
// 
// BEIRF: pixkit-image contains image processing related methods which have been published (on articles, e.g., journal or conference papers). 
//	In addition, some frequently used related tools are also involved.
// 
//////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

		// peer group filtering
		bool PGF1999(const cv::Mat &src,cv::Mat &dst,int &blocksize,double sigma=1.,int alpha=16);
				
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
			bool FloydSteinberg1975(const cv::Mat &src,cv::Mat &dst);
			bool Jarvis1976(const cv::Mat &src, cv::Mat &dst);
			bool Stucki1981(const cv::Mat &src, cv::Mat &dst);
			bool ShiauFan1996(const cv::Mat &src, cv::Mat &dst);
			bool Ostromoukhov2001(const cv::Mat &src, cv::Mat &dst);
			bool ZhouFang2003(const cv::Mat &src, cv::Mat &dst);
		}

		/// iterative
		namespace iterative{
			// efficient DBS
			bool LiebermanAllebach1997(const cv::Mat &src1b, cv::Mat &dst1b,double *coeData=NULL,int FilterSize=7,bool cppmode=false);
			bool dualmetricDBS2002(const cv::Mat &src1b,cv::Mat &dst1b);

			// Electrostatic halftoning
			bool ElectrostaticHalftoning2010(const cv::Mat &src, cv::Mat &dst, int InitialCharge, int Iterations, int GridForce, int Shake, int Debug);

		}

		/// Ordered Dither related
		namespace ordereddithering{

			// conventional od methods
			enum DitherArray_TYPE { DispersedDot, ClusteredDot };
			bool Ulichney1987(const cv::Mat &src, cv::Mat &dst, DitherArray_TYPE = DispersedDot);

			bool KackerAllebach1998(const cv::Mat &src1b, cv::Mat &dst1b);
		}

		/// Dot diffusion related
		namespace dotdiffusion{

			class CLiuGuo2015{

			public:


				/*
				*	@param	pthfname_resources: Please indicate this to `PIXKIT_ROOT/data/LiuGuo2015/data_LiuGuo2015.xml` for the performance of the paper. 
				*/
				CLiuGuo2015(std::string pthfname_resources);
				virtual	~CLiuGuo2015();

				// halftoning process
				bool process(const cv::Mat &src1b, cv::Mat &dst1b);

				// Get the processing orders of the pixels.
				void getPointList(const cv::Size imgSize);

			private:
				class CPARAMS{
				public:
					/*
					*	@[0]	threshold
					*	@[1]	afa
					*	@[2]	beta
					*/
					float	coe[3];
				};

				cv::Size	cmsize;		// class matrix size.
				cv::Mat		cct;		// class tiling
				cv::Mat		cct_ori;	// original class tiling
				std::vector<std::vector<CPARAMS>>	paramsmap;	// parameters
				std::vector<std::vector<cv::Point>>	pointlist;	// processing order location list.

				/*
				*	read class tiling (CT)
				*/
				bool ctread(const cv::Mat &src, const cv::Size cmsize, cv::Mat &cct1b);

				/*
				*	read the map of parameters
				*	@param	paramsmap[grayscale][order]
				*/
				bool read_paramsmap(std::vector<cv::Mat> &vec_src, std::vector<std::vector<CPARAMS>> &paramsmap);

			};

			class CNADDCT{
			public:
				int				m_CT_height;	// CT's height and width
				int				m_CT_width;
				unsigned char	**m_ct;
				std::vector<std::vector<cv::Point>>	pointList;
				cv::Size		imgSize_pointList;
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

			bool GuoLiu2009(const cv::Mat &src, cv::Mat &dst,const int ClassMatrixSize);
			
			bool LippensPhilips2007(const cv::Mat &src, cv::Mat &dst);

			bool Knuth1987(const cv::Mat &src, cv::Mat &dst);

			bool MeseVaidyanathan2000(const cv::Mat &src, cv::Mat &dst, int ClassMatrixSize = 8);

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
			*			int seed	= 123456789,imgsize=512;
			*			const	int	DIM_NUM=2,N=1000;
			*			double		r[DIM_NUM*N];	// where DIM_NUM=2 and N=500 in this case.
			*			double it_diff;	int it_num;	double energy;	
			*			pixkit::halftoning::ungrouped::cvt_ (dst, imgsize,2, 500, 1000, 1, 0, 10000, 40, 1, &seed, r, &it_num, &it_diff, &energy );
			*
			*			// more examples can be found at http://people.sc.fsu.edu/~jburkardt/cpp_src/cvt/cvt_prb.cpp
			*/
			void cvt_(cv::Mat &dst, const int imageSize, int dim_num, int n, int batch, int init, int sample, int sample_num, 
				int it_max, int it_fixed, int *seed, double *r, int *it_num, double *it_diff, double *energy);

			/*
			* @brief	This function generates a two-component Gaussian model to fit the Nasanen's HVS model.
			*			
			* @ref		S. H. Kim and J. P. Allebach, "Impact of HVS Models on Model-based halftoning," IEEE TIP, vol. 11, no. 3, March 2002.
			*
			* @param	dst: dst model.
			* @param	others: Please refer to the paper.
			*/
			bool generateTwoComponentGaussianModel(cv::Mat &dst1d,float k1=40.8,float k2=9.03,float sd1=0.0384,float sd2=0.105);

		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// Multitoning
	namespace multitoning{
		namespace ordereddithering{

			// dms
			bool DMS2012_genDitherArray(std::vector<cv::Mat> &vec_DA1b, int daSize, int nColors);
			bool DMS2012(const cv::Mat &src1b, const std::vector<cv::Mat> &vec_DA,cv::Mat &dst1b);

			// generate green noise
			bool ChanduStanichWuTrager2014_genDitherArray(std::vector<cv::Mat> &vec_DA1b, int daSize, int nColors,float D_min);
			bool ChanduStanichWuTrager2014(const cv::Mat &src1b, const std::vector<cv::Mat> &vec_DA1b,cv::Mat &dst1b);

		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Image compression
	namespace comp{

		// BTC
		bool	DBSBTC2011(cv::Mat src, cv::Mat &dst, int blockSize = 8);
		bool	DDBTC2014(const cv::Mat &src,cv::Mat &dst,int blockSize);
		enum	ODBTC_TYPE{ODBTC_TYPE_ClusteredDot,ODBTC_TYPE_DispersedDot};
		bool	ODBTC(const cv::Mat &src,cv::Mat &dst,int blockSize,ODBTC_TYPE type);
		enum	EDBTC_TYPE{EDBTC_TYPE_Floyd,EDBTC_TYPE_Jarvis,EDBTC_TYPE_Stucki};
		bool	EDBTC(const cv::Mat &src,cv::Mat &dst,int blockSize,EDBTC_TYPE type);
		bool	BTC(const cv::Mat &src,cv::Mat &dst,int blockSize);
		bool	YangTsai1998(const cv::Mat &src3b, cv::Mat &dst3b, const int K = 256);
		
		// ColorBTC
		namespace ColorBTC{
			bool	CCC1986(const cv::Mat &src,cv::Mat &dst, int BlockSize);
			bool	FS_BMO2014(const cv::Mat &src,cv::Mat &dst, int BlockSize, int MoreCompressFlag=1, int THBO=15);
			bool	IBTC_KQ2014(const cv::Mat &src,cv::Mat &dst, int BlockSize);
		}

		// JPEG
		bool	JPEG(const cv::Mat &src1b, cv::Mat &dst1b, const int jpeg_quality);
	}

	//////////////////////////////////////////////////////////////////////////
	/// Image enhancement related
	namespace enhancement{
		
		/// Local methods
		namespace local{
			bool	LCE_BSESCS2014(const cv::Mat &src,cv::Mat &dst,cv::Size blockSize);
			bool	Lal2014(const cv::Mat &src,cv::Mat &dst, cv::Size title, float L = 0.03,float K1 = 10,float K2 =0.5);
			bool	MSRCP2014(const cv::Mat &src,cv::Mat &dst);
			bool	WangZhengHuLi2013(const cv::Mat &src,cv::Mat &dst);
			bool	Kimori2013(cv::Mat &src,cv::Mat &dst,cv::Size B, int N = 8);
			bool	POHE2013(const cv::Mat &src,cv::Mat &dst,const cv::Size blockSize,const cv::Mat &sum=cv::Mat(),const cv::Mat &sqsum=cv::Mat());
			bool    LiWangGeng2011(const cv::Mat & ori,cv::Mat &ret);
			bool	Sundarami2011(const cv::Mat &src,cv::Mat &dst, cv::Size N, float L = 0.03, float phi = 0.5);
			bool	LiuJinChenLiuLi2011(const cv::Mat &src,cv::Mat &dst,const cv::Size N);
			bool    NRCIR2009(const cv::Mat ori,cv::Mat &ret);
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
			bool    LeeLeeKim2013(const cv::Mat ori,cv::Mat &ret,double alpha=2.5);
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
		float IW_PSNR(const cv::Mat &src1,const cv::Mat &src2);
		// signal similarity for halftone images
		float HPSNR(const cv::Mat &src1, const cv::Mat &src2,const int ksize=7);

		bool GaussianDiff(cv::InputArray &_src1,cv::InputArray &_src2,double sd=1.);
	
		// Get averaged power spectrum density 
		bool PowerSpectrumDensity(cv::InputArray &_src,cv::OutputArray &_dst, bool flag_display=true);	
		bool spectralAnalysis_Bartlett(cv::InputArray &_src,cv::OutputArray &_dst1f,const cv::Size specSize,const int rounds=10,const bool rand_sample=false,bool flag_display=true);
		bool RAPSD(const cv::Mat Spectrum1f, cv::Mat &RAPSD1f, cv::Mat &Anisotropy1f);

		// image similarity
		float SSIM(const cv::Mat &src1, const cv::Mat &src2);	
		float MSSIM(const cv::Mat &src1, const cv::Mat &src2, int HVSsize=11,  double* lu_co_st=NULL);
		float MS_SSIM(const cv::Mat &src1, const cv::Mat &src2, int HVSsize=11);
		float IW_SSIM(const cv::Mat &src1,const cv::Mat &src2);
		float GMSD(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst);
	}

}
#endif
