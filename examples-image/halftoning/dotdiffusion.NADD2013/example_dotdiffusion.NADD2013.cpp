// opencv related headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// system related headers
#include <iostream>

// pixkit-image
#include "../../../modules/pixkit-image/include/pixkit-image.hpp"

void main(){
	 

	//////////////////////////////////////////////////////////////////////////
	///// Select a CT Size
	std::cout	<<	"Select a size of CT:"	<<	std::endl
				<<	"\t1) 256"	<<	std::endl
				<<	"\t2) 512"		<<	std::endl;
	short	CTSize;
	std::cin	>>	CTSize;
	if(CTSize!=1&&CTSize!=2){
		CV_Error(CV_StsBadArg,"");
	}
	CTSize	=	CTSize*256;
	char	name[30];
	sprintf(name,"ct%.3d.map",CTSize);
	

	//////////////////////////////////////////////////////////////////////////
	///// [case01] ct generation and save
	pixkit::halftoning::dotdiffusion::CNADDCT	cct;
 	cct.generation(cv::Size(CTSize,CTSize));	// assign the size to ct
 	cct.save(name);	// Thus, this save (and entire [case01] can be perform 
	                // only once for many images with different sizes.
					// Maps of sizes 256 and 512 are also saved in 
					// pixkit/data/NADD2013.


	//////////////////////////////////////////////////////////////////////////
	///// [case02] load ct and perform nadd halftoning
	// load .map file
	cv::Mat	src,dst;
//	pixkit::halftoning::dotdiffusion::CNADDCT	cct;	// required when without [case01]
	cct.load(name);
	// load image 
	src	=	cv::imread("../../../data/lena.bmp",CV_LOAD_IMAGE_GRAYSCALE);
	if(!src.empty()){
		// process
		pixkit::halftoning::dotdiffusion::NADD2013(src,dst,cct);

		// write
		cv::imwrite("output.bmp",dst);

		// show results
		cv::namedWindow("src");
		cv::namedWindow("dst");
		cv::moveWindow("src",0,0);
		cv::moveWindow("dst",src.cols,0);
		cv::imshow("src",src);
		cv::imshow("dst",dst);
		cv::waitKey(0);
	}


}