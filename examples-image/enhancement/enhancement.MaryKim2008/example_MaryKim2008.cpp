// opencv related headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// system related headers
#include <iostream>

// pixkit-image
#include "../../../modules/pixkit-image/include/pixkit-image.hpp"


void main(){
	cv::Mat	src,dst;
	src	=	cv::imread("../../../data/lena.bmp",CV_LOAD_IMAGE_GRAYSCALE);
	if(!src.empty()){
		// process
		if(pixkit::enhancement::local::Lal2014(src,dst,cv::Size(41,41),0.03)){
			// write output
			cv::imwrite("output.bmp",dst);
			// show image
			cv::imshow("src",src);
			cv::imshow("dst",dst);			
			cv::waitKey(0);
		}
	}
}