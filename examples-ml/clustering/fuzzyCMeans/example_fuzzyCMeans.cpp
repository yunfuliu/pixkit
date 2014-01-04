// opencv related headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// system related headers
#include <iostream>

// pixkit-image
#include "../../../modules/pixkit-ml/include/pixkit-ml.hpp"

void main(){



	std::vector<std::vector<double>>	src(6,std::vector<double>(2,0)),dst;
	src[0][0]=1;	src[0][1]=1;
	src[1][0]=1;	src[1][1]=2;
	src[2][0]=2;	src[2][1]=1;
	src[3][0]=1+5;	src[3][1]=1+5;
	src[4][0]=1+5;	src[4][1]=2+5;
	src[5][0]=2+5;	src[5][1]=1+5;

	pixkit::clustering::fuzzyCMeans(src,dst,0,2,2.0,1);



	

	std::cout		<<	std::endl;


}