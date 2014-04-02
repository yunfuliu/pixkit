// opencv related headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// system related headers
#include <iostream>

// pixkit-image
#include "../../../modules/pixkit-ml/include/pixkit-ml.hpp"

void main(){

	// sample point definition (2d case)
	std::vector<std::vector<double>>	src(6,std::vector<double>(2,0)),dst;
	src[0][0] = 1;	src[0][1] = 1;
	src[1][0] = 1;	src[1][1] = 2;
	src[2][0] = 2;	src[2][1] = 1;
	src[3][0] = 10;	src[3][1] = 3;
	src[4][0] = 10;	src[4][1] = 5;
	src[5][0] = 12;	src[5][1] = 4;
	
	// unsupervised clustering with fuzzy c means
	pixkit::clustering::fuzzyCMeans(src,dst,1,2,2.0,4);

	// output display
	for(int i = 0; i < src.size(); i++){
		for(int j = 0; j < src[0].size(); j++){
			std::cout << dst[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout	<<	std::endl;

	system("pause");
}