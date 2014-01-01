// opencv related headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// system related headers
#include <iostream>

// pixkit-image
#include "../include/pixkit-image.hpp"

void describe(std::string str){
	std::cout	<<	"\t"	<<	str	<<	"\t... ";
}
int checkout(bool input,bool compare){
	if(compare==input){
		std::cout	<<	"ok"	<<	std::endl;
		return 0;
	}else{
		std::cout	<<	"x"		<<	std::endl;
		return 1;
	}
}

int main(int argc,char* argv[]){
	
	cv::Mat	src,dst;
	int	nCount	=	0;	// calc the counts of errors


	//////////////////////////////////////////////////////////////////////////
	///// ** (test) empty input, [output] should be false
	std::cout	<<	"(test) empty input, [output] should be false"	<<	std::endl;

	// [pixkit::halftoning::dotdiffusion::GuoLiu2009]
	describe("[pixkit::halftoning::dotdiffusion::GuoLiu2009]");
	nCount+=checkout(pixkit::halftoning::dotdiffusion::GuoLiu2009(src,dst,8),false);

	// [pixkit::halftoning::dotdiffusion::NADD2013]
	pixkit::halftoning::dotdiffusion::CNADDCT cct;
	cct.generation(cv::Size(256,256));
	describe("[pixkit::halftoning::dotdiffusion::NADD2013]");
	nCount+=checkout(pixkit::halftoning::dotdiffusion::NADD2013(src,dst,cct),false);
	std::cout	<<	std::endl	<<	std::endl;


	//////////////////////////////////////////////////////////////////////////
	///// ** (test) dst type, [output] should be CV_8UC1
	std::cout	<<	"(test) dst type, [output] should be CV_8UC1"	<<	std::endl;

	// [pixkit::halftoning::dotdiffusion::GuoLiu2009]
	describe("[pixkit::halftoning::dotdiffusion::GuoLiu2009]");
	pixkit::halftoning::dotdiffusion::GuoLiu2009(src,dst,8);
	nCount+=checkout(true,dst.type()==CV_8UC1?true:false);

	// [pixkit::halftoning::dotdiffusion::NADD2013]
	describe("[pixkit::halftoning::dotdiffusion::NADD2013]");
	pixkit::halftoning::dotdiffusion::NADD2013(src,dst,cct);
	nCount+=checkout(true,dst.type()==CV_8UC1?true:false);
	std::cout	<<	std::endl	<<	std::endl;

	
	//////////////////////////////////////////////////////////////////////////
	///// ** (test) check channel types
	std::cout	<<	"(test) check channel types"	<<	std::endl;
	int	chennel_types[]={CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F};	// 10 types
	for(int i=0;i<10;i++){
		src.create(cv::Size(512,512),chennel_types[i]);
		src.setTo(rand()%256);
		std::cout << chennel_types[i];

		// [pixkit::halftoning::dotdiffusion::GuoLiu2009]
		describe("[pixkit::halftoning::dotdiffusion::GuoLiu2009]");
		nCount+=checkout(pixkit::halftoning::dotdiffusion::GuoLiu2009(src,dst,8),chennel_types[i]==CV_8UC1?true:false);

		// [pixkit::halftoning::dotdiffusion::NADD2013]
		describe("[pixkit::halftoning::dotdiffusion::NADD2013]");
		nCount+=checkout(pixkit::halftoning::dotdiffusion::NADD2013(src,dst,cct),chennel_types[i]==CV_8UC1?true:false);
	}
	std::cout	<<	std::endl	<<	std::endl;


	//////////////////////////////////////////////////////////////////////////
	///// ** (test) various input sizes
	std::cout	<<	"(test) various input sizes"	<<	std::endl;
	std::vector<cv::Size>	input_sizes;
	input_sizes.push_back(cv::Size(0,0));		// false
	input_sizes.push_back(cv::Size(0,1));		// false
	input_sizes.push_back(cv::Size(1,0));		// false
	input_sizes.push_back(cv::Size(1,1));		// true
	input_sizes.push_back(cv::Size(100,200));	// true
	input_sizes.push_back(cv::Size(200,100));	// true
	input_sizes.push_back(cv::Size(1001,99));	// true
	input_sizes.push_back(cv::Size(501,277));	// true
	for(int i=0;i<input_sizes.size();i++){
		src.create(input_sizes[i],CV_8UC1);
		src.setTo(rand()%256);
		std::cout << input_sizes[i];

		// [pixkit::halftoning::dotdiffusion::GuoLiu2009]
		describe("[pixkit::halftoning::dotdiffusion::GuoLiu2009]");
		nCount+=checkout(pixkit::halftoning::dotdiffusion::GuoLiu2009(src,dst,8),i<3?false:true);

		// [pixkit::halftoning::dotdiffusion::NADD2013]
		describe("[pixkit::halftoning::dotdiffusion::NADD2013]");
		nCount+=checkout(pixkit::halftoning::dotdiffusion::NADD2013(src,dst,cct),i<3?false:true);
	}
	std::cout	<<	std::endl	<<	std::endl;


	//////////////////////////////////////////////////////////////////////////
	///// ** (test) various parameters
	std::cout	<<	"(test) various parameters"	<<	std::endl;
	for(int i=0;i<=20;i++){
		src.create(cv::Size(16,16),CV_8UC1);
		src.setTo(rand()%256);
		std::cout << i;

		// [pixkit::halftoning::dotdiffusion::GuoLiu2009]
		describe("[pixkit::halftoning::dotdiffusion::GuoLiu2009]");
		nCount+=checkout(pixkit::halftoning::dotdiffusion::GuoLiu2009(src,dst,i),i==8||i==16);	// supports only 8 and 16 cm sizes
	}
	std::cout	<<	std::endl	<<	std::endl;

	//////////////////////////////////////////////////////////////////////////
	///// ** (test) check output should be black/white
	std::cout	<<	"(test) check output should be black/white"	<<	std::endl;
	int	error_pixel;
	for(int i=0;i<10;i++){
		src.create(cv::Size(256,256),CV_8UC1);
		src.setTo(rand()%256);
		std::cout << i;

		// [pixkit::halftoning::dotdiffusion::GuoLiu2009]
		describe("[pixkit::halftoning::dotdiffusion::GuoLiu2009]");
		pixkit::halftoning::dotdiffusion::GuoLiu2009(src,dst,8);
		error_pixel=0;
		for(int m=0;m<dst.rows;m++){
			for(int n=0;n<dst.cols;n++){
				if(dst.data[m*dst.cols+n]!=0&&dst.data[m*dst.cols+n]!=255){
					error_pixel++;
				}
			}
		}
		nCount+=checkout(true,error_pixel==0?true:false);

		// [pixkit::halftoning::dotdiffusion::NADD2013]
		describe("[pixkit::halftoning::dotdiffusion::NADD2013]");
		pixkit::halftoning::dotdiffusion::NADD2013(src,dst,cct);
		error_pixel=0;
		for(int m=0;m<dst.rows;m++){
			for(int n=0;n<dst.cols;n++){
				if(dst.data[m*dst.cols+n]!=0&&dst.data[m*dst.cols+n]!=255){
					error_pixel++;
				}
			}
		}
		nCount+=checkout(true,error_pixel==0?true:false);

	}
	std::cout	<<	std::endl	<<	std::endl;


	//////////////////////////////////////////////////////////////////////////
	std::cout	<<	nCount	<<	"\terror tests"	<<	std::endl;
	return nCount;
}