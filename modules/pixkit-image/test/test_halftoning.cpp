// opencv related headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// system related headers
#include <iostream>
#include <ctime>

// pixkit-image
#include "../include/pixkit-image.hpp"

//////////////////////////////////////////////////////////////////////////
// others
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

//////////////////////////////////////////////////////////////////////////
bool help(){

	std::cout	<<	"[usage]: .exe [delay]"	<<	std::endl
		<<	"\tdelay:\t>0: show demo images with [delay] ms"	<<	std::endl
		<<	"\t\t0: show no image"	<<	std::endl;
	return false;
}

//////////////////////////////////////////////////////////////////////////
///// functions
typedef bool (*predicate)(cv::Mat&,cv::Mat&);
std::vector<predicate>	function_vec;
bool dd_GuoLiu2009(cv::Mat &src,cv::Mat &dst){
	describe("[pixkit::halftoning::dotdiffusion::GuoLiu2009]");
	return pixkit::halftoning::dotdiffusion::GuoLiu2009(src,dst,8);
}
bool dd_NADD2013(cv::Mat &src,cv::Mat &dst){
	describe("[pixkit::halftoning::dotdiffusion::NADD2013]");
	pixkit::halftoning::dotdiffusion::CNADDCT cct;
	cct.generation(cv::Size(256,256));
	return pixkit::halftoning::dotdiffusion::NADD2013(src,dst,cct);
}
///// construction
void construct(){
	function_vec.push_back(dd_GuoLiu2009);
	function_vec.push_back(dd_NADD2013);
}


int main(int argc,char* argv[]){


	if(argc<2){
		return help();
	}
	short	delay	=	-1;	// default
	if(argc>1){
		delay	=	atoi(argv[1]);
	}
	srand(time(NULL));


	//////////////////////////////////////////////////////////////////////////
	///// initial
	construct();
	cv::Mat	src,dst;
	int	nCount	=	0;	// calc the counts of errors


	//////////////////////////////////////////////////////////////////////////
	///// ** (test) empty input, [output] should be false
	std::cout	<<	"(test) empty input, [output] should be false"	<<	std::endl;
	for(int k=0;k<function_vec.size();k++){
		nCount+=checkout(function_vec[k](src,dst),false);
	}
	std::cout	<<	std::endl	<<	std::endl;


	//////////////////////////////////////////////////////////////////////////
	///// ** (test) dst type, [output] should be CV_8UC1
	std::cout	<<	"(test) dst type, [output] should be CV_8UC1"	<<	std::endl;
	src.create(cv::Size(256,256),CV_8UC1);
	src.setTo(rand()%256);
	for(int k=0;k<function_vec.size();k++){
		nCount+=checkout(function_vec[k](src,dst),dst.type()==CV_8UC1?true:false);
		if(delay>0){
			cv::imshow("output",dst);
			cv::waitKey(delay);
		}
	}
	std::cout	<<	std::endl	<<	std::endl;

	
	//////////////////////////////////////////////////////////////////////////
	///// ** (test) check channel types
	std::cout	<<	"(test) check channel types"	<<	std::endl;
	int	chennel_types[]={CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F};	// 10 types
	for(int k=0;k<function_vec.size();k++){
		for(int i=0;i<10;i++){
			src.create(cv::Size(256,256),chennel_types[i]);
			src.setTo(rand()%256);
			std::cout << chennel_types[i];
			nCount+=checkout(function_vec[k](src,dst),chennel_types[i]==CV_8UC1?true:false);
		}
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
	for(int k=0;k<function_vec.size();k++){
		for(int i=0;i<input_sizes.size();i++){
			src.create(input_sizes[i],CV_8UC1);
			src.setTo(rand()%256);
			std::cout << input_sizes[i];
			nCount+=checkout(function_vec[k](src,dst),i<3?false:true);
			if(delay>0){
				cv::imshow("output",dst);
				cv::waitKey(delay);
			}
		}
	}
	std::cout	<<	std::endl	<<	std::endl;


	//////////////////////////////////////////////////////////////////////////
	///// ** (test) check output should be black/white
	std::cout	<<	"(test) check output should be black/white"	<<	std::endl;
	int	error_pixel;
	for(int k=0;k<function_vec.size();k++){

		for(int i=0;i<10;i++){
			src.create(cv::Size(256,256),CV_8UC1);
			int	randv	=	rand()%256;
			src.setTo(randv);
			std::cout << randv;

			// do test
			function_vec[k](src,dst);
			error_pixel=0;
			for(int m=0;m<dst.rows;m++){
				for(int n=0;n<dst.cols;n++){
					if(dst.data[m*dst.cols+n]!=0&&dst.data[m*dst.cols+n]!=255){
						error_pixel++;
					}
				}
			}
			nCount+=checkout(true,error_pixel==0?true:false);
			if(delay>0){
				cv::imshow("output",dst);
				cv::waitKey(delay);
			}
		}
	}
	std::cout	<<	std::endl	<<	std::endl;


	//////////////////////////////////////////////////////////////////////////
	std::cout	<<	nCount	<<	" errors."	<<	std::endl;
	return nCount;
}