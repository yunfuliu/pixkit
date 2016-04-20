#include "../include/pixkit-file.hpp"
#include <string>
#include <fstream>

using namespace cv;

pixkit::vecS pixkit::loadStrList(CStr &fName){
	std::ifstream fIn(fName);
	std::string line;
	pixkit::vecS strs;
	while(getline(fIn, line) && line.size())
		strs.push_back(line);
	return strs;
}

bool pixkit::write_vecMat(std::string fname,const std::vector<cv::Mat> &vec){
	FileStorage	fs(fname,FileStorage::WRITE);
	fs	<<	"size"	<<	((int)vec.size());
	for(int i=0;i<vec.size();i++){
		fs	<<	format("mat_%d",i)	<<	vec[i];
	}
	fs.release();
	return true;
}

bool pixkit::read_vecMat(std::string fname,std::vector<cv::Mat> &vec){
	FileStorage	fs(fname,FileStorage::READ);
	int	vec_size;
	fs["size"]	>>	vec_size;
	vec.resize(vec_size);
	Mat	aa;
	for(int i=0;i<vec.size();i++){
		fs[format("mat_%d",i)]	>>	vec[i];
	}
	fs.release();
	return true;
}