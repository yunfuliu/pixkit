// system related headers
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

#ifndef __PIXKIT_FILE_HPP__
#define __PIXKIT_FILE_HPP__
namespace pixkit{
	typedef const std::string	CStr;
	typedef std::vector<std::string>	vecS;
	vecS loadStrList(CStr &fName);

	// vecMat
	bool write_vecMat(std::string fname,const std::vector<cv::Mat> &vec);
	bool read_vecMat(std::string fname,std::vector<cv::Mat> &vec);
}
#endif