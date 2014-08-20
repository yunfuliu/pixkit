// system related headers
#include <vector>

#ifndef __PIXKIT_FILE_HPP__
#define __PIXKIT_FILE_HPP__
namespace pixkit{
	typedef const std::string	CStr;
	typedef std::vector<std::string>	vecS;
	vecS loadStrList(CStr &fName);
}
#endif