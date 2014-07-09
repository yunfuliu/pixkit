// system related headers
#include <iostream>
#include <vector>
#include <fstream>

#ifndef __PIXKIT_FILE_HPP__
#define __PIXKIT_FILE_HPP__

typedef const std::string	CStr;
typedef std::vector<std::string>	vecS;

vecS loadStrList(CStr &fName){
	std::ifstream fIn(fName);
	std::string line;
	vecS strs;
	while(getline(fIn, line) && line.size())
		strs.push_back(line);
	return strs;
}

#endif