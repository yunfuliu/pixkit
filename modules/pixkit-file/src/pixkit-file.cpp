#include "../include/pixkit-file.hpp"
#include <string>
#include <fstream>

pixkit::vecS pixkit::loadStrList(CStr &fName){
	std::ifstream fIn(fName);
	std::string line;
	pixkit::vecS strs;
	while(getline(fIn, line) && line.size())
		strs.push_back(line);
	return strs;
}