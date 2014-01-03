//////////////////////////////////////////////////////////////////////////
// 
// SOURCE CODE: https://github.com/yunfuliu/pixkit
// 
// BEIRF: pixkit-ml contains machine learning related methods which have been published (on articles, e.g., journal or conference papers). 
//	In addition, some frequently used related tools are also involved.
// 
//////////////////////////////////////////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

#ifndef __PIXKIT_ML_HPP__
#define __PIXKIT_ML_HPP__

namespace pixkit{
	
	namespace clustering{
		/**
		* @brief		clustering the input data by using this function
		* @brief		paper: V. K. Dehariya, S. K. Shrivastava, R. C. Jain, "Clustering Of Image Data Set Using K-Means And Fuzzy K-Means Algorithms", IEEE International Conf. of CICN, 2010
		*
		* @author		HuangYu Liu (yf6204220@hotmail.com)
		* @date			Jan. 3, 2014
		* @version		1.0
		*
		* @param		src : Input data. The data type is vector.
		* @param		dst : Output data. The data type is vector.
		* @param		seedNum : How many seed do the user want to spread
		* @param		K : How many universe do the user want to cluster, for instance, if you want to segment the input data to two different parts, then you must set the argument to 2.
		* @param		m : The fuzzy coefficient. If set this argument close to 1.0, then the cluster effect like K-Means clustering.
		* @param		iterNum : Set the iteration number. If set this argument bigger, then the error of result will be much smaller.
		* @param		initialPosi : The argument is relate to the type, which is the next argument. The initial position can define by the user or computer.
		* @param		type : If user set the type to 1, then the initial position will define by the user. If user set the type to 2, then the initial position will define random by the computer.
		* @return		bool: true: successful, false: failure
		*/
		bool fuzzyCMeans(const vector<vector<double>> &src, vector<vector<double>> &dst, const int seedNum, const int K, const double m, const int iterNum, vector<vector<double>> &initialPosi = vector<vector<double>>(), TYPE type = FUZZYCM_RANDPOS);
	}

}
#endif