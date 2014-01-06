#include "../include/pixkit-ml.hpp"
#include <ctime>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

using namespace std;

//Calculate the euclidean distance between the two points
double radius(const vector<vector<double>> &src, const int c1, const vector<vector<double>> &randomPosi, const int k1){
	double euclideanDistance = 0.0;

	//Calculate the euclidean distance in multi-dimension
	for(int d = 0; d < src[0].size(); d++){
		euclideanDistance += fabs((double)(src[c1][d] - randomPosi[k1][d]));
	}

	return euclideanDistance;
}
bool pixkit::clustering::fuzzyCMeans(const std::vector<std::vector<double>> &src, std::vector<std::vector<double>> &dst, const int seedNum, const int K, const double m, const int iterNum, std::vector<std::vector<double>> &initialPosi, pixkit::clustering::FUZZYCM_TYPE type, bool debug){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(iterNum <= 0){
		cout << "The iteration number must setting greater than 0" << endl;
		return false;
	}else if(K <= 0){
		cout << "The parameter K must setting greater than 0" << endl;
		return false;
	}else if(seedNum <= 0){
		cout << "The seed number must setting greater than 0" << endl;
		return false;
	}else if(m <= 1.0){
		cout << "The parameter m must setting greater than 1.0" << endl;
		return false;
	}else if(src.size() <= 0 || src[0].size() <= 0){
		cout << "Please input the data" << endl;
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	///// process
	int i, j, k, d, s;
	double vectorSize = src.size();		//	The vector height
	double dimension = src[0].size();	//	The vector width
	double minValue = 50000.0;			//	The minimum distance
	double distance = 0.0;
	srand(time(NULL));

	//	Destination vector size defined by the source vector size
	dst = vector<vector<double>>(vectorSize, vector<double>(dimension, 0));

	vector<double> weightSum(vectorSize, 0);

	vector<double> srcLabel(vectorSize, 0);

	vector<vector<double>> colorSum(K, vector<double>(dimension, 0));
	vector<vector<double>> sum(K, vector<double>(dimension, 0));

	vector<vector<double>> fcmWeight(vectorSize, vector<double>(K, 0));

	vector<double> dimensionMaxValue(dimension, 0);
	vector<vector<double>> fcmError(seedNum, vector<double>(K, 0));

	vector<double> minError(seedNum, 9999999999);

	for(s = 0; s < seedNum; s++){
		if(type == 1){
			initialPosi = vector<vector<double>>(K, vector<double>(dimension, 0));

			//	In default case, the initial point define by the computer using the random function
			for(d = 0; d < dimension; d++){
				dimensionMaxValue[d] = 0.0;

				//	Detect the dimension of input database
				for(i = 0; i < vectorSize; i++){
					if(src[i][d] > dimensionMaxValue[d]){
						dimensionMaxValue[d] = src[i][d];
					}
				}
			}

			//	Starting to generate the random initial point
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for(i = 0; i < K; i++){
				for(d = 0; d < dimension; d++){
					initialPosi[i][d] = rand() % (int)dimensionMaxValue[d];
				}
			}
		}

		int iter;
		for(iter = 0; iter < iterNum; iter++){
			//	Initialize
			for(i = 0; i < K; i++){
				for(d = 0; d < dimension; d++){
					colorSum[i][d] = 0.0;
					sum[i][d] = 0.0;
				}
			}

			for(i = 0; i < vectorSize; i++){
				weightSum[i] = 0.0;
			}

			double demSum = 0.0, dstRaidusValue;

			//	Fuzzy C-Means Clustering
			for(i = 0; i < vectorSize; i++){
				minValue = 50000.0;

				for(k = 0; k < K; k++){
					dstRaidusValue = radius(src, i, initialPosi, k);

					demSum = 0.0;
					for(int kk = 0; kk < K; kk++){
						distance = radius(src, i, initialPosi, kk);
						demSum += pow((dstRaidusValue / (distance + 0.0000000001)), 2.0/(m-1.0));
					}
					//	Calculate the relationship from this destinate point to the other points.
					fcmWeight[i][k] = 1.0/(demSum + 0.0000000001);
				}
			}

			for(i = 0; i < vectorSize; i++){
				for(d = 0; d < dimension; d++){
					for(k = 0; k < K; k++){
						//	Calculate the destinate point
						colorSum[k][d] += src[i][d] * pow(fcmWeight[i][k], m);

						//	Calculate the sum of the point to the other point
						sum[k][d] += pow(fcmWeight[i][k], m);
					}
				}
			}

			//	Calculate the new focus
			for(k = 0; k < K; k++){
				for(d = 0; d < dimension; d++){
					initialPosi[k][d] = colorSum[k][d] / (sum[k][d] + 0.0000000001);
				}
			}
		}

		//	Find which points is closet to the initial point or focus point, then label it.
		for(int i = 0; i < vectorSize; i++){
			minValue = 50000.0;

			for(k = 0; k < K; k++){
				distance = radius(src, i, initialPosi, k);

				if(distance < minValue){
					minValue = distance;
					srcLabel[i] = k;
				}
			}
		}

		//	Calculate the error
		for(i = 0; i < vectorSize; i++){
			for(d = 0; d < dimension; d++){
				fcmError[s][srcLabel[i]] += fabs((double)(src[i][d] - initialPosi[srcLabel[i]][d]));
			}
		}
	}

	//	Find the minimum error
	for(i = 0; i < seedNum; i++){
		for(k = 0; k < K; k++){
			if(fcmError[i][k] > 0.0){
				if(fcmError[i][k] < minError[i]){
					minError[i] = fcmError[i][k];
				}
			}
		}
	}

	if(debug){
		cout << "=========== Minimum Error ===========" << endl;
		for(i = 0; i < seedNum; i++){
			cout << "Seed index = " << i << " || Error: " << (double)minError[i] << endl;
		}
		cout << "=====================================" << endl;
	}

	//	Output the Results
	for(i = 0; i < vectorSize; i++){
		for(d = 0; d < dimension; d++){
			dst[i][d] = colorSum[srcLabel[i]][d] / sum[srcLabel[i]][d];
		}
	}
	return true;
}