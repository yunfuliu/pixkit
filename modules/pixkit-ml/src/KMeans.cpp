#include "../include/pixkit-ml.hpp"
#include <ctime>

using namespace std;

void calculate(vector<vector<double>> &input, vector<vector<double>> &out, vector<vector<double>> &sum){

	vector<int>	lable(input.size(),(-1));	//	Labeling
	vector<int>	num(out.size(),0);			//	Check that how many points in each group
	double dist = 0,min=5000.0;


	#pragma region	Calculate distance
		for(int i=0;i<input.size();i++){	//	Calculate distance and label that which point correspond to which group
			for(int j=0;j<out.size();j++){
				for(int k=0;k<input[0].size();k++){

					dist+=fabs( input[i][k] - out[j][k] );	//	Calculate the distance between the input data and mean
				}

				if(dist<min){	//	Compare the minimum value and labeling it

					min=dist;
					dist=0;
					lable[i]=j;
				}	
			}
			dist=0;
			min=5000.0;
		}
	#pragma endregion

	#pragma region	Calculate the sum, update mean value
		for(int i=0;i<input.size();i++){
			for(int j=0;j<input[0].size();j++){
				int la=lable[i];
				sum[la][j]+=input[i][j];
			}
			num[lable[i]]++;
		}

		for(int i=0;i<out.size();i++){
			for(int j=0;j<out[0].size();j++){
				sum[i][j]=sum[i][j]/num[i];	//	Update mean value
			}
		}
	#pragma endregion
}
bool pixkit::clustering::KMeans(std::vector<std::vector<double>> &src, std::vector<std::vector<double>> &dst, int K, int iter, pixkit::clustering::KM_TYPE type){

	double vectorSize = src.size();		//	The vector height
	double dimension = src[0].size();	//	The vector width
	
	vector<double> dimensionMaxValue(dimension, 0);
	dst = vector<vector<double>>(K, vector<double>(src[0].size(), 0));

	srand(time(NULL));

	if( src[0].size() != dst[0].size()){	//	Determind the dimension between the input data and the output data is same or not

		cout<<"The input data and the output need be the same dimension\n";
		return false;
	}

	if(type == pixkit::clustering::KM_RANDPOS){	//	Generate random initial points

		//	In default case, the initial point define by the computer using the random function
		for(int d = 0; d < dimension; d++){
			dimensionMaxValue[d] = 0.0;

			//	Detect the dimension of input database
			for(int i = 0; i < vectorSize; i++){
				if(src[i][d] > dimensionMaxValue[d]){
					dimensionMaxValue[d] = src[i][d];
				}
			}
		}

		for(int i = 0; i < K; i++){
			for(int d = 0; d < dimension; d++){
				dst[i][d] = rand() % (int)dimensionMaxValue[d];
			}
		}
	}

	vector<double> oldlength(dst.size(),1);
	vector<double> newlength(dst.size(),0);
	vector<vector<double>>	sum(dst.size(),vector<double>(dst[0].size(),0));	//	Calculate the sum
	
	if(iter != -1){

		while (iter !=0 ){

			calculate(src,dst,sum);	//	Calculate new mean value

			for(int i=0;i<dst.size();i++){
				for(int j=0;j<dst[0].size();j++){
					dst[i][j]=sum[i][j];
					sum[i][j]=0;
				}
			}
			
			iter--;
		}
	}
	else 
	{
		int error=1;
		while(error !=0){

			calculate(src,dst,sum);
			error=0;

			for(int i=0;i<dst.size();i++){
				for(int j=0;j<dst[0].size();j++){
					newlength[i]+=fabs(dst[i][j]-sum[i][j]);
				}
				error=(newlength[i]/oldlength[i] ==0)? error:error+1;
			}

			for(int i=0;i<dst.size();i++){
				for(int j=0;j<dst[0].size();j++){
					dst[i][j]=sum[i][j];
					sum[i][j]=0;
				}
				oldlength[i]=newlength[i];
				newlength[i]=0;
			}
		}
	}
	return true;
}