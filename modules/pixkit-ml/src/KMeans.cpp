#include "../include/pixkit-ml.hpp"
#include <ctime>

using namespace std;

void calculate(vector<vector<double>> &input,vector<vector<double>> &out,vector<vector<double>> &sum){

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
bool pixkit::clustering::kMean(vector<vector<double>> &input, vector<vector<double>> &out, int iter, bool initial){

	srand(time(NULL));

	if( input[0].size() != out[0].size()){	//	Determind the dimension between the input data and the output data is same or not

		cout<<"The input data and the output need be the same dimension\n";
		return false;
	}

	if(initial == 0){	//	Generate random initial points

		int ranmax,ranmin;
		cout<<"Input random maximum\n";
		cin>>ranmax;
		cout<<"Input rendom minimum\n";
		cin>>ranmin;

		for(int i=0;i<out.size();i++){
			for(int j=0;j<out[0].size();j++){
				out[i][j]=(rand() % ranmax)+ranmin;
			}
		}
	}
	vector<double> oldlength(out.size(),1);
	vector<double> newlength(out.size(),0);
	vector<vector<double>>	sum(out.size(),vector<double>(out[0].size(),0));	//	Calculate the sum
	
	if(iter != -1){

		while (iter !=0 ){

			calculate(input,out,sum);	//	Calculate new mean value

			for(int i=0;i<out.size();i++){
				for(int j=0;j<out[0].size();j++){
					out[i][j]=sum[i][j];
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

			calculate(input,out,sum);
			error=0;

			for(int i=0;i<out.size();i++){
				for(int j=0;j<out[0].size();j++){
					newlength[i]+=fabs(out[i][j]-sum[i][j]);
				}
				error=(newlength[i]/oldlength[i] ==0)? error:error+1;
			}

			for(int i=0;i<out.size();i++){
				for(int j=0;j<out[0].size();j++){
					out[i][j]=sum[i][j];
					sum[i][j]=0;
				}
				oldlength[i]=newlength[i];
				newlength[i]=0;
			}
				
			
		}
		
	}
	return true;
}