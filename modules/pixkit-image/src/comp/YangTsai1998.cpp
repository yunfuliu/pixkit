#include "../../include/pixkit-image.hpp"
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

bool table_construction(const Mat &src3b,vector<vector<float>> &table){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if (src3b.type() != CV_8UC3){
		CV_Assert(false);
	}	

	//////////////////////////////////////////////////////////////////////////
	vector<float>	temp_vec(4,0);

	temp_vec[0] = src3b.ptr<Vec3b>(0)[0][0];
	temp_vec[1] = src3b.ptr<Vec3b>(0)[0][1];
	temp_vec[2] = src3b.ptr<Vec3b>(0)[0][2];
	temp_vec[3] = 1;
	table.push_back(temp_vec);

	for (int i = 2; i < src3b.rows; i+=2){
		for (int j = 2; j < src3b.cols; j += 2){
			int a=0;
			while (true){
				if (table[a][0] == src3b.ptr<Vec3b>(i)[j][0]){
					if (table[a][1] == src3b.ptr<Vec3b>(i)[j][1]){
						if (table[a][2] == src3b.ptr<Vec3b>(i)[j][2]){
							table[a][3]++;
							break;
						}
					}
				}

				a++;
				if (a==table.size()){
					temp_vec[0] = src3b.ptr<Vec3b>(i)[j][0];
					temp_vec[1] = src3b.ptr<Vec3b>(i)[j][1];
					temp_vec[2] = src3b.ptr<Vec3b>(i)[j][2];
					temp_vec[3] = 1;
					table.push_back(temp_vec);
					a = 0;
					break;
				}
			}
		}
	}

	return true;
}
bool count_var(double box_index, vector<vector<float>> &box_range, vector<vector<float>> &table){

	int B_sum = 0, G_sum = 0, R_sum = 0;
	float moment1 = 0, moment2 = 0;

	for (int i = 0; i < table.size(); i++){
		if (table[i][0] < box_range[box_index][0] && table[i][0] >= box_range[box_index][1]){
			moment1 += table[i][0] * table[i][3];
			moment2 += table[i][0] * table[i][0] * table[i][3];
			B_sum += table[i][3];
		}
	}
	box_range[box_index][6] = moment2/B_sum - moment1*moment1/B_sum/B_sum;
	box_range[box_index][11] = moment1 / B_sum;
	moment1 = 0;
	moment2 = 0;

	for (int i = 0; i < table.size(); i++){
		if (table[i][1] < box_range[box_index][2] && table[i][1] >= box_range[box_index][3]){
			moment1 += table[i][1] * table[i][3];
			moment2 += table[i][1] * table[i][1] * table[i][3];
			G_sum += table[i][3];
		}
	}
	box_range[box_index][7] = moment2 / G_sum - moment1*moment1 / G_sum / G_sum;
	box_range[box_index][12] = moment1 / G_sum;
	moment1 = 0;
	moment2 = 0;

	for (int i = 0; i < table.size(); i++){
		if (table[i][2] < box_range[box_index][4] && table[i][2] >= box_range[box_index][5]){
			moment1 += table[i][2] * table[i][3];
			moment2 += table[i][2] * table[i][2] * table[i][3];
			R_sum += table[i][3];
		}
	}
	box_range[box_index][8] = moment2 / R_sum - moment1*moment1 / R_sum / R_sum;
	box_range[box_index][13] = moment1 / R_sum;

	return true;
}
bool largest_plane(double box_index, vector<vector<float>> &box_range){

	double max = box_range[box_index][6];
	double max2;
	if (max < box_range[box_index][7]){
		max2 = max;
		max = box_range[box_index][7];
	}else{
		max2 = box_range[box_index][7];
	}
	if (max < box_range[box_index][8]){
		max2 = max;
		max = box_range[box_index][8];
	}else if (max2<box_range[box_index][8]){
		max2 = box_range[box_index][8];
	}
	box_range[box_index][9] = max + max2;
	box_range[box_index][10] = max;
	return true;
}
int moment_pre_thresholding(int t1, int t2,double box_index, vector<vector<float>> &box_range, vector<vector<float>> &table){

	double p0 = 0, total = 0, c0 = 0, c1 = 0, h0 = 0, h1 = 0, m1 = 0, m2 = 0, m3 = 0, mean11 = 0, mean12 = 0, mean21 = 0, mean22 = 0, var1 = 0, var2 = 0;

	for (int i = 0; i < table.size(); i++){
		if (table[i][box_index] < t2 && table[i][box_index] >= t1){
			total += table[i][3];
			m1 += table[i][box_index] * table[i][3];
			m2 += table[i][box_index] * table[i][box_index] * table[i][3];
			m3 += table[i][box_index] * table[i][box_index] * table[i][box_index] * table[i][3];
		}
	}
	m1 = m1 / total;
	m2 = m2 / total;
	m3 = m3 / total;

	c0 = (-m1*m3 + m2*m2) / (m1*m1 - m2);
	c1 = (-m1*m2 + m3) / (m1*m1 - m2);
	h0 = (-c1 - sqrt(c1*c1 - 4 * c0)) / 2;
	h1 = (-c1 + sqrt(c1*c1 - 4 * c0)) / 2;
	p0 = (h1 - m1) / (h1 - h0)*total;

	for (int k = t1; k < t2; k++){
		total = 0;
		for (int i = 0; i < table.size(); i++){
			if (table[i][box_index] < k && table[i][box_index] >= t1){
				total += table[i][3];
				if (total>p0){
					return k;
				}
			}
		}
	}
}
bool box_splitting(vector<vector<float>> &box_range, vector<vector<float>> &table){
	double box_index1 = 0,box_index2=0;
	for (int i = 0; i < box_range.size(); i++){
		if (box_range[box_index1][9] < box_range[i][9]){
			box_index1 = i;
		}
	}
	int t1, t2,split_point;

	vector<float> memorry(14, 0);
	for (int i = 0; i < 14; i++){
		memorry[i] = box_range[box_index1][i];
	}

	if (box_range[box_index1][10] == box_range[box_index1][8]){
		box_index2 = 2;
		t2 = box_range[box_index1][4];
		t1 = box_range[box_index1][5];
		split_point = moment_pre_thresholding(t1, t2, box_index2, box_range, table);
		box_range.erase(box_range.begin() + box_index1);
		memorry[5] = split_point;
		box_range.push_back(memorry);
		memorry[5] = t1;
		memorry[4] = split_point;
		box_range.push_back(memorry);

	}else if (box_range[box_index1][10] == box_range[box_index1][7]){
		box_index2 = 1;
		t2 = box_range[box_index1][2];
		t1 = box_range[box_index1][3];
		split_point = moment_pre_thresholding(t1, t2, box_index2, box_range, table);
		box_range.erase(box_range.begin() + box_index1);
		memorry[3] = split_point;
		box_range.push_back(memorry);
		memorry[3] = t1;
		memorry[2] = split_point;
		box_range.push_back(memorry);
	}else{
		box_index2 = 0;
		t2 = box_range[box_index1][0];
		t1 = box_range[box_index1][1];
		split_point = moment_pre_thresholding(t1, t2, box_index2, box_range, table);
		box_range.erase(box_range.begin() + box_index1);
		memorry[1] = split_point;
		box_range.push_back(memorry);
		memorry[1] = t1;
		memorry[0] = split_point;
		box_range.push_back(memorry);
	}

	for (int i = box_range.size() - 2; i < box_range.size(); i++){
		count_var(i, box_range, table);
		largest_plane(i, box_range);
	}

	return true;
}
bool pixkit::comp::YangTsai1998(const cv::Mat &src3b, cv::Mat &dst3b,const int K){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src3b.type()==CV_8UC1){
		CV_Error(CV_StsUnsupportedFormat,"this function supports only color image (CV_8UC3) imgae.");
	}
	if(K<=0){
		CV_Error(CV_StsBadArg,"K should >=1.");
	}

	//////////////////////////////////////////////////////////////////////////
	///// init
	dst3b = src3b.clone();

	//////////////////////////////////////////////////////////////////////////
	///// table construction 	
	vector<vector<float>> table;
	table_construction(src3b, table);

	//////////////////////////////////////////////////////////////////////////
	///// box construction
	vector<vector<float>> box_range;
	// box_range(b_high,b_low,g_high,g_low,r_high,r_low,b_var,g_var,r_var,largest_plane,b_mean,g_mean,r_mean)
	vector<float> start(14, 0);
	start[0] = 256;	// init
	start[2] = 256;	// init
	start[4] = 256;	// init
	box_range.push_back(start);
	count_var(0,box_range,table);
	largest_plane(0, box_range);
	// number of representative colors
	for (int i = 0; i < K-1; i++){
		box_splitting(box_range, table);
	}

	//////////////////////////////////////////////////////////////////////////
	///// color mapping 
	for (int i = 0; i < src3b.rows; i++){
		for (int j = 0; j < src3b.cols; j++){
			for (int m = 0; m<K;m++){

				if (box_range[m][0]>src3b.ptr<Vec3b>(i)[j][0] && box_range[m][1] <= src3b.ptr<Vec3b>(i)[j][0]){
					if (box_range[m][2]>src3b.ptr<Vec3b>(i)[j][1] && box_range[m][3] <= src3b.ptr<Vec3b>(i)[j][1]){
						if (box_range[m][4]>src3b.ptr<Vec3b>(i)[j][2] && box_range[m][5] <= src3b.ptr<Vec3b>(i)[j][2]){

							dst3b.ptr<Vec3b>(i)[j][0] = box_range[m][11];
							dst3b.ptr<Vec3b>(i)[j][1] = box_range[m][12];
							dst3b.ptr<Vec3b>(i)[j][2] = box_range[m][13];

						}
					}
				}
			}
		}
	}

	return true;
}
