#include "../include/pixkit-cv.hpp"

using namespace std;

bool pixkit::labeling::twoPass(const cv::Mat &src,cv::Mat &dst,const int offset){

	//////////////////////////////////////////////////////////////////////////
	///// EXCEPTION
	if(src.type()!=CV_8UC1){
		CV_Error(CV_StsBadArg,"[xxx] src's type should be CV_8UC1.");
	}

	//////////////////////////////////////////////////////////////////////////
	///// INITIALIZATION
	cv::Mat	tdst;
	tdst	=	src.clone();
	tdst.convertTo(tdst,CV_32SC1);	// type of dst: int

	//////////////////////////////////////////////////////////////////////////
	/*標籤為從1開始*/
	int LableNumber = 1;
	int C[5],min,temp;
	int W,A,Q,E,S;
	/*正規化用標籤*/
	std::vector<int> ObjectIndex;
	/*第一次掃描*/
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			
			C[0] = tdst.ptr<int>(i)[j];
			if(C[0]<128)
				continue;
			min = src.rows*src.cols;
			if(j-1 <0){
				C[1] = 0;
			}else{
				C[1] = tdst.ptr<int>(i)[j-1];
			}
			if (i-1<0 || j-1 <0){
				C[2] = 0;
			}else{
				C[2] = tdst.ptr<int>(i-1)[j-1];
			}
			if(i-1<0){
				C[3] = 0;
			}else{
				C[3] = tdst.ptr<int>(i-1)[j];
			}
			if (i-1<0 || j+1 >=src.cols){
				C[4] = 0;
			}else{
				C[4] = tdst.ptr<int>(i-1)[j+1];
			}

			if(C[1] ==0 && C[2] ==0 && C[3] ==0 && C[4] ==0){
				C[0] = LableNumber;
				LableNumber++;
			}else{
				for(int k=1;k<=4;k++){
					if(C[k]<min && C[k] != 0){
						min = C[k];
					}
				}
				C[0] = min;
			}
			tdst.ptr<int>(i)[j] = C[0];

		}
	}

	//////////////////////////////////////////////////////////////////////////
	/*LableNumber != 1 代表有前景物件存在
	LableNumber == 1 代表無前景物件
	使用動態新增Table記憶體*/
	bool **Table = NULL;
	int *refTeble = NULL;
	if(LableNumber != 1){
		Table = new bool *[LableNumber];
		refTeble = new int [LableNumber];
		for(int i=0;i<LableNumber;i++){
			Table[i] = new bool [LableNumber];
			refTeble[i] = 0;
			for(int j=0;j<LableNumber;j++){
				Table[i][j] = 0;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//第二次掃描
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){	
			S = tdst.ptr<int>(i)[j];

			if(S == 0)
				continue;

			if(i-1<0){
				W = 0;
			}else{
				W = tdst.ptr<int>(i-1)[j];
			}
			if(i-1<0 || j-1<0){
				Q = 0;
			}else{
				Q = tdst.ptr<int>(i-1)[j-1];
			}
			if(j-1<0){
				A = 0;
			}else{
				A = tdst.ptr<int>(i)[j-1];
			}
			if (i-1<0 || j+1 >=src.cols){
				E = 0;
			}else{
				E = tdst.ptr<int>(i-1)[j+1];
			}

			Table[S][S] = 1;
			if(S != W && W != 0){
				Table[S][W] = 1;
				Table[W][S] = 1;
			}else if(S!=A && A!=0){
				Table[S][A] = 1;
				Table[A][S] = 1;
			}else if(S!=Q && Q!=0){
				Table[S][Q] = 1;
				Table[Q][S] = 1;
			}else if (S!=E && E!=0){
				Table[S][E] = 1;
				Table[E][S] = 1;
			}					
		}
	}
	//Equivalent Table
	//因為標籤值是從1開始，所以i,j從1開始
	for(int i=1;i<LableNumber;i++){
		for (int j=1;j<LableNumber;j++){
			if(Table[i][j] ==1){
				for(int ii = 1;ii<LableNumber;ii++){
					if(Table[j][ii] == 1){
						Table[i][ii] = 1;
					}
				}
			}
		}
	}
	/*建立比對對照表*/
	for(int i=1;i<LableNumber;i++){
		for (int j=1;j<LableNumber;j++){
			if(Table[i][j] != 0){
				refTeble[i] = j;
				break;
			}
		}
	}

	/*利用對照表去Refine*/
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			temp = tdst.ptr<int>(i)[j];
			if(temp != 0)
				tdst.ptr<int>(i)[j] = refTeble[temp];
		}
	}
	//LableNumber != 1 代表有前景物件存在
	//刪除Table記憶體
	if(LableNumber != 1)
	{
		for(int i=0;i<LableNumber;i++)
			delete []Table[i];
		delete []Table;
		delete []refTeble;
	}

	int labelValue = 0,labelIndex = 0;
	bool flag = true;

	/*蒐集所有標籤值並把標籤給正規化*/
	for(int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			labelValue = tdst.ptr<int>(i)[j];
			if(labelValue != 0){
				if(ObjectIndex.size() == 0){
					ObjectIndex.push_back(labelValue);
					labelIndex++;
					tdst.ptr<int>(i)[j] = offset;
				}else{
					flag = true;
					int k = 0;
					for(k=0;k<ObjectIndex.size();k++){
						if(ObjectIndex[k] == labelValue){
							flag = false;
							tdst.ptr<int>(i)[j] = k+offset;
							break;
						}
					}
					if(flag == true){
						ObjectIndex.push_back(labelValue);
						labelIndex++;
						tdst.ptr<int>(i)[j] = k+offset;
					}
				}
			}
		}
	}

	dst	=	tdst.clone();
	return true;
}


