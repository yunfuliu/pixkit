#include <ctime>
#include "../../include/pixkit-image.hpp"
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
	/*
	 * @param		src: input image
	 * @param		dst: output image
	 * @param		BlockSize: the height and width of block
	 *
	 */	
bool pixkit::comp::CCC1986(cv::Mat &src,cv::Mat &dst, int BlockSize){

	if(src.cols%BlockSize!=0 || src.rows%BlockSize!=0){
		printf("input size error\n");
		system("pause");
		exit(0);
	}

	Mat B(src.size(),CV_32FC1),G(src.size(),CV_32FC1),R(src.size(),CV_32FC1),Lumin(src.size(),CV_32FC1);
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			B.ptr<float>(i)[j]=(float)src.at<Vec3b>(i,j)[0];
			G.ptr<float>(i)[j]=(float)src.at<Vec3b>(i,j)[1];
			R.ptr<float>(i)[j]=(float)src.at<Vec3b>(i,j)[2];
			Lumin.ptr<float>(i)[j]=0.3*R.ptr<float>(i)[j]+0.59*G.ptr<float>(i)[j]+0.11*B.ptr<float>(i)[j];
		}
	}

	double LuminMean=0;
	double CountAmount[2]={0},QuantB[2]={0},QuantG[2]={0},QuantR[2]={0}; //0 small, 1 big
	Mat tdst(src.size(),src.type());
	for(int i=0; i<src.rows; i+=BlockSize){
		for(int j=0; j<src.cols; j+=BlockSize){
			//Average
			LuminMean=0;
			for(int m=i; m<i+BlockSize; m++)
				for(int n=j; n<j+BlockSize; n++)
					LuminMean+=Lumin.ptr<float>(m)[n];
			LuminMean/=(BlockSize*BlockSize);

			CountAmount[0]=0;
			CountAmount[1]=0;
			for(int m=i; m<i+BlockSize; m++){
				for(int n=j; n<j+BlockSize; n++){
					if(Lumin.ptr<float>(m)[n]>=LuminMean)
						CountAmount[1]++;
					else
						CountAmount[0]++;
				}
			}

			//Quant
			for(int m=0; m<2; m++){
				QuantB[m]=0;
				QuantG[m]=0;
				QuantR[m]=0;
			}
			for(int m=i; m<i+BlockSize; m++){
				for(int n=j; n<j+BlockSize; n++){
					if(Lumin.ptr<float>(m)[n]>=LuminMean){
						QuantB[1]+=B.ptr<float>(m)[n];
						QuantG[1]+=G.ptr<float>(m)[n];
						QuantR[1]+=R.ptr<float>(m)[n];
					}						
					else{
						QuantB[0]+=B.ptr<float>(m)[n];
						QuantG[0]+=G.ptr<float>(m)[n];
						QuantR[0]+=R.ptr<float>(m)[n];
					}
				}
			}
			for(int m=0; m<2; m++){
				QuantB[m]/=CountAmount[m];
				QuantG[m]/=CountAmount[m];
				QuantR[m]/=CountAmount[m];
			}

			//output
			for(int m=i; m<i+BlockSize; m++){
				for(int n=j; n<j+BlockSize; n++){
					if(Lumin.ptr<float>(m)[n]>=LuminMean){
						tdst.at<Vec3b>(m,n)[0]=QuantB[1];
						tdst.at<Vec3b>(m,n)[1]=QuantG[1];
						tdst.at<Vec3b>(m,n)[2]=QuantR[1];
					}
					else{
						tdst.at<Vec3b>(m,n)[0]=QuantB[0];
						tdst.at<Vec3b>(m,n)[1]=QuantG[0];
						tdst.at<Vec3b>(m,n)[2]=QuantR[0];
					}
				}
			}

		}
	}

	dst=tdst.clone();

	return true;
}