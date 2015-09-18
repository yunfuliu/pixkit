#include <ctime>
#include "../../include/pixkit-image.hpp"
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

	/*
	 * @param		src: input image
	 * @param		dst: output image
	 * @param		BlockSize: the height and width of block
	 * @param		MoreCopressFlag: 0=FS-SBM 1=use FS-BMO 
	 * @param		THBO: If the absolute distance between the first quantization level and the second quantization level
	 *					  is less than or equal to THBO, the mean of the component values in the decomposed block is used to encode it.
	 *
	 */
bool pixkit::comp::FS_BMO2014(cv::Mat &src,cv::Mat &dst, int BlockSize, int MoreCompressFlag, int THBO){

	if(src.cols%BlockSize!=0 || src.rows%BlockSize!=0){
		printf("input size error\n");
		system("pause");
		exit(0);
	}

	int **FSBitmap = new int *[BlockSize];
	for(int i=0; i<BlockSize; i++)
		FSBitmap[i] = new int [BlockSize];

	double CountAmount[2]={0},Quant[3][2]={0},EuDistance[2]={0}; //0 small, 1 big
	Mat tdst(src.size(),src.type());

	double ChannelMean[3]={0};
	int OutputCode[3]; // 0=use BMO tech , 1=use SBM tech
	int CountOutputBits=0;

	for(int i=0; i<src.rows; i+=BlockSize){
		for(int j=0; j<src.cols; j+=BlockSize){
			for(int channel=0; channel<3; channel++){
				
				for(int m=i; m<i+BlockSize; m++)
					for(int n=j; n<j+BlockSize; n++)
						ChannelMean[channel]+=src.at<Vec3b>(m,n)[channel];
				ChannelMean[channel]/=(BlockSize*BlockSize);

				CountAmount[0]=0;
				CountAmount[1]=0;
				for(int m=i; m<i+BlockSize; m++){
					for(int n=j; n<j+BlockSize; n++){
						if(src.at<Vec3b>(m,n)[channel]>=ChannelMean[channel])
							CountAmount[1]++;
						else
							CountAmount[0]++;
					}
				}
			
				//Quant
				for(int quantlevel=0; quantlevel<2; quantlevel++)
					Quant[channel][quantlevel]=0;
				for(int m=i; m<i+BlockSize; m++){
					for(int n=j; n<j+BlockSize; n++){
						if(src.at<Vec3b>(m,n)[channel]>=ChannelMean[channel])
							Quant[channel][1]+=src.at<Vec3b>(m,n)[channel];
						else
							Quant[channel][0]+=src.at<Vec3b>(m,n)[channel];
					}
				}
				for(int quantlevel=0; quantlevel<2; quantlevel++)
					Quant[channel][quantlevel]/=CountAmount[quantlevel];

			}

			//Recalculate
			CountAmount[0]=0;
			CountAmount[1]=0;
			for(int m=i; m<i+BlockSize; m++){
				for(int n=j; n<j+BlockSize; n++){
					EuDistance[0]=pow(src.at<Vec3b>(m,n)[0]-Quant[0][0],2)+pow(src.at<Vec3b>(m,n)[1]-Quant[1][0],2)+pow(src.at<Vec3b>(m,n)[2]-Quant[2][0],2);
					EuDistance[1]=pow(src.at<Vec3b>(m,n)[0]-Quant[0][1],2)+pow(src.at<Vec3b>(m,n)[1]-Quant[1][1],2)+pow(src.at<Vec3b>(m,n)[2]-Quant[2][1],2);
					if(EuDistance[0]<=EuDistance[1]){
						FSBitmap[m%BlockSize][n%BlockSize]=0;
						CountAmount[0]++;
					}
					else{
						FSBitmap[m%BlockSize][n%BlockSize]=1;
						CountAmount[1]++;
					}
				}
			}

			//New Quant
			for(int m=0; m<2; m++){
				Quant[0][m]=0;
				Quant[1][m]=0;
				Quant[2][m]=0;
			}
			for(int m=i; m<i+BlockSize; m++){
				for(int n=j; n<j+BlockSize; n++){
					if(FSBitmap[m%BlockSize][n%BlockSize]==1){
						Quant[0][1]+=src.at<Vec3b>(m,n)[0];
						Quant[1][1]+=src.at<Vec3b>(m,n)[1];
						Quant[2][1]+=src.at<Vec3b>(m,n)[2];
					}						
					else{
						Quant[0][0]+=src.at<Vec3b>(m,n)[0];
						Quant[1][0]+=src.at<Vec3b>(m,n)[1];
						Quant[2][0]+=src.at<Vec3b>(m,n)[2];
					}
				}
			}
			for(int quantlevel=0; quantlevel<2; quantlevel++){
				Quant[0][quantlevel]/=CountAmount[quantlevel];
				Quant[1][quantlevel]/=CountAmount[quantlevel];
				Quant[2][quantlevel]/=CountAmount[quantlevel];
			}

			//more compress			
			if(MoreCompressFlag==1){
				CountOutputBits+=3;
				for(int channel=0; channel<3; channel++){
					if(abs(Quant[channel][0]-Quant[channel][1])<=THBO)
						OutputCode[channel]=0;
					else
						OutputCode[channel]=1;
				}
				if(OutputCode[0]==0&&OutputCode[1]==0&&OutputCode[2]==0)
					CountOutputBits+=3*8;
				else{
					CountOutputBits+=BlockSize*BlockSize;
					for(int channel=0; channel<3; channel++){
						if(OutputCode[channel]==0)
							CountOutputBits+=8;
						else
							CountOutputBits+=2*8;
					}
				}
			}

			//output
			if(MoreCompressFlag==1){
				for(int m=i; m<i+BlockSize; m++){
					for(int n=j; n<j+BlockSize; n++){
						for(int channel=0; channel<3; channel++){
							if(OutputCode[channel]==0)
								tdst.at<Vec3b>(m,n)[channel]=ChannelMean[channel];
							else{
								if(FSBitmap[m%BlockSize][n%BlockSize]==1)
									tdst.at<Vec3b>(m,n)[channel]=Quant[channel][1];
								else
									tdst.at<Vec3b>(m,n)[channel]=Quant[channel][0];
							}
						}					
					}
				}
			}
			else{
				for(int m=i; m<i+BlockSize; m++){
					for(int n=j; n<j+BlockSize; n++){
						for(int channel=0; channel<3; channel++){
							if(FSBitmap[m%BlockSize][n%BlockSize]==1)
								tdst.at<Vec3b>(m,n)[channel]=Quant[channel][1];
							else
								tdst.at<Vec3b>(m,n)[channel]=Quant[channel][0];
						}
					}
				}
			}
		}
	}

	dst=tdst.clone();
	delete [] FSBitmap;
	return true;
}
