#include "../include/pixkit-image.hpp"

//////////////////////////////////////////////////////////////////////////
bool pixkit::edgedetection::Sobel(const cv::Mat &src, cv::Mat &dst){

	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	// get magnitude and angle by Sobel operator
	const	double	Sobel_V[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}},
					Sobel_H[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};

	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		dst.data[i*dst.cols]=0;
		dst.data[i*dst.cols+(src.cols-1)]=0;
	}
	for(int j=0;j<src.cols;j++){
		dst.data[j]=0;
		dst.data[(src.rows-1)*dst.cols+j]=0;
	}
	for(int i=1;i<src.rows-1;i++){
		for(int j=1;j<src.cols-1;j++){
			float	Sh=0.,Sv=0.;	// ¤ô¥­­Èand««ª½­È
			for(int m=-1;m<=1;m++){
				for(int n=-1;n<=1;n++){
					Sh+=(double)Sobel_H[m+1][n+1]*src.data[(i+m)*src.cols+(j+n)];
					Sv+=(double)Sobel_V[m+1][n+1]*src.data[(i+m)*src.cols+(j+n)];
				}
			}
			// get mag
			float	tempv	=	sqrt((double)Sh*Sh+(double)Sv*Sv);
			if(tempv>255){
				dst.data[i*dst.cols+j]=255.;
			}else if(tempv<0){
				dst.data[i*dst.cols+j]=0.;
			}else{
				dst.data[i*dst.cols+j]	=	(int)(tempv	+0.5);
			}
		}
	}

	return true;
}

