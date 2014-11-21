#include "../../include/pixkit-image.hpp"

void new_integral(const cv::Mat &src, int **integral_image)
{

	int i,j;

	integral_image[1][1]=static_cast < int >(src.data[0*src.cols+0]);


	for (i = 1 ; i< src.cols ; i++)
	{

		integral_image[1][i+1]=static_cast < int >(src.data[0*src.cols+i])+integral_image[1][i];

	}

	for (i = 1 ; i< src.rows ; i++)
	{
		integral_image[i+1][1]=static_cast < int >(src.data[i*src.cols+0])+integral_image[i][1];
	}

	for (i = 1 ; i< src.rows ; i++)
	{
		for (j = 1 ; j< src.cols ; j++)
		{
			integral_image[i+1][j+1]=static_cast < int >(src.data[i*src.cols+j])+integral_image[i+1][j]+integral_image[i][j+1]-integral_image[i][j];

		}
	}

}


double new_local_mean(int x,int y,int windowsize,int **integral_image)
{

	int d= (windowsize/2)+1,s;
	double localmean;
	s=(integral_image[x+d-1][y+d-1]+integral_image[x-d][y-d])-(integral_image[x-d][y+d-1]+integral_image[x+d-1][y-d]);
	localmean=s/(windowsize*windowsize);
	return localmean;


}

bool pixkit::thresholding::LAT2011(const cv::Mat &src,cv::Mat &dst,int windowSize, double k){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src.type()!=CV_8UC1){
		CV_Error(CV_StsBadArg,"[pixkit::thresholding::LAT2011] src's type should be CV_8UC1.");
	}


	//////////////////////////////////////////////////////////////////////////
	///// integral
	int **new_integral_image = new int *[src.rows+1];
	for(int i=0;i<=src.rows;i++){
		new_integral_image[i] = new int [src.cols+1];
	}
	new_integral( src, new_integral_image);

	//////////////////////////////////////////////////////////////////////////
	///// initial
	dst.create(src.size(),src.type());
	dst.setTo(0);


	//////////////////////////////////////////////////////////////////////////
	///// process
	int startstep= (windowSize/2)+1;
	int endstep=startstep-1;
	double localmean,deviation,T;

	for (int i=startstep ; i<=src.rows-endstep-1;i++){
		for (int j=startstep ; j<=src.cols-endstep-1;j++){

			localmean=new_local_mean(i+1,j+1, windowSize,new_integral_image);
			deviation=src.data[i*src.cols+j]-localmean;
			T=localmean*(1+k*((deviation/(1-deviation))-1));

			if (src.data[i*src.cols+j]<=T){
				dst.data[i*dst.cols+j]=0;

			}else{ 
				dst.data[i*dst.cols+j]=255;
			}

		}
	}


	for(int i=0;i<src.rows;i++)
		delete []new_integral_image[i];
	delete []new_integral_image;

	return	true;
}

