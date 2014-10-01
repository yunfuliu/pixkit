/*
MSRCP
*/
# include <stdlib.h>   
# include <stdio.h>   
# include <math.h>   
# include <string.h>   
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv\cv.h>
#include "../../include/pixkit-image.hpp"
# define MAX_RETINEX_SCALES    8       
# define MIN_GAUSSIAN_SCALE   16    
# define MAX_GAUSSIAN_SCALE  250      

typedef struct{   
	int     scale;          
	int     nscales;       
	int     scales_mode;                
} RetinexParams;   

# define RETINEX_UNIFORM 0   
# define RETINEX_LOW     1   
# define RETINEX_HIGH    2   

static float RetinexScales[MAX_RETINEX_SCALES];   

/*  
* Private variables.  
*/   
static RetinexParams rvals =   
{   
	240,             /* Scale */   

	//select number of scales
	3,         //default        
	//	2,
	//	1,

	//select Retinex processing mode
	RETINEX_UNIFORM, //default
	// RETINEX_LOW,
	//RETINEX_HIGH,
};   

# define clip( val, minv, maxv )    (( val = (val < minv ? minv : val ) ) > maxv ? maxv : val )   

/*  
* calculate scale values for desired distribution.  
*/   
void retinex_scales_distribution( float* scales, int nscales, int mode, int s)   
{   
	if ( nscales == 1 )   
	{ /* For one filter we choose the median scale */   
		scales[0] =  (float)s / 2;   //default
		// scales[0] =  15;	// strong on detail and dynamic range compression, weak color and tonal rendition
		// scales[0] =  240;	// inverse
	}   
	else if (nscales == 2)   
	{ /* For two filters we choose the median and maximum scale */   
		scales[0] = (float) s / 2;   
		scales[1] = (float) s;   
	}   
	else   
	{   
		float size_step = (float) s / (float) nscales;   
		int   i;   

		switch( mode )   
		{   
		case RETINEX_UNIFORM:   
			for ( i = 0; i < nscales; ++i )   
				scales[i] = 2.0f + (float)i * size_step;   
			break;   

		case RETINEX_LOW:   
			size_step = (float)log(s - 2.0f) / (float) nscales;   
			for ( i = 0; i < nscales; ++i )   
				scales[i] = 2.0f + (float)pow (10, (i * size_step) / log (10.));   
			break;   

		case RETINEX_HIGH:   
			size_step = (float) log(s - 2.0) / (float) nscales;   
			for ( i = 0; i < nscales; ++i )   
				scales[i] = s - (float)pow (10, (i * size_step) / log (10.));   
			break;   

		default:   
			break;   
		}   
	}   
}   
void compute_min_max( float *psrc, float *min, float *max, int size, int bytes ){
	*min =psrc[0];
	*max =psrc[0];
	for (int i = 0; i < size; i+= bytes )   
	{   
		if(psrc[i]<*min)*min=psrc[i];
		if(psrc[i]>*max)*max=psrc[i];
	} 
}

bool SimplestColorBalance(float * ori,int size,float upperthresh,float lowerthresh){

	int totalarea=size;
	upperthresh=upperthresh*totalarea;
	lowerthresh=lowerthresh*totalarea;
	int histogramsize=256;
	float *histogram=NULL;
	histogram = (float *)malloc (histogramsize * sizeof (float));
	memset( histogram, 0, histogramsize * sizeof (float) );  

	int min=histogramsize;
	int max=0;

	//compute histogram
	for(int i=0;i<totalarea;i++){
		histogram[(int)ori[i]]++;
	}

	//compute erase range
	int lowercut=0;
	double number=0;
	for(int i=0;i<histogramsize;i++){
		number=number+histogram[i];
		if(lowerthresh>number)lowercut++;
		else break;
	}

	int uppercut=0;
	number=0;
	for(int i=histogramsize-1;i>=0;i--){
		number=number+histogram[i];
		if(upperthresh>number)uppercut++;
		else break;
	}

	//erase
	int Vmin=(lowercut);
	int Vmax=(histogramsize-1)-uppercut;

	for(int i=0;i<totalarea;i++){

		if(ori[i]<Vmin)ori[i]=Vmin;
		if(ori[i]>Vmax)ori[i]=Vmax;

	}





	for(int i=0;i<totalarea;i++){
		float a=((float)ori[i]-Vmin)/(Vmax-Vmin)*(histogramsize-1)+0;
		ori[i] = (unsigned char)clip(a, 0, 255 );   
	}

	return 1;
}
void MSRCP_Main( unsigned char * src, int width, int height, int NumChannel){
	int           scale;
	int           i=0;
	int			  j=0;    
	int           pos;   
	int           channel;  //channel index 
	int           channelsize  = ( width * height );  
	float         weight;    
	float         mini, range, maxi;   
	int std_rate=3;			//std search range
	float         *dst  = NULL; 
	float         *MSR_Nor = NULL;
	int         *IntensitySum = NULL;
	cv::Mat MatPsrc;
	std::vector<cv::Mat> GaussianIn(NumChannel);//

	//zeros
	for(int i=0;i<NumChannel;i++){
		GaussianIn[i]=cv::Mat::zeros(height,width,CV_8UC1);
	}
	cv::Mat Intensity=cv::Mat::zeros(height,width,CV_8UC1);
	MatPsrc=cv::Mat::zeros(height,width,CV_8UC1);
	cv::Mat GaussianOut=cv::Mat::zeros(height,width,CV_8UC1);


	/* Allocate all the memory needed for algorithm*/  
	MSR_Nor = (float *)malloc (channelsize * sizeof (float));  
	IntensitySum = (int *)malloc (channelsize * sizeof (int));  
	dst = (float *)malloc (channelsize * sizeof (float));   

	/* set all the memory needed 0*/  
	memset( dst, 0, channelsize * sizeof (float) );   
	memset( MSR_Nor, 0, channelsize * sizeof (float) );
	memset( IntensitySum, 0, channelsize * sizeof (int) );

	if (dst == NULL)   
	{   
		printf( "Failed to allocate memory" );   
		return;   
	}   

	/*  
	Calculate the scales of filtering according to the  
	number of filter and their distribution.  
	*/   
	retinex_scales_distribution( RetinexScales,   
		rvals.nscales, rvals.scales_mode, rvals.scale ); 

	/*  
	change data arrangement
	*/   
	weight = 1.0f / (float) rvals.nscales;   
	pos = 0;   
	for ( channel = 0; channel < NumChannel; channel++ ){  
		for ( i = 0, pos = channel; i <channelsize ; i++, pos += NumChannel ){
			GaussianIn[channel].data[i]=(src[pos] );
		}  
	}

	/*  
	compute intensity channel
	*/
	for ( i=0; i<channelsize; i++ ){
		IntensitySum[i]= ((GaussianIn[0].data[i]+GaussianIn[1].data[i]+GaussianIn[2].data[i]));
		Intensity.data[i]= IntensitySum[i]/3.;
	}

	/*  
	multiscale retinex
	*/
	for ( scale = 0; scale < rvals.nscales; scale++ ){
		int GauWidth=((int)RetinexScales[scale]*std_rate%2)==0?((int)RetinexScales[scale]*std_rate-1):((int)RetinexScales[scale]*std_rate);
		int GauHeight=((int)RetinexScales[scale]*std_rate%2)==0?((int)RetinexScales[scale]*std_rate-1):((int)RetinexScales[scale]*std_rate);
		cv::Size a(GauWidth,GauHeight);
		cv::GaussianBlur(Intensity,GaussianOut,a,(int)RetinexScales[scale]);
		for ( i = 0; i < channelsize; i++ ){ 
			if(Intensity.data[i]==0) dst[i] +=0;
			else if(Intensity.data[i]!=0&&GaussianOut.data[i]==0)dst[i] += weight * log( (float)Intensity.data[i] );
			else
				dst[i] += weight * ( log( (float)IntensitySum[i] )-log((float)GaussianOut.data[i]) );  
		}   
	}

	/*  
	get the range from the result of MSR
	*/
	compute_min_max( dst,&mini,&maxi, channelsize, 1 );
	range = maxi - mini;

	/*  
	normalize the MSR range to 0~255
	*/
	for ( i = 0; i < channelsize; i++ ){        
		float c = 255 * ( dst[i] - mini ) / range; 
		MSR_Nor[i]=clip( c, 0, 255 ); 
	}   

	/*  
	SimplestColorBalance
	*/
	SimplestColorBalance(MSR_Nor,channelsize,0.1,0.1);

	/*  
	according to the rate between normalized  MSR and intensity channel,revise and output the final result
	*/
	int B=0;
	float A=0;
	float minimum=0;
	for ( i = 0; i < channelsize*NumChannel; i += NumChannel ){
		B=0;
		A=0;
		if(src[i]>B)B=src[i];
		if(src[i+1]>B)B=src[i+1];
		if(src[i+2]>B)B=src[i+2];
		int temp=src[i]+src[i+1]+src[i+2];	//intensity channel(before /3)
		if(temp==0&&B==0){ A=1;}
		else if(temp==0&&B>0) { 
			A=(255.0f/B);
		}
		else if(temp>0&&B==0)
			A=(float)MSR_Nor[i/NumChannel]/(temp/3.);
		else  {
			A=(255.0f/B)<(MSR_Nor[i/NumChannel]/(temp/3.))?
				(255.0f/B):(MSR_Nor[i/NumChannel]/(temp/3.));
		}

		int a= (A*src[i]+0.5);
		int b= (A*src[i+1]+0.5);
		int c =(A*src[i+2]+0.5);
		src[i]=  (unsigned char)clip( a,0,255 ); 
		src[i+1]= (unsigned char)clip( b,0,255 ); 
		src[i+2]=  (unsigned char)clip(  c,0,255 ); 
	}  

	//free memory
	free (dst);  
	free (MSR_Nor);   
	free (IntensitySum);  
}   
/*  
* This function is the heart of the algo.  
* (a)  Filterings at several scales and sumarize the results.  
* (b)  Calculation of the final values.  
*/   
bool pixkit::enhancement::local::MSRCP2014(const cv::Mat &src,cv::Mat &Return_Image,int Nscale)   
{   
	rvals.nscales=Nscale;
	/////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src.type()!=CV_8UC3||Nscale<1||Nscale>3){
		CV_Assert(false);
	}

	IplImage * orig= &IplImage(src);
	unsigned char * sImage, * dImage;   
	int x, y;   
	int nWidth, nHeight, step;  

	if ( orig == NULL )	{   
		printf( "Could not get image. Program exits!\n" );  
		CV_Assert(false);
	}   
	nWidth =orig->width;   
	nHeight = orig->height;   
	step = orig->widthStep/sizeof( unsigned char );   

	Return_Image=cv::Mat::zeros(nHeight,nWidth,CV_8UC3);
	sImage = new unsigned char[nHeight*nWidth*3];  
	dImage = new unsigned char[nHeight*nWidth*3];   

	if ( orig->nChannels == 3 )   
	{   
		for ( y = 0; y < nHeight; y++ )   
			for ( x = 0; x < nWidth; x++ )   
			{   
				sImage[(y*nWidth+x)*orig->nChannels] = orig->imageData[y*step+x*orig->nChannels];   
				sImage[(y*nWidth+x)*orig->nChannels+1] = orig->imageData[y*step+x*orig->nChannels+1];   
				sImage[(y*nWidth+x)*orig->nChannels+2] = orig->imageData[y*step+x*orig->nChannels+2];   
			}   
	}  

	memcpy( dImage, sImage, nWidth*nHeight*orig->nChannels );  
	MSRCP_Main( dImage, nWidth, nHeight, orig->nChannels );   
	for ( y = 0; y < nHeight*nWidth* orig->nChannels; y=y+3 ){
		Return_Image.data[y]=dImage[y];
		Return_Image.data[y+1]=dImage[y+1];
		Return_Image.data[y+2]=dImage[y+2];
	}


	delete [] sImage;  
	delete [] dImage;   
	return true;
}   


