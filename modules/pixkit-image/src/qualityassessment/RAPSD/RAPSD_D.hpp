#include "stdio.h"
#include "tchar.h"
#include "cv.hpp"

#ifndef _RAPSD_D_H_
#define _RAPSD_D_H_

/*-------------------------------------------------------------------------
Perform a 2D FFT inplace given a complex 2D array
The direction dir, 1 for forward, -1 for reverse
The size of the array (nx,ny)
Return false if there are memory problems or
the dimensions are not powers of 2
*/


struct COMPLEX {
	double real;
	double imag;
};


int FFT(int dir,int m,double *x,double *y);
int FFT2D(COMPLEX **c,int nx,int ny,int dir);
void FFTEvaluation(cv::Mat &src, std::vector< std::vector< double > >& temp, COMPLEX ** FFTInfo);
double APSEvaluation(std::vector< std::vector< double > > &FImage, std::vector< double > &APS, std::vector< double > &AN, double dr, int im);
void Calculate_RadiallyAveragedPowerSpectralDensity( cv::Mat src , int CurGrayscale , double **RAPSD , double **Anso );
int Round(double Value);

#endif