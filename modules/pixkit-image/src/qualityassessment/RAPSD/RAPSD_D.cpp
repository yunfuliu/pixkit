#include "RAPSD_D.hpp"
//// "CurGrayscale" for the single tone value
void Calculate_RadiallyAveragedPowerSpectralDensity( cv::Mat src , int CurGrayscale , double **RAPSD , double **Anso )
{
	//int sampleOfSize = Img->height >> 1 ; //// or Img->width
	//int sampleOfSize = Img->width;

	//cv::Mat src(Img,0);
	cv::Mat dst;
	int sampleOfSize = src.rows;

	dst.create(sampleOfSize,sampleOfSize,src.type());

	std::vector< std::vector< double > > dst_FFT (dst.rows, std::vector< double >(dst.cols));
	std::vector< std::vector< double > > FImage (dst.rows, std::vector< double >(dst.cols));

	COMPLEX * FFTData	= new COMPLEX	[dst.rows * dst.cols];
	COMPLEX **FFTInfo	= new COMPLEX *	[dst.rows];

	for (int i=0;i<dst.rows;i++)
		FFTInfo[i] = &FFTData[i*dst.cols];

	double dr = 1;
	// for dr == 1, dd means the maximum radius of the image, only for (width == height), still need to be modified
	int dd = (int)sqrt( (double)2*( (dst.cols/2)*(dst.cols/2) ) );
	int Number_APS = (int)((dd)/dr);
	std::vector< double > APS (Number_APS);
	std::vector< double > AN (Number_APS);

	for (int x=0;x<dst.rows;x++){
		for (int y=0;y<dst.cols;y++){
			dst_FFT[x][y] = 0;
			dst.data[x*dst.cols + y] = 0;
		}
	}	

	for (int x=0;x<dst.rows;x++){
		for (int y=0;y<dst.cols;y++){
			if(src.data[ (x*dst.rows)+y]==NULL )
				src.data[ (x*dst.rows)+y]=0;
			dst.data[x*dst.cols + y] = src.data[(x*dst.rows)+y];
		}
	}

	FFTEvaluation(dst, FImage, FFTInfo);

	for (int x=0;x<dst.rows;x++){
		for (int y=0;y<dst.cols;y++){
			dst_FFT[x][y] += FImage[x][y];
			dst.ptr<float>(y)[x] = dst_FFT[x][y];
		}
	}
	//// "CurGrayscale" for the single tone value
	APSEvaluation(dst_FFT, APS, AN, 1, CurGrayscale);

	//////////////////////////////////////////////////////////////////////////
	//// add by Zara
	//////////////////////////////////////////////////////////////////////////

	double *RAPSD_t = (double * ) calloc ( Number_APS , sizeof(double) );
	double *Anso_t  = (double * ) calloc ( Number_APS , sizeof(double) );

	for (int i = 0; i < Number_APS ; i++)
	{
		RAPSD_t[i] = APS[i];
		Anso_t[i] = AN[i];
	}

	*RAPSD = RAPSD_t;
	*Anso = Anso_t;


}
// FFT Evaluation (The value of FFT result is only for RAPSD calculation !! )	[2/14/2014 °ê¥°]
void FFTEvaluation(cv::Mat &src, std::vector< std::vector< double > >& temp, COMPLEX ** FFTInfo)
{
	//std::vector< std::vector< double > > temp (src.rows, std::vector< double >(src.cols));
	for (int i=0; i<src.rows; i++){
		for (int j=0; j<src.cols; j++){
			FFTInfo[i][j].real = src.data[i*src.cols + j]/255.0;
			FFTInfo[i][j].imag = 0.0;
		}
	}

	if ( FFT2D(FFTInfo, src.rows, src.cols,1)==false ){
		system("pause");
	}
	
	// P = | F(u,v) | ^ 2
	for (int i=0; i<src.rows; i++)
		for (int j=0; j<src.cols; j++)
			temp[i][j] = ((FFTInfo[i][j].real * FFTInfo[i][j].real) + (FFTInfo[i][j].imag * FFTInfo[i][j].imag))/(double)(src.rows*src.cols);

	// swap the quarter blocks
	for (int i=0; i<(int)(src.rows/2); i++){
		for (int j=0; j<(int)(src.cols/2); j++){
			double tempv = temp[i][j];
			temp[i][j] = temp[i+(int)(src.rows/2)][j+(int)(src.cols/2)];
			temp[i+(int)(src.rows/2)][j+(int)(src.cols/2)] = tempv;
		}
	}
	for (int i=0; i<(int)(src.rows/2); i++){
		for (int j=(int)(src.cols/2); j<src.cols; j++){
			double tempv = temp[i][j];
			temp[i][j] = temp[i+(int)(src.rows/2)][j-(int)(src.cols/2)];
			temp[i+(int)(src.rows/2)][j-(int)(src.cols/2)] = tempv;
		}
	}
	// output (visual) result
//	writeFilterFile("FFT_test.xls", temp);
}

//*
// Evaluation for RAPSD	[2/14/2014 °ê¥°]
// parameter :	(1) "FImage", the average FFT (image) spectrum data , (2) "APS" vector for storing RAPSD result, 
//				(3) "AN" vector for storing anisotropy result , (4) "dr" for the value for width range (length) of delta radius assignment
//				(5) "im" for the single tone value			
double APSEvaluation(std::vector< std::vector< double > > &FImage, std::vector< double > &APS, std::vector< double > &AN, double dr, int im)
{
	cv::Mat temp;
	temp.create(static_cast<int>(FImage.size()), static_cast<int>(FImage[0].size()), 0);

	double g = (double)im / 256.0;
	double g2 = g * (1 - g);

	for (int i=0; i<temp.rows; i++)
		for (int j=0; j<temp.cols; j++)
			temp.data[i*temp.cols + j] = 0;

	double pi = 3.14159265;
	
	// for dr == 1, dd means the maximum radius of the image, only for (width == height), still need to be modified
	int dd = (int)sqrt( (double)2 * ( (FImage[0].size()/2) * (FImage[0].size()/2) ) );	
	int Number_APS = (int)((dd)/dr);
	// int Number_APS = APS.size();

	//std::cout << "dr = " << dr << "\tim = " << im << "\n";
	//std::cout << "dd = " << dd << "\tnum_aps = " <<  Number_APS << "\n";


	for (int p=0; p<Number_APS; p++)
		APS[p] = 0;

	for (int p=0; p<Number_APS; p++)
	{
		for (int i=0; i<temp.rows; i++)
			for (int j=0; j<temp.cols; j++)
				temp.data[i*temp.cols + j] = 0;

		double TotalE = 0;
		int Counter = 0;
		double r = p * dr;

		// RAPSD
		for (double theta=0; theta<360; theta+=0.1)
		{
			double epi = (double)theta * pi / 180.0;
			int x =  Round(r * cos(epi));
			int y =  Round(r * sin(epi));

			if ((int)(FImage.size()/2.0)+x < 0 || (int)(FImage.size()/2.0)+x >= FImage.size())
				continue;

			if ((int)(FImage[0].size()/2.0)+y < 0 || (int)(FImage[0].size()/2.0)+y >= FImage[0].size())
				continue;

			if (temp.data[((int)(temp.rows/2.0)+x) * temp.cols + ((int)(temp.cols/2.0)+y)] == 255)
				continue;

			TotalE = TotalE + FImage[(int)(FImage.size()/2.0)+x][(int)(FImage[0].size()/2.0)+y];
			Counter = Counter + 1;
			temp.data[((int)(temp.rows/2.0)+x) * temp.cols + ((int)(temp.cols/2.0)+y)] = 255;
		}
		if (Counter != 0 && TotalE != 0)
			APS[p] = (TotalE / Counter);
	}


	// Anisotropy
	for (int p=0; p<Number_APS; p++)
	{
		for (int i=0; i<temp.rows; i++)
			for (int j=0; j<temp.cols; j++)
				temp.data[i*temp.cols + j] = 0;

		double TotalE = 0;
		int Counter = 0;
		double r = p * dr;

		for (double theta=0; theta<360; theta+=0.1)
		{
			double epi = (double)theta * pi / 180.0;
			int x =  Round(r * cos(epi));
			int y =  Round(r * sin(epi));

			if ((int)(FImage.size()/2.0)+x < 0 || (int)(FImage.size()/2.0)+x >= FImage.size())
				continue;

			if ((int)(FImage[0].size()/2.0)+y < 0 || (int)(FImage[0].size()/2.0)+y >= FImage[0].size())
				continue;

			if (temp.data[((int)(temp.rows/2.0)+x) * temp.cols + ((int)(temp.cols/2.0)+y)] == 255)
				continue;

			TotalE = TotalE + 
				(APS[p] - FImage[(int)(FImage.size()/2.0)+x][(int)(FImage[0].size()/2.0)+y]) * 
				(APS[p] - FImage[(int)(FImage.size()/2.0)+x][(int)(FImage[0].size()/2.0)+y]);

			Counter = Counter + 1;
			temp.data[((int)(temp.rows/2.0)+x) * temp.cols + ((int)(temp.cols/2.0)+y)] = 255;
		}

		if (Counter != 0 && TotalE != 0)
			AN[p] = 10*log10((TotalE / (Counter-1)) / (APS[p]*APS[p]));
		APS[p] /= g2;
	}

	return 0;
}
//*/
int Round(double Value)
{
	int RV = 0;

	RV = (int)floor(Value+0.5);

	return RV;
}

/*-------------------------------------------------------------------------
Perform a 2D FFT inplace given a complex 2D array
The direction dir, 1 for forward, -1 for reverse
The size of the array (nx,ny)
Return false if there are memory problems or
the dimensions are not powers of 2
*/

int FFT2D(COMPLEX **c,int nx,int ny,int dir)
{
	int i,j;
	int m;
	double *real,*imag;

	if (nx == 512 && ny == 512)
		m = 9;
	else if (nx == 256 && ny == 256)
		m = 8;
	else if (nx == 128 && ny == 128)
		m = 7;
	else if (nx == 64 && ny == 64)
		m = 6;
	else if (nx == 32 && ny == 32)
		m = 5;
	else if (nx == 16 && ny == 16)
		m = 4;
	else if (nx == 8 && ny == 8)
		m = 3;
	else if (nx == 4 && ny == 4)
		m = 2;
	else if (nx == 2 && ny == 2)
		m = 1;
	else
		return false;

	/* Transform the rows */
	real = (double *)malloc(nx * sizeof(double));
	imag = (double *)malloc(nx * sizeof(double));
	if (real == NULL || imag == NULL)
		return(false);
	//if (!Powerof2(nx,&m,&twopm) || twopm != nx)
	//	return(false);
	for (j=0;j<ny;j++) {
		for (i=0;i<nx;i++) {
			real[i] = c[i][j].real;
			imag[i] = c[i][j].imag;
		}
		FFT(dir,m,real,imag);
		for (i=0;i<nx;i++) {
			c[i][j].real = real[i];
			c[i][j].imag = imag[i];
		}
	}
	free(real);
	free(imag);

	/* Transform the columns */
	real = (double *)malloc(ny * sizeof(double));
	imag = (double *)malloc(ny * sizeof(double));
	if (real == NULL || imag == NULL)
		return(false);
	//if (!Powerof2(ny,&m,&twopm) || twopm != ny)
	//	return(false);
	for (i=0;i<nx;i++) {
		for (j=0;j<ny;j++) {
			real[j] = c[i][j].real;
			imag[j] = c[i][j].imag;
		}
		FFT(dir,m,real,imag);
		for (j=0;j<ny;j++) {
			c[i][j].real = real[j];
			c[i][j].imag = imag[j];
		}
	}
	free(real);
	free(imag);

	return(true);
}

/*-------------------------------------------------------------------------
This computes an in-place complex-to-complex FFT
x and y are the real and imaginary arrays of 2^m points.
dir =  1 gives forward transform
dir = -1 gives reverse transform

Formula: forward
N-1
---
1   \          - j k 2 pi n / N
X(n) = ---   >   x(k) e                    = forward transform
N   /                                n=0..N-1
---
k=0

Formula: reverse
N-1
---
\          j k 2 pi n / N
X(n) =       >   x(k) e                    = forward transform
/                                n=0..N-1
---
k=0
*/

int FFT(int dir,int m,double *x,double *y)
{
	long nn,i,i1,j,k,i2,l,l1,l2;
	double c1,c2,tx,ty,t1,t2,u1,u2,z;

	/* Calculate the number of points */
	nn = 1;
	for (i=0;i<m;i++)
		nn *= 2;

	/* Do the bit reversal */
	i2 = nn >> 1;
	j = 0;
	for (i=0;i<nn-1;i++) {
		if (i < j) {
			tx = x[i];
			ty = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = tx;
			y[j] = ty;
		}
		k = i2;
		while (k <= j) {
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l=0;l<m;l++) {
		l1 = l2;
		l2 <<= 1;
		u1 = 1.0;
		u2 = 0.0;
		for (j=0;j<l1;j++) {
			for (i=j;i<nn;i+=l2) {
				i1 = i + l1;
				t1 = u1 * x[i1] - u2 * y[i1];
				t2 = u1 * y[i1] + u2 * x[i1];
				x[i1] = x[i] - t1;
				y[i1] = y[i] - t2;
				x[i] += t1;
				y[i] += t2;
			}
			z =  u1 * c1 - u2 * c2;
			u2 = u1 * c2 + u2 * c1;
			u1 = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (dir == 1)
			c2 = -c2;
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for forward transform */
	//if (dir == 1) {
	//	for (i=0;i<nn;i++) {
	//		x[i] /= (double)nn;
	//		y[i] /= (double)nn;
	//	}
	//}

	return(true);
}
