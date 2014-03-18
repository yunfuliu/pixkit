#include "../include/pixkit-image.hpp"

//////////////////////////////////////////////////////////////////////////
///// Local contrast enhancement
bool pixkit::enhancement::local::LiuJinChenLiuLi2011(const cv::Mat &src,cv::Mat &dst,const cv::Size N){

	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[enhancement::local::LiuJinChenLiuLi2011] image should be grayscale");
	}
	if(N.width>src.cols||N.height>src.rows){
		CV_Error(CV_StsBadArg,"[enhancement::local::LiuJinChenLiuLi2011] parameter N should < image size");
	}

	//////////////////////////////////////////////////////////////////////////
	const int	nColors	=	256;	// how many colors in the input image
	dst.create(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	// define the stepsize
	float		tempv1	=	(float)src.cols/N.width,
				tempv2	=	(float)src.rows/N.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	stepsize((int)tempv1,(int)tempv2);
	
	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<std::vector<float>>>	Tinput(N.height,std::vector<std::vector<float>>(N.width,std::vector<float>(nColors,0)));
	// get cdf of each block (Step 1)
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){

			// compute pdf, then compute cdf to store in Tinput
			std::vector<float>	pdf(nColors,0);
			int	temp_count=0;
			for(int m=0;m<stepsize.height;m++){
				for(int n=0;n<stepsize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						pdf[(int)src.data[(i+m)*src.cols+(j+n)]]++;
						temp_count++;
					}					
				}
			}
			for(int m=0;m<nColors;m++){
				pdf[m]/=(float)temp_count;
			}

			// get cdf
			Tinput[i/stepsize.height][j/stepsize.width][0]=pdf[0];
			for(int m=1;m<nColors;m++){
				Tinput[i/stepsize.height][j/stepsize.width][m] = Tinput[i/stepsize.height][j/stepsize.width][m-1] + pdf[m];
				if(Tinput[i/stepsize.height][j/stepsize.width][m]>1.){
					Tinput[i/stepsize.height][j/stepsize.width][m]=1.;
				}
				assert(Tinput[i/stepsize.height][j/stepsize.width][m]>=0.&&Tinput[i/stepsize.height][j/stepsize.width][m]<=1.);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// get enhanced result (Step 3)
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			// enhance each pixel (A: current; B: right; C: top; D: top-right)
			float	enh_A=-1,enh_B=-1,enh_C=-1,enh_D=-1;	// the reason why not use the 0 to instead of -1 is for the following decision (to check whether that block had been accessed or not)
			enh_A	=	Tinput[i/stepsize.height][j/stepsize.width][(int)(src.data[i*src.cols+j]+0.5)];	// enh_x here denotes only the enhanced result
			if((float)(j+stepsize.width)/stepsize.width<N.width){
				enh_B	=	Tinput[i/stepsize.height][(j+stepsize.width)/stepsize.width][(int)(src.data[i*src.cols+j]+0.5)];
			}
			if((float)(i+stepsize.height)/stepsize.height<N.height){
				enh_C	=	Tinput[(i+stepsize.height)/stepsize.height][j/stepsize.width][(int)(src.data[i*src.cols+j]+0.5)];
			}
			if((float)(i+stepsize.height)/stepsize.height<N.height&&(float)(j+stepsize.width)/stepsize.width<N.width){
				enh_D	=	Tinput[(i+stepsize.height)/stepsize.height][(j+stepsize.width)/stepsize.width][(int)(src.data[i*src.cols+j]+0.5)];
			}
			
			// enhancement
			double	weight_A	=	(stepsize.height+1-(i%stepsize.height+1))	*	(stepsize.width+1-(j%stepsize.width+1)),	// this is to represent the weight for each block only
					weight_B	=	(stepsize.height+1-(i%stepsize.height+1))	*	(j%stepsize.width+1),
					weight_C	=	(i%stepsize.height+1)						*	(stepsize.width+1-(j%stepsize.width+1)),
					weight_D	=	(i%stepsize.height+1)						*	(j%stepsize.width+1);

			double	temp_dst		=	(double)(1./((enh_A==-1?0:weight_A)+(enh_B==-1?0:weight_B)+(enh_C==-1?0:weight_C)+(enh_D==-1?0:weight_D)))	*	// this equation is additional added since the paper did not give the process when meet the boundary of an image and the normalize term is bigger than the sum of all the weights. 
										((double)	(enh_A==-1?0:enh_A)		*	weight_A	+	// also, this strategy is to make sure that only the accessed parts are added in this calculation.									
										(double)	(enh_B==-1?0:enh_B)		*	weight_B	+		
										(double)	(enh_C==-1?0:enh_C)		*	weight_C	+			
										(double)	(enh_D==-1?0:enh_D)		*	weight_D);
			
			assert(temp_dst>=0&&temp_dst<=255.);
			dst.data[i*src.cols+j]	=	(int)((temp_dst	*	255.)+0.5);	// (Step 2)

		}
	}

	return true;
}
bool pixkit::enhancement::local::LambertiMontrucchioSanna2006(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S){

	//////////////////////////////////////////////////////////////////////////
	// initialization
	const	short	nColors	=	256;	// 影像之顏色數量.

	//////////////////////////////////////////////////////////////////////////
	// transformation (block size)
	float	tempv1	=	(float)src.cols/B.width,
			tempv2	=	(float)src.rows/B.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	blocksize((int)tempv1,(int)tempv2);
	tempv1	=	(float)src.cols/S.width;
	tempv2	=	(float)src.rows/S.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	stepsize((int)tempv1,(int)tempv2);

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(blocksize.height>src.rows||blocksize.width>src.cols||blocksize.height==1||blocksize.width==1){	// block size should be even (S3P5-Step.2)
		return false;
	}
	if(stepsize.height>blocksize.height/2||stepsize.width>blocksize.width/2){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
// 	// output image
// 	outputImage=new double*[width];
// 	for (int i=0;i<width;i++)
// 		outputImage[i]=new double[length];
	// transformation functions (S4P3-step a)
	std::vector<std::vector<std::vector<float>>>	Tinput(S.height,std::vector<std::vector<float>>(S.width,std::vector<float>(nColors,0)));
	std::vector<std::vector<std::vector<float>>>	Toutput(S.height,std::vector<std::vector<float>>(S.width,std::vector<float>(nColors,0)));
// 	Tinput = new double**[S+LPF_SIZE-1];
// 	Toutput = new double**[S+LPF_SIZE-1];
// 	for (int i=0;i<S+LPF_SIZE-1;i++){
// 		Tinput[i] = new double*[S+LPF_SIZE-1];
// 		Toutput[i] = new double*[S+LPF_SIZE-1];
// 		for (int j=0;j<S+LPF_SIZE-1;j++){
// 			Tinput[i][j] = new double[256];
// 			Toutput[i][j] = new double[256];
// 		}
// 	}

	//////////////////////////////////////////////////////////////////////////
	// get transformation functions (S4P3-step b)
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){

			// computing PDF
			std::vector<float>	pdf(nColors,0);
			int	temp_count=0;
			for(int m=0;m<blocksize.height;m++){
				for(int n=0;n<blocksize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						pdf[(int)src.data[(i+m)*src.cols+(j+n)]]++;
						temp_count++;
					}					
				}
			}
			for(int m=0;m<nColors;m++){
				pdf[m]/=(float)temp_count;
			}

			// computing CDF that is stored in Tinput
			Tinput[i/stepsize.height][j/stepsize.width][0]=pdf[0];
			for(int m=1;m<nColors;m++){
				Tinput[i/stepsize.height][j/stepsize.width][m] = Tinput[i/stepsize.height][j/stepsize.width][m-1] + pdf[m];
				if(Tinput[i/stepsize.height][j/stepsize.width][m]>1.){
					Tinput[i/stepsize.height][j/stepsize.width][m]=1.;
				}
				assert(Tinput[i/stepsize.height][j/stepsize.width][m]>=0.&&Tinput[i/stepsize.height][j/stepsize.width][m]<=1.);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// copy
	for(int i=0;i<S.height;i++){
		for(int j=0;j<S.width;j++){
			for(int m=0;m<nColors;m++){
				Toutput[i][j][m]	=	Tinput[i][j][m];
			}
		}
	}
	// refine the transformation functions
	int delta = 1;
	const	double	sx	=	log((double)S.width/B.width)/log(2.0);
	const	double	sy	=	log((double)S.height/B.height)/log(2.0);
	double	s	=	sx>sy?sy:sx;
	for(int times=0;times<s;times++){

		// horizontal direction (S4P3-step c)
		for(int i=0;i<S.height;i++){
			for(int j=delta;j<S.width-delta;j++){		
				for(int m=0;m<nColors;m++){
					Toutput[i][j][m] = 0;
					Toutput[i][j][m] += Tinput[i][j-delta][m]/4.;
					Toutput[i][j][m] += Tinput[i][j][m]/2;
					Toutput[i][j][m] += Tinput[i][j+delta][m]/4.;
					assert(Toutput[i][j][m]>=0&&Toutput[i][j][m]<=1);
				}
			}
		}

		// vertical direction (S4P3-step d)
		for(int i=delta;i<S.height-delta;i++){
			for(int j=0;j<S.width;j++){				
				for(int m=0;m<nColors;m++){
					Tinput[i][j][m] = 0;
					Tinput[i][j][m] += Toutput[i-delta][j][m]/4.;
					Tinput[i][j][m] += Toutput[i][j][m]/2.;
					Tinput[i][j][m] += Toutput[i+delta][j][m]/4.;
					assert(Tinput[i][j][m]>=0&&Tinput[i][j][m]<=1);
				}
			}
		}

		delta *= 2;
	}

	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<float>>	tdst(src.rows,std::vector<float>(src.cols,0));
	std::vector<std::vector<short>>	accu_count(src.rows,std::vector<short>(src.cols,0));	// the add number of each pixel
	// enhancement
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){
			for(int m=0;m<blocksize.height;m++){
				for(int n=0;n<blocksize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						tdst[i+m][j+n]	+=	Tinput[i/stepsize.height][j/stepsize.width][(int)src.data[(i+m)*src.cols+(j+n)]]	*	((float)nColors-1);
						accu_count[i+m][j+n]	++;
					}					
				}
			}
		}
	}
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			tdst[i][j]	/=	(float)accu_count[i][j];
			assert(tdst[i][j]>=0.&&tdst[i][j]<=nColors-1.);
		}
	}



	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(),src.type());
	for(int i=0;i<dst.rows;i++){
		for(int j=0;j<dst.cols;j++){
			dst.data[i*dst.cols+j]	=	(uchar)(tdst[i][j]	+0.5);
		}
	}

	return true;
}
bool pixkit::enhancement::local::YuBajaj2004(const cv::Mat &src,cv::Mat &dst,const int blockheight,const int blockwidth,const float C,bool anisotropicMode,const float R){

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8UC1){
		return false;
	}
	if(blockheight>src.rows||blockheight%2==0){
		return false;
	}
	if(blockwidth>src.cols||blockwidth%2==0){
		return false;
	}
	if(anisotropicMode){
		if(R<0.01||R>0.1){
			return false;
		}
	}else{
		if(C>1||C<0){
			return false;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// initialization
	cv::Mat	tdst;
	tdst.create(src.size(),src.type());
	const	float	w	=	255.;

	//////////////////////////////////////////////////////////////////////////
	// get max, min, and avg
	cv::Mat	maxmap(src.size(),src.type()),
			minmap(src.size(),src.type()),
			avgmap(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			
			float	maxv	=	src.data[i*src.cols+j];
			float	minv	=	src.data[i*src.cols+j];
			float	avgv	=	0.;
			int		avgv_count	=	0;
			for(int m=-blockheight/2;m<=blockheight/2;m++){
				for(int n=-blockwidth/2;n<=blockwidth/2;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						uchar	&currv	=	src.data[(i+m)*src.cols+(j+n)];
						if(currv>maxv){
							maxv=currv;
						}
						if(currv<minv){
							minv=currv;
						}
						avgv+=currv;
						avgv_count++;
					}
				}
			}
			avgv	/=	(float)	avgv_count;

			maxmap.data[i*maxmap.cols+j]	=	maxv;
			minmap.data[i*minmap.cols+j]	=	minv;
			avgmap.data[i*avgmap.cols+j]	=	avgv;		

		}
	}

	//////////////////////////////////////////////////////////////////////////
	// smoothing
	if(anisotropicMode){

		for(int i=0;i<src.rows;i++){
			for(int j=1;j<src.cols;j++){

				// avg
				avgmap.data[i*avgmap.cols+j]	+=	(avgmap.data[i*avgmap.cols+(j-1)] - avgmap.data[i*avgmap.cols+j]) * exp(-R * fabs((float)(avgmap.data[i*avgmap.cols+(j-1)] - avgmap.data[i*avgmap.cols+j])));
				// min
				if(minmap.data[i*minmap.cols+(j-1)] < minmap.data[i*minmap.cols+j]){
					minmap.data[i*minmap.cols+j]	+=	(minmap.data[i*minmap.cols+(j-1)] - minmap.data[i*minmap.cols+j]) * exp(-R * fabs((float)(minmap.data[i*minmap.cols+(j-1)] - minmap.data[i*minmap.cols+j])));	
				}
				// max
				if(maxmap.data[i*maxmap.cols+(j-1)] > maxmap.data[i*maxmap.cols+j]){
					maxmap.data[i*maxmap.cols+j]	+=	(maxmap.data[i*maxmap.cols+(j-1)] - maxmap.data[i*maxmap.cols+j]) * exp(-R * fabs((float)(maxmap.data[i*maxmap.cols+(j-1)] - maxmap.data[i*maxmap.cols+j])));
				}

			}
		}

	}else{

		for(int i=0;i<src.rows;i++){
			for(int j=1;j<src.cols;j++){

				// avg
				avgmap.data[i*avgmap.cols+j]	=	(uchar)((float)(1.-C)*avgmap.data[i*avgmap.cols+j]	+	(float)C*avgmap.data[i*avgmap.cols+(j-1)]	+0.5);
				// min
				if(minmap.data[i*minmap.cols+(j-1)] < minmap.data[i*minmap.cols+j]){
					minmap.data[i*minmap.cols+j]	=	(uchar)((float)(1.-C)*minmap.data[i*minmap.cols+j]	+	(float)C*minmap.data[i*minmap.cols+(j-1)]	+0.5);
				}
				// max
				if(maxmap.data[i*maxmap.cols+(j-1)] > maxmap.data[i*maxmap.cols+j]){
					maxmap.data[i*maxmap.cols+j]	=	(uchar)((float)(1.-C)*maxmap.data[i*maxmap.cols+j]	+	(float)C*maxmap.data[i*maxmap.cols+(j-1)]	+0.5);
				}

			}
		}

	}

	//////////////////////////////////////////////////////////////////////////
	// enhancement
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			// get Inew and Anew
			float	Inew	=	w	*	(float)(src.data[i*src.cols+j] - minmap.data[i*src.cols+j] )/(float)(maxmap.data[i*src.cols+j]  - minmap.data[i*src.cols+j] );
			float	Anew	=	w	*	(float)(avgmap.data[i*src.cols+j] - minmap.data[i*src.cols+j] )/(float)(maxmap.data[i*src.cols+j]  - minmap.data[i*src.cols+j] );

			// get afa
			float	afa	=	(Anew-Inew)/128.;

			// get, afa, beta, gamma
			float	a,b,c;
			a	=	afa	/	(2.	*	w);
			b	=	(float)afa / w * src.data[i*src.cols+j]	-	afa	-	1.;
			c	=	(float)afa / (2.*w) * src.data[i*src.cols+j] * src.data[i*src.cols+j]	-	(float)afa	* src.data[i*src.cols+j] + (float)src.data[i*src.cols+j];			

			// get result
			float	tempv;
			if(afa<-0.000001||afa>0.000001){
				tempv	=	(-b-sqrt((float)b*b-(float)4.*a*c))/(2.*a);
			}else{
				tempv	=	src.data[i*src.cols+j];
			}
			if(tempv>255.){
				tempv=255.;
			}
			if(tempv<0.){
				tempv=0.;
			}
			tdst.data[i*tdst.cols+j]	=	(uchar)	tempv+0.5;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}
bool pixkit::enhancement::local::KimKimHwang2001(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S){

	//////////////////////////////////////////////////////////////////////////
	// initialization
	const	short	nColors	=	256;	// 影像之顏色數量.
	// transformation
	cv::Size	blocksize	=	cv::Size(src.cols/B.width,src.rows/B.height);
	cv::Size	stepsize	=	cv::Size(src.cols/S.width,src.rows/S.height);

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(blocksize.height>src.rows||blocksize.width>src.cols||blocksize.height==1||blocksize.width==1){	// block size should be even (S3P5-Step.2)
		return false;
	}
	if(stepsize.height>blocksize.height/2||stepsize.width>blocksize.width/2){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<float>>	tdst(src.rows,std::vector<float>(src.cols,0));
	std::vector<std::vector<short>>	accu_count(src.rows,std::vector<short>(src.cols,0));	// the add number of each pixel
	// process (S3P5-Steps 3 and 4)
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){
			
			// get pdf
			std::vector<float>	pdf(nColors,0.);
			int	temp_count	=	0;
			for(int m=0;m<blocksize.height;m++){
				for(int n=0;n<blocksize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						pdf[(int)src.data[(i+m)*src.cols+(j+n)]]++;
						temp_count++;
					}

				}
			}
			for(int m=0;m<nColors;m++){
				pdf[m]/=(double)temp_count;
			}

			// get cdf
			std::vector<float>	cdf(nColors,0.);
			cdf[0]=pdf[0];
			for(int m=1;m<nColors;m++){
				cdf[m]=cdf[m-1]+pdf[m];
				if(cdf[m]>1.){
					cdf[m]=1;
				}
				assert(cdf[m]>=0.&&cdf[m]<=1.);
			}

			// get enhanced result and accumulate 
			for(int m=0;m<blocksize.height;m++){
				for(int n=0;n<blocksize.width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						tdst[i+m][j+n]	+=	(float)cdf[(int)src.data[(i+m)*src.cols+(j+n)]]*(nColors-1);
						accu_count[i+m][j+n]++;
					}
				}
			}			
		}
	}
	// process (S3P5-Step5)
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			tdst[i][j]	/=	(float)accu_count[i][j];
			assert(tdst[i][j]>=0&&tdst[i][j]<=255.);
			
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// BERF (blocking effect reduction filter)
	// for vertical
	for(int i=stepsize.height;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j++){

			// S3P5-Step6 and S3P7
			if (fabs(tdst[i][j]-tdst[i-1][j])>=3){
 				double avg=(tdst[i][j]+tdst[i-1][j])/2.;
 				tdst[i][j]=avg;
 				tdst[i-1][j]=avg;
			}
		}
	}
	// for horizontal
	for(int i=0;i<src.rows;i++){
		for(int j=stepsize.width;j<src.cols;j+=stepsize.width){

			// S3P5-Step6 and S3P7
			if (fabs(tdst[i][j]-tdst[i][j-1])>=3){
				double avg=(tdst[i][j]+tdst[i][j-1])/2.;
				tdst[i][j]=avg;
				tdst[i][j-1]=avg;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			dst.data[i*dst.cols+j]	=	tdst[i][j];
		}
	}

	return true;
}
bool pixkit::enhancement::local::Stark2000(const cv::Mat &src,cv::Mat &dst,const int blockheight,const int blockwidth,const float alpha,const float beta){

	//////////////////////////////////////////////////////////////////////////
	if(blockheight%2==0){
		return false;
	}
	if(blockwidth%2==0){
		return false;
	}
	if(alpha<0||alpha>1){
		return false;
	}
	if(beta<0||beta>1){
		return false;
	}
	if(src.type()!=CV_8UC1){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	// processing
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			//////////////////////////////////////////////////////////////////////////
			// calc histogram for each pixel
			int numCount=0;
			double hist[256]={0};
			for(int m=-blockheight/2;m<=blockheight/2;m++){
				for(int n=-blockwidth/2;n<=blockwidth/2;n++){
					if( (i+m>=0)&&(j+n>=0)&&(i+m<src.rows)&&(j+n<src.cols) ){
						numCount++;
						hist[(int)src.data[(i+m)*src.cols+(j+n)]]++;
					}
				}
			}
			// change to pdf
			for(int m=0;m<256;m++){
				hist[m] /= numCount;
			}

			//////////////////////////////////////////////////////////////////////////
			// 計算輸出值  ps. 需正規化 -1/2~1/2
			double normalizedinput=((float)src.data[i*src.cols+j]/255.)-0.5;
			assert(normalizedinput>=-0.5&&normalizedinput<=0.5);	
			double output	=	0.;
			for(int c=0;c<256;c++){

				// calc q
				double	q1=0., 
					q2=0., 
					d=normalizedinput-(((double)c/255.)-0.5);
				// for q1 (Eq. (13))
				if(d>0){
					q1	=	0.5*pow((double)2.*d,(double)alpha);
				}else if (d<0){
					q1	=	-0.5*pow((double)fabs(2.*d),(double)alpha);
				}
				// for q2 (Eq. (13))
				if(d>0){
					q2	=	0.5*2.*d;
				}else if (d<0){
					q2	=	-0.5*fabs(2.*d);
				}
				// Eq. (16)
				double	q	=	q1-beta*q2+beta*normalizedinput;

				// Eq. (5)
				output += hist[c]*q;
			}
			// normalize output
			output	=	255.*(output+0.5);
			if(output>255){
				output=255;
			}
			if(output<0){
				output=0;
			}
			tdst.data[i*tdst.cols+j]=(uchar)(output+0.5);
		}
	}

	dst	=	tdst.clone();

	return true;
}
bool pixkit::enhancement::local::LocalHistogramEqualization1992(const cv::Mat &src,cv::Mat &dst,const cv::Size blocksize){

	//////////////////////////////////////////////////////////////////////////
	if(blocksize.height%2==0){
		return false;
	}
	if(blocksize.width%2==0){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst(src.size(),src.type());
	const int nColors	=	256;

	//////////////////////////////////////////////////////////////////////////
	// processing
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){

			std::vector<double>	Histogram(nColors,0);	// 256個灰階值

			// 進行統計
			int temp_count=0;
			for(int m=-blocksize.height/2;m<=blocksize.height/2;m++){
				for(int n=-blocksize.width/2;n<=blocksize.width/2;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						Histogram[(int)(src.data[(i+m)*src.cols+(j+n)])]++;
						temp_count++;
					}					
				}
			}
			for(int graylevel=0;graylevel<nColors;graylevel++){
				Histogram[graylevel]/=(double)(temp_count);
			}

			// 將Histogram改為累積分配函數
			for(int graylevel=1;graylevel<nColors;graylevel++){
				Histogram[graylevel]+=Histogram[graylevel-1];
			}

			// 取得新的輸出值
			double	tempv	=	Histogram[(int)(src.data[i*src.cols+j])];
			if(tempv>1){
				tempv=1.;
			}
			assert(tempv>=0.&&tempv<=1.);
			tdst.data[i*src.cols+j]=tempv*(nColors-1.);	// 最多延展到255		

		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}

bool pixkit::enhancement::local::Pizer1987(cv::Mat &src,cv::Mat &dst, cv::Size title, float L )
{
	///////////////////////////////////////////////////////////////////////////////////////////////////
	if(src.type()!=CV_8UC1){
		return false;
	}
	if(L>1 || L<=0)
		return false;

	if(title.height > (src.rows/4) || title.width > (src.cols/4))
		return false;
	//////////////////////////////////////////////////////////////////////////////////////////////////
	const int nColors = 256;
	int x = src.cols/title.width, y = src.rows/title.height;

	dst = cvCreateMat(src.rows,src.cols,src.type());

	std::vector<std::vector<float>> hist(title.height*title.width,std::vector<float> (nColors,0)); //儲存每個title的轉移函式
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//計算每個title的轉移函式
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for(int m = 0;m<title.height;m++)
		for(int n = 0;n<title.width;n++)
		{
			int i,j,i1=(m+1)*y,j1=(n+1)*x;

			if( (m+1) == title.height )
				i1 = src.rows;
			if((n+1) == title.width )
				j1 = src.cols;

			int Count = 0;
			for(i=m*y;i<i1;i++)
				for(j=n*x;j<j1;j++)
				{
					hist[m*title.width+n][(int)src.data[i*src.cols+j]]++;
					Count++;
				}

				int limt = (int) (Count*L + 0.5);  //計算需要裁切的限制

				float over_limit = 0;
				for(int k=1;k<256;k++)
				{

					if(hist[m*title.width+n][k] > limt)
					{
						over_limit += (hist[m*title.width+n][k]-limt);
						hist[m*title.width+n][k] = limt;
					}
				}

				over_limit /= nColors;

				for(int k=0;k<256;k++)
				{
					hist[m*title.width+n][k] += over_limit;
					hist[m*title.width+n][k] = (hist[m*title.width+n][k]/Count)*(nColors-1);
				}

				for(int k=1;k<256;k++)
					hist[m*title.width+n][k] += hist[m*title.width+n][k-1];
		}
		//////////////////////////////////////////////////////////////////////////
		//計算輸出
		///////////////////////////////////////////////////////////////////////
		int a1=0,a2=x/2,b1=0,b2=y/2;  //a表示x軸方向.b表示y軸方向
		for(int i=0;i<src.rows;i++)
		{
			a2 = x/2 , a1 = 0;
			for(int j=0;j<src.cols;j++)
			{
				if(j>a2)
				{
					a1=a2;
					a2+=x;

					if(a2>=src.cols)
						a2 = src.cols-1;
				}
				if(i>b2)
				{
					b1 = b2;
					b2 += y;
					if(b2>=src.rows)
						b2 = src.rows-1;
				}

				int p1=a1/x,p2=a2/x,q1=b1/y,q2=b2/y;
				if(p2 >= title.width)
					p2 = title.width-1;
				if(q2 >= title.height)
					q2 = title.height-1;

				float a=(float)(a2-j)/(a2-a1), b=(float)(b2-i)/(b2-b1);
				int v = (int)src.data[i*src.cols+j];
				dst.data[i*dst.cols+j] = (unsigned char) (b*(a*hist[q1*title.width+p1][v] + (1-a)*hist[q1*title.width+p2][v]) + (1-b)*(a*hist[q2*title.width+p1][v] + (1-a)*hist[q2*title.width+p2][v]));
			}
		}

		return true;
}
bool pixkit::enhancement::local::Lal2014(cv::Mat &src,cv::Mat &dst, cv::Size title, float L,float K1 ,float K2 )
{
	///////////////////////////////////////////////////////////////////////////////////////////////////
	if(src.type()!=CV_8UC1){
		return false;
	}
	if(L>1 || L<=0)
		return false;

	if(title.height > (src.rows/4) || title.width > (src.cols/4))
		return false;

	if(K2>1 || K2<0)
		return false;
	//////////////////////////////////////////////////////////////////////////////////////////////////
	const int nColors = 256;
	int x = src.cols/title.width, y = src.rows/title.height;

	dst = cvCreateMat(src.rows,src.cols,src.type());
	cv::Mat temp = cvCreateMat(src.rows,src.cols,src.type());
	std::vector<std::vector<float>> hist(title.height*title.width,std::vector<float> (nColors,0)); //儲存每個title的轉移函式
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//計算Sigmoid轉換
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for(int i=0;i<temp.rows;i++)
		for(int j=0;j<temp.cols;j++)
		{
			float t = (double)src.data[i*src.cols+j]/(nColors-1);
			float o = t + K1*t/(1.0-exp(K1*(K2+t)));

			temp.data[i*temp.cols+j] = o*(nColors-1);
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//計算每個title的轉移函式
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for(int m = 0;m<title.height;m++)
			for(int n = 0;n<title.width;n++)
			{
				int i,j,i1=(m+1)*y,j1=(n+1)*x;

				if( (m+1) == title.height )
					i1 = src.rows;
				if((n+1) == title.width )
					j1 = src.cols;

				int Count = 0;
				for(i=m*y;i<i1;i++)
					for(j=n*x;j<j1;j++)
					{
						hist[m*title.width+n][(int)temp.data[i*temp.cols+j]]++;
						Count++;
					}

					int limt = (int) (Count*L + 0.5);  //計算需要裁切的限制

					float over_limit = 0;
					for(int k=1;k<256;k++)
					{

						if(hist[m*title.width+n][k] > limt)
						{
							over_limit += (hist[m*title.width+n][k]-limt);
							hist[m*title.width+n][k] = limt;
						}
					}

					over_limit /= nColors;

					for(int k=0;k<256;k++)
					{
						hist[m*title.width+n][k] += over_limit;
						hist[m*title.width+n][k] = (hist[m*title.width+n][k]/Count)*(nColors-1);
					}

					for(int k=1;k<256;k++)
						hist[m*title.width+n][k] += hist[m*title.width+n][k-1];
			}
			//////////////////////////////////////////////////////////////////////////
			//計算輸出
			///////////////////////////////////////////////////////////////////////
			int a1=0,a2=x/2,b1=0,b2=y/2;  //a表示x軸方向.b表示y軸方向
			for(int i=0;i<src.rows;i++)
			{
				a2 = x/2 , a1 = 0;
				for(int j=0;j<src.cols;j++)
				{
					if(j>a2)
					{
						a1=a2;
						a2+=x;

						if(a2>=src.cols)
							a2 = src.cols-1;
					}
					if(i>b2)
					{
						b1 = b2;
						b2 += y;
						if(b2>=src.rows)
							b2 = src.rows-1;
					}

					int p1=a1/x,p2=a2/x,q1=b1/y,q2=b2/y;
					if(p2 >= title.width)
						p2 = title.width-1;
					if(q2 >= title.height)
						q2 = title.height-1;

					float a=(float)(a2-j)/(a2-a1), b=(float)(b2-i)/(b2-b1);
					int v = (int)src.data[i*src.cols+j];
					dst.data[i*dst.cols+j] = (unsigned char) (b*(a*hist[q1*title.width+p1][v] + (1-a)*hist[q1*title.width+p2][v]) + (1-b)*(a*hist[q2*title.width+p1][v] + (1-a)*hist[q2*title.width+p2][v]));
				}
			}

			return true;
}

bool pixkit::enhancement::local::Sundarami2011(cv::Mat &src,cv::Mat &dst, cv::Size N, float L, float phi)
{
	//////////////////////////////////////////////////////////////////////////////
	if(src.type()!=CV_8UC1){
		return false;
	}
	if(L>1 || L<=0)
		return false;

	if(N.height >= src.rows-1 || N.width >= src.cols-1)
		return false;

	if(phi>1 || phi<0)
		return false;

	if(N.height%2==0 || N.width%2==0)
		return false;
	//////////////////////////////////////////////////////////////////////////////
	int limt = (int) (N.height*N.width*L + 0.5);
	int x = N.width/2, y = N.height/2;

	dst = cvCreateMat(src.rows,src.cols,src.type());

	for(int i=0;i<dst.rows;i++)
		for(int j=0;j<dst.cols;j++)
		{
			std::vector<float> hist(256,0);

			float Total = 0;
			for(int m=i-y;m<=i+y;m++)
				for(int n=j-x;n<=j+x;n++)
				{
					if(m>=0 && m<dst.rows && n>=0 && n<dst.cols)
					{
						hist[(int)src.data[m*src.cols+n]]++;
						Total++;
					}
				}

				//////////////////////////////////////////////////////////////////////////
				//調整Histogram
				//////////////////////////////////////////////////////////////////////////
				float u = Total/256.0; //代表均勻分布
				for(int k=0;k<256;k++)
				{
					hist[k] = (1.0/(1.0+phi))*hist[k] + (phi/(1+phi))*u;
				}

				float over_limit = 0;
				for(int k=1;k<256;k++)
				{
					if(hist[k] > limt)
					{
						over_limit += (hist[k]-limt);
						hist[k] = limt;
					}
				}

				over_limit /= 256;

				for(int k=0;k<256;k++)
					hist[k] += over_limit;

				for(int k=1;k<256;k++)
					hist[k] += hist[k-1];

				dst.data[i*dst.cols+j] = (unsigned char) ( (hist[(int)src.data[i*src.cols+j]]*255.0/Total + 0.5) ); 
		}
		return true;
}


//////////////////////////////////////////////////////////////////////////
///// Global contrast enhancement
bool pixkit::enhancement::global::WadudKabirDewanChae2007(const cv::Mat &src, cv::Mat &dst, const int x){

	//////////////////////////////////////////////////////////////////////////
	//	exception process
	if (src.type()!=CV_8U){
		return false;
	}
	//////////////////////////////////////////////////////////////////////////
	dst=src.clone();
	//////////////////////////////////////////////////////////////////////////
	//	step	1	:	histogram partition
	//	get histogram
	int		hist[256]={0};
	for (int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			hist[(int)src.data[i*src.cols+j]]++;
		}
	}
	//	smooth histogram
	int smoothfilter[3]={1,1,1};
	for(int i=0;i<256;i++){
		int		tempnum=0;
		float	sum=0;
		for (int j=-1;j<2;j++){
			if ((i+j)>=0 && (i+j)<256){
				sum+=(float)hist[i+j]*(float)smoothfilter[j+1];
				tempnum++;
			}	
		}
		tempnum*=smoothfilter[0];
		hist[i]=(int)(sum/(float)tempnum+0.5);
	}
	//	get minima for sub-histogram
	int	count=0;				//	pointer of minima array.
	int	minima[256]={0};		//	儲存最小值
	bool PartitionFlag=false;	//	true:進行histogram分區, 並判斷是否符合高斯的68.3%分布
	bool SubHistFlag=false;		//	true:histogram分區後, low 換到 high histogram 的68.3%判斷
	bool SubHistFlag2=false;
	double sumFactor=0.;			//	sum of factor.
	double range[256]={0};
	int q=0;
	for (int i=0;i<256;i++){
		//	get first non-zero number
		if (hist[i-1]==0 && hist[i]!=0 || (i==0 && hist[0]!=0)){
			minima[count]=i;
			count++;
			PartitionFlag=true;
		}	
		//	get minima number
		if (hist[i]<hist[i-1] && hist[i]<hist[i+1]){
			minima[count]=i;
			count++;
			PartitionFlag=true;
		}
		//	get last non-zero number && i==0, hist[0]!=0
		if ((hist[i]!=0 && hist[i+1]==0) || (i==255 && hist[0]!=0)){
			minima[count]=i;
			count++;
			PartitionFlag=true;
		}
		if (count==1){					//	第一個minima不進行分區
			PartitionFlag=false;
		}
		if (minima[0]==minima[1]){		//	修正上面判斷BUG
			count=1;
			PartitionFlag=false;
		}
		//	judge is (mean +- standard deviation) satisfy 68.3% of GL or not.
		int a=0;		
		while (PartitionFlag){
			double	sum=0, mean=0, sd=0, temp=0;
			//	get mean
			for (int k=minima[count-2];k<=minima[count-1];k++){
				mean+=(double)hist[k]*k;
				sum+=(double)hist[k];
			}
			mean/=sum;
			//	get standard deviation
			for (int k=minima[count-2];k<=minima[count-1];k++){
				sd+=(pow((double)k-mean,2)*(double)hist[k]);
			}
			sd=sqrt(sd/sum);
			//	judge 68.3% for (mean +- sd)
			for (int k=(int)(mean-sd+0.5);k<=(int)(mean+sd+0.5);k++){
				temp+=(double)hist[k];
			}
			temp/=sum;
			if (temp>=0.683){
				if (SubHistFlag){		//	(mean+sd) 至 high-minima的高斯分布判定
					if(SubHistFlag2){
						count+=3;
						SubHistFlag2=false;
					}else{
						count+=2;
					}
					SubHistFlag=false;
					a=0;
				}else{
					PartitionFlag=false;
				}					
			}else{						//	low-minima 至 (mean-sd)的高斯分布判定.
				if(a>0){
					for (int m=0;m<=a;m++){
						minima[count+m+2]=minima[count+m];
						SubHistFlag2=true;
					}
				}
				minima[count+1]=minima[count-1];
				minima[count]=(int)(mean+sd+0.5);
				minima[count-1]=(int)(mean-sd+0.5);
				SubHistFlag=true;
				a++;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//	step	2	:	gray level allocation by cumulative frequencies (CF)
	//////////////////////////////////////////////////////////////////////////
	for (int i=1;i<count;i++){
		double	sumA=0.;
		for (int j=minima[i-1];j<minima[i];j++){
			sumA+=(double)hist[j];
		}
		if(sumA!=0){
			double a=log10(sumA);
			range[i]=(minima[i]-minima[i-1])*pow(a,x);
			sumFactor+=range[i];
		}
	}
	double	a=0.;
	for (int i=0;i<count;i++){
		range[i]=range[i]*255./sumFactor;
		a+=range[i];
		range[i]=a;
	}
	//////////////////////////////////////////////////////////////////////////
	//	step	3	:	histogram equalization
	//////////////////////////////////////////////////////////////////////////
	double	cdf[256]={0.};
	for(int i=1;i<count;i++){
		double	sumCdf=0.;
		double	sumGL=0.;
		for (int j=minima[i-1];j<minima[i];j++){
			sumGL+=(double)hist[j];
		}
		for (int j=minima[i-1];j<minima[i];j++){
			sumCdf+=(double)hist[j]/sumGL;
			cdf[j]=sumCdf;
		}
		for (int j=minima[i];j<256;j++){
			cdf[j]=1;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	//	image output
	//////////////////////////////////////////////////////////////////////////
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			int a=0;
			for(int k=1;k<count;k++){
				if (minima[k-1]<(int)src.data[i*src.cols+j] && minima[k]>=(int)src.data[i*src.cols+j]){
					a=k;
					break;
				}
			}
			dst.data[i*src.cols+j]=(uchar)(cdf[(int)src.data[i*src.cols+j]]*(range[a]-range[a-1])+range[a-1]);
		}
	}

	return true;
}
bool pixkit::enhancement::global::GlobalHistogramEqualization1992(const cv::Mat &src,cv::Mat &dst){

	const int nColors	=	256;

	std::vector<double>	hist(nColors,0);	// initialize histogram 

	// 進行統計
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			hist[(int)(src.data[i*src.cols+j])]++;
		}
	}
	for(int graylevel=0;graylevel<nColors;graylevel++){
		hist[graylevel]/=(double)(src.rows*src.cols);
	}

	// 將Histogram改為累積分配函數
	for(int graylevel=1;graylevel<nColors;graylevel++){
		hist[graylevel]+=hist[graylevel-1];
	}

	// 取得新的輸出值
	cv::Mat	tdst(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			double	tempv	=	hist[(int)(src.data[i*src.cols+j])];
			if(tempv>1){
				tempv=1.;
			}
			assert(tempv>=0.&&tempv<=1.);
			tdst.data[i*src.cols+j]=tempv*(nColors-1.);	// 最多延展到255			
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}
bool pixkit::enhancement::global::MaryKim2008(const cv::Mat &src, cv::Mat &dst,int MorD , int r){

	//////////////////////////////////////////////////////////////////////////
	//	exception process
	if (src.type()!=CV_8U){
		return false;
	}
	if(MorD < 1 || MorD > 2){
		return false;
	}
	if(r>64){
		return false;
	}
	////////////////////////////////////////////////////////////////////////////
	const int nColors	=	256;
	float Pmax = 0,Pmin = 1.0, Xm=0 , Xg=0 ,Beta=0;
	std::vector <float> pdf(nColors,0),cdf(nColors,0);

	//統計灰階分佈pdf
	cv::Mat	tdst(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			pdf[(int)src.data[i*src.cols+j]]++;	
		}
	}

	//計算出平均值,並找出積率分佈中的最大最小值
	for(int i=0;i<nColors;i++){
		pdf[i] /= (float)(src.rows*src.rows);
		Xm += i*pdf[i];

		if(pdf[i]>Pmax){
			Pmax = pdf[i];
		}

		if(pdf[i]<Pmin){
			Pmin = pdf[i];
		}
	}
	//影像灰階值的中位數
	Xg = (0+nColors-1)/2.0;
	//計算Beta,用來更新權重
	Beta = Pmax*abs(Xm-Xg)/(nColors-1);

	std::vector <int> segmentation(pow(2,(double)r)+1,0);

	segmentation[0] = -1;
	segmentation[pow(2,(double)r)] = nColors-1;
	/////////////////////////////////////////////////////////////////////////////////////////
	//Histogram Segmentation Module
	/////////////////////////////////////////////////////////////////////////////////////////////
	if(MorD == 1){//用平均值分割

		int Xl,Xu;

		for(int i=0;i<r;i++){
			int location = pow(2,(double)(r-i))/2;
			int interval = location;
			for(int j=0;j<pow(2,(double)i);j++){
				Xl = segmentation[location-interval]+1;
				Xu = segmentation[location+interval];

				float m=0,sum=0;
				for(int k=Xl;k<=Xu;k++)
				{
					m += k*pdf[k];
					sum += pdf[k];
				}

				segmentation[location] =(int)(m/sum+0.5);
				location += 2*interval;
			}
		}
	}else if(MorD == 2){//用中位數分割

		for(int i=0;i<nColors;i++){
			cdf[i] = pdf[i];
		}

		for(int i=1;i<nColors;i++){
			cdf[i] += cdf[i-1];
		}

		int Xl,Xu;

		for(int i=0;i<r;i++){
			int location = pow(2,(double)(r-i))/2;
			int interval = location;
			for(int j=0;j<pow(2,(double)i);j++){
				Xl = segmentation[location-interval]+1;
				Xu = segmentation[location+interval]; 

				float m = (cdf[Xl]+cdf[Xu])/2 , m_min = 1.0;

				for(int k=Xl;k<=Xu;k++){
					if(abs(cdf[k]-m) < m_min){
						m_min = abs(cdf[k]-m);
						segmentation[location] = k;
					}
				}
				location += 2*interval;
			}
		}
	}
	/////////////////////////////////////////////////////////////
	//權重更新模組
	/////////////////////////////////////////////////////////////
	for(int i=0;i<pow(2,(double)r);i++){
		int Xl = segmentation[i]+1, Xu = segmentation[i+1];

		float alpha = 0;

		for(int j=Xl;j<=Xu;j++){
			alpha += pdf[j];
		}

		for(int j=Xl;j<=Xu;j++){
			pdf[j] =  Pmax*pow((pdf[j]-Pmin)/(Pmax-Pmin),alpha)+Beta;
		}
	}
	//將權重做正規劃
	float sum = 0;
	for(int i=0;i<nColors;i++){
		sum += pdf[i];
	}

	for(int i=0;i<nColors;i++){
		pdf[i] /= sum;
	}
	/////////////////////////////////////////////////////////////////////
	//將每一區段作值方圖等化
	////////////////////////////////////////////////////////////////////////
	for(int i=0;i<pow(2,(double)r);i++){
		int Xl = segmentation[i]+1, Xu = segmentation[i+1];
		float t_weight=0,D_range=Xu-Xl;
		//將每個區段的CDF正規劃到1
		for(int j=Xl;j<=Xu;j++){
			t_weight += pdf[j];
		}

		for(int j=Xl;j<=Xu;j++){
			cdf[j] = pdf[j]/t_weight;
		}

		for(int j=Xl+1;j<=Xu;j++){
			cdf[j] += cdf[j-1];
		}

		for(int j=Xl;j<=Xu;j++){
			cdf[j] = cdf[j]*D_range + Xl;
		}
	}

	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			tdst.data[i*src.cols+j] = cdf[src.data[i*src.cols+j]]+0.5;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}
