#include "pixkit-image.hpp"


//////////////////////////////////////////////////////////////////////////
bool	pixkit::attack::addGaussianNoise(const cv::Mat &src,cv::Mat &dst,const double sd){
	
	//////////////////////////////////////////////////////////////////////////
	if(src.empty()){
		return false;
	}
	if(src.type()!=CV_8U){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const	double	PI			=	3.1415926;
	const	int		MAXVALUE	=	255;

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst;
	tdst	=	src.clone();

	//////////////////////////////////////////////////////////////////////////
	///// get cdf [output] dis
	double	cdf[257]={0};
	double	fm=0.;	// 分母
	for(int i=-128;i<=128;i++){
		if(sd==0.){
			if(i!=0){	// 使得以下計算error最小值為i==0時
				cdf[i+128]	=	0.;
			}else{
				cdf[i+128]	=	1.;
			}
		}else{
			cdf[i+128]=1./sqrt((double)2.*PI*sd*sd)*exp((double)-0.5*i*i/sd/sd);	// get pdf
		}
		fm+=cdf[i+128];	// get fm
	}
	for(int i=0;i<257;i++){
		cdf[i]/=fm;	// normalize
	}
	for(int i=1;i<257;i++){
		cdf[i]+=cdf[i-1];	// cdf
	}

	//////////////////////////////////////////////////////////////////////////
	// add noise
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			///// obtain noise, 垂直為white noise, 以找到對應的水平雜訊強度, 故以尋找最小值方式進行, [output] noise
			double	rand_value=(double)rand()/RAND_MAX;	// white noise from 0 to 1
			double	minv=9999999.;
			double	noise_position=0;
			for(int k=0;k<257;k++){
				if(cdf[k]!=0){
					double	temp=fabs(rand_value-cdf[k]);
					if(temp<minv){
						minv=temp;
						noise_position=k;
					}
				}
			}
			double	noise_mag=noise_position-128.;

			///// add noise
			tdst.data[i*tdst.cols+j]+=noise_mag;
			if(tdst.data[i*tdst.cols+j]>MAXVALUE){
				tdst.data[i*tdst.cols+j]=MAXVALUE;
			}
			if(tdst.data[i*tdst.cols+j]<0){
				tdst.data[i*tdst.cols+j]=0;
			}
		}
	}

	dst	=	tdst.clone();
	return true;
}
bool	pixkit::attack::addWhiteNoise(const cv::Mat &src,cv::Mat &dst,const double maxMag){


	//////////////////////////////////////////////////////////////////////////
	if(src.empty()){
		return false;
	}
	if(src.type()!=CV_8U){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const	int		MAXVALUE	=	255;

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst;
	tdst	=	src.clone();

	//////////////////////////////////////////////////////////////////////////
	///// add noise
	for(int i=0;i<tdst.rows;i++){
		for(int j=0;j<tdst.cols;j++){

			// get noise
			double	noise_mag=(double)rand()/RAND_MAX;	// white noise from 0 to 1
			noise_mag	*=maxMag*2.-maxMag;	// 上下兩倍

			// add noise
			double	temp_output	=	(double)tdst.data[i*tdst.cols+j]	+	noise_mag;
			if(temp_output>=MAXVALUE){
				tdst.data[i*tdst.cols+j]=MAXVALUE;
			}else if(temp_output<=0){
				tdst.data[i*tdst.cols+j]=0;
			}else{
				tdst.data[i*tdst.cols+j]=(int)(temp_output+0.5);
			}
		}
	}

	dst	=	tdst.clone();
	return true;
}
//////////////////////////////////////////////////////////////////////////
bool pixkit::filtering::medianfilter(const cv::Mat &src,cv::Mat &dst,cv::Size blocksize){

	//////////////////////////////////////////////////////////////////////////
	if(blocksize.width>src.cols||blocksize.height>src.rows){
		return false;	}
	if(blocksize.width%2==0||blocksize.height%2==0){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	const	int	half_block_height	=	blocksize.height/2;
	const	int	half_block_width	=	blocksize.width/2;
	std::vector<uchar>	temp_img(blocksize.height*blocksize.width,0);
	// process
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			for(int m=-half_block_height;m<=half_block_height;m++){
				for(int n=-half_block_width;n<=half_block_width;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						temp_img[(m+half_block_height)*blocksize.width+(n+half_block_width)]=src.data[(i+m)*src.cols+(j+n)];
					}else{
						temp_img[(m+half_block_height)*blocksize.width+(n+half_block_width)]=0;
					}
					
				}
			}
			// ordering
			for(int m=0;m<blocksize.height*blocksize.width;m++){
				for(int n=0;n<blocksize.height*blocksize.width-1;n++){
					if(temp_img[n]>temp_img[n+1]){
						double temp=temp_img[n+1];
						temp_img[n+1]=temp_img[n];
						temp_img[n]=temp;
					}	//將value陣列內數值由大至小排列
				}
			}
			tdst.data[i*tdst.cols+j]	=	temp_img[(blocksize.height*blocksize.width-1)/2];
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}

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
			float	Sh=0.,Sv=0.;	// 水平值and垂直值
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

//////////////////////////////////////////////////////////////////////////
bool pixkit::halftoning::dotdiffusion::GuoLiu2009(const cv::Mat &src, cv::Mat &dst,const int ClassMatrixSize){
	
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(ClassMatrixSize!=8&&ClassMatrixSize!=16){
		CV_Error(CV_StsBadArg,"[halftoning::dotdiffusion::GuoLiu2009] accepts only 8 and 16 these two class matrix sizes");
	}
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::dotdiffusion::GuoLiu2009] accepts only grayscale image");
	}

	//////////////////////////////////////////////////////////////////////////
	// initialization
	const int	DiffusionMaskSize=3;
	const int	nColors	=	256;
	// define class matrix and diffused weighting
	const	int		ClassMatrix8[8][8]={{22,5,57,8,45,30,36,19},{40,58,32,18,1,43,29,38},{34,4,62,42,20,16,48,37},{28,7,21,56,15,3,49,11},{6,23,35,17,55,51,50,44},{47,12,39,26,25,27,63,61},{14,46,41,31,2,33,60,13},{9,24,52,0,53,54,59,10}};
	const	int		ClassMatrix16[16][16]={	{204,0,5,33,51,59,23,118,54,69,40,160,169,110,168,188},{3,6,22,36,60,50,74,115,140,82,147,164,171,142,220,214},{14,7,42,16,63,52,94,56,133,152,158,177,179,208,222,1},{15,26,43,75,79,84,148,81,139,136,166,102,217,219,226,4},{17,39,72,92,103,108,150,135,157,193,190,100,223,225,227,13},{28,111,99,87,116,131,155,112,183,196,181,224,232,228,12,21},{47,120,91,105,125,132,172,180,184,205,175,233,245,8,20,41},{76,65,129,137,165,145,178,194,206,170,229,244,246,19,24,49},{80,73,106,138,176,182,174,197,218,235,242,249,247,18,48,68},{101,107,134,153,185,163,202,173,231,241,248,253,44,88,70,45},{123,141,149,61,195,200,221,234,240,243,254,38,46,77,104,109},{85,96,156,130,203,215,230,250,251,252,255,53,62,93,86,117},{151,167,189,207,201,216,236,239,25,31,34,113,83,95,124,114},{144,146,191,209,213,237,238,29,32,55,64,97,126,78,128,159},{187,192,198,212,9,10,30,35,58,67,90,71,122,127,154,161},{199,210,211,2,11,27,37,57,66,89,98,121,119,143,162,186}};
	const	double	coe8[3][3]={{0.47972,1,0.47972},{1,0,1},{0.47972,1,0.47972}},
					coe16[3][3]={{0.38459,1,0.38459},{1,0,1},{0.38459,1,0.38459}};

	// = = = = = get class matrix and diffused weighting = = = = = //
	std::vector<std::vector<int>>		CM(ClassMatrixSize,std::vector<int>(ClassMatrixSize,0));
	std::vector<std::vector<double>>	DW(DiffusionMaskSize,std::vector<double>(DiffusionMaskSize,0));
	if(ClassMatrixSize==8){
		for(int i=0;i<ClassMatrixSize;i++){
			for(int j=0;j<ClassMatrixSize;j++){
				CM[i][j]=ClassMatrix8[i][j];
			}
		}
		for(int i=0;i<DiffusionMaskSize;i++){
			for(int j=0;j<DiffusionMaskSize;j++){
				DW[i][j]=coe8[i][j];
			}
		}
	}else if(ClassMatrixSize==16){
		for(int i=0;i<ClassMatrixSize;i++){
			for(int j=0;j<ClassMatrixSize;j++){
				CM[i][j]=ClassMatrix16[i][j];
			}
		}
		for(int i=0;i<DiffusionMaskSize;i++){
			for(int j=0;j<DiffusionMaskSize;j++){
				DW[i][j]=coe16[i][j];
			}
		}
	}

	// = = = = = processing = = = = = //
	std::vector<std::vector<int>>	ProcOrder(src.rows,std::vector<int>(src.cols,0));
	std::vector<std::vector<double>>	tdst(src.rows,std::vector<double>(src.cols,0));
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			tdst[i][j]	=	src.data[i*src.cols+j];
		}
	}
	// 取得處理順序
	for(int i=0;i<src.rows;i+=ClassMatrixSize){
		for(int j=0;j<src.cols;j+=ClassMatrixSize){
			for(int m=0;m<ClassMatrixSize;m++){
				for(int n=0;n<ClassMatrixSize;n++){
					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
						ProcOrder[i+m][j+n]=CM[m][n];
					}
				}
			}
		}
	}
	// 進行dot_diffusion
	int		OSCW=DiffusionMaskSize/2;
	int		OSCL=DiffusionMaskSize/2;
	int		number=0;
	while(number!=ClassMatrixSize*ClassMatrixSize){
		for(int i=0;i<src.rows;i++){
			for(int j=0;j<src.cols;j++){
				if(ProcOrder[i][j]==number){
					// 取得error
					double	error;
					if(tdst[i][j]<(float)(nColors-1.)/2.){
						error=tdst[i][j];
						tdst[i][j]=0.;
					}else{
						error=tdst[i][j]-(nColors-1.);
						tdst[i][j]=(nColors-1.);
					}
					// 取得分母
					double	fm=0.;
					for(int m=-OSCW;m<=OSCW;m++){
						for(int n=-OSCL;n<=OSCL;n++){
							if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// 在影像範圍內
								if(ProcOrder[i+m][j+n]>number){		// 可以擴散的區域
									fm+=DW[m+OSCW][n+OSCL];
								}
							}
						}
					}
					// 進行擴散
					for(int m=-OSCW;m<=OSCW;m++){
						for(int n=-OSCL;n<=OSCL;n++){
							if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// 在影像範圍內
								if(ProcOrder[i+m][j+n]>number){		// 可以擴散的區域								
									tdst[i+m][j+n]+=error*DW[m+OSCW][n+OSCL]/fm;
								}
							}
						}
					}
				}
			}
		}
		number++;
	}


	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			dst.data[i*dst.cols+j]	=	(uchar)tdst[i][j]+0.5;
			assert(dst.data[i*dst.cols+j]==0||dst.data[i*dst.cols+j]==(nColors-1));
		}
	}

	return true;
}

//////////////////////////////////////////////////////////////////////////
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
		if ((i-1)>=0 && (i+1)<256 || i==0 || i==255){
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
			if (hist[i]!=0 && hist[i+1]==0 || i==255){
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

	std::vector<double>	Histogram(nColors,0);	// 256個灰階值

	// 進行統計
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			Histogram[(int)(src.data[i*src.cols+j])]++;
		}
	}
	for(int graylevel=0;graylevel<nColors;graylevel++){
		Histogram[graylevel]/=(double)(src.rows*src.cols);
	}

	// 將Histogram改為累積分配函數
	for(int graylevel=1;graylevel<nColors;graylevel++){
		Histogram[graylevel]+=Histogram[graylevel-1];
	}

	// 取得新的輸出值
	cv::Mat	tdst(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
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

//////////////////////////////////////////////////////////////////////////
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
bool pixkit::enhancement::local::JiaoSunSha2010(const cv::Mat &src,cv::Mat &dst,const cv::Size BlockSize,const cv::Size StepSize,const short mode){



	std::cout	<<	"this function is not completed yet!"	<<	std::endl;
	return false;
	// TEN還沒有實現, 目前寄送mail過去詢問該怎麼處理看似bug的問題. Sept. 3, 2013. 



	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		return false;
	}
	if(BlockSize.width>src.cols||BlockSize.height>src.rows){
		return false;
	}
	if(BlockSize.width%2==1||BlockSize.height%2==1){	// should be in even size
		return false;
	}
	if(StepSize.width%2==1||StepSize.height%2==1){	// should be in even size
		return false;
	}
	if(StepSize.height>BlockSize.height||StepSize.width>StepSize.width){
		return false;
	}
	if(mode!=1&&mode!=2){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	const int	nColors	=	256;	// how many colors in the input image
	dst.create(src.size(),src.type());

	//////////////////////////////////////////////////////////////////////////
	std::vector<std::vector<short int>>		SM(src.rows,std::vector<short int>(src.cols,0));
	std::vector<std::vector<float>>			temp_dst(src.rows,std::vector<float>(src.cols,0));
	// get temp result
	for(int i=0;i<src.rows;i+=StepSize.height){
		for(int j=0;j<src.cols;j+=StepSize.width){

			//////////////////////////////////////////////////////////////////////////
			// various parameters
			short int	opt_p	=	-1,
						opt_q	=	-1;
			float		opt_cost=	0.;
			std::vector<std::vector<float>>	temp_dstblock(BlockSize.height,std::vector<float>(BlockSize.width,0));
			for(short int p=1;p<=100;p++){
				for(float q=1;q<=3;q+=0.1){

					//////////////////////////////////////////////////////////////////////////
					// initialization
					for(int m=0;m<BlockSize.height;m++){
						for(int n=0;n<BlockSize.width;n++){
							temp_dstblock[m][n]	=	0.;
						}
					}

					//////////////////////////////////////////////////////////////////////////
					// process for each block
					for(int m=0;m<BlockSize.height;m++){
						for(int n=0;n<BlockSize.width;n++){

							if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){

								double	norm_src	=	(double)src.data[(i+m)*src.cols+(j+n)]/(nColors-1);
								double	tempv		=	pow((double)	log((double)(1.+norm_src))	/	log((double)(1.+p))	,1./q);
								assert(tempv>=0.&&tempv<=1.);

								// result
								temp_dstblock[m][n]	+=	tempv*255.;
							}
						}
					}

					//////////////////////////////////////////////////////////////////////////
					// get opt p and q with either EME or TEN simulations
					float	current_cost;
					if(mode==1){	// use EME

						// calculate
						float	local_maxv	=	temp_dstblock[0][0],
								local_minv	=	temp_dstblock[0][0];
						for(int m=0;m<BlockSize.height;m++){
							for(int n=0;n<BlockSize.width;n++){
								if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// this's very important to make sure that do not access out of the scope 
									if(temp_dstblock[m][n]>local_maxv){
										local_maxv	=	temp_dstblock[m][n];
									}
									if(temp_dstblock[m][n]<local_minv){
										local_minv	=	temp_dstblock[m][n];
									}
								}
							}
						}
						// calc EME (Eq. 4)
						current_cost	=	local_maxv/local_minv;

					}else{	// use TEN	***********************************************************************************************************************

					}
					if(current_cost>opt_cost){	// the higher the better
						opt_cost	=	current_cost;
						opt_p		=	p;
						opt_q		=	q;
					}

				}
			}

			//////////////////////////////////////////////////////////////////////////
			// get final block each
			for(int m=0;m<BlockSize.height;m++){
				for(int n=0;n<BlockSize.width;n++){

					if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){

						double	norm_src	=	(double)src.data[(i+m)*src.cols+(j+n)]/(nColors-1);
						double	tempv		=	pow((double)	log((double)(1.+norm_src))	/	log((double)(1.+opt_p))	,1./opt_q);
						assert(tempv>=0.&&tempv<=1.);

						// result
						temp_dst[i+m][j+n]	+=	tempv*255.;
						SM[i+m][j+n]		+=	1;
					}
				}
			}

		}
	}

	//////////////////////////////////////////////////////////////////////////
	// get final result
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			dst.data[i*dst.cols+j]	=	(int)((double)temp_dst[i][j]/SM[i][j]	+0.5);
			assert((float)temp_dst[i][j]/SM[i][j]>=0&&(float)temp_dst[i][j]/SM[i][j]<nColors);
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
//////////////////////////////////////////////////////////////////////////
float pixkit::qualityassessment::EME(const cv::Mat &src,const cv::Size nBlocks,const short mode){

	//////////////////////////////////////////////////////////////////////////
	// exceptions
	if(src.type()!=CV_8U){
		return false;
	}
	if(nBlocks.width>src.cols||nBlocks.height>src.rows){
		return false;
	}
	if(mode!=1&&mode!=2){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	// param
	const	float	c	=	0.0001;

	//////////////////////////////////////////////////////////////////////////
	// define the stepsize
	float	tempv1	=	(float)src.cols/nBlocks.width,
			tempv2	=	(float)src.rows/nBlocks.height;
	if((int)tempv1!=tempv1){
		tempv1	=	(int)	tempv1+1.;
	}
	if((int)tempv2!=tempv2){
		tempv2	=	(int)	tempv2+1.;
	}
	cv::Size	stepsize((int)tempv1,(int)tempv2);

	//////////////////////////////////////////////////////////////////////////
	// estimate
	int		count	=	0;
	float	eme		=	0.;
	for(int i=0;i<src.rows;i+=stepsize.height){
		for(int j=0;j<src.cols;j+=stepsize.width){

			// get local max and min
			float	local_maxv	=	src.data[i*src.cols+j],
					local_minv	=	src.data[i*src.cols+j];		
			if(mode==1){	// standard mode

				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){

						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							if(src.data[(i+m)*src.cols+(j+n)]>local_maxv){
								local_maxv	=	src.data[(i+m)*src.cols+(j+n)];
							}
							if(src.data[(i+m)*src.cols+(j+n)]<local_minv){
								local_minv	=	src.data[(i+m)*src.cols+(j+n)];
							}
						}
					}
				}

			}else if(mode==2){	// BTC's mode

				// find first moment and second moment
				double	moment1=0.,moment2=0.;
				int		count_mom=0;
				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){
						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							moment1+=src.data[(i+m)*src.cols+(j+n)];
							moment2+=src.data[(i+m)*src.cols+(j+n)]*src.data[(i+m)*src.cols+(j+n)];
							count_mom++;
						}
						
					}
				}
				moment1/=(double)count_mom;
				moment2/=(double)count_mom;

				// find variance
				double	sd=sqrt(moment2-moment1*moment1);

				// find num of higher than moment1
				int	q=0;
				for(int m=0;m<stepsize.height;m++){
					for(int n=0;n<stepsize.width;n++){
						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){
							if(src.data[(i+m)*src.cols+(j+n)]>=moment1){
								q++;
							}
						}
					}
				}
				int		m_q=count_mom-q;
				local_minv=moment1-sd*sqrt((double)q/m_q),
				local_maxv=moment1+sd*sqrt((double)m_q/q);
				if(local_minv>255){
					local_minv=255;
				}
				if(local_minv<0){
					local_minv=0;
				}
				if(local_maxv>255){
					local_maxv=255;
				}
				if(local_maxv<0){
					local_maxv=0;
				}
			}else{
				assert(false);
			}

			// calc EME (Eq. 2) -totally same
			if(local_maxv!=local_minv){
				eme	+=	log((double)local_maxv/(local_minv+c));
			}
			count++;

		}
	}

	return (float)20.*eme/count;
}
float pixkit::qualityassessment::TEN(const cv::Mat &src){

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	est;	
	// process
	edgedetection::Sobel(src,est);

	//////////////////////////////////////////////////////////////////////////
	// estimation
	double	ten	=	0.;
	for(int i=0;i<est.rows;i++){
		for(int j=0;j<est.cols;j++){
			ten	+=	est.data[i*est.cols+j]	*	est.data[i*est.cols+j];	// eq. 6
		}
	}

	return (double)ten/(est.rows*est.cols);
}
float pixkit::qualityassessment::AMBE(const cv::Mat &src1,const cv::Mat &src2){

	//////////////////////////////////////////////////////////////////////////
	if((src1.rows!=src2.rows)||(src2.cols!=src2.cols)){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	double	mean1=0.,mean2=0.;
	for(int i=0;i<src1.rows;i++){
		for(int j=0;j<src1.cols;j++){
			mean1	+=	(double)src1.data[i*src1.cols+j];
			mean2	+=	(double)src2.data[i*src1.cols+j];
		}
	}
	mean1	/=	(double)(src1.cols*src1.rows);
	mean2	/=	(double)(src2.cols*src2.rows);

	return abs((double)(mean1-mean2));
}
float	pixkit::qualityassessment::PSNR(const cv::Mat &src1,const cv::Mat &src2){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src1.empty()||src2.empty()){
		CV_Error(CV_HeaderIsNull,"[qualityassessment::PSNR] image is empty");
	}
	if(src1.type()!=src2.type()){
		CV_Error(CV_StsBadArg,"[qualityassessment::PSNR] both types of image do not match");
	}
	if(src1.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[qualityassessment::PSNR] image should be grayscale");
	}

	//////////////////////////////////////////////////////////////////////////
	///// derive psnr
	double	total_err=0.;
	for(int i=0;i<src1.rows;i++){
		for(int j=0;j<src1.cols;j++){
			total_err+=(src1.data[i*src1.cols+j]-src2.data[i*src1.cols+j])*(src1.data[i*src1.cols+j]-src2.data[i*src1.cols+j]);
		}
	}

	// = = = = = Return PSNR = = = = = //
	return 10*log10((double)(src1.cols)*(src1.rows)*(255.)*(255.)/total_err);
}