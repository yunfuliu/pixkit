#include "../include/pixkit-image.hpp"

using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////
///// Local contrast enhancement
bool pixkit::enhancement::local::LCE_BSESCS2014(const cv::Mat &src,cv::Mat &dst,cv::Size blockSize){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	Mat	tsrc	=	src.clone();
	int	nC		=	3;	// number of channels
	if(src.type()==CV_8UC1){
		nC	=	1;
	}else if(src.type()==CV_8UC3){
		// do nothing
	}else{
		CV_Assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	int	half_block_height	=	blockSize.height/2;
	int	half_block_width	=	blockSize.width/2;
	const	int	nColors		=	256;
	// for hist
	const	int	chaninx		=	0;
	const	int histSize	=	nColors;
	float hranges[] = { 0, nColors };
	const float* ranges[] = { hranges };

	//////////////////////////////////////////////////////////////////////////
	///// get separated channels
	Mat	channels1b[3],dst_channels1b[3];
	if(nC==1){
		channels1b[0]	=	tsrc.clone();
	}else if(nC==3){
		split(tsrc,channels1b);
	}else{
		CV_Assert(false);
	}
	for(int c=0;c<nC;c++){
		dst_channels1b[c].create(channels1b[c].size(),channels1b[c].type());
		dst_channels1b[c].setTo(0);
	}

	//////////////////////////////////////////////////////////////////////////
	///// process
	for(int c=0;c<nC;c++){
		for(int i=0;i<src.rows;i++){
			for(int j=0;j<src.cols;j++){

				uchar	&currv	=	channels1b[c].ptr<uchar>(i)[j];
				uchar	&dstv	=	dst_channels1b[c].ptr<uchar>(i)[j];

				//////////////////////////////////////////////////////////////////////////
				// get block region (tl: top-left; br: bottom-right);
				int		tl_corner_i	=	(i-half_block_height)	<0			?	0			:	(i-half_block_height);
				int		tl_corner_j	=	(j-half_block_width)	<0			?	0			:	(j-half_block_width);
				int		br_corner_i	=	(i+half_block_height)	>=src.rows	?	src.rows-1	:	(i+half_block_height);
				int		br_corner_j	=	(j+half_block_width)	>=src.cols	?	src.cols-1	:	(j+half_block_width);
				int		block_width	=	br_corner_j	-	tl_corner_j	+1;
				int		block_height=	br_corner_i	-	tl_corner_i	+1;
				Rect	roi(tl_corner_j,tl_corner_i,block_width,block_height);
				Mat		block	=	channels1b[c](roi);

				//////////////////////////////////////////////////////////////////////////
				///// get hist
				Mat hist;
				cv::calcHist(&block, 1, &chaninx, Mat(),hist, 1, &histSize, ranges,true,false);

				//////////////////////////////////////////////////////////////////////////
				// get mean
				float	meanv=0.;
				for(int ci=0;ci<nColors;ci++){
					meanv	+=	hist.ptr<float>(ci)[0]	*	(float)ci;
				}
				meanv	=	cvFloor(meanv	/	((float)block_width*block_height));

				//////////////////////////////////////////////////////////////////////////
				///// histogram clipping
				// get T_CR
				float	T_CR;
				if(currv<=meanv){	// get T_L
					float	T_L	=0.;
					for(int ci=0;ci<=meanv;ci++){
						T_L	+=	hist.ptr<float>(ci)[0];
					}
					T_L	=	cvFloor(T_L/(meanv+1))	+1;
					T_CR	=	T_L;
				}else{	// get T_U
					float	T_U	=0.;
					for(int ci=meanv+1;ci<nColors;ci++){
						T_U	+=	hist.ptr<float>(ci)[0];
					}

					T_U	=	cvFloor((float)T_U	/	(nColors-1-meanv))	+1;
					T_CR	=	T_U;
				}
				// clip hist, the idea of clahe
				for(int ci=0;ci<nColors;ci++){
					if(hist.ptr<float>(ci)[0]>=T_CR){
						hist.ptr<float>(ci)[0]	=	T_CR;
					}
				}

				//////////////////////////////////////////////////////////////////////////
				///// get pdf
				// get nt
				float	nt	=	0.;
				if(currv<=meanv){
					for(int ci=0;ci<=meanv;ci++){
						nt	+=	hist.ptr<float>(ci)[0];
					}
				}else{
					for(int ci=meanv+1;ci<nColors;ci++){
						nt	+=	hist.ptr<float>(ci)[0];
					}
				}
				// enhancement with cdf
				if(currv<=meanv){	// f_L
					float	cdfv	=	0;
					for(int ci=0;ci<=currv;ci++){
						cdfv	+=	hist.ptr<float>(ci)[0];
					}
					cdfv/=nt;
					dstv	=	cvFloor(meanv*cdfv);
				}else{	// f_U
					float	cdfv	=	0;
					for(int ci=currv;ci<nColors;ci++){
						cdfv	+=	hist.ptr<float>(ci)[0];
					}
					cdfv	=	(nt	-	cdfv)/nt;
					dstv	=	cvFloor(((float)nColors-meanv-2)*cdfv)	+	meanv	+	1;
				}
				CV_Assert(dstv>=0&&dstv<nColors);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// convert back to bgr
	if(nC==1){
		dst	=	dst_channels1b[0].clone();	// dst_channels1b	to	dst
	}else if(nC==3){
		merge(dst_channels1b,3,dst);		// dst_channels1b	to	dst
	}else{
		CV_Assert(false);
	}
	return true;
}
bool pixkit::enhancement::local::Lal2014(const cv::Mat &src,cv::Mat &dst, cv::Size title, float L,float K1 ,float K2 ){
	///////////////////////////////////////////////////////////////////////////////////////////////////
	if(src.type()!=CV_8UC1){
		return false;
	}

	if(L>1 || L<=0){
		return false;
	}

	if(title.height > (src.rows/4) || title.width > (src.cols/4)){
		return false;
	}

	if(K2>1 || K2<0){
		return false;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	const int nColors = 256;
	int x = src.cols/title.width, y = src.rows/title.height;

	dst = cvCreateMat(src.rows,src.cols,src.type());
	cv::Mat temp = cvCreateMat(src.rows,src.cols,src.type());
	std::vector<std::vector<float>> hist(title.height*title.width,std::vector<float> (nColors,0)); //�x�s�C��title���ಾ�禡
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//�p��Sigmoid�ഫ
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for(int i=0;i<temp.rows;i++){
		for(int j=0;j<temp.cols;j++){
			float t = (double)src.data[i*src.cols+j]/(nColors-1);
			float o = t + K1*t/(1.0-exp(K1*(K2+t)));

			temp.data[i*temp.cols+j] = static_cast<uchar>(o*(nColors-1));
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//�p���C��title���ಾ�禡
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

				int limt = (int) (Count*L + 0.5);  //�p���ݭn����������

				float over_limit = 0;
				for(int k=0;k<256;k++)
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
		//�p�����X
		///////////////////////////////////////////////////////////////////////
		int a1=0,a2=x/2,b1=0,b2=y/2;  //a����x�b���V.b����y�b���V
		for(int i=0;i<src.rows;i++)
		{
			a2 = x/2 , a1 = 0;
			for(int j=0;j<src.cols;j++)
			{
				if(j>a2)
				{
					a1=a2;
					a2+=x;

					if(a2/x == title.width)
						a2 = src.cols-1;
				}
				if(i>b2)
				{
					b1 = b2;
					b2 += y;
					if(b2/y == title.height)
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
bool pixkit::enhancement::local::Kimori2013(cv::Mat &src,cv::Mat &dst,cv::Size B, int N){
	std::vector <std::vector<std::vector<float>>> Ob( (int)N,std::vector<std::vector<float>> (src.rows,std::vector<float> (src.cols,0)) );
	std::vector <std::vector<std::vector<float>>> Cb( (int)N,std::vector<std::vector<float>> (src.rows,std::vector<float> (src.cols,0)) );

	int h = B.height, w = B.width;
	const int nColors = 256;
	dst = cvCreateMat(src.rows,src.cols,src.type());

	//計算各個旋轉影像
	for(int k=0;k<N;k++){
		cv::Mat t_src,t_OP,t_CL,t_ob,t_cb; 
		double degree = -180.0*k/N; // rotate degree

		cv::Mat map_matrix = getRotationMatrix2D(cv::Point2f(src.cols/2, src.rows/2),degree,1.0);
		cv::warpAffine(src,t_src,map_matrix,cv::Size(src.cols, src.rows));

		cv::Mat element = getStructuringElement(cv::MORPH_RECT,B);
		cv::morphologyEx(t_src,t_OP,cv::MORPH_OPEN,element);
		cv::morphologyEx(t_src,t_CL,cv::MORPH_CLOSE,element);

		map_matrix = getRotationMatrix2D(cv::Point2f(src.cols/2, src.rows/2),-degree,1.0);
		cv::warpAffine(t_OP,t_ob,map_matrix,cv::Size(src.cols, src.rows));
		cv::warpAffine(t_CL,t_cb,map_matrix,cv::Size(src.cols, src.rows));

		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++){
				Ob[k][i][j] = t_ob.data[i*t_ob.cols+j];
				Cb[k][i][j] = t_cb.data[i*t_cb.cols+j];
			}
	}

	//RMP計算Top-hat增強
	std::vector <std::vector<float>> WTH(src.rows,std::vector<float> (src.cols,0));
	std::vector <std::vector<float>> BTH(src.rows,std::vector<float> (src.cols,0));

	for(int i=0;i<src.rows;i++)
		for(int j=0;j<src.cols;j++){

			WTH[i][j] = Ob[0][i][j];
			BTH[i][j] = Cb[0][i][j];


			for(int k=1;k<N;k++){

				if(Ob[k][i][j] > WTH[i][j]){
					WTH[i][j] = Ob[k][i][j];
				}

				if(Cb[k][i][j] > BTH[i][j]){
					BTH[i][j] = Cb[k][i][j];
				}
			}


			WTH[i][j] = src.data[i*src.cols+j] - WTH[i][j];
			BTH[i][j] = BTH[i][j] -  src.data[i*src.cols+j];

			if(WTH[i][j] > nColors-1)
				WTH[i][j] = (float)(nColors-1);

			if(BTH[i][j] > nColors-1)
				BTH[i][j] = (float)(nColors-1);

			if(WTH[i][j] < 0)
				WTH[i][j] = 0.0;

			if(BTH[i][j] < 0)
				BTH[i][j] = 0.0;
		}

		//直方圖等化
		std::vector<float> WTH_hist(nColors,0);
		std::vector<float> BTH_hist(nColors,0);

		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++){
				WTH_hist[(int)(WTH[i][j]+0.5)]++;
				BTH_hist[(int)(BTH[i][j]+0.5)]++;
			}
			//計算CDF
			for(int k=1;k<nColors;k++){
				WTH_hist[k] += WTH_hist[k-1];
				BTH_hist[k] += BTH_hist[k-1];
			}
			//正歸化CDF
			for(int k=0;k<nColors;k++){
				WTH_hist[k] /= (src.rows*src.cols);
				BTH_hist[k] /= (src.rows*src.cols);
			}

			for(int i=0;i<src.rows;i++)
				for(int j=0;j<src.cols;j++){
					WTH[i][j] = (WTH_hist[(int)WTH[i][j]]-WTH_hist[0])/(WTH_hist[nColors-1]-WTH_hist[0])*(nColors-1); 
					BTH[i][j] = (BTH_hist[(int)BTH[i][j]]-BTH_hist[0])/(BTH_hist[nColors-1]-BTH_hist[0])*(nColors-1); 
				}

				//--------------------------------------------------------
				cv::Mat t_Ob = cvCreateMat(src.rows,src.cols,src.type()); 
				cv::Mat t_Cb = cvCreateMat(src.rows,src.cols,src.type()); 

				for(int i=0;i<src.rows;i++)
					for(int j=0;j<src.cols;j++){
						float temp = src.data[i*src.cols+j]+WTH[i][j] - BTH[i][j];

						if(temp >= nColors-1)
							temp = nColors - 1;
						if(temp < 0)
							temp = 0;

						dst.data[i*dst.cols+j] = temp;


						t_Ob.data[i*t_Ob.cols+j] = WTH[i][j];
						t_Cb.data[i*t_Cb.cols+j] = BTH[i][j];

					}

					return true;
}
bool pixkit::enhancement::local::LiWangGeng2011(const cv::Mat & ori,cv::Mat &ret){

	if(ori.type()!=CV_8UC3){
		printf("The input type should be CV_8UC3");
		CV_Assert(false);
	}

	float alpha=0;
	unsigned char *Lmax=new unsigned char [3];
	double MA_max [3]={0};
	double MA_min [3]={99};
	memset( Lmax, 0, 3 * sizeof (unsigned char) );   
	ret=cv::Mat::zeros(ori.rows,ori.cols,CV_8UC3);
	cv::Mat RL=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC3);//equation (16) annd (11)
	cv::Mat Li=cv::Mat::zeros(ori.rows,ori.cols,CV_8UC1);//equation (10)
	cv::Mat F=cv::Mat::zeros(ori.rows,ori.cols,CV_8UC1);
	cv::Mat MA=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC3);
	cv::Mat AI=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC3);

	//equation (16), calculate the "max" term in the dominator
	for(int j=0;j<ori.cols*ori.rows*3;j=j+3){
		if(ori.data[j]>Lmax[0])Lmax[0]=ori.data[j];
		if(ori.data[j+1]>Lmax[1])Lmax[1]=ori.data[j+1];
		if(ori.data[j+2]>Lmax[2])Lmax[2]=ori.data[j+2];
	}

	//equation (16), calculate the RL(x,y) matrix and prepare for equation (10)
	for(int j=0;j<ori.cols*ori.rows*3;j=j+3){
		float a=(1-0.5*(ori.data[j]/(float)Lmax[0]))*ori.data[j];
		float b=(1-0.5*(ori.data[j+1]/(float)Lmax[1]))*ori.data[j+1];
		float c=(1-0.5*(ori.data[j+2]/(float)Lmax[2]))*ori.data[j+2];
		float d=0;
		if(a>d)d=a;
		if(b>d)d=b;
		if(c>d)d=c;
		Li.data[j/3]=cv::saturate_cast <unsigned char>(d);
	}

	//equation (9), calculate F
	//You can adjust the bilateralFilter parameter
	cv::bilateralFilter(Li,F,-1,3,11);

	//equation 12
	float FinalMin=FLT_MAX;
	float FinalMax=0;
	double k=1;
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			MA.at<cv::Vec3f>(i,j)[0]=log((0.5*(ori.at<cv::Vec3b>(i,j)[0]/(float)Lmax[0]))*ori.at<cv::Vec3b>(i,j)[0]+k);
			MA.at<cv::Vec3f>(i,j)[1]=log((0.5*(ori.at<cv::Vec3b>(i,j)[1]/(float)Lmax[1]))*ori.at<cv::Vec3b>(i,j)[1]+k);
			MA.at<cv::Vec3f>(i,j)[2]=log((0.5*(ori.at<cv::Vec3b>(i,j)[2]/(float)Lmax[2]))*ori.at<cv::Vec3b>(i,j)[2]+k);
			if(MA.at<cv::Vec3f>(i,j)[0]>MA_max[0])MA_max[0]=MA.at<cv::Vec3f>(i,j)[0];
			if(MA.at<cv::Vec3f>(i,j)[1]>MA_max[1])MA_max[1]=MA.at<cv::Vec3f>(i,j)[1];
			if(MA.at<cv::Vec3f>(i,j)[2]>MA_max[2])MA_max[2]=MA.at<cv::Vec3f>(i,j)[2];
			if(MA.at<cv::Vec3f>(i,j)[0]<MA_min[0])MA_min[0]=MA.at<cv::Vec3f>(i,j)[0];
			if(MA.at<cv::Vec3f>(i,j)[1]<MA_min[1])MA_min[1]=MA.at<cv::Vec3f>(i,j)[1];
			if(MA.at<cv::Vec3f>(i,j)[2]<MA_min[2])MA_min[2]=MA.at<cv::Vec3f>(i,j)[2];
			RL.at<cv::Vec3f>(i,j)[0]=(1-0.5*(ori.at<cv::Vec3b>(i,j)[0]/(float)Lmax[0]))*ori.at<cv::Vec3b>(i,j)[0];
			RL.at<cv::Vec3f>(i,j)[1]=(1-0.5*(ori.at<cv::Vec3b>(i,j)[1]/(float)Lmax[1]))*ori.at<cv::Vec3b>(i,j)[1];
			RL.at<cv::Vec3f>(i,j)[2]=(1-0.5*(ori.at<cv::Vec3b>(i,j)[2]/(float)Lmax[2]))*ori.at<cv::Vec3b>(i,j)[2];
		}
	}
	//normilize equation (12)
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			MA.at<cv::Vec3f>(i,j)[0]=MA.at<cv::Vec3f>(i,j)[0]/(MA_max[0]-MA_min[0])*255;
			MA.at<cv::Vec3f>(i,j)[1]=MA.at<cv::Vec3f>(i,j)[1]/(MA_max[1]-MA_min[1])*255;
			MA.at<cv::Vec3f>(i,j)[2]=MA.at<cv::Vec3f>(i,j)[2]/(MA_max[2]-MA_min[2])*255;
		}
	}

	//normilize equation (13)
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			ret.at<cv::Vec3b>(i,j)[0]=cv::saturate_cast <unsigned char>(RL.at<cv::Vec3f>(i,j)[0]/F.at<uchar>(i,j)*MA.at<cv::Vec3f>(i,j)[0]);
			ret.at<cv::Vec3b>(i,j)[1]=cv::saturate_cast <unsigned char>(RL.at<cv::Vec3f>(i,j)[1]/F.at<uchar>(i,j)*MA.at<cv::Vec3f>(i,j)[1]);
			ret.at<cv::Vec3b>(i,j)[2]=cv::saturate_cast <unsigned char>(RL.at<cv::Vec3f>(i,j)[2]/F.at<uchar>(i,j)*MA.at<cv::Vec3f>(i,j)[2]);
		}
	}
	return 1;
}
bool pixkit::enhancement::local::Sundarami2011(const cv::Mat &src,cv::Mat &dst, cv::Size N, float L, float phi){

	//////////////////////////////////////////////////////////////////////////////
	if(src.type()!=CV_8UC1){
		return false;
	}
	if(L>1 || L<=0){
		return false;
	}
	if(N.height >= src.rows-1 || N.width >= src.cols-1){
		return false;
	}
	if(phi>1 || phi<0){
		return false;
	}
	if(N.height%2==0 || N.width%2==0){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////////
	int limt = (int) (N.height*N.width*L + 0.5);
	int x = N.width/2, y = N.height/2;
	dst = cvCreateMat(src.rows,src.cols,src.type());
	for(int i=0;i<dst.rows;i++){
		for(int j=0;j<dst.cols;j++){
			std::vector<float> hist(256,0);

			float Total = 0;
			for(int m=i-y;m<=i+y;m++){
				for(int n=j-x;n<=j+x;n++){
					if(m>=0 && m<dst.rows && n>=0 && n<dst.cols){
						hist[(int)src.data[m*src.cols+n]]++;
						Total++;
					}
				}
			}

			//////////////////////////////////////////////////////////////////////////
			//�վ�Histogram
			//////////////////////////////////////////////////////////////////////////
			float u = Total/256.0; //�N�����ä���
			for(int k=0;k<256;k++){
				hist[k] = (1.0/(1.0+phi))*hist[k] + (phi/(1+phi))*u;
			}

			float over_limit = 0;
			for(int k=1;k<256;k++){
				if(hist[k] > limt){
					over_limit += (hist[k]-limt);
					hist[k] = limt;
				}
			}

			over_limit /= 256;

			for(int k=0;k<256;k++){
				hist[k] += over_limit;
			}

			for(int k=1;k<256;k++){
				hist[k] += hist[k-1];
			}

			dst.data[i*dst.cols+j] = (unsigned char) ( (hist[(int)src.data[i*src.cols+j]]*255.0/Total + 0.5) ); 
		}
	}
	return true;
}
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
	const	short	nColors	=	256;	// �v�����C���ƶq.

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
	cv::Size	blockSize((int)tempv1,(int)tempv2);
	// step size
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
	if(blockSize.height>src.rows||blockSize.width>src.cols||blockSize.height==1||blockSize.width==1){	// block size should be even (S3P5-Step.2)
		return false;
	}
// 	if(stepsize.height>blockSize.height/2||stepsize.width>blockSize.width/2){	// step size should be smaller than 1/2 of the block size. 
// 		return false;
// 	}

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
			for(int m=0;m<blockSize.height;m++){
				for(int n=0;n<blockSize.width;n++){
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
			for(int m=0;m<blockSize.height;m++){
				for(int n=0;n<blockSize.width;n++){
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
bool pixkit::enhancement::local::FAHE2006(const cv::Mat &src1b,cv::Mat &dst1b,cv::Size blockSize){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(blockSize.height%2==0){
		return false;
	}
	if(blockSize.width%2==0){
		return false;
	}
	if(src1b.type()!=CV_8UC1){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::FAHE2006] allows only grayscale image.");
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst1b(src1b.size(),src1b.type());
	const int nColors	=	256;

	//////////////////////////////////////////////////////////////////////////
	std::vector<int>	hist(nColors,0);	// histogram for 256 grayscales
	// processing
	for(int i=0;i<src1b.rows;i++){

		// assign a new hist for the current row
		hist.assign(nColors,0);
		int		nNeighbors=0;

		// process each col
		for(int j=0;j<src1b.cols;j++){

			// init			
			auto	&currv	=	src1b.ptr<uchar>(i)[j];

			//////////////////////////////////////////////////////////////////////////
			///// get pdf hist
			if(j==0){	// the first column				
				for(int m=-blockSize.height/2;m<=blockSize.height/2;m++){
					for(int n=-blockSize.width/2;n<=blockSize.width/2;n++){
						if(i+m>=0&&i+m<src1b.rows&&j+n>=0&&j+n<src1b.cols){
							auto	&neiv	=	src1b.ptr<uchar>(i+m)[j+n];
							hist[neiv]++;
							nNeighbors++;
						}					
					}
				}
			}else{		// rest columns

				int	idxPreLeftCol	=	j-blockSize.width/2-1;
				int	idxRightCol		=	j+blockSize.width/2;

				// for the previous first col
				if(idxPreLeftCol>=0){	// if the first col exists
					for(int m=-blockSize.height/2;m<=blockSize.height/2;m++){
						if(i+m>=0&&i+m<src1b.rows){
							auto	&neiv	=	src1b.ptr<uchar>(i+m)[idxPreLeftCol];
							hist[neiv]--;
							nNeighbors--;
							CV_DbgAssert(hist[neiv]>=0);
						}					
					}					
				}
				CV_DbgAssert(nNeighbors>=0);

				// for the last column
				if(idxRightCol<src1b.cols){	// if the last col exists
					for(int m=-blockSize.height/2;m<=blockSize.height/2;m++){
						if(i+m>=0&&i+m<src1b.rows){
							auto	&neiv	=	src1b.ptr<uchar>(i+m)[idxRightCol];
							hist[neiv]++;
							nNeighbors++;
						}
					}					
				}
			}

			//////////////////////////////////////////////////////////////////////////
			///// get cdf hist
			int	curr_g_cdf_hist;
			if(currv<nColors/2){
				curr_g_cdf_hist	=	0;
				for(int graylevel=0;graylevel<=currv;graylevel++){
					curr_g_cdf_hist+=hist[graylevel];
				}
			}else{
				curr_g_cdf_hist	=	nNeighbors;
				CV_DbgAssert(nNeighbors>=0&&nNeighbors<=(blockSize.height*blockSize.width));
				for(int graylevel=nColors-1;graylevel>currv;graylevel--){
					curr_g_cdf_hist-=hist[graylevel];
				}
			}

			//////////////////////////////////////////////////////////////////////////
			///// et enhanced pixel value
			double	cdf	=	(double)curr_g_cdf_hist/nNeighbors;	// cdf hist to cdf
			if(cdf>1){
				cdf=1.;
			}
			CV_Assert(cdf>=0.&&cdf<=1.);
			tdst1b.ptr<uchar>(i)[j]	=	cvRound((double)cdf*(nColors-1));

		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst1b	=	tdst1b.clone();

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

			maxmap.data[i*maxmap.cols+j]	=	static_cast<uchar>(maxv);
			minmap.data[i*minmap.cols+j]	=	static_cast<uchar>(minv);
			avgmap.data[i*avgmap.cols+j]	=	static_cast<uchar>(avgv);		

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
			tdst.data[i*tdst.cols+j]	=	static_cast<uchar>(cvRound(tempv));
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}
bool pixkit::enhancement::local::KimKimHwang2001(const cv::Mat &src,cv::Mat &dst,const cv::Size B,const cv::Size S){

	//////////////////////////////////////////////////////////////////////////
	// initialization
	const	short	nColors	=	256;	// �v�����C���ƶq.
	// transformation
	cv::Size	blockSize	=	cv::Size(src.cols/B.width,src.rows/B.height);
	cv::Size	stepsize	=	cv::Size(src.cols/S.width,src.rows/S.height);

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(blockSize.height>src.rows||blockSize.width>src.cols||blockSize.height==1||blockSize.width==1){	// block size should be even (S3P5-Step.2)
		return false;
	}
	if(stepsize.height>blockSize.height/2||stepsize.width>blockSize.width/2){
		return false;
	}
	if (stepsize.height <= 0 || stepsize.width <= 0){
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
			for(int m=0;m<blockSize.height;m++){
				for(int n=0;n<blockSize.width;n++){
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
			for(int m=0;m<blockSize.height;m++){
				for(int n=0;n<blockSize.width;n++){
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
			dst.data[i*dst.cols+j]	=	static_cast<uchar>(tdst[i][j]);
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
			// �p�����X��  ps. �ݥ��W�� -1/2~1/2
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
bool pixkit::enhancement::local::CLAHEnon1987(const cv::Mat &src,cv::Mat &dst, cv::Size nBlock, float L ){
	///////////////////////////////////////////////////////////////////////////////////////////////////
	if(src.type()!=CV_8UC1){
		return false;
	}
	if(L>1 || L<=0){
		return false;
	}

	if(nBlock.height > (src.rows/4) || nBlock.width > (src.cols/4)){
		return false;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	const int nColors = 256;
	int x = src.cols/nBlock.width, y = src.rows/nBlock.height;

	dst = cvCreateMat(src.rows,src.cols,src.type());

	std::vector<std::vector<float>> hist(nBlock.height*nBlock.width,std::vector<float> (nColors,0)); //儲存每個title的轉移函式
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//計算每個title的轉移函式
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for(int m = 0;m<nBlock.height;m++)
		for(int n = 0;n<nBlock.width;n++)
		{
			int i,j,i1=(m+1)*y,j1=(n+1)*x;

			if( (m+1) == nBlock.height )
				i1 = src.rows;
			if((n+1) == nBlock.width )
				j1 = src.cols;

			int Count = 0;
			for(i=m*y;i<i1;i++)
				for(j=n*x;j<j1;j++)
				{
					hist[m*nBlock.width+n][(int)src.data[i*src.cols+j]]++;
					Count++;
				}

				int limt = (int) (Count*L + 0.5);  //計算需要裁切的限制

				float over_limit = 0;
				for(int k=0;k<256;k++)
				{

					if(hist[m*nBlock.width+n][k] > limt)
					{
						over_limit += (hist[m*nBlock.width+n][k]-limt);
						hist[m*nBlock.width+n][k] = limt;
					}
				}

				over_limit /= nColors;

				for(int k=0;k<256;k++)
				{
					hist[m*nBlock.width+n][k] += over_limit;
					hist[m*nBlock.width+n][k] = (hist[m*nBlock.width+n][k]/Count)*(nColors-1);
				}

				for(int k=1;k<256;k++)
					hist[m*nBlock.width+n][k] += hist[m*nBlock.width+n][k-1];
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

				if(a2/x == nBlock.width)
					a2 = src.cols-1;
			}
			if(i>b2)
			{
				b1 = b2;
				b2 += y;
				if(b2/y == nBlock.height)
					b2 = src.rows-1;
			}

			int p1=a1/x,p2=a2/x,q1=b1/y,q2=b2/y;
			if(p2 >= nBlock.width)
				p2 = nBlock.width-1;
			if(q2 >= nBlock.height)
				q2 = nBlock.height-1;

			float a=(float)(a2-j)/(a2-a1), b=(float)(b2-i)/(b2-b1);
			int v = (int)src.data[i*src.cols+j];

			dst.data[i*dst.cols+j] = (unsigned char) (b*(a*hist[q1*nBlock.width+p1][v] + (1-a)*hist[q1*nBlock.width+p2][v]) + (1-b)*(a*hist[q2*nBlock.width+p1][v] + (1-a)*hist[q2*nBlock.width+p2][v]));
		}
	}

	return true;
}
bool pixkit::enhancement::local::CLAHE1987(const cv::Mat &src,cv::Mat &dst, cv::Size blockSize, float L){
	///////////////////////////////////////////////////////////////////////////////////////////////////
	if(src.type()!=CV_8UC1){
		return false;
	}
	if(L>1 || L<=0){
		return false;
	}

	if(blockSize.height > src.rows-1 || blockSize.width > src.cols-1){
		return false;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int limt = (int) (blockSize.height*blockSize.width*L + 0.5);
	int x = blockSize.width/2, y = blockSize.height/2;

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

				float over_limit = 0;
				for(int k=0;k<256;k++)
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
bool pixkit::enhancement::local::AHE1974(const cv::Mat &src1b,cv::Mat &dst1b,const cv::Size blockSize){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(blockSize.height%2==0){
		return false;
	}
	if(blockSize.width%2==0){
		return false;
	}
	if(src1b.type()!=CV_8UC1){
		CV_Error(CV_StsBadArg,"[pixkit::enhancement::local::AHE1974] allows only grayscale image.");
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst1b(src1b.size(),src1b.type());
	const int nColors	=	256;

	//////////////////////////////////////////////////////////////////////////
	std::vector<int>	hist(nColors,0);	// histogram for 256 grayscales
	// processing
	for(int i=0;i<src1b.rows;i++){
		for(int j=0;j<src1b.cols;j++){

			hist.assign(nColors,0);
			auto	&currv	=	src1b.ptr<uchar>(i)[j];

			// get pdf hist
			int nNeighbors=0;
			for(int m=-blockSize.height/2;m<=blockSize.height/2;m++){
				for(int n=-blockSize.width/2;n<=blockSize.width/2;n++){
					if(i+m>=0&&i+m<src1b.rows&&j+n>=0&&j+n<src1b.cols){
						auto	&neiv	=	src1b.ptr<uchar>(i+m)[j+n];
						hist[neiv]++;
						nNeighbors++;
					}					
				}
			}

			// get cdf hist
			for(int graylevel=1;graylevel<=currv;graylevel++){	// calc only to the current value rather than the theoretical maximum for saving complexity. 
				hist[graylevel]+=hist[graylevel-1];
			}

			// get enhanced pixel value
			double	cdf	=	(double)hist[currv]/nNeighbors;	// cdf hist to cdf
			if(cdf>1){
				cdf=1.;
			}
			CV_Assert(cdf>=0.&&cdf<=1.);
			tdst1b.ptr<uchar>(i)[j]	=	cvRound((double)cdf*(nColors-1));

		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst1b	=	tdst1b.clone();

	return true;
}

//////////////////////////////////////////////////////////////////////////
///// Global contrast enhancement
bool pixkit::enhancement::global::RajuNair2014(const cv::Mat &src,cv::Mat &dst){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	Mat	tsrc3b	=	src.clone();
	bool	flag_gray	=	false;
	if(src.type()==CV_8UC1){
		cvtColor(tsrc3b,tsrc3b,CV_GRAY2BGR);
		flag_gray	=	true;
	}else if(src.type()==CV_8UC3){
		// do nothing
	}else{
		CV_Assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	const	float	K		=	128.;
	const	float	E		=	255.;
	const	int		nColors	=	256;
	// for hist
	const	int	chaninx		=	0;
	const	int histSize	=	256;
	float hranges[] = { 0, 256 };
	const float* ranges[] = { hranges };

	//////////////////////////////////////////////////////////////////////////
	///// convert color
	Mat	src3b_hsv;
	cvtColor(tsrc3b,src3b_hsv,CV_BGR2HSV);

	//////////////////////////////////////////////////////////////////////////
	///// get M
	Mat	channel[3];
	split(src3b_hsv,channel);
	// get hist
	Mat hist;
	cv::calcHist(&channel[2], 1, &chaninx, Mat(),hist, 1, &histSize, ranges,true,false);
	// get M
	float	M=0.;
	for(int ci=0;ci<nColors;ci++){
		M	+=	hist.ptr<float>(ci)[0]	*	(float)ci;
	}
	M	=	cvRound(M	/	((float)channel[2].rows*channel[2].cols));

	//////////////////////////////////////////////////////////////////////////
	///// enhancement
	Mat	eV	=	channel[2].clone();	// enhanced V
	eV.setTo(0);
	for(int i=0;i<channel[2].rows;i++){
		for(int j=0;j<channel[2].cols;j++){
			//
			uchar	&X	=	channel[2].ptr<uchar>(i)[j];
			uchar	&dstv	=	eV.ptr<uchar>(i)[j];
			//
			float	f_dst_value;
			if(X<M){
				float	ud1x	=	1.-((M-(float)X)/M);
				f_dst_value	=	K*ud1x;
			}else if(X>=M){
				float	ud2x	=	(E-(float)X)/(E-M);
				f_dst_value	=	E-K*ud2x;
			}else{
				CV_Assert(false);
			}
			CV_DbgAssert(f_dst_value>=0.&&f_dst_value<=255.);
			dstv	=	cvRound(f_dst_value);
		}
	}	

	//////////////////////////////////////////////////////////////////////////
	///// convert back to bgr
	channel[2]	=	eV.clone();				// eV			to	channel[2]
	merge(channel,3,src3b_hsv);				// channel		to	src3b_hsv
	cvtColor(src3b_hsv,tsrc3b,CV_HSV2BGR);	// src3b_hsv	to	tsrc3b
	if(flag_gray){	
		cvtColor(tsrc3b,dst,CV_BGR2GRAY);	// tsrc3b		to	dst1b
	}else{
		dst	=	tsrc3b.clone();			// tsrc3b		to	dst1b	
	}

	return true;
}
bool pixkit::enhancement::global::LeeLeeKim2013(const cv::Mat ori,cv::Mat &ret,double alpha){

	if(ori.type()!=CV_8UC3){
		printf("The type of input image should be CV_8UC3.\n");
		CV_Assert(false);
	}

	std::vector <cv::Mat> YUV(3);
	cv::Mat YUV_Mat;
	cv::Mat_<cv::Vec3b> Result=cv::Mat::zeros(ori.rows,ori.cols,ori.type());
	cv::Mat_<double> temp_h_l_k=cv::Mat::zeros(256,256,CV_64FC1);			//temp for equation(2)
	cv::Mat_<double> h_l_k=cv::Mat::zeros(256,256,CV_64FC1);				//equation(2)
	cv::Mat_<double> m_l=cv::Mat::zeros(256,256,CV_64FC1);					//equation(30)
	cv::Mat_<double> oneT_u_l_k_Inverse=cv::Mat::zeros(256,256,CV_64FC1);	//part of the dominator of equation(21)
	std::vector<cv::Mat>u_l_k(256);											//equation(28)					
	for(int i=0;i<256;i++){													
		u_l_k[i]=cv::Mat::zeros(256,256,CV_64FC1);	
	}
	cv::Mat_<double> phi_max=cv::Mat::zeros(1,256,CV_64FC1);			//phi_max equation(21)	
	cv::Mat_<double> dl=cv::Mat::zeros(256,256,CV_64FC1);				//equation(22)
	cv::Mat_<double> sl=cv::Mat::zeros(1,256,CV_64FC1);					//numerator of equation(23)
	cv::Mat_<double> wl=cv::Mat::zeros(1,256,CV_64FC1);					//equation(23)
	cv::Mat_<double> d=cv::Mat::zeros(1,256,CV_64FC1);					//equation(24)
	cv::Mat_<double> TransFun=cv::Mat::zeros(1,256,CV_64FC1);			//equation(25)
	
	int patch=1;	//ideal value 
	int patch_x=patch;
	int patch_y=patch;
	cv::cvtColor(ori,YUV_Mat,CV_BGR2YUV);
	cv::split(YUV_Mat,YUV);
	//------------------------reflect margin to solve the margin problem------------------------
	unsigned char type=0;
	cv::Mat Yenlarged;
	copyMakeBorder(YUV[0],Yenlarged ,patch_x,patch_x,
		patch_x,patch_x, cv::BORDER_REFLECT_101);
	//------------------------reflect margin to solve the margin problem------------------------
	
	//calculate  h(k,k+layer) before equation(2)
	for(int n=patch;n<ori.rows-patch;n++){
		for(int m=patch;m<ori.cols-patch;m++){
			if(Yenlarged.ptr<uchar>(n+1)[m]>Yenlarged.ptr<uchar>(n)[m])
			temp_h_l_k.ptr<double>(Yenlarged.ptr<uchar>(n)[m])[abs(Yenlarged.ptr<uchar>(n)[m]-Yenlarged.ptr<uchar>(n+1)[m])]++;
			if(Yenlarged.ptr<uchar>(n-1)[m]>Yenlarged.ptr<uchar>(n)[m])
			temp_h_l_k.ptr<double>(Yenlarged.ptr<uchar>(n)[m])[abs(Yenlarged.ptr<uchar>(n)[m]-Yenlarged.ptr<uchar>(n-1)[m])]++;
			if(Yenlarged.ptr<uchar>(n)[m+1]>Yenlarged.ptr<uchar>(n)[m])
			temp_h_l_k.ptr<double>(Yenlarged.ptr<uchar>(n)[m])[abs(Yenlarged.ptr<uchar>(n)[m]-Yenlarged.ptr<uchar>(n)[m+1])]++;
			if(Yenlarged.ptr<uchar>(n)[m-1]>Yenlarged.ptr<uchar>(n)[m])
			temp_h_l_k.ptr<double>(Yenlarged.ptr<uchar>(n)[m])[abs(Yenlarged.ptr<uchar>(n)[m]-Yenlarged.ptr<uchar>(n)[m-1])]++;
		}
	}

	//calculate equation(2), (layer,k)
	for(int layer=1;layer<256;layer++){//n:layer
		for(int k=0;k<=255-layer;k++){//m:k
			h_l_k.ptr<double>(layer)[k]=log10(temp_h_l_k.ptr<double>(k)[layer]+ temp_h_l_k.ptr<double>(k+layer)[layer] +1);
		}
	}

	//use (29) (30) to calculate (32) m_l
	for(int layer=1;layer<256;layer++){//n:layer
		for(int k=1;k<=255;k++){//m:k
			int min=k-1>255-layer?255-layer:k-1;
			int i=k-layer>0?k-layer:0;
			for(;i<=min;i++){
				m_l.ptr<double>(k)[layer]=m_l.ptr<double>(k)[layer]+h_l_k.ptr<double>(layer)[i];
			}
		}
	}

	//use (28) to calculate (31) u_l_k
	for(int layer=1;layer<256;layer++){//n:layer
		for(int k=1;k<=255;k++){//m:k
			int min=k>256-layer?256-layer:k;
			int max=k-layer>0?k-layer:0;
			oneT_u_l_k_Inverse.ptr<double>(layer)[k]=1./(min-max);		//the diagonal elements of layer-th
		}
	}

	// calculate (21) 
	for(int layer=1;layer<=255;layer++){
		double min=0;
		cv::Mat imageROI=m_l(cv::Rect(1,layer,255,1));	//cv::Rect(400,10,50,50) x y length height
		cv::minMaxLoc(imageROI, &min);
		double front=0;
		double back=0;
		for(int k=1;k<=255;k++){
			front=front+oneT_u_l_k_Inverse.ptr<double>(layer)[k]*m_l.ptr<double>(layer)[k];
			back=back+min*oneT_u_l_k_Inverse.ptr<double>(layer)[k];
		}
		double denominator=(front-back);
		if(front-back==0.)continue;
		else phi_max.ptr<double>(0)[layer]=255/(front-back);
		//calculate (22)
		for(int k=1;k<=255;k++){
			dl.ptr<double>(layer)[k]=phi_max.ptr<double>(0)[layer]*oneT_u_l_k_Inverse.ptr<double>(layer)[k]*
				(m_l.ptr<double>(k)[layer]-min);
		}
	}

	//calculate sl (numerator of equation(23))
	for(int layer=1;layer<256;layer++){
		for(int k=1;k<=255;k++){
			sl.ptr<double>(0)[layer]=sl.ptr<double>(0)[layer]+h_l_k.ptr<double>(layer)[k];
		}
	}


	//calculate equation(23)
	double max;
	cv::Mat imageROI=sl(cv::Rect(1,0,255,1));	//cv::Rect(400,10,50,50) x y length height
	cv::minMaxLoc(imageROI,NULL, &max);
	//calculate sl
	for(int layer=1;layer<256;layer++){
		wl.ptr<double>(0)[layer]=pow(sl.ptr<double>(0)[layer]/max,alpha);				//(23)
	}


	//calculate d ,equation (24)
	double temp_24=0;
	for(int layer=1;layer<256;layer++){
		temp_24=temp_24+wl.ptr<double>(0)[layer];	
	}
	for(int layer=1;layer<256;layer++){
		for(int k=1;k<=255;k++){
			d.ptr<double>(0)[layer]=d.ptr<double>(0)[layer]+dl.ptr<double>(k)[layer]*wl.ptr<double>(0)[k];
		}
		d.ptr<double>(0)[layer]=d.ptr<double>(0)[layer]/temp_24;	
			
	}

	//calculate TransFun ,equation (25)
	TransFun.ptr<double>(0)[0]=0;
	for(int k=1;k<=255;k++){
		for(int i=0;i<=k-1;i++){
			TransFun.ptr<double>(0)[k]=TransFun.ptr<double>(0)[k]+d.ptr<double>(0)[i+1];
		}
	}
	cv::normalize(TransFun,TransFun,0,255,32);

	//According to the TransFun, transform the Input Y[0]
	for(int n=0;n<ori.rows;n++){
		for(int m=0;m<ori.cols;m++){
			YUV[0].ptr<uchar>(n)[m]=TransFun.ptr<double>(0)[YUV[0].ptr<uchar>(n)[m]];
		}
	}

	cv::merge(YUV,ret);
	cv::cvtColor(ret,ret,CV_YUV2BGR);
	return 1;

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

	//�έp�Ƕ����Gpdf
	cv::Mat	tdst(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			pdf[(int)src.data[i*src.cols+j]]++;	
		}
	}

	//�p���X������,�ç��X�n�v���G�����̤j�̤p��
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
	//�v���Ƕ��Ȫ�������
	Xg = (0+nColors-1)/2.0;
	//�p��Beta,�Ψӧ��s�v��
	Beta = Pmax*abs(Xm-Xg)/(nColors-1);

	std::vector <int> segmentation(pow(2,(double)r)+1,0);

	segmentation[0] = -1;
	segmentation[pow(2,(double)r)] = nColors-1;
	/////////////////////////////////////////////////////////////////////////////////////////
	//Histogram Segmentation Module
	/////////////////////////////////////////////////////////////////////////////////////////////
	if(MorD == 1){//�Υ����Ȥ���

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
	}else if(MorD == 2){//�Τ����Ƥ���

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
	//�v�����s�Ҳ�
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
	//�N�v�������W��
	float sum = 0;
	for(int i=0;i<nColors;i++){
		sum += pdf[i];
	}

	for(int i=0;i<nColors;i++){
		pdf[i] /= sum;
	}
	/////////////////////////////////////////////////////////////////////
	//�N�C�@�Ϭq�@�Ȥ��ϵ���
	////////////////////////////////////////////////////////////////////////
	for(int i=0;i<pow(2,(double)r);i++){
		int Xl = segmentation[i]+1, Xu = segmentation[i+1];
		float t_weight=0,D_range=Xu-Xl;
		//�N�C�ӰϬq��CDF���W����1
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
	int	minima[256]={0};		//	�x�s�̤p��
	bool PartitionFlag=false;	//	true:�i��histogram����, �çP�_�O�_�ŦX������68.3%����
	bool SubHistFlag=false;		//	true:histogram���ϫ�, low ���� high histogram ��68.3%�P�_
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
		if (count==1){					//	�Ĥ@��minima���i������
			PartitionFlag=false;
		}
		if (minima[0]==minima[1]){		//	�ץ��W���P�_BUG
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
				if (SubHistFlag){		//	(mean+sd) �� high-minima�����������P�w
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
			}else{						//	low-minima �� (mean-sd)�����������P�w.
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

	if(src.type()!=CV_8UC1){
		CV_Assert(CV_StsUnmatchedFormats,"[pixkit::enhancement::global::GlobalHistogramEqualization1992] src should be CV_8UC1.");
	}

	const int nColors	=	256;

	std::vector<double>	hist(nColors,0);	// initialize histogram 

	// �i���έp
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			hist[(int)(src.data[i*src.cols+j])]++;
		}
	}
	for(int graylevel=0;graylevel<nColors;graylevel++){
		hist[graylevel]/=(double)(src.rows*src.cols);
	}

	// �NHistogram�אּ�ֿn���t����
	for(int graylevel=1;graylevel<nColors;graylevel++){
		hist[graylevel]+=hist[graylevel-1];
	}

	// ���o�s�����X��
	cv::Mat	tdst(src.size(),src.type());
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			double	tempv	=	hist[(int)(src.data[i*src.cols+j])];
			if(tempv>1){
				tempv=1.;
			}
			assert(tempv>=0.&&tempv<=1.);
			tdst.data[i*src.cols+j]=static_cast<uchar>(tempv*(nColors-1.));	// �̦h���i��255			
		}
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst.clone();

	return true;
}

