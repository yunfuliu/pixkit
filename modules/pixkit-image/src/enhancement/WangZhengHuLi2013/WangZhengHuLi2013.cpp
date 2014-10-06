
#include "../../../include/pixkit-image.hpp"
bool Mirror_Reflect(const cv::Mat ori,cv::Mat &ret,int Pixel){

	int height=ori.rows;
	int width=ori.cols;
	int new_height=height+Pixel*2;
	int new_width=width+Pixel*2;
	ret.create(new_height,new_width,ori.type());

	//set the ori-iimage to the center area
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			ret.at<uchar>(i+Pixel,j+Pixel)=ori.at<uchar>(i,j);
		}
	}

	//left-right reflect
	for(int i=0;i<height;i++){
		for(int j=0;j<Pixel;j++){
		
			//left reflect
			ret.at<uchar>((i+Pixel),j)=
				ret.at<uchar>(i+Pixel,Pixel*2-j-1);
			//right reflect
			ret.at<uchar>(i+Pixel,+width+Pixel+j)=
				ret.at<uchar>(i+Pixel,width+Pixel-1-j);
		}
	}

	//treat the result of left-right reflect as the input of up-down reflect
	for(int j=0;j<width+Pixel*2;j++){
		for(int i=0;i<Pixel;i++){
			
			ret.at<uchar>(i,j)=ret.at<uchar>((Pixel*2-i-1),j);
			ret.at<uchar>((height+Pixel+i),j)=ret.at<uchar>((height+Pixel-1-i),j);
		}
	}
	return 1;
}	
bool pixkit::enhancement::local::WangZhengHuLi2013(const cv::Mat &ori,cv::Mat &ret){
	if(ori.type()!=CV_8UC3){
		printf("The type of input image should be CV_8UC3.\n");
		CV_Assert(false);
	}
	std::vector <cv::Mat> HSV(3);
	cv::Mat_<cv::Vec3b> Result=cv::Mat::zeros(ori.rows,ori.cols,ori.type());
	cv::Mat_<cv::Vec3f> reflectance_f=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC1);	//equation (13)
	cv::Mat_<float> L1g=cv::Mat::zeros(ori.rows,ori.cols,CV_32FC1);					//equation (14)
	cv::Mat_<uchar> Lm=cv::Mat::zeros(ori.rows,ori.cols,CV_8UC1);					//equation (22)
	cv::Mat_<float> clv=cv::Mat::zeros(1,256,CV_32FC1);								//equation (17)
	cv::Mat_<float> cfz=cv::Mat::zeros(1,256,CV_32FC1);								//equation (18)
	cv::Mat_<double> Qhat_matrix=cv::Mat::zeros(256,256,CV_64FC1);					//equation (6)
	cv::Mat_<uchar> illumination=cv::Mat::zeros(ori.rows,ori.cols,CV_8UC1);			//equation (12)
	int patch=7;	//ideal value 
	int patch_x=patch;
	int patch_y=patch;
	cv::split(ori,HSV);
	double Qhat=0;

	int Gmax=0;
	int Gmin=255;
	int win=0;
	for(int index=0;index<ori.rows*ori.cols;index++){
		if(HSV[2].data[index]>Gmax)Gmax=HSV[2].data[index];
		if(HSV[2].data[index]<Gmin)Gmin=HSV[2].data[index];
	}
	win=(Gmax-Gmin)/32;

	//------------------------reflectance------------------------
	unsigned char type=0;
	Mirror_Reflect(HSV[2],HSV[2],patch_x);
	//------------------------reflectance------------------------
	int new_height=ori.rows+patch_y*2;
	int new_width=ori.cols+patch_x*2;


	//calculate equation (6)
	for(int n=1;n<ori.rows-1;n++){
		for(int m=1;m<ori.cols-1;m++){
			Qhat_matrix.at<double>(HSV[2].at<uchar>(n,m),HSV[2].at<uchar>(n+1,m))++;
			Qhat_matrix.at<double>(HSV[2].at<uchar>(n,m),HSV[2].at<uchar>(n-1,m))++;
			Qhat_matrix.at<double>(HSV[2].at<uchar>(n,m),HSV[2].at<uchar>(n,m+1))++;
			Qhat_matrix.at<double>(HSV[2].at<uchar>(n,m),HSV[2].at<uchar>(n,m-1))++;
			Qhat_matrix.at<double>(HSV[2].at<uchar>(n,m),HSV[2].at<uchar>(n,m))++;
		}
	}

	//calculate equation (12)
	for(int a=patch_y;a<new_height-patch_y;a++){
		for(int b=patch_x;b<new_width-patch_x;b++){
			int CenterX=b;
			int CenterY=a;
			//calculate bright pass filter
			double Q=0;
			double W=0;
			for(int y=CenterY-patch_y;y<=CenterY+patch_y;y++){
				for(int x=CenterX-patch_x;x<=CenterX+patch_x;x++){
					if(HSV[2].at<uchar>(CenterY,CenterX)>HSV[2].at<uchar>(y,x))continue;
					//G(x,y) HSV[2].at<uchar>(CenterY,CenterX)
					//G(i,j) HSV[2].at<uchar>(y,x)

					//calculate Q
					Qhat=0;
					int l=HSV[2].at<uchar>(y,x);
					int k=HSV[2].at<uchar>(CenterY,CenterX);
					for(int i=l-win;i<=l+win;i++){
						if(i<0||i>255){continue;}
						Qhat	=	Qhat	+	Qhat_matrix.at<double>(k,i);			
					}
					Q=Q+HSV[2].at<uchar>(y,x)*Qhat/(2*win+1);
					W=W+Qhat/(2*win+1);
				}
			}

			Q=ceil(Q/W);

			//calculate reflectance and illumination, b-patch_x and a-patch_y are due to the margin problem
			if(Q==(double)0){
				reflectance_f.at<cv::Vec3f>(a-patch_y,b-patch_x)[0]=1;
				reflectance_f.at<cv::Vec3f>(a-patch_y,b-patch_x)[1]=1;
				reflectance_f.at<cv::Vec3f>(a-patch_y,b-patch_x)[2]=1;
			}
			reflectance_f.at<cv::Vec3f>(a-patch_y,b-patch_x)[0]=(float)(ori.at<cv::Vec3b>(a-patch_y,b-patch_x)[0]/Q);
			reflectance_f.at<cv::Vec3f>(a-patch_y,b-patch_x)[1]=(float)(ori.at<cv::Vec3b>(a-patch_y,b-patch_x)[1]/Q);
			reflectance_f.at<cv::Vec3f>(a-patch_y,b-patch_x)[2]=(float)(ori.at<cv::Vec3b>(a-patch_y,b-patch_x)[2]/Q);
			illumination.at<uchar>(a-patch_y,b-patch_x)=Q;
		}
	}

	//equation (14)
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			L1g.at<float>(i,j)=log((float)(illumination.at<uchar>(i,j)+1));

		}
	}


	//equation (17)
	double L1gTotal = 0;
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			L1gTotal = L1gTotal + L1g.at<float>(i,j);
		}
	}
	for(int z=0;z<256;z++){
		float v=z;
		double L1gU = 0;
		for(int i=0;i<ori.rows;i++){
			for(int j=0;j<ori.cols;j++){
				if(v>=illumination.at<uchar>(i,j)){
					L1gU=L1gU+L1g.at<float>(i,j);
				}
			}
		}
		clv.at<float>(0,z)=L1gU/L1gTotal;
	}

	double siTotal=0;//equation (18)
	for(int i=0;i<256;i++){
		siTotal=siTotal+log((float)(i+1));
	}
	for(int n=0;n<256;n++){
		double temp=0;
		for(int m=0;m<=n;m++){
			temp=temp+log((float)(m+1));
		}
		cfz.at<float>(0,n)=temp/siTotal;
	}

	//equation (20) (21) (22)
	for(int i=0;i<ori.rows;i++){
		for(int j=0;j<ori.cols;j++){
			float minmin=255;
			for(int z=0;z<256;z++){
				if(abs(cfz.at<float>(0,z)-clv.at<float>(0,illumination.at<uchar>(i,j)))<minmin){
					minmin=abs(cfz.at<float>(0,z)-clv.at<float>(0,illumination.at<uchar>(i,j)));
					Lm.at<uchar>(i,j)=z;
				}
			}
		}
	}

	//equation (23)
	for(int channel=0;channel<ori.channels();channel++){
		for(int i=0;i<ori.rows;i++){
			for(int j=0;j<ori.cols;j++){
				Result.at<cv::Vec3b>(i,j)[channel]=cv::saturate_cast<uchar>(reflectance_f.at<cv::Vec3f>(i,j)[channel]*Lm.at<uchar>(i,j));
			}
		}
	}
	ret=Result.clone();
	return 1;
}