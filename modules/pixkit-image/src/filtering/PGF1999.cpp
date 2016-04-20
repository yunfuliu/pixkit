#include "../../include/pixkit-image.hpp"

void gaussianweight(int &blocksize,double &sigma,double **kernel){

	double  sum=0.0;
	double r=0.0;
	for(int x=-blocksize/2;x<=blocksize/2;x++){
		for(int	y=-blocksize/2;y<=blocksize/2;y++){
			r=sqrt(double(x*x+y*y));
			kernel[x+blocksize/2][y+blocksize/2]=(exp(-((r*r)/(2*sigma)))/(2*3.1415926*sigma));
			sum+=kernel[x+blocksize/2][y+blocksize/2];
		}
	}

	for(int x=0;x<blocksize;x++){
		for(int	y=0;y<blocksize;y++){
			kernel[x][y]/=sum;
			
		}
	}

}

bool pixkit::filtering::PGF1999(const cv::Mat &src,cv::Mat &dst,int &blocksize,double sigma,int alpha){//peer group filter(source image,output image,gauss blocksize,gauss standard variance,first derivative thershold)

	//create dst
	dst=src.clone();
	//////////////////////////////////////////////////////////////////////////
	//judge coefficient 
	if(blocksize%2==0){
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	//grayimage,colorimage
	double ***rgbimagestorage = new double **[src.channels()];
		for (int i = 0 ; i < src.channels(); i++){
			rgbimagestorage [i] = new double *[src.rows];
			for(int j = 0 ; j <	src.rows ; j++){
				rgbimagestorage [i][j] = new double [src.cols];
				for(int k = 0 ; k <	src.cols ; k++){
					rgbimagestorage [i][j][k]=0;
				}
			}
		}
	if(src.channels()==3){
		for (int i = 0 ; i < src.channels(); i++){
			for(int j = 0 ; j <	src.rows ; j++){
				for(int k = 0 ; k <	src.cols ; k++){
					rgbimagestorage [i][j][k]=src.at<cv::Vec3b>(j,k)[i];
				}
			}
		}
	}
	double **imagestorage  = new double *[src.rows];
	for (int i = 0 ; i < src.rows; i++){
		imagestorage [i] = new double [src.cols];
		for(int j = 0 ; j <	src.cols ; j++){
			imagestorage [i][j]=src.at<unsigned char>(i,j);
		}
	}
	
	
	//////////////////////////////////////////////////////////////////////////
	//gaussion filter
	double **kernel=new double *[blocksize];
	for(int i=0;i<blocksize;i++){
		kernel[i]=new double [blocksize];
		for(int j=0;j<blocksize;j++){
			kernel[i][j]=0;
		}
	}
	gaussianweight(blocksize,sigma,kernel);
	//////////////////////////////////////////////////////////////////////////
	double *totaldistance=new double [blocksize*blocksize];
	for(int i=0;i<blocksize*blocksize;i++){
		totaldistance[i]=0.;

	}

	//////////////////////////////////////////////////////////////////////////
	int fishmaxlocation=0;
	double tempstorage=0.,maxfish=-99999.,distance=0.,tempdistance=-99999;;
	// process
	for(int sourcey=0;sourcey<src.rows;sourcey++){
		for(int sourcex=0;sourcex<src.cols;sourcex++){
			if(sourcey+blocksize>(src.rows-blocksize/2)||sourcex+blocksize>(src.cols-blocksize/2)){
				continue;
			}else{
				
				int *totallocation=new int [blocksize*blocksize];
				for(int i=0;i<blocksize*blocksize;i++){

					if(i==(blocksize*blocksize/2)){
						totallocation[i]=0;
					}else{
						if(i<(blocksize*blocksize/2)){
							totallocation[i]=i+1;
						}else{
							totallocation[i]=i;
						}
					}

				}
				double *fisherdisciminant=new double [blocksize*blocksize];
				for(int i=0;i<blocksize*blocksize;i++){
					fisherdisciminant[i]=0.;

				}
				double *a1function=new double [blocksize*blocksize];
				for(int i=0;i<blocksize*blocksize;i++){
					a1function[i]=0.;

				}
				double *a2function=new double [blocksize*blocksize];
				for(int i=0;i<blocksize*blocksize;i++){
					a2function[i]=0.;

				}
				double *s1function=new double [blocksize*blocksize];
				for(int i=0;i<blocksize*blocksize;i++){
					s1function[i]=0.;

				}
				double *s2function=new double [blocksize*blocksize];
				for(int i=0;i<blocksize*blocksize;i++){
					s2function[i]=0.;

				}
				double *distancefirstderivate=new double [blocksize*blocksize/3];
				for(int i=0;i<blocksize/2;i++){
					distancefirstderivate[i]=-1;

				}
				double *distancelastderivate=new double [blocksize*blocksize/3];
				for(int i=0;i<blocksize/2;i++){
					distancelastderivate[i]=-1;

				}
				//distance compute
				if(src.channels()==3){
					for(int i=0;i<blocksize;i++){
						for(int j=0;j<blocksize;j++){
							for(int k=0;k<src.channels();k++){
							distance+=pow(fabs(rgbimagestorage[k][sourcey+blocksize/2][sourcex+blocksize/2]-rgbimagestorage[k][sourcey+i][sourcex+j]),3);
							}
							distance=pow(distance,(0.33333333));
							totaldistance[i*blocksize+j]=distance;
							distance=0;
						}
					}
				}else{
					for(int i=0;i<blocksize;i++){
						for(int j=0;j<blocksize;j++){
							distance=fabs(imagestorage[sourcey+blocksize/2][sourcex+blocksize/2]-imagestorage[sourcey+i][sourcex+j]);
							totaldistance[i*blocksize+j]=distance;
						}
					}
				}
				//(distance/location) order
				int k=0,w=0,templocation=0;
				for (int i =(blocksize*blocksize)-1; i > 0;--i){
					for (int j = 0; j < i; ++j){
						if (totaldistance[j] > totaldistance[j+1]){
							tempstorage=totaldistance[j];
							totaldistance[j]=totaldistance[j+1];
							totaldistance[j+1]=tempstorage;
							templocation=totallocation[j];
							totallocation[j]=totallocation[j+1];
							totallocation[j+1]=templocation;
						}	
					}
				}
				//first derivate to judge impulse noise
				int firstderivatelocation=0,lastderivatelocationfont=0;
				int locationderivate=-1;
				for (int i =0; i<(blocksize*blocksize);i++){
					lastderivatelocationfont=(int)((((double)(2*(blocksize*blocksize-1)))/blocksize)+0.5);
					if(i<((blocksize/2))){
						distancefirstderivate[i]=fabs(totaldistance[i+1]-totaldistance[i]);
						if(distancefirstderivate[i]>alpha){
							locationderivate=i;
						}
					}
					if(blocksize*blocksize-2<=(i)&&(i+1)<(blocksize*blocksize)){
						distancelastderivate[i-blocksize*blocksize+2]=fabs(totaldistance[i+1]-totaldistance[i]);
						if(distancelastderivate[i-blocksize*blocksize+2]>alpha){
							locationderivate=i;
						}
					}
				}
			
				//Fisher's discriminant estimation
				for(int i=1;i<=blocksize*blocksize-1;i++){
					for(int j=0;j<=i-1;j++){
						a1function[i]+=(double)((1.0/i)*totaldistance[j]);

					}
				}
				for(int i=1;i<=blocksize*blocksize-1;i++){
					for(int j=i;j<=blocksize*blocksize-1;j++){
						a2function[i]+=(double)((1.0/(blocksize*blocksize-i))*totaldistance[j]);

					}
				}
				for(int i=1;i<=blocksize*blocksize-1;i++){
					for(int j=0;j<=i-1;j++){
						s1function[i]+=pow((a1function[j]-totaldistance[j]),2);
					}
				}
				for(int i=1;i<=blocksize*blocksize-1;i++){
					for(int j=i;j<=blocksize*blocksize-1;j++){
						s2function[i]+=pow((a2function[j]-totaldistance[j]),2);
					}
				}
				maxfish=-9999;
				for(int i=1;i<blocksize*blocksize;i++){

					if((s1function[i]>0.000000000)||(s2function[i]>0.00000000000)){
						fisherdisciminant[i]=pow((double)(a1function[i]-a2function[i]),2)/(s1function[i]+s2function[i]);

					}else{
						s1function[i]=1;
						s2function[i]=0;
						fisherdisciminant[i]=pow((double)(a1function[i]-a2function[i]),2)/(s1function[i]+s2function[i]);
					}

					if(fisherdisciminant[i]>=maxfish){
						maxfish=fisherdisciminant[i];
						fishmaxlocation=i;
					}
				}
				int firstfishlocation=0;
				if(fishmaxlocation<=((blocksize/2))&&locationderivate<((blocksize/2)+2)&&locationderivate!=-1){
					fishmaxlocation=blocksize*blocksize;
					firstfishlocation=locationderivate+1;	
				}

				//peer group
				int ylocation=0,xlocation=0;
				int *peergrouplocation=new int [fishmaxlocation];
				for(int i=0;i<fishmaxlocation;i++){
					peergrouplocation[i]=totallocation[i];
				}
				int *peergroup=new int [fishmaxlocation];
				for(int i=firstfishlocation;i<fishmaxlocation;i++){
				peergroup[i]=0;
				}
				int **rgbpeergroup=new int *[fishmaxlocation];
				for(int i=0;i<fishmaxlocation;i++){
					rgbpeergroup[i]=new int [src.channels()];
					for(int j=0;j<src.channels();j++){
						rgbpeergroup[i][j]=0;
					}
				}

				if(src.channels()==3){
					for(int i=firstfishlocation;i<fishmaxlocation;i++){
						xlocation=totallocation[i]%blocksize;
						ylocation=totallocation[i]/blocksize;
						if(xlocation==0&&ylocation==0){
							xlocation=blocksize/2;
							ylocation=blocksize/2;

						}else{
							if(totallocation[i]<=(blocksize*blocksize/2)){
								if(totallocation[i]==blocksize){
									ylocation--;
									xlocation=blocksize-1;
								}else{
									xlocation--;
								}
							}
						}
						for(int j=0;j<src.channels();j++){
							rgbpeergroup[i][j]=rgbimagestorage[j][sourcey+ylocation][sourcex+xlocation];
						}
					}
				}else{
					for(int i=firstfishlocation;i<fishmaxlocation;i++){
						xlocation=totallocation[i]%blocksize;
						ylocation=totallocation[i]/blocksize;
						if(xlocation==0&&ylocation==0){
							peergroup[i]=imagestorage[sourcey+blocksize/2+ylocation][sourcex+blocksize/2+xlocation];

						}else{
							if(totallocation[i]<=(blocksize*blocksize/2)){
								if(totallocation[i]==blocksize){
									ylocation--;
									xlocation=blocksize-1;
								}else{
									xlocation--;
								}
							}
							peergroup[i]=imagestorage[sourcey+ylocation][sourcex+xlocation];
						}
					}
				}
				//use gaussion filter to peer group 
				if(src.channels()==3){
					int weightlocationx=0,weightlocationy=0;
					double weighttotal=0.,colortotal=0.;
					for(int channel=0;channel<src.channels();channel++){
						for(int i=firstfishlocation;i<fishmaxlocation;i++){
							weightlocationx=totallocation[i]%blocksize;
							weightlocationy=totallocation[i]/blocksize;
							if(totallocation[i]==0){
								weightlocationx=blocksize/2;
								weightlocationy=blocksize/2;
							}else{
								if(totallocation[i]<=(blocksize*blocksize/2)){
									if(totallocation[i]==blocksize){
										weightlocationy--;
										weightlocationx=blocksize-1;
									}else{
										weightlocationx--;
									}
								}
							}
							colortotal+=rgbpeergroup[i][channel]*kernel[weightlocationy][weightlocationx];
							weighttotal+=kernel[weightlocationy][weightlocationx];
						}
					rgbimagestorage[channel][sourcey+blocksize/2][sourcex+blocksize/2]=colortotal/weighttotal;
					colortotal=0;
					weighttotal=0;
					}
				}else{
					int weightlocationx=0,weightlocationy=0;
					double weighttotal=0.,colortotal=0.;
					for(int i=firstfishlocation;i<fishmaxlocation;i++){
						weightlocationx=totallocation[i]%blocksize;
						weightlocationy=totallocation[i]/blocksize;
						if(totallocation[i]==0){
							weightlocationx=blocksize/2;
							weightlocationy=blocksize/2;
						}else{
							if(totallocation[i]<=(blocksize*blocksize/2)){
								if(totallocation[i]==blocksize){
									weightlocationy--;
									weightlocationx=blocksize-1;
								}else{
									weightlocationx--;
								}
							}
						}
						colortotal+=peergroup[i]*kernel[weightlocationy][weightlocationx];
						weighttotal+=kernel[weightlocationy][weightlocationx];		
				}
				imagestorage[sourcey+blocksize/2][sourcex+blocksize/2]=colortotal/weighttotal;

				}
				///////////////////////////////////////////////////////////////////////////////////
				for(int i=0;i<fishmaxlocation;i++){
				delete	rgbpeergroup[i];
				}

				delete	[]rgbpeergroup;
				delete	[]totallocation;
				delete	[]peergrouplocation;
				delete	[]peergroup;

				delete	[]fisherdisciminant;
				delete	[]a1function;
				delete	[]a2function;
				delete	[]s1function;
				delete	[]s2function;
				delete	[]distancefirstderivate;
				delete	[]distancelastderivate;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	if(src.channels()==3){
		for(int channel=0;channel<src.channels();channel++){
			for(int i=0;i<src.rows;i++){
				for(int j=0;j<src.cols;j++){
				dst.at<cv::Vec3b>(i,j)[channel]=rgbimagestorage[channel][i][j]+0.5;
				}
			}
		}

	}else{
		for(int i=0;i<src.rows;i++){
			for(int j=0;j<src.cols;j++){
				dst.at<unsigned char>(i,j)=imagestorage[i][j]+0.5;
			}
		}
	}

	for(int i = 0 ; i < src.channels(); i++){
		for(int j=0;j<src.rows;j++){
			delete []rgbimagestorage[i][j];
		}
			delete []rgbimagestorage[i];
	}
	for(int i=0;i<blocksize;i++){
		delete	[]kernel[i];
	}
	for (int i = 0 ; i < src.rows; i++){
		delete	[]imagestorage[i];
	}
	delete	[]imagestorage;
	delete	[]kernel;
	delete	[]totaldistance;
	delete	[]rgbimagestorage;
	
	return true;
}