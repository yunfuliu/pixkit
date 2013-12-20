#include "../pixkit-image.hpp"

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
