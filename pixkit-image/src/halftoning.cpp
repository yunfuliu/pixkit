#include "../pixkit-image.hpp"

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
