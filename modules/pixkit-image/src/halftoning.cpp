//////////////////////////////////////////////////////////////////////////
//
// halftoning.cpp
//
//////////////////////////////////////////////////////////////////////////
#include "../include/pixkit-image.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>

using namespace	cv;

////////////////////////////////////////////////////////////////////////
//	error diffusion
//////////////////////////////////////////////////////////////////////////

// Floyd-Steinberg halftoning processing
bool pixkit::halftoning::errordiffusion::FloydSteinberg1975(const cv::Mat &src,cv::Mat &dst){

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::errordiffusion::FloydSteinberg1975] accepts only grayscale image");
	}
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[halftoning::errordiffusion::FloydSteinberg1975] image is empty");
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	Mat	tdst1f	=	src.clone();
	tdst1f.convertTo(tdst1f,CV_32FC1);
	//raster scan
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){

			double	error;
			if(tdst1f.ptr<float>(i)[j] >= 128){	
				error = tdst1f.ptr<float>(i)[j]-255;	//error value
				tdst1f.ptr<float>(i)[j] = 255;
			}
			else{				
				error = tdst1f.ptr<float>(i)[j];	//error value
				tdst1f.ptr<float>(i)[j] = 0;
			}

			if(j+1<src.cols){
				tdst1f.ptr<float>(i)[j+1] += error*	0.4375;
			}
			if((i+1<src.rows)&&(j-1>=0)){
				tdst1f.ptr<float>(i+1)[j-1] += error*	0.1875;
			}
			if(i+1<src.rows){
				tdst1f.ptr<float>(i+1)[j] += error*	0.3125;
			}
			if((i+1<src.rows)&&(j+1<src.cols)){
				tdst1f.ptr<float>(i+1)[j+1] += error*	0.0625;
			}
		}
	}
	tdst1f.convertTo(dst,CV_8UC1);

	return true;
}

// Jarvis halftoning processing
bool pixkit::halftoning::errordiffusion::Jarvis1976(const cv::Mat &src, cv::Mat &dst)
{
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::errordiffusion::Jarvis1976] accepts only grayscale image");
		return false;
	}
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[halftoning::errordiffusion::Jarvis1976] image is empty");
		return false;
	}

	/////////////////////////////////////////////////////////////////////////
	const float Errorkernel_Jarvis[3][5] = {	
		0,	0,	0,	7,	5,	
		3,	5,	7,	5,	3,	
		1,	3,	5,	3,	1
	};
	const int HalfSize = 2;

	//copy Input to RegImage
	cv::Mat tdst1f = src.clone();
	tdst1f.convertTo(tdst1f, CV_32FC1);

	// processing
	//raster scan
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){

			float error;
			if(tdst1f.ptr<float>(i)[j] >= 128){	
				error = tdst1f.ptr<float>(i)[j]-255;	//error value
				tdst1f.ptr<float>(i)[j] = 255;
			}
			else{				
				error = tdst1f.ptr<float>(i)[j];	//error value
				tdst1f.ptr<float>(i)[j] = 0;
			}

			float sum = 0;
			for(int x=0; x<=HalfSize; x++){
				if(i+x<0 || i+x>=src.rows)	continue;
				for(int y=-HalfSize; y<=HalfSize; y++){
					if(j+y<0 || j+y>=src.cols)	continue;
					sum += Errorkernel_Jarvis[x][y + HalfSize];
				}
			}

			for(int x=0; x<=HalfSize; x++){
				if(i+x<0 || i+x>=src.rows)	continue;
				for(int y=-HalfSize; y<=HalfSize; y++){
					if(j+y<0 || j+y>=src.cols)	continue;
					if(x!=0 || y!=0)
						tdst1f.ptr<float>(i+x)[j+y] += (error * Errorkernel_Jarvis[x][y + HalfSize] / sum);
				}
			}
		}
	}

	tdst1f.convertTo(dst,CV_8UC1);
	return true;
}

// Stucki halftoning processing
bool pixkit::halftoning::errordiffusion::Stucki1981(const cv::Mat &src, cv::Mat &dst)
{
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::errordiffusion::Stucki1981] accepts only grayscale image");
		return false;
	}
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[halftoning::errordiffusion::Stucki1981] image is empty");
		return false;
	}

	/////////////////////////////////////////////////////////////////////////
	const float ErrorKernel_Stucki[3][5] = {	
		0,	0,	0,	8,	4,	
		2,	4,	8,	4,	2,	
		1,	2,	4,	2,	1	
	};
	const int HalfSize = 2;

	//copy Input to RegImage
	cv::Mat tdst1f = src.clone();
	tdst1f.convertTo(tdst1f, CV_32FC1);

	// processing
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){

			float error;
			if(tdst1f.ptr<float>(i)[j] >= 128){	
				error = tdst1f.ptr<float>(i)[j]-255;	//error value
				tdst1f.ptr<float>(i)[j] = 255;
			}
			else{				
				error = tdst1f.ptr<float>(i)[j];	//error value
				tdst1f.ptr<float>(i)[j] = 0;
			}

			float sum = 0;
			for(int x=0; x<=HalfSize; x++){
				if(i+x<0 || i+x>=src.rows)	continue;
				for(int y=-HalfSize; y<=HalfSize; y++){
					if(j+y<0 || j+y>=src.cols)	continue;
					sum += ErrorKernel_Stucki[x][y + HalfSize];
				}
			}

			for(int x=0; x<=HalfSize; x++){
				if(i+x<0 || i+x>=src.rows)	continue;
				for(int y=-HalfSize; y<=HalfSize; y++){
					if(j+y<0 || j+y>=src.cols)	continue;
					if(x!=0 || y!=0)
						tdst1f.ptr<float>(i+x)[j+y] += (error * ErrorKernel_Stucki[x][y + HalfSize] / sum);
				}
			}
		}
	}

	tdst1f.convertTo(dst,CV_8UC1);
	return true;
}

// Shiau-Fan halftoning processing
bool pixkit::halftoning::errordiffusion::ShiauFan1996(const cv::Mat &src, cv::Mat &dst)
{
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::errordiffusion::ShiauFan1996] accepts only grayscale image");
		return false;
	}
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[halftoning::errordiffusion::ShiauFan1996] image is empty");
		return false;
	}

	/////////////////////////////////////////////////////////////////////////
	const float ErrorKernel_ShiauFan[2][7] = {	
		0,	0,	0,	0,	8,	0,	0,	
		1,	1,	2,	4,	0,	0,	0,	
	};

	int HalfSize = 3;

	//copy Input to RegImage
	cv::Mat tdst1f = src.clone();
	tdst1f.convertTo(tdst1f, CV_32FC1);

	// processing
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){

			float error;
			if(tdst1f.ptr<float>(i)[j] >= 128){	
				error = tdst1f.ptr<float>(i)[j]-255;	//error value
				tdst1f.ptr<float>(i)[j] = 255;
			}
			else{				
				error = tdst1f.ptr<float>(i)[j];	//error value
				tdst1f.ptr<float>(i)[j] = 0;
			}

			float sum = 0;
			for(int x=0; x<=1; x++){
				if(i+x<0 || i+x>=src.rows)	continue;
				for(int y=-HalfSize; y<=HalfSize; y++){
					if(j+y<0 || j+y>=src.cols)	continue;
					sum += ErrorKernel_ShiauFan[x][y + HalfSize];
				}
			}

			for(int x=0; x<=1; x++){
				if(i+x<0 || i+x>=src.rows)	continue;
				for(int y=-HalfSize; y<=HalfSize; y++){
					if(j+y<0 || j+y>=src.cols)	continue;
					if(x!=0 || y!=0)
						tdst1f.ptr<float>(i+x)[j+y] += (error * ErrorKernel_ShiauFan[x][y + HalfSize] / sum);
				}
			}
		}
	}

	tdst1f.convertTo(dst,CV_8UC1);
	return true;
}

// Ostromoukhov halftoning processing
bool pixkit::halftoning::errordiffusion::Ostromoukhov2001(const cv::Mat &src1b, cv::Mat &dst1b){
	
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src1b.type()!=CV_8UC1){
		CV_Error(CV_BadNumChannels,"[halftoning::errordiffusion::Ostromoukhov2001] accepts only grayscale image");
	}

	///////////////////////////////////////////////////////////////////////////
	// error kernel coefficient (A_10, A_-11, A_01)
	const int Ostromoukhov_EDcoefficient[128][3]={
	13,		0,		5,
	13,		0,		5,
	21,		0,		10,
	7,		0,		4,
	8,		0,		5,
	47,		3,		28,
	23,		3,		13,
	15,		3,		8,
	22,		6,		11,
	43,		15,		20,
	7,		3,		3,
	501,	224,	211,
	249,	116,	103,
	165,	80,		67,
	123,	62,		49,
	489,	256,	191,
	81,		44,		31,
	483,	272,	181,
	60,		35,		22,
	53,		32,		19,
	237,	148,	83,
	471,	304,	161,
	3,		2,		1,
	481,	314,	185,
	354,	226,	155,
	1389,	866,	685,
	227,	138,	125,
	267,	158,	163,
	327,	188,	220,
	61,		34,		45,
	627,	338,	505,
	1227,	638,	1075,
	20,		10,		19,
	1937,	1000,	1767,
	977,	520,	855,
	657,	360,	551,
	71,		40,		57,
	2005,	1160,	1539,
	337,	200,	247,
	2039,	1240,	1425,
	257,	160,	171,
	691,	440,	437,
	1045,	680,	627,
	301,	200,	171,
	177,	120,	95,
	2141,	1480,	1083,
	1079,	760,	513,
	725,	520,	323,
	137,	100,	57,
	2209,	1640,	855,
	53,		40,		19,
	2243,	1720,	741,
	565,	440,	171,
	759,	600,	209,
	1147,	920,	285,
	2311,	1880,	513,
	97,		80,		19,
	335,	280,	57,
	1181,	1000,	171,
	793,	680,	95,
	599,	520,	57,
	2413,	2120,	171,
	405,	360,	19,
	2447,	2200,	57,
	11,		10,		0,
	158,	151,	3,
	178,	179,	7,
	1030,	1091,	63,
	248,	277,	21,
	318,	375,	35,
	458,	571,	63,
	878,	1159,	147,
	5,		7,		1,
	172,	181,	37,
	97,		76,		22,
	72,		41,		17,
	119,	47,		29,
	4,		1,		1,
	4,		1,		1,
	4,		1,		1,
	4,		1,		1,
	4,		1,		1,
	4,		1,		1,
	4,		1,		1,
	4,		1,		1,
	4,		1,		1,
	65,		18,		17,
	95,		29,		26,
	185,	62,		53,
	30,		11,		9,
	35,		14,		11,
	85,		37,		28,
	55,		26,		19,
	80,		41,		29,
	155,	86,		59,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	5,		3,		2,
	305,	176,	119,
	155,	86,		59,
	105,	56,		39,
	80,		41,		29,
	65,		32,		23,
	55,		26,		19,
	335,	152,	113,
	85,		37,		28,
	115,	48,		37,
	35,		14,		11,
	355,	136,	109,
	30,		11,		9,
	365,	128,	107,
	185,	62,		53,
	25,		8,		7,
	95,		29,		26,
	385,	112,	103,
	65,		18,		17,
	395,	104,	101,
	4,		1,		1	};

	//////////////////////////////////////////////////////////////////////////
	///// initialize
	int sizeOfKernel = 3; // fixed size for Ostromoukhov's method
	Mat	errBuff1d	=	src1b.clone();	// copy src1b to tdst1b
	errBuff1d.convertTo(errBuff1d,CV_64FC1);
	dst1b.create(src1b.size(),src1b.type());
	Mat	errKernel1d(Size(sizeOfKernel,sizeOfKernel),CV_64FC1);
	errKernel1d.setTo(0);
	int HalfSize = (static_cast<int>(sizeOfKernel) - 1) / 2;


	//////////////////////////////////////////////////////////////////////////
	///// processing (serpentine scan)
	for(int i=0; i<errBuff1d.rows; i++){		

		if(i%2==0){ // for even rows
			for(int j=0; j<errBuff1d.cols; j++){

				if(errBuff1d.ptr<double>(i)[j] >= 128){
					dst1b.ptr<uchar>(i)[j] = static_cast< uchar >(255);
				}else{
					dst1b.ptr<uchar>(i)[j] = static_cast< uchar >(0);
				}
				int grayscale = static_cast<int>(src1b.data[i* src1b.cols + j]);
				if(src1b.data[i* src1b.cols + j] >= 128){
					grayscale = 255 - static_cast<int>(src1b.data[i* src1b.cols + j]);
				}

				// error value
				double error = errBuff1d.ptr<double>(i)[j] - static_cast< double >(dst1b.ptr<uchar>(i)[j]);

				// assign coefficients of Error Kernel with corresponding grayscale
				errKernel1d.ptr<double>(1)[2] = Ostromoukhov_EDcoefficient[grayscale][0];
				errKernel1d.ptr<double>(2)[0] = Ostromoukhov_EDcoefficient[grayscale][1];
				errKernel1d.ptr<double>(2)[1] = Ostromoukhov_EDcoefficient[grayscale][2];

				double sum = 0;
				for(int x=-HalfSize; x<=HalfSize; x++){
					if(i+x<0 || i+x>=errBuff1d.rows)	continue;
					for(int y=-HalfSize; y<=HalfSize; y++){
						if(j+y<0 || j+y>=errBuff1d.cols)	continue;
						sum += errKernel1d.ptr<double>(x + HalfSize)[y + HalfSize];
					}
				}

				for(int x=-HalfSize; x<=HalfSize; x++){
					if(i+x<0 || i+x>=errBuff1d.rows)	continue;
					for(int y=-HalfSize; y<=HalfSize; y++){
						if(j+y<0 || j+y>=errBuff1d.cols)	continue;
						if(x!=0 || y!=0)
							errBuff1d.ptr<double>(i+x)[j+y] += (error * errKernel1d.ptr<double>(x + HalfSize)[y + HalfSize] / sum);
					}
				}
			}

		}else{  // for odd rows
			for(int j=errBuff1d.cols-1; j>=0; j--){

				if(errBuff1d.ptr<double>(i)[j] >= 128){
					dst1b.ptr<uchar>(i)[j] = static_cast< uchar >(255);
				}else{
					dst1b.ptr<uchar>(i)[j] = static_cast< uchar >(0);
				}		
				int grayscale = static_cast<int>(src1b.data[i* src1b.cols + j]);
				if(src1b.data[i* src1b.cols + j] >= 128){
					grayscale = 255 - static_cast<int>(src1b.data[i* src1b.cols + j]);
				}

				// error value
				double error = errBuff1d.ptr<double>(i)[j] - static_cast< double >(dst1b.ptr<uchar>(i)[j]);

				// assign coefficients of Error Kernel with corresponding grayscale
				errKernel1d.ptr<double>(1)[2] = Ostromoukhov_EDcoefficient[grayscale][0];
				errKernel1d.ptr<double>(2)[0] = Ostromoukhov_EDcoefficient[grayscale][1];
				errKernel1d.ptr<double>(2)[1] = Ostromoukhov_EDcoefficient[grayscale][2];

				double sum = 0;
				for(int x=-HalfSize; x<=HalfSize; x++){
					if(i+x<0 || i+x>=errBuff1d.rows)	continue;
					for(int y=-HalfSize; y<=HalfSize; y++){
						if(j+y<0 || j+y>=errBuff1d.cols)	continue;
						sum += errKernel1d.ptr<double>(x + HalfSize)[2*HalfSize - (y + HalfSize)];
					}
				}
				for(int x=-HalfSize; x<=HalfSize; x++){
					if(i+x<0 || i+x>=errBuff1d.rows)	continue;
					for(int y=-HalfSize; y<=HalfSize; y++){
						if(j+y<0 || j+y>=errBuff1d.cols)	continue;
						if(x!=0 || y!=0)
							errBuff1d.ptr<double>(i+x)[j+y] += (error * errKernel1d.ptr<double>(x + HalfSize)[2*HalfSize - (y + HalfSize)] / sum);
					}
				}
			}
		}
	}

	return true;
}

// Zhou-Fang halftoning processing
bool pixkit::halftoning::errordiffusion::ZhouFang2003(const cv::Mat &src1b, cv::Mat &dst1b){
	
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src1b.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::errordiffusion::ZhouFang2003] accepts only grayscale image");
	}	

	///////////////////////////////////////////////////////////////////////////
	// error kernel coefficient (A_10, A_-11, A_01)
	const int ZhouFang_EDcoefficient[128][3]={
		13,			 0 ,      5,         //0.722222, 0.000000 ,0.277778    
		1300249,     0 , 499250,         //0.722562, 0.000000 ,0.277438    
		214114,     287 ,  99357,         //0.682418, 0.000915 ,0.316668    
		351854,       0 , 199965,         //0.637626, 0.000000 ,0.362374    
		801100,       0 , 490999,         //0.619999, 0.000000 ,0.380001    
		606569,   37983 , 355446,         //0.606570, 0.037983 ,0.355447    
		593140,   75967 , 330891,         //0.593141, 0.075967 ,0.330892    
		579711,  113951 , 306337,         //0.579712, 0.113951 ,0.306337    
		283141,   75967 , 140891,         //0.566283, 0.151934 ,0.281783    
		552853,  189918 , 257228,         //0.552854, 0.189918 ,0.257228    
		704075,  297466 , 303694,         //0.539424, 0.227902 ,0.232674    
		76188,   33644 ,  33025,         //0.533317, 0.235508 ,0.231175    
		527209,  243114 , 229676,         //0.527210, 0.243114 ,0.229676    
		521101,  250719 , 228178,         //0.521102, 0.250720 ,0.228178    
		  514994,  258325 , 226679,         //0.514995, 0.258326 ,0.226679    
		   508886,  265931 , 225181,         //0.508887, 0.265932 ,0.225181    
		 502779,  273537 , 223682,         //0.502780, 0.273538 ,0.223682    
		  496671,  281143 , 222184,         //0.496672, 0.281144 ,0.222184    
		  490564,  288749 , 220686,         //0.490564, 0.288749 ,0.220686    
		   484456,  296355 , 219187,         //0.484457, 0.296356 ,0.219187    
		   478349,  303961 , 217689,         //0.478349, 0.303961 ,0.217689    
		  472242,  311567 , 216190,         //0.472242, 0.311567 ,0.216190    
			46613,   31917 ,  21469,         //0.466135, 0.319173 ,0.214692    
		  467003,  317873 , 215123,         //0.467003, 0.317873 ,0.215123    
		  467872,  316573 , 215554,         //0.467872, 0.316573 ,0.215554    
			22321,   15013 ,  10285,         //0.468741, 0.315273 ,0.215985    
		   469610,  313973 , 216416,         //0.469610, 0.313973 ,0.216416    
		   470479,  312673 , 216847,         //0.470479, 0.312673 ,0.216847    
			52372,   34597 ,  24142,         //0.471348, 0.311373 ,0.217278    
		   472217,  310073 , 217709,         //0.472217, 0.310073 ,0.217709    
		   473086,  308773 , 218140,         //0.473086, 0.308773 ,0.218140    
		   157985,  102491 ,  72857,         //0.473955, 0.307473 ,0.218571    
			47482,   30617 ,  21900,         //0.474825, 0.306173 ,0.219002    
		   472921,  298013 , 229065,         //0.472921, 0.298013 ,0.229065    
		  471018,  289853 , 239128,         //0.471018, 0.289853 ,0.239128    
		   469115,  281693 , 249191,         //0.469115, 0.281693 ,0.249191    
		   467211,  273533 , 259254,         //0.467212, 0.273534 ,0.259255    
		   465308,  265374 , 269317,         //0.465308, 0.265374 ,0.269317    
		   463405,  257214 , 279380,         //0.463405, 0.257214 ,0.279380    
		   153834,   83018 ,  96481,         //0.461502, 0.249054 ,0.289443    
			45959,   24089 ,  29950,         //0.459599, 0.240895 ,0.299506    
		   452279,  286018 , 261701,         //0.452280, 0.286019 ,0.261702    
		  444960,  331142 , 223897,         //0.444960, 0.331142 ,0.223897    
		   437641,  376266 , 186092,         //0.437641, 0.376266 ,0.186092    
		   43024,   42131 ,  14826,         //0.430322, 0.421390 ,0.148288    
		   427011,  421930 , 151058,         //0.427011, 0.421930 ,0.151058    
		   211850,  211235 ,  76914,         //0.423701, 0.422471 ,0.153828    
		   210195,  211505 ,  78299,         //0.420391, 0.423011 ,0.156598    
		   208540,  211775 ,  79684,         //0.417081, 0.423551 ,0.159368    
		   413769,  424091 , 162139,         //0.413769, 0.424091 ,0.162139    
		   410459,  424631 , 164909,         //0.410459, 0.424631 ,0.164909    
		   407148,  425171 , 167679,         //0.407149, 0.425172 ,0.167679    
		   403838,  425711 , 170449,         //0.403839, 0.425712 ,0.170449    
		   400528,  426251 , 173219,         //0.400529, 0.426252 ,0.173219    
		   397217,  426792 , 175990,         //0.397217, 0.426792 ,0.175990    
		   393907,  427332 , 178760,         //0.393907, 0.427332 ,0.178760    
		   195298,  213936 ,  90765,         //0.390597, 0.427873 ,0.181530    
		   193643,  214206 ,  92150,         //0.387287, 0.428413 ,0.184300    
		   383976,  428953 , 187070,         //0.383976, 0.428953 ,0.187070    
		   380665,  429493 , 189841,         //0.380665, 0.429493 ,0.189841    
		   377355,  430033 , 192611,         //0.377355, 0.430033 ,0.192611    
		   374044,  430573 , 195381,         //0.374045, 0.430574 ,0.195381    
		   370734,  431113 , 198151,         //0.370735, 0.431114 ,0.198151    
		   367424,  431654 , 200921,         //0.367424, 0.431654 ,0.200921    
			36411,   43219 ,  20369,         //0.364114, 0.432194 ,0.203692    
		   366696,  445475 , 187828,         //0.366696, 0.445475 ,0.187828    
		   369279,  458755 , 171964,         //0.369280, 0.458756 ,0.171964    
		   185931,  236018 ,  78050,         //0.371863, 0.472037 ,0.156100    
		   374445,  485317 , 140236,         //0.374446, 0.485318 ,0.140236    
		   188514,  249299 ,  62186,         //0.377029, 0.498599 ,0.124372    
		   379611,  511879 , 108509,         //0.379611, 0.511880 ,0.108509    
		   382194,  525159 ,  92645,         //0.382195, 0.525160 ,0.092645    
			38477,   53843 ,   7678,         //0.384778, 0.538441 ,0.076782    
		   388829,  533848 ,  77321,         //0.388830, 0.533849 ,0.077321    
		   392881,  529256 ,  77861,         //0.392882, 0.529257 ,0.077861    
		   396933,  524664 ,  78401,         //0.396934, 0.524665 ,0.078401    
		   400986,  520072 ,  78941,         //0.400986, 0.520073 ,0.078941    
			40503,   51547 ,   7948,         //0.405038, 0.515480 ,0.079482    
		   399240,  493681 , 107078,         //0.399240, 0.493681 ,0.107078    
		   393441,  471883 , 134674,         //0.393442, 0.471884 ,0.134674    
			38764,   45008 ,  16227,         //0.387644, 0.450085 ,0.162272    
		   381845,  428284 , 189869,         //0.381846, 0.428285 ,0.189869    
		   376047,  406484 , 217468,         //0.376047, 0.406484 ,0.217468    
		   370249,  384683 , 245066,         //0.370250, 0.384684 ,0.245066    
		   364451,  362883 , 272664,         //0.364452, 0.362884 ,0.272665    
			35865,   34108 ,  30026,         //0.358654, 0.341083 ,0.300263    
		   356905,  343874 , 299219,         //0.356906, 0.343875 ,0.299220    
		   355157,  346665 , 298176,         //0.355158, 0.346666 ,0.298177    
		   353409,  349456 , 297133,         //0.353410, 0.349457 ,0.297134    
		   351661,  352247 , 296090,         //0.351662, 0.352248 ,0.296091    
		   349913,  355038 , 295047,         //0.349914, 0.355039 ,0.295048    
		   348165,  357829 , 294004,         //0.348166, 0.357830 ,0.294005    
		   346417,  360620 , 292961,         //0.346418, 0.360621 ,0.292962    
		   344669,  363411 , 291918,         //0.344670, 0.363412 ,0.291919    
		   342921,  366202 , 290875,         //0.342922, 0.366203 ,0.290876    
			34117,   36899 ,  28983,         //0.341173, 0.368994 ,0.289833    
		   342623,  367794 , 289581,         //0.342624, 0.367795 ,0.289582    
		   172037,  183297 , 144665,         //0.344075, 0.366595 ,0.289331    
		   345524,  365395 , 289079,         //0.345525, 0.365396 ,0.289080    
		   346975,  364195 , 288828,         //0.346976, 0.364196 ,0.288829    
		  348425,  362996 , 288577,         //0.348426, 0.362997 ,0.288578    
		  174938,  180898 , 144163,         //0.349877, 0.361797 ,0.288327    
		   35132,   36059 ,  28807,         //0.351327, 0.360597 ,0.288076    
		  346970,  363719 , 289309,         //0.346971, 0.363720 ,0.289310    
		  342614,  366841 , 290543,         //0.342615, 0.366842 ,0.290544    
		  338258,  369963 , 291777,         //0.338259, 0.369964 ,0.291778    
		  333902,  373085 , 293011,         //0.333903, 0.373086 ,0.293012    
		   16477,   18810 ,  14712,         //0.329547, 0.376208 ,0.294246    
		  330357,  376874 , 292767,         //0.330358, 0.376875 ,0.292768    
		  331169,  377542 , 291288,         //0.331169, 0.377542 ,0.291288    
		  331980,  378209 , 289810,         //0.331980, 0.378209 ,0.289810    
		  332791,  378876 , 288331,         //0.332792, 0.378877 ,0.288332    
		   33360,   37954 ,  28685,         //0.333603, 0.379544 ,0.286853    
		  334876,  378285 , 286838,         //0.334876, 0.378285 ,0.286838    
		  168074,  188513 , 143412,         //0.336149, 0.377027 ,0.286825    
		  337421,  375767 , 286810,         //0.337422, 0.375768 ,0.286811    
		  338694,  374509 , 286796,         //0.338694, 0.374509 ,0.286796    
		  169983,  186625 , 143391,         //0.339967, 0.373251 ,0.286783    
		  341239,  371991 , 286768,         //0.341240, 0.371992 ,0.286769    
		  342512,  370733 , 286754,         //0.342512, 0.370733 ,0.286754    
		  171892,  184737 , 143370,         //0.343785, 0.369475 ,0.286741    
		  345057,  368215 , 286726,         //0.345058, 0.368216 ,0.286727    
		  346330,  366957 , 286712,         //0.346330, 0.366957 ,0.286712    
		  173801,  182849 , 143349,         //0.347603, 0.365699 ,0.286699    
		  348875,  364439 , 286684,         //0.348876, 0.364440 ,0.286685    
		  175074,  181590 , 143335,         //0.350149, 0.363181 ,0.286671    
		175710,  180961 , 143328,         //0.351421, 0.361923 ,0.286657    
		35269,   36066 ,  28664          //0.352694, 0.360664 ,0.286643 
	};

	const int randomScale[128]={0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15, 15, 16, 17, 18, 18, 19, 20,   21,   21,   22,   23,   24,   24,   25,   26,   27,   27,   28,   29,   30,   31,   32,   33,   34,   34,   35,   36,   37,   38,   38,   39,   40,   41,   42,   42,   43,   44,   45,   46,   46,   47,   48,   49,   50,   53,   56,   59,   62,   65,   68,   71,   75,   78,   81,   84,   87,   90,   93,   96,   100,   100,   100,   100,   100,   100,   91,   83,   75,   66,   58,   50,   41,   33,   25,   17,   21,   26,   31,   35,   40,   45,   50,   54,   58,   62,   66,   70,   71,   73,   75,   77,   79,   80,   81,   83,   84,   86,   87,   88,   90,   91,   93,   94,   95,   97,   98,   100	};

	//////////////////////////////////////////////////////////////////////////
	///// initialization
	int sizeOfKernel = 3; // fixed size for Ostromoukhov's method
	Mat	errBuff1d	=	src1b.clone();
	errBuff1d.convertTo(errBuff1d,CV_64FC1);
	dst1b.create(src1b.size(),src1b.type());
	Mat	errKernel1d(Size(sizeOfKernel,sizeOfKernel),CV_64FC1);
	errKernel1d.setTo(0);
	int HalfSize = (static_cast<int>(sizeOfKernel) - 1) / 2;

	// get halftone image
	srand(0);	// for reproducibility

	// processing (serpentine scan)
	for(int i=0; i<dst1b.rows; i++){	
		if(i%2==0){ // for even rows
			for(int j=0; j<dst1b.cols; j++){

				int grayscale = static_cast<int>(src1b.data[i* src1b.cols + j]);
				if(src1b.data[i* src1b.cols + j] >= 128){
					grayscale = 255 - static_cast<int>(src1b.data[i* src1b.cols + j]);
				}
				double threshold = 128 + (rand() % 128)*(randomScale[grayscale]/100.0);

				if(errBuff1d.ptr<double>(i)[j] >= threshold){
					dst1b.ptr<uchar>(i)[j] = static_cast< uchar >(255);
				}else{
					dst1b.ptr<uchar>(i)[j] = static_cast< uchar >(0);
				}

				// error value
				double error = errBuff1d.ptr<double>(i)[j] - static_cast< double >(dst1b.ptr<uchar>(i)[j]);

				// assign coefficients of Error Kernel with corresponding grayscale
				errKernel1d.ptr<double>(1)[2] = ZhouFang_EDcoefficient[grayscale][0];
				errKernel1d.ptr<double>(2)[0] = ZhouFang_EDcoefficient[grayscale][1];
				errKernel1d.ptr<double>(2)[1] = ZhouFang_EDcoefficient[grayscale][2];

				double sum = 0;
				for(int x=-HalfSize; x<=HalfSize; x++){
					if(i+x<0 || i+x>=dst1b.rows)	continue;
					for(int y=-HalfSize; y<=HalfSize; y++){
						if(j+y<0 || j+y>=dst1b.cols)	continue;
						sum += errKernel1d.ptr<double>(x + HalfSize)[y + HalfSize];
					}
				}

				for(int x=-HalfSize; x<=HalfSize; x++){
					if(i+x<0 || i+x>=dst1b.rows)	continue;
					for(int y=-HalfSize; y<=HalfSize; y++){
						if(j+y<0 || j+y>=dst1b.cols)	continue;
						if(x!=0 || y!=0)
							errBuff1d.ptr<double>(i+x)[j+y] += (error * errKernel1d.ptr<double>(x + HalfSize)[y + HalfSize] / sum);
					}
				}
			}

		}else{  // for odd rows
			for(int j=dst1b.cols-1; j>=0; j--){

				int grayscale = static_cast<int>(src1b.data[i* src1b.cols + j]);
				if(src1b.data[i* src1b.cols + j] >= 128){
					grayscale = 255 - static_cast<int>(src1b.data[i* src1b.cols + j]);
				}
				double threshold = 128 + (rand() % 128)*(randomScale[grayscale]/100.0);

				if(errBuff1d.ptr<double>(i)[j] >= threshold){
					dst1b.ptr<uchar>(i)[j] = static_cast< uchar >(255);
				}else{
					dst1b.ptr<uchar>(i)[j] = static_cast< uchar >(0);
				}

				// error value
				double error = errBuff1d.ptr<double>(i)[j] - static_cast< double >(dst1b.ptr<uchar>(i)[j]);

				// assign coefficients of Error Kernel with corresponding grayscale
				errKernel1d.ptr<double>(1)[2] = ZhouFang_EDcoefficient[grayscale][0];
				errKernel1d.ptr<double>(2)[0] = ZhouFang_EDcoefficient[grayscale][1];
				errKernel1d.ptr<double>(2)[1] = ZhouFang_EDcoefficient[grayscale][2];

				double sum = 0;
				for(int x=-HalfSize; x<=HalfSize; x++){
					if(i+x<0 || i+x>=dst1b.rows)	continue;
					for(int y=-HalfSize; y<=HalfSize; y++){
						if(j+y<0 || j+y>=dst1b.cols)	continue;
						sum += errKernel1d.ptr<double>(x + HalfSize)[2*HalfSize - (y + HalfSize)];
					}
				}

				for(int x=-HalfSize; x<=HalfSize; x++){
					if(i+x<0 || i+x>=dst1b.rows)	continue;
					for(int y=-HalfSize; y<=HalfSize; y++){
						if(j+y<0 || j+y>=dst1b.cols)	continue;
						if(x!=0 || y!=0)
							errBuff1d.ptr<double>(i+x)[j+y] += (error * errKernel1d.ptr<double>(x + HalfSize)[2*HalfSize - (y + HalfSize)] / sum);
					}
				}
			}

		}
	}
	return true;
}


//////////////////////////////////////////////////////////////////////////
// iterative
//////////////////////////////////////////////////////////////////////////
bool pixkit::halftoning::iterative::LiebermanAllebach1997(const cv::Mat &src1b, cv::Mat &dst1b,double *coeData,int FilterSize,bool cppmode){

	//////////////////////////////////////////////////////////////////////////
	/// exceptions
	if(src1b.type()!=CV_8UC1){
		assert(false);
	}
	if(FilterSize==1){
		assert(false);
	}else if(FilterSize%2==0){
		assert(false);
	}


	//////////////////////////////////////////////////////////////////////////
	/// initialization
	int	m_Height	=	src1b.rows;
	int	m_Width		=	src1b.cols;
	Mat	dst1f;
	dst1f.create(src1b.size(),CV_32FC1);


	//////////////////////////////////////////////////////////////////////////
	/// get coe
	double	**	coe	=	new	double	*	[FilterSize];
	bool	UseOutSideCoe=false;
	if(coeData==NULL){	// default filter
		// default coe
		double	coetemp[7][7]={	{-0.001743412,	0.001445723,	0.007113962,	0.008826460,	0.006254165,	-0.000353216,	-0.003116813},	
								{0.000833244,	0.009929635,	0.023728512,	0.029311034,	0.022933593,	0.009711988,	0.000118289},	
								{0.005842017,	0.023691596,	0.048267152,	0.059043883,	0.046412889,	0.023315857,	0.005651780},	
								{0.009138026,	0.032894130,	0.064200407,	0.078773930,	0.060303292,	0.029523382,	0.007565880},	
								{0.007942508,	0.028669495,	0.053826022,	0.065024427,	0.050496807,	0.023933323,	0.005585837},	
								{0.002404652,	0.014542416,	0.029730907,	0.035492112,	0.027318902,	0.011351292,	0.001239805},	
								{-0.001666554,	0.002380156,	0.009867343,	0.012292728,	0.009216210,	0.002268413,	-0.001534185}};
		// copy coetemp to coe
		coeData	=	new	double	[FilterSize*FilterSize];
		for(int i=0;i<FilterSize;i++){
			coe[i]=&coeData[i*FilterSize];
			for(int j=0;j<FilterSize;j++){
				coe[i][j]=coetemp[i][j];
			}
		}
		UseOutSideCoe=false;
	}else{	// use external loaded filter, rather than the above default one
		for(int i=0;i<FilterSize;i++){
			coe[i]=&coeData[i*FilterSize];
		}
		UseOutSideCoe=true;
	}

	//////////////////////////////////////////////////////////////////////////
	/// get autocorrelation. Notably, the process depends upon the cppmode, it means whether the 'coe' is cpp or p as defined in paper. 
	int	exFS;
	int	tempFS;
	if(cppmode){
		exFS=FilterSize;
		tempFS=FilterSize/2;
	}else{
		exFS=FilterSize*2-1;
		tempFS=FilterSize-1;
	}
	double	*	autocoeData	=	new	double		[exFS*exFS];
	double	**	autocoe		=	new	double	*	[exFS];
	if(cppmode){
		for(int i=0;i<exFS;i++){
			autocoe[i]=&autocoeData[i*exFS];
			for(int j=0;j<exFS;j++){
				autocoe[i][j]=coe[i][j];
			}
		}
	}else{
		for(int i=0;i<exFS;i++){
			autocoe[i]=&autocoeData[i*exFS];
			for(int j=0;j<exFS;j++){
				autocoe[i][j]=0.;
			}
		}
		for(int i=0;i<FilterSize;i++){
			for(int j=0;j<FilterSize;j++){
				for(int m=-tempFS;m<=tempFS;m++){
					for(int n=-tempFS;n<=tempFS;n++){
						if(i+m<FilterSize&&i+m>=0&&j+n<FilterSize&&j+n>=0){
							autocoe[m+tempFS][n+tempFS]+=coe[i][j]*coe[i+m][j+n];
						}
					}
				}
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////
	/// load original image
	double	*	oriData	=	new	double		[m_Height*m_Width];
	double	**	ori		=	new	double	*	[m_Height];
	for(int i=0;i<m_Height;i++){
		ori[i]=&oriData[i*m_Width];
		for(int j=0;j<m_Width;j++){
			ori[i][j]=src1b.ptr<uchar>(i)[j];
		}
	}
	// get halftone image
	srand(7);	// for reproducibility 
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			double	temp=((double)rand())/32767.;
			dst1f.ptr<float>(i)[j]=temp<0.5?0.:255.;
		}
	}


	//////////////////////////////////////////////////////////////////////////
	/// Change grayscale to absorb
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			ori[i][j]=1.-ori[i][j]/255.;
			dst1f.ptr<float>(i)[j]=1.-dst1f.ptr<float>(i)[j]/255.;
		}
	}

	/// get error matrix
	double	*	emData	=	new	double		[m_Height*m_Width];
	double	**	em		=	new	double	*	[m_Height];
	for(int i=0;i<m_Height;i++){
		em[i]=&emData[i*m_Width];
		for(int j=0;j<m_Width;j++){
			em[i][j]=dst1f.ptr<float>(i)[j]-ori[i][j];
		}
	}

	/// get cross correlation
	double	*	crosscoeData	=	new	double		[m_Height*m_Width];
	double	**	crosscoe		=	new	double	*	[m_Height];
	for(int i=0;i<m_Height;i++){
		crosscoe[i]=&crosscoeData[i*m_Width];
		for(int j=0;j<m_Width;j++){
			crosscoe[i][j]=0.;
			for(int m=i-tempFS;m<=i+tempFS;m++){
				for(int n=j-tempFS;n<=j+tempFS;n++){
					if(m>=0&&m<m_Height&&n>=0&&n<m_Width){
						crosscoe[i][j]+=em[m][n]*autocoe[tempFS+m-i][tempFS+n-j];
					}
				}
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////
	/// DBS
	int		BenefitPixelNumber;
	double	dE[10],a0[10],a1[10];
	while(1){
		BenefitPixelNumber=0;
		for(int i=0;i<m_Height;i++){	// entire image
			for(int j=0;j<m_Width;j++){

				// = = = = = trial part = = = = = //
				// initialize psnr		0: original psnr, 1~8: Swap, 9: Toggel.
				// 8 1 2
				// 7 0 3
				// 6 5 4	
				for(int m=0;m<10;m++){
					dE[m]=0.;
					a0[m]=0.;
					a1[m]=0.;
				}
				// change the delta error as per different replacement methods
				for(int mode=1;mode<10;mode++){
					int		m,n;
					if(mode>=1&&mode<=8){
						// set position
						if(mode==1){
							m=1;	n=0;
						}else if(mode==2){
							m=1;	n=1;
						}else if(mode==3){
							m=0;	n=1;
						}else if(mode==4){
							m=-1;	n=1;
						}else if(mode==5){
							m=-1;	n=0;
						}else if(mode==6){
							m=-1;	n=-1;
						}else if(mode==7){
							m=0;	n=-1;
						}else if(mode==8){
							m=1;	n=-1;
						}
						// get dE
						if(i+m>=0&&i+m<m_Height&&j+n>=0&&j+n<m_Width){							
							if(dst1f.ptr<float>(i)[j]==1){
								a0[mode]=-1;
							}else{
								a0[mode]=1;
							}
							if(dst1f.ptr<float>(i+m)[j+n]==1){
								a1[mode]=-1;
							}else{
								a1[mode]=1;
							}
							if(dst1f.ptr<float>(i)[j]!=dst1f.ptr<float>(i+m)[j+n]){
								dE[mode]=(a0[mode]*a0[mode]+a1[mode]*a1[mode])*autocoe[tempFS][tempFS]+2.*a0[mode]*crosscoe[i][j]+2.*a1[mode]*crosscoe[i+m][j+n]+2.*a0[mode]*a1[mode]*autocoe[tempFS+m][tempFS+n];
							}
						}
					}else if(mode==9){
						if(dst1f.ptr<float>(i)[j]==1){
							a0[mode]=-1;
						}else{
							a0[mode]=1;
						}
						dE[mode]=autocoe[tempFS][tempFS]+2.*a0[mode]*crosscoe[i][j];
					}
				}
				// get minimum delta error and its position
				int		tempMinNumber	=0;
				double	tempMindE		=dE[0];
				for(int x=1;x<10;x++){
					if(dE[x]<tempMindE){
						tempMindE		=dE[x];
						tempMinNumber	=x;
					}
				}

				// = = = = = update part = = = = = //
				if(tempMindE<0){	// error is reduce
					// update hft image
					dst1f.ptr<float>(i)[j]	=1.-dst1f.ptr<float>(i)[j];
					if(tempMinNumber>=1&&tempMinNumber<=8){
						// get position
						int nm,nn;
						if(tempMinNumber==1){
							nm=1;	nn=0;
						}else if(tempMinNumber==2){
							nm=1;	nn=1;
						}else if(tempMinNumber==3){
							nm=0;	nn=1;
						}else if(tempMinNumber==4){
							nm=-1;	nn=1;
						}else if(tempMinNumber==5){
							nm=-1;	nn=0;
						}else if(tempMinNumber==6){
							nm=-1;	nn=-1;
						}else if(tempMinNumber==7){
							nm=0;	nn=-1;
						}else if(tempMinNumber==8){
							nm=1;	nn=-1;
						}
						// update hft image
						dst1f.ptr<float>(i+nm)[j+nn]	=1.-dst1f.ptr<float>(i+nm)[j+nn];
						// update cross correlation
						for(int m=-tempFS;m<=tempFS;m++){
							for(int n=-tempFS;n<=tempFS;n++){
								if(i+m>=0&&i+m<m_Height&&j+n>=0&&j+n<m_Width){
									crosscoe[i+m][j+n]+=a0[tempMinNumber]*autocoe[tempFS+m][tempFS+n];
								}
								if(i+m+nm>=0&&i+m+nm<m_Height&&j+n+nn>=0&&j+n+nn<m_Width){
									crosscoe[i+m+nm][j+n+nn]+=a1[tempMinNumber]*autocoe[tempFS+m][tempFS+n];
								}
							}
						}
					}else if(tempMinNumber==9){
						// update cross correlation
						for(int m=-tempFS;m<=tempFS;m++){
							for(int n=-tempFS;n<=tempFS;n++){
								if(i+m>=0&&i+m<m_Height&&j+n>=0&&j+n<m_Width){
									crosscoe[i+m][j+n]+=a0[tempMinNumber]*autocoe[tempFS+m][tempFS+n];
								}
							}
						}
					}
					BenefitPixelNumber++;
				}
			}
		}
		if(BenefitPixelNumber==0){
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Change absorb to grayscale
	dst1b.create(src1b.size(),CV_8UC1);
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			dst1b.ptr<uchar>(i)[j]=(1.-dst1f.ptr<float>(i)[j])*255.;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// release space
	delete	[]	crosscoeData;
	delete	[]	crosscoe;
	delete	[]	emData;
	delete	[]	em;
	delete	[]	oriData;
	delete	[]	ori;
	delete	[]	autocoe;
	delete	[]	autocoeData;
	delete	[]	coe;
	if(UseOutSideCoe==false){
		delete	[]	coeData;		
	}
	
	return true;
}

bool pixkit::halftoning::iterative::dualmetricDBS2002(const cv::Mat &src1b, cv::Mat &dst1b){

	//////////////////////////////////////////////////////////////////////////
	/// exceptions
	if(src1b.type()!=CV_8UC1){
		assert(false);
	}

	//////////////////////////////////////////////////////////////////////////
	///// get filter
	vector<Mat>	cpp1d;
	cpp1d.push_back(Mat());
	cpp1d.push_back(Mat());
	pixkit::halftoning::ungrouped::generateTwoComponentGaussianModel(cpp1d[0],43.2,38.7,0.0219,0.0598);
	pixkit::halftoning::ungrouped::generateTwoComponentGaussianModel(cpp1d[1],19.1,42.7,0.0330,0.0569);
	int FilterSize	=	cpp1d[0].rows;
	int	exFS=FilterSize;
	int	tempFS=FilterSize/2;

	//////////////////////////////////////////////////////////////////////////
	///// get weight
	Mat	weightmap1f(Size(2,256),CV_32FC1);
	for(int i=0;i<256;i++){
		float	gray	=	(float)i/255.;
		// get weight
		if(gray<=0.25){
			weightmap1f.ptr<float>(i)[0]	=	std::sqrtf(1-((float)4.*gray-1.)*((float)4.*gray-1.));
		}else if(gray>0.25&&gray<=0.75){
			weightmap1f.ptr<float>(i)[0]	=	std::fabsf((float)4.*gray-2);
		}else{
			weightmap1f.ptr<float>(i)[0]	=	std::sqrtf(1-((float)4.*gray-3.)*((float)4.*gray-3.));
		}
		weightmap1f.ptr<float>(i)[1]	=	1.-weightmap1f.ptr<float>(i)[0];
	}

	//////////////////////////////////////////////////////////////////////////
	/// initialization
	int	m_Height	=	src1b.rows;
	int	m_Width		=	src1b.cols;
	Mat	dst1f;
	dst1f.create(src1b.size(),CV_32FC1);

	//////////////////////////////////////////////////////////////////////////
	// get halftone image
	srand(0);
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			double	temp=((double)rand())/32767.;
			dst1f.ptr<float>(i)[j]=temp<0.5?0.:255.;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Change grayscale to absorb
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			dst1f.ptr<float>(i)[j]	=	1.-dst1f.ptr<float>(i)[j]/255.;			
		}
	}
	/// get error matrix
	Mat	em1f(Size(m_Width,m_Height),CV_32FC1);
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			double	oriv			=	1.-((double)src1b.ptr<uchar>(i)[j])/255.;			
			em1f.ptr<float>(i)[j]=dst1f.ptr<float>(i)[j]-oriv;
		}
	}
	/// get cross correlation
	vector<Mat>	c_ep1d;
	c_ep1d.push_back(Mat(Size(m_Width,m_Height),CV_64FC1));
	c_ep1d.push_back(Mat(Size(m_Width,m_Height),CV_64FC1));
	c_ep1d[0].setTo(0);
	c_ep1d[1].setTo(0);
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			for(int m=i-tempFS;m<=i+tempFS;m++){
				for(int n=j-tempFS;n<=j+tempFS;n++){
					if(m>=0&&m<m_Height&&n>=0&&n<m_Width){
						c_ep1d[0].ptr<double>(i)[j]+=em1f.ptr<float>(m)[n]*cpp1d[0].ptr<double>(tempFS+m-i)[tempFS+n-j];
						c_ep1d[1].ptr<double>(i)[j]+=em1f.ptr<float>(m)[n]*cpp1d[1].ptr<double>(tempFS+m-i)[tempFS+n-j];
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// DBS process
	int		BenefitPixelNumber;
	double	delta_E[10],a0[10],a1[10];
	while(1){
		BenefitPixelNumber=0;
		for(int i=0;i<m_Height;i++){	// entire image
			for(int j=0;j<m_Width;j++){

				//////////////////////////////////////////////////////////////////////////
				///// get weight
				int		currv	=	cvRound((float)dst1f.ptr<float>(i)[j]*255.);

				// = = = = = trial part = = = = = //
				// initialize psnr		0: original psnr, 1~8: Swap, 9: Toggle.
				// 8 1 2
				// 7 0 3
				// 6 5 4	
				for(int m=0;m<10;m++){
					delta_E[m]=0.;
					a0[m]=0.;
					a1[m]=0.;
				}
				// change the delta error as per different replacement methods
				for(int mode=1;mode<10;mode++){
					int		m,n;
					if(mode>=1&&mode<=8){
						// set position
						if(mode==1){
							m=1;	n=0;
						}else if(mode==2){
							m=1;	n=1;
						}else if(mode==3){
							m=0;	n=1;
						}else if(mode==4){
							m=-1;	n=1;
						}else if(mode==5){
							m=-1;	n=0;
						}else if(mode==6){
							m=-1;	n=-1;
						}else if(mode==7){
							m=0;	n=-1;
						}else if(mode==8){
							m=1;	n=-1;
						}
						// get dE
						if(i+m>=0&&i+m<m_Height&&j+n>=0&&j+n<m_Width){			
							// get weight to neighbor
							int	currv_nei	=	cvRound((float)dst1f.ptr<float>(i+m)[j+n]*255.);

							// get error
							if(dst1f.ptr<float>(i)[j]==1){
								a0[mode]=-1;
							}else{
								a0[mode]=1;
							}
							if(dst1f.ptr<float>(i+m)[j+n]==1){
								a1[mode]=-1;
							}else{
								a1[mode]=1;
							}
							if(dst1f.ptr<float>(i)[j]!=dst1f.ptr<float>(i+m)[j+n]){
								for(int w_idx=0;w_idx<2;w_idx++){
									delta_E[mode]+=(a0[mode]*a0[mode]	*	weightmap1f.ptr<float>(currv)[w_idx]*weightmap1f.ptr<float>(currv)[w_idx]	+	a1[mode]*a1[mode]	*	weightmap1f.ptr<float>(currv_nei)[w_idx]*weightmap1f.ptr<float>(currv_nei)[w_idx])	*	cpp1d[w_idx].ptr<double>(tempFS)[tempFS]	+
										2.*	a0[mode]	*	weightmap1f.ptr<float>(currv)[w_idx]	*	a1[mode]	*weightmap1f.ptr<float>(currv_nei)[w_idx]	*	cpp1d[w_idx].ptr<double>(tempFS+m)[tempFS+n]	+
										2.*	a0[mode]	*	weightmap1f.ptr<float>(currv)[w_idx]	*	c_ep1d[w_idx].ptr<double>(i)[j]	+
										2.*	a1[mode]	*	weightmap1f.ptr<float>(currv_nei)[w_idx]	*	c_ep1d[w_idx].ptr<double>(i+m)[j+n];
								}
							}
						}
					}else if(mode==9){
						if(dst1f.ptr<float>(i)[j]==1){
							a0[mode]=-1;
						}else{
							a0[mode]=1;
						}

						for(int w_idx=0;w_idx<2;w_idx++){
							delta_E[mode]	+=	(a0[mode]*a0[mode]	*	weightmap1f.ptr<float>(currv)[w_idx]*weightmap1f.ptr<float>(currv)[w_idx]	)	*	cpp1d[w_idx].ptr<double>(tempFS)[tempFS]	+
								2.*	a0[mode]	*	weightmap1f.ptr<float>(currv)[w_idx]	*	c_ep1d[w_idx].ptr<double>(i)[j];
						}

					}
				}
				// get minimum delta error and its position
				int		tempMinNumber	=0;
				double	tempMindE		=delta_E[0];
				for(int x=1;x<10;x++){
					if(delta_E[x]<tempMindE){
						tempMindE		=delta_E[x];
						tempMinNumber	=x;
					}
				}

				// = = = = = update part = = = = = //
				if(tempMindE<0){	// error is reduce
					// update hft image
					dst1f.ptr<float>(i)[j]	=1.-dst1f.ptr<float>(i)[j];
					if(tempMinNumber>=1&&tempMinNumber<=8){
						// get position
						int nm,nn;
						if(tempMinNumber==1){
							nm=1;	nn=0;
						}else if(tempMinNumber==2){
							nm=1;	nn=1;
						}else if(tempMinNumber==3){
							nm=0;	nn=1;
						}else if(tempMinNumber==4){
							nm=-1;	nn=1;
						}else if(tempMinNumber==5){
							nm=-1;	nn=0;
						}else if(tempMinNumber==6){
							nm=-1;	nn=-1;
						}else if(tempMinNumber==7){
							nm=0;	nn=-1;
						}else if(tempMinNumber==8){
							nm=1;	nn=-1;
						}
						// update hft image
						dst1f.ptr<float>(i+nm)[j+nn]	=1.-dst1f.ptr<float>(i+nm)[j+nn];
						// get weight to neighbor
						int	currv_nei	=	cvRound((float)dst1f.ptr<float>(i+nm)[j+nn]*255.);

						// update cross correlation
						for(int m=-tempFS;m<=tempFS;m++){
							for(int n=-tempFS;n<=tempFS;n++){
								if(i+m>=0&&i+m<m_Height&&j+n>=0&&j+n<m_Width){
									for(int w_idx=0;w_idx<2;w_idx++){
										c_ep1d[w_idx].ptr<double>(i+m)[j+n]+=a0[tempMinNumber]*weightmap1f.ptr<float>(currv)[w_idx]*cpp1d[w_idx].ptr<double>(tempFS+m)[tempFS+n];
									}
								}
								if(i+m+nm>=0&&i+m+nm<m_Height&&j+n+nn>=0&&j+n+nn<m_Width){
									for(int w_idx=0;w_idx<2;w_idx++){
										c_ep1d[w_idx].ptr<double>(i+m+nm)[j+n+nn]+=a1[tempMinNumber]*weightmap1f.ptr<float>(currv_nei)[w_idx]*cpp1d[w_idx].ptr<double>(tempFS+m)[tempFS+n];
									}
								}
							}
						}
					}else if(tempMinNumber==9){
						// update cross correlation
						for(int m=-tempFS;m<=tempFS;m++){
							for(int n=-tempFS;n<=tempFS;n++){
								if(i+m>=0&&i+m<m_Height&&j+n>=0&&j+n<m_Width){
									for(int w_idx=0;w_idx<2;w_idx++){
										c_ep1d[w_idx].ptr<double>(i+m)[j+n]+=a0[tempMinNumber]*weightmap1f.ptr<float>(currv)[w_idx]*cpp1d[w_idx].ptr<double>(tempFS+m)[tempFS+n];
									}
								}
							}
						}
					}
					BenefitPixelNumber++;
				}
			}
		}
		//		cout	<<	BenefitPixelNumber	<<	endl;
		if(BenefitPixelNumber==0){
			break;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	/// Change absorb to grayscale
	dst1b.create(src1b.size(),CV_8UC1);
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			dst1b.ptr<uchar>(i)[j]=(1.-dst1f.ptr<float>(i)[j])*255.;
		}
	}

	return true;
}

bool pixkit::halftoning::iterative::ElectrostaticHalftoning2010(const cv::Mat &src, cv::Mat &dst, int InitialCharge, int Iterations, int GridForce, int Shake, int Debug){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[pixkit::halftoning::ElectrostaticHalftoning] image should be grayscale");
	}
	if(InitialCharge!=0&&InitialCharge!=1){
		printf("[pixkit::halftoning::ElectrostaticHalftoning] InitialCharge should be 0 or 1");
		system("pause");
		exit(0);
	}
	if(Iterations<1){
		printf("[pixkit::halftoning::ElectrostaticHalftoning] Iterations should be bigger than 1");
		system("pause");
		exit(0);
	}
	if(GridForce!=0&&GridForce!=1){
		printf("[pixkit::halftoning::ElectrostaticHalftoning] GridForce should be 0 or 1");
		system("pause");
		exit(0);
	}
	if(Shake!=0&&Shake!=1){
		printf("[pixkit::halftoning::ElectrostaticHalftoning] Shake should be 0 or 1");
		system("pause");
		exit(0);
	}
	if(Shake==1&&Iterations<=64){
		printf("[pixkit::halftoning::ElectrostaticHalftoning] Iterations should be bigger than 64");
		system("pause");
		exit(0);
	}
	if(Debug!=0&&Debug!=1&&Debug!=2){
		printf("[pixkit::halftoning::ElectrostaticHalftoning] Debug should be 0, 1 or 2");
		system("pause");
		exit(0);
	}

	char out_file[50];
	double **image_in = new double*[src.rows];
	for(int i=0;i<src.rows;i++)
		image_in[i] = new double [src.cols];
	int **image_tmp = new int*[src.rows];
	for(int i=0;i<src.rows;i++)
		image_tmp[i] = new int [src.cols];
	Mat real_dst(src.rows,src.cols,CV_8UC1);

	//////////////////////////////////////////////////////////////////////////
	///// Initialization
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			image_in[i][j]=(double)src.data[i*src.cols+j]/255;
			image_tmp[i][j]=255;
			real_dst.data[i*src.cols+j]=255;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// Find the number of Particle
	double CountParticle=0;
	for(int i=0; i<src.rows; i++)
		for(int j=0; j<src.cols; j++)
			CountParticle=CountParticle+(1-image_in[i][j]);
	printf("The number of black pixel(charge) = %d\n",(int)CountParticle);

	//////////////////////////////////////////////////////////////////////////
	///// Initialize the Particle's position
	double *Particle_Y = new double[(int)CountParticle];
	double *Particle_X = new double[(int)CountParticle];
	int Particle=CountParticle;
	while(Particle>0){
		int RandY=rand()%src.rows;
		int RandX=rand()%src.cols;
		if(image_tmp[RandY][RandX]!=0){
			if(InitialCharge==1){
				int RandNumber=rand()%256;
				if(RandNumber>src.data[RandY*src.cols+RandX]){
					image_tmp[RandY][RandX]=0;
					if(Debug==1)
						real_dst.data[RandY*src.cols+RandX]=0;
					Particle--;
				}
			}
			else if(InitialCharge==0){
				image_tmp[RandY][RandX]=0;
				if(Debug==1)
					real_dst.data[RandY*src.cols+RandX]=0;
				Particle--;
			}
		}
	}
	if(Debug==1)
		cv::imwrite("output.bmp", real_dst);
	else if(Debug==2){
		sprintf_s(out_file,".\\output\\0.bmp");
		cv::imwrite(out_file, real_dst);
	}

	//////////////////////////////////////////////////////////////////////////
	///// Record the Particle's position
	int ParticleNumber=0;
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			if(image_tmp[i][j]==0){
				Particle_Y[ParticleNumber]=(double)i;
				Particle_X[ParticleNumber]=(double)j;
				ParticleNumber++;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// Create Forcefield Table
	printf("Create Forcefield Table, \n");
	double **forcefield_y = new double*[src.rows];
	for(int i=0;i<src.rows;i++)
		forcefield_y[i] = new double [src.cols];
	double **forcefield_x = new double*[src.rows];
	for(int i=0;i<src.rows;i++)
		forcefield_x[i] = new double [src.cols];
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			forcefield_y[i][j]=0;
			forcefield_x[i][j]=0;
			for(int y=0; y<src.rows; y++){
				for(int x=0; x<src.cols; x++){
					if(!(i==y&&j==x)){
						forcefield_y[i][j]+=(1-image_in[y][x])*(y-i)/((y-i)*(y-i)+(x-j)*(x-j));
						forcefield_x[i][j]+=(1-image_in[y][x])*(x-j)/((y-i)*(y-i)+(x-j)*(x-j));
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	///// process
	double instead_y,instead_x;
	Particle=CountParticle;
	for(int iterations=1; iterations<=Iterations; iterations++){
		printf("Iterations %d\n",iterations);

		for(int NowCharge=0; NowCharge<Particle; NowCharge++){
			double NewPosition_Y=0,NewPosition_X=0;
			double GridForce_Y=0,GridForce_X=0;

			//Attraction(by using bilinear interpolation)
			if(Particle_Y[NowCharge]-(int)Particle_Y[NowCharge]==0&&Particle_X[NowCharge]-(int)Particle_X[NowCharge]==0){
				NewPosition_Y=forcefield_y[(int)Particle_Y[NowCharge]][(int)Particle_X[NowCharge]];
				NewPosition_X=forcefield_x[(int)Particle_Y[NowCharge]][(int)Particle_X[NowCharge]];
			}
			else{
				int Bilinear_y1=Particle_Y[NowCharge];
				int Bilinear_x1=Particle_X[NowCharge];
				int Bilinear_y2=Bilinear_y1+1;
				int Bilinear_x2=Bilinear_x1+1;
				if(Bilinear_y1+1<src.rows&&Bilinear_x1+1<src.cols){
					NewPosition_Y=forcefield_y[Bilinear_y1][Bilinear_x1]*((double)Bilinear_x2-Particle_X[NowCharge])*((double)Bilinear_y2-Particle_Y[NowCharge])
						+forcefield_y[Bilinear_y1][Bilinear_x2]*(Particle_X[NowCharge]-(double)Bilinear_x1)*((double)Bilinear_y2-Particle_Y[NowCharge])
						+forcefield_y[Bilinear_y2][Bilinear_x1]*((double)Bilinear_x2-Particle_X[NowCharge])*(Particle_Y[NowCharge]-(double)Bilinear_y1)
						+forcefield_y[Bilinear_y2][Bilinear_x2]*(Particle_X[NowCharge]-(double)Bilinear_x1)*(Particle_Y[NowCharge]-(double)Bilinear_y1);
					NewPosition_X=forcefield_x[Bilinear_y1][Bilinear_x1]*((double)Bilinear_x2-Particle_X[NowCharge])*((double)Bilinear_y2-Particle_Y[NowCharge])
						+forcefield_x[Bilinear_y1][Bilinear_x2]*(Particle_X[NowCharge]-(double)Bilinear_x1)*((double)Bilinear_y2-Particle_Y[NowCharge])
						+forcefield_x[Bilinear_y2][Bilinear_x1]*((double)Bilinear_x2-Particle_X[NowCharge])*(Particle_Y[NowCharge]-(double)Bilinear_y1)
						+forcefield_x[Bilinear_y2][Bilinear_x2]*(Particle_X[NowCharge]-(double)Bilinear_x1)*(Particle_Y[NowCharge]-(double)Bilinear_y1);
				}
			}

			//Repulsion
			for(int OtherCharge=0; OtherCharge<Particle; OtherCharge++){
				if(NowCharge!=OtherCharge){
					instead_y=Particle_Y[OtherCharge]-Particle_Y[NowCharge];
					instead_x=Particle_X[OtherCharge]-Particle_X[NowCharge];
					if(!(instead_y==0&&instead_x==0)){
						NewPosition_Y-=instead_y/(instead_y*instead_y+instead_x*instead_x);
						NewPosition_X-=instead_x/(instead_y*instead_y+instead_x*instead_x);
					}
				}
			}

			//Add GridForce to find discrete particle locations
			double real_y=Particle_Y[NowCharge]-(int)Particle_Y[NowCharge];
			double real_x=Particle_X[NowCharge]-(int)Particle_X[NowCharge];
			if(real_y==0&&real_x==0){
				GridForce_Y=0;
				GridForce_X=0;
			}
			else{
				if(real_y<0.5){
					if(real_x<0.5){
						real_y=(0-real_y);
						real_x=(0-real_x);
					}
					else{
						real_y=(0-real_y);
						real_x=(1-real_x);
					}	
				}
				else{
					if(real_x<0.5){
						real_y=(1-real_y);
						real_x=(0-real_x);
					}
					else{
						real_y=(1-real_y);
						real_x=(1-real_x);
					}	
				}
				double vector3=sqrt(real_y*real_y+real_x*real_x);
				if(real_y==0)
					GridForce_Y=0;
				else
					GridForce_Y=3.5*real_y/(vector3*(1+pow(vector3,8)*10000));
				if(real_x==0)
					GridForce_X=0;
				else
					GridForce_X=3.5*real_x/(vector3*(1+pow(vector3,8)*10000));
			}

			//resault (new position of particles)
			if(GridForce==0){
				Particle_Y[NowCharge]=Particle_Y[NowCharge]+0.1*NewPosition_Y;
				Particle_X[NowCharge]=Particle_X[NowCharge]+0.1*NewPosition_X;
			}
			else if(GridForce==1){
				Particle_Y[NowCharge]=Particle_Y[NowCharge]+0.1*(NewPosition_Y+GridForce_Y);
				Particle_X[NowCharge]=Particle_X[NowCharge]+0.1*(NewPosition_X+GridForce_X);
			}

			//Shake
			if(Shake==1&&iterations%10==0&&Iterations>64){
				Particle_Y[NowCharge]+=(log10((double)Iterations)/log10(2.0)-6)*exp(iterations/1000.0)/10;
				Particle_X[NowCharge]+=(log10((double)Iterations)/log10(2.0)-6)*exp(iterations/1000.0)/10;
			}

			if(Particle_Y[NowCharge]<0)
				Particle_Y[NowCharge]=0;
			if(Particle_Y[NowCharge]>=src.rows)
				Particle_Y[NowCharge]=src.rows-1;
			if(Particle_X[NowCharge]<0)
				Particle_X[NowCharge]=0;
			if(Particle_X[NowCharge]>=src.cols)
				Particle_X[NowCharge]=src.cols-1;

		}

		//Output
		for(int y=0; y<src.rows; y++){
			for(int x=0; x<src.cols; x++){
				real_dst.data[y*src.cols+x]=255;
				image_tmp[y][x]=255;
			}
		}
		int output_position;
		int out_Y,out_X;
		double count_errorY=0,count_errorX=0;
		for(int NowCharge=0; NowCharge<Particle; NowCharge++){
			out_Y=Particle_Y[NowCharge]+0.5;
			out_X=Particle_X[NowCharge]+0.5;
			if(out_Y>=src.rows)
				out_Y=src.rows-1;
			if(out_X>=src.cols)
				out_X=src.cols-1;
			image_tmp[out_Y][out_X]=0;
		}

		for(int y=0; y<src.rows; y++)
			for(int x=0; x<src.cols; x++)
				real_dst.data[y*src.cols+x]=image_tmp[y][x];

		if(Debug==1)
			cv::imwrite("output.bmp",real_dst);
		else if(Debug==2){
			sprintf_s(out_file,".\\output\\%d.bmp",iterations);
			cv::imwrite(out_file,real_dst);
		}
	}

	dst=real_dst.clone();

	delete [] image_in;
	delete [] image_tmp;

	return true;

}

//////////////////////////////////////////////////////////////////////////
//	ordered dithering
//////////////////////////////////////////////////////////////////////////

bool pixkit::halftoning::ordereddithering::Ulichney1987(const cv::Mat &src, cv::Mat &dst, DitherArray_TYPE odtype)
{	
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::ordereddithering::Ulichney1987] accepts only grayscale image");
		return false;
	}
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[halftoning::ordereddithering::Ulichney1987] image is empty");
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	dst.create(src.size(), src.type());
	const double classical_4[8][8] = { 0.567,	0.635,	0.608,	0.514,	0.424,	0.365,	0.392,	0.486, 	0.847,	0.878,	0.910,	0.698,	0.153,	0.122,	0.090,	0.302,	0.820,	0.969,	0.941,	0.667,	0.180,	0.031,	0.059,	0.333,	0.725,	0.788,	0.757,	0.545,	0.275,	0.212,	0.243,	0.455,	0.424,	0.365,	0.392,	0.486,	0.567,	0.635,	0.608,	0.514,	0.153,	0.122,	0.090,	0.302,	0.847,	0.878,	0.910,	0.698,	0.180,	0.031,	0.059,	0.333,	0.820,	0.969,	0.941,	0.667,	0.275,	0.212,	0.243,	0.455,	0.725,	0.788,	0.757,	0.545	};
	const double bayer_5[8][8] = { 0.513,	0.272,	0.724,	0.483,	0.543,	0.302,	0.694,	0.453,	0.151,	0.755,	0.091,	0.966,	0.181,	0.758,	0.121,	0.936,	0.634,	0.392,	0.574,	0.332,	0.664,	0.423,	0.604,	0.362,	0.060,	0.875,	0.211,	0.815,	0.030,	0.906,	0.241,	0.845,	0.543,	0.302,	0.694,	0.453,	0.513,	0.272,	0.724,	0.483,	0.181,	0.758,	0.121,	0.936,	0.151,	0.755,	0.091,	0.966,	0.664,	0.423,	0.604,	0.362,	0.634,	0.392,	0.574,	0.332,	0.030,	0.906,	0.241,	0.845,	0.060,	0.875,	0.211,	0.815	};
	std::string NameOfFilterFile;
	const int sizeOfArray = 8;

	std::vector< std::vector<double> > DitherArray( sizeOfArray, std::vector<double>(sizeOfArray) );

	switch (odtype)
	{
	case 0:		//classical-4
		for (int i=0; i<sizeOfArray; i++){
			for (int j=0; j<sizeOfArray; j++){
				DitherArray[i][j] = bayer_5[i][j];
			}
		}
		break;
	case 1:		//bayer-5
		for (int i=0; i<sizeOfArray; i++){
			for (int j=0; j<sizeOfArray; j++){
				DitherArray[i][j] = classical_4[i][j];
			}
		}
		break;
	default:	//default condition : use dither array of classical-4
		for (int i=0; i<sizeOfArray; i++){
			for (int j=0; j<sizeOfArray; j++){
				DitherArray[i][j] = bayer_5[i][j];
			}
		}
	}

	//Normalization of Dither Array for gray scale threshold values
	for(int i=0; i<sizeOfArray; i++){
		for(int j=0; j<sizeOfArray; j++){
			DitherArray[i][j] *= 255;
		}
	}

	//Dither processing
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
				if(src.data[(i)*src.cols + (j)] >= DitherArray[i%sizeOfArray][j%sizeOfArray])
					dst.data[i*dst.cols+j] = 255;
				else
					dst.data[i*dst.cols+j] = 0;
		}
	}
	return true;
}


//////////////////////////////////////////////////////////////////////////
//	ordered dithering
//////////////////////////////////////////////////////////////////////////

// Hft_OD_KackerAllebach1998 processing
bool pixkit::halftoning::ordereddithering::KackerAllebach1998(const cv::Mat &src1b, cv::Mat &dst1b){

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src1b.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::ordereddithering::KackerAllebach1998] accepts only grayscale image");
	}
	dst1b.create(src1b.size(),src1b.type());

	//////////////////////////////////////////////////////////////////////////
	// initialization: set initial parameters
	const int	sizeDitherArray=32;	// defined in their Section 4. 
	const int	nDitherArray=4;		// 4 is defined for their experiments. This value should >=3 for easing checkerboard pattern and periodicity as described in Section 2.3.
	const int	DAs[4][32][32]={
	{ {172,40,126,240,213,72,49,171,94,207,69,153,251,17,96,166,222,106,47,158,224,14,138,239,36,192,168,99,116,200,37,237}, {84,23,193,56,12,140,226,146,126,242,25,53,194,118,186,58,24,253,203,92,60,114,194,55,153,82,228,11,216,253,83,74}, {232,147,101,166,86,203,189,64,8,179,107,81,139,1,237,146,133,75,7,128,191,244,97,211,71,23,129,144,43,150,6,124}, {210,114,248,1,113,31,255,104,41,220,199,150,226,41,206,102,33,177,239,169,27,42,123,3,159,248,197,52,95,182,164,220}, {76,16,217,186,120,230,21,77,212,115,16,30,171,123,86,163,197,95,215,80,107,149,183,223,110,177,65,205,236,24,108,61}, {150,53,93,206,43,131,180,162,151,133,191,93,58,215,46,225,45,119,16,161,17,252,64,46,137,31,16,120,155,134,34,201}, {195,29,160,68,144,58,236,96,1,51,240,176,232,143,73,136,173,64,129,196,228,175,91,207,234,102,169,87,56,225,243,79}, {127,247,175,225,107,12,170,200,68,250,38,83,112,10,184,28,248,208,2,34,53,142,109,10,77,190,255,210,4,172,100,146}, {7,98,78,39,189,243,85,30,139,120,123,21,158,211,99,122,162,113,185,75,98,238,20,181,127,25,143,71,111,47,187,11}, {219,166,118,23,133,69,222,108,239,183,207,224,72,199,37,63,85,223,57,152,213,165,52,221,154,45,200,162,92,216,251,130}, {63,141,229,195,55,154,100,44,165,10,148,49,171,112,254,235,18,105,243,0,201,122,66,249,116,62,227,15,135,32,67,193}, {185,48,75,255,176,214,18,184,81,59,97,29,132,2,161,147,138,177,10,152,29,83,141,4,96,39,239,105,180,233,157,22},{12,159,111,88,1,116,66,203,128,231,197,215,247,91,79,50,187,115,92,215,230,48,176,194,216,188,122,78,54,147,106,86}, {128,240,210,30,234,168,143,250,160,17,105,154,179,66,204,33,240,71,39,182,131,106,245,34,73,148,167,253,7,202,42,242}, {198,38,101,150,130,46,93,23,35,57,136,85,0,124,136,221,102,249,148,70,81,15,204,88,125,60,28,208,69,119,176,217}, {85,167,20,57,252,190,221,109,188,201,245,45,192,226,9,172,24,156,4,212,164,27,154,170,241,13,223,130,97,231,18,60},
	  {108,218,181,117,158,11,79,153,233,2,73,168,209,54,145,117,59,86,193,127,235,56,98,220,112,48,180,157,36,188,163,135}, {6,61,246,89,204,137,38,97,174,144,28,114,94,103,184,71,241,225,36,125,50,252,188,40,137,194,91,249,107,3,83,205}, {229,139,70,43,27,186,241,68,122,222,46,135,236,19,161,222,26,134,76,195,105,141,80,6,67,238,22,143,50,216,241,28}, {175,95,167,126,214,102,229,17,53,254,181,208,62,37,199,126,47,149,173,14,228,21,213,198,146,121,206,55,174,117,152,89}, {51,13,202,148,59,80,180,157,192,76,9,152,78,244,5,174,212,95,63,237,117,159,60,178,103,43,76,32,224,64,191,140}, {161,119,238,20,253,45,113,141,32,110,127,202,170,140,115,82,34,251,183,6,94,0,132,245,14,231,182,156,131,40,18,247}, {65,211,36,198,219,134,3,223,246,163,88,40,25,90,252,52,160,108,137,218,153,202,111,35,165,124,217,92,255,168,73,110}, {192,82,106,90,155,72,209,61,19,234,55,226,211,132,190,171,233,19,42,54,166,88,206,72,51,196,82,26,103,235,205,48}, {236,7,142,243,52,121,169,99,187,145,104,173,65,13,69,8,100,204,124,227,210,26,191,237,140,3,151,67,9,186,84,156}, {44,131,200,27,185,232,11,203,90,129,31,196,118,136,238,145,218,30,91,67,104,12,121,169,98,250,198,213,164,138,33,218}, {74,177,214,84,160,39,112,249,49,5,214,158,24,184,80,244,58,157,175,235,142,81,62,38,228,14,115,41,232,59,20,118}, {227,114,19,254,101,68,135,149,179,70,242,87,56,201,37,120,0,207,113,189,49,182,248,156,87,29,178,78,144,94,250,174}, {151,32,96,189,5,208,230,22,164,99,190,47,147,224,178,163,103,75,21,133,4,221,199,134,109,165,242,129,2,209,196,74}, {178,61,145,125,233,57,195,79,220,119,254,15,110,93,50,66,230,193,41,90,212,101,22,54,205,70,219,104,51,151,111,35}, {77,247,219,44,173,89,109,8,35,138,42,170,209,244,128,5,217,139,245,167,63,149,123,179,44,8,187,26,172,246,15,227}, {197,9,100,162,25,130,246,155,185,65,231,121,74,33,183,155,84,13,116,181,31,229,77,251,142,89,234,125,62,159,87,132} },
	{ {185,22,140,215,92,182,66,125,222,95,7,193,153,108,24,174,236,57,98,205,49,136,169,33,110,241,50,173,78,221,44,157}, {105,57,241,201,15,34,251,199,79,163,240,54,46,210,77,118,140,28,161,251,8,92,216,63,192,17,96,212,133,114,64,250}, {168,38,130,72,117,165,49,141,22,175,114,134,227,88,196,250,43,222,186,108,124,197,154,13,150,131,239,165,33,202,11,101}, {206,231,7,195,149,225,107,5,55,211,67,30,201,17,160,6,101,64,87,143,40,59,115,227,207,73,0,5,91,162,187,142}, {17,86,254,101,176,237,88,187,243,99,149,255,124,93,69,248,171,203,19,213,190,245,75,25,83,181,117,144,218,82,121,26}, {139,184,50,62,27,42,129,158,219,122,14,180,41,218,187,135,117,237,48,132,158,3,178,138,76,244,209,63,225,8,179,238}, {71,118,155,217,83,18,203,70,27,61,233,165,80,143,26,57,35,152,96,233,69,31,223,104,200,12,127,41,191,109,56,46}, {198,232,5,135,171,146,223,113,180,138,45,193,51,111,225,194,11,81,163,0,122,104,169,237,22,155,173,89,21,135,158,212}, {29,165,245,105,191,252,56,9,230,85,106,214,4,156,241,120,250,149,185,228,199,53,147,65,48,249,97,184,253,226,67,90}, {58,90,69,23,123,36,76,162,209,32,248,173,129,89,16,75,177,23,87,44,95,37,211,182,116,51,141,33,77,167,36,148}, {226,132,179,45,234,95,184,109,127,150,20,99,189,38,171,130,210,62,141,238,136,244,10,80,131,220,2,206,123,14,194,239}, {116,209,150,99,202,137,244,4,195,240,70,141,228,205,55,219,105,3,100,170,156,115,196,178,24,213,164,102,59,231,112,74}, {170,8,254,52,167,13,153,90,46,212,42,116,9,162,65,151,35,201,224,17,30,226,98,58,239,42,150,85,204,174,133,25}, {76,34,82,220,62,29,207,67,121,172,58,183,97,255,28,191,233,79,180,125,47,86,0,145,112,72,127,252,7,157,50,215}, {140,195,158,122,238,189,84,253,225,159,23,234,131,83,114,19,143,73,112,246,198,160,187,232,169,12,181,30,224,40,95,249}, {246,91,110,43,145,106,178,51,147,11,108,197,216,44,177,243,92,166,9,60,70,104,37,120,242,94,194,137,109,67,183,145}, 
	  {14,185,212,5,229,20,128,36,77,93,247,55,153,3,208,123,196,52,217,149,229,207,74,2,52,203,231,61,205,166,120,27}, {64,103,134,198,172,73,164,221,200,176,118,33,69,139,81,103,31,133,183,42,21,176,164,217,152,22,80,44,248,16,88,215}, {125,236,30,45,247,97,59,241,136,211,18,188,230,252,157,220,15,235,99,82,250,126,66,91,111,132,188,100,126,191,235,155}, {161,181,216,78,154,119,28,107,43,63,161,128,91,170,10,61,201,245,173,117,56,31,199,9,254,208,161,1,143,55,35,72}, {7,93,56,111,12,210,183,147,4,83,236,104,50,40,113,146,128,47,2,206,139,239,105,192,49,78,31,222,237,170,113,132}, {229,202,144,255,194,84,226,252,169,217,192,26,178,205,235,77,94,166,68,154,220,59,182,119,228,96,177,88,68,200,20,221}, {80,122,36,166,68,20,52,115,75,34,138,224,70,156,190,215,21,228,109,38,87,16,235,1,148,19,136,247,151,43,79,249}, {175,24,188,130,238,103,139,202,126,94,19,151,120,86,2,62,134,186,197,248,174,124,73,114,167,214,66,53,8,184,98,144}, {100,221,61,4,213,179,41,233,10,175,246,208,39,54,172,255,32,118,51,146,25,190,209,39,243,25,107,192,164,251,60,29}, {242,140,112,157,90,29,152,249,57,160,98,130,243,107,198,100,142,223,81,10,102,242,137,78,156,127,229,37,204,121,218,175}, {11,53,206,245,74,123,219,110,84,196,26,64,185,92,159,18,240,41,177,219,163,32,85,48,172,58,145,75,85,133,49,111}, {189,86,39,168,195,15,65,186,137,46,227,167,14,232,48,125,189,152,71,94,128,63,232,181,216,0,101,236,154,68,16,162}, {147,234,23,135,102,47,231,171,39,213,102,148,34,218,76,207,60,1,110,200,253,15,153,40,106,197,13,174,1,240,188,211}, {66,121,204,180,253,151,89,129,71,254,119,81,179,113,138,27,84,230,214,37,182,116,142,247,71,134,222,119,203,106,35,87}, {193,45,79,6,60,32,242,210,3,190,54,246,204,6,168,155,244,129,146,103,53,24,208,6,89,186,28,93,251,72,126,142}, {214,97,160,224,115,176,108,12,144,159,18,131,65,223,96,38,199,74,13,193,159,227,82,124,163,234,47,148,21,168,230,54} },
	{ {175,21,137,238,18,201,170,68,165,240,100,31,126,173,22,210,47,153,223,19,44,239,169,60,114,250,14,175,46,133,185,214}, {54,109,209,88,148,38,83,112,23,213,74,152,194,231,101,248,137,116,62,199,134,150,100,26,215,144,107,196,228,67,8,63}, {196,73,33,228,120,191,219,35,130,181,50,252,61,67,41,80,177,79,33,253,90,7,232,189,94,39,77,159,97,147,249,90}, {247,183,158,48,7,100,254,142,56,225,123,8,145,110,202,166,2,236,185,159,122,175,72,209,129,200,52,244,23,32,109,157}, {92,143,107,131,243,166,76,195,155,2,207,88,189,240,25,152,217,82,17,106,221,57,110,19,163,4,218,184,137,191,223,206}, {13,42,235,64,203,28,15,91,236,103,136,174,14,118,54,182,69,126,141,48,193,36,248,86,142,114,236,80,95,49,128,59}, {174,224,193,83,116,179,138,251,47,67,25,247,96,80,209,94,0,198,171,232,87,152,227,204,180,69,46,119,170,253,106,14}, {123,71,149,164,38,233,52,124,168,211,112,195,231,162,142,222,111,9,242,75,131,3,99,124,12,31,155,211,6,39,180,239}, {207,25,55,97,2,201,105,188,11,84,151,59,1,49,35,226,179,58,45,162,29,187,60,169,217,251,197,62,229,117,74,154}, {104,245,221,133,194,156,78,144,241,32,245,184,129,101,176,19,81,153,220,98,210,253,113,25,54,144,108,23,186,140,214,93}, {45,182,15,171,216,65,21,43,227,96,41,73,145,215,29,237,134,206,119,17,70,156,140,241,192,78,130,69,163,48,202,27}, {144,81,117,33,90,255,119,205,135,161,16,197,222,114,55,168,66,3,195,147,34,229,50,121,40,226,5,245,89,235,13,134}, {208,167,240,51,129,178,150,58,187,70,202,107,0,9,121,192,249,108,88,239,185,85,10,181,102,206,173,151,36,83,250,172}, {2,63,142,213,11,71,28,99,249,1,125,235,158,183,87,228,42,160,27,171,124,219,136,63,160,26,110,220,196,122,102,59}, {133,95,230,189,108,234,203,218,50,85,172,63,80,206,94,17,127,225,37,74,58,20,198,249,234,73,93,57,165,24,188,223}, {175,38,21,159,86,43,164,132,115,143,211,32,22,219,151,76,190,102,212,253,113,166,99,147,46,13,132,255,141,79,161,40}, 
	  {246,115,199,53,122,243,16,77,9,229,176,105,240,131,170,202,50,139,64,152,237,39,5,184,116,177,193,208,7,49,243,111}, {81,68,225,104,188,218,97,170,191,37,91,157,53,3,68,35,118,244,10,93,204,174,55,242,86,226,36,101,231,150,28,182}, {9,126,168,30,1,139,62,149,252,121,68,196,162,111,212,85,233,163,178,47,81,127,66,140,29,153,20,125,62,200,91,132}, {212,186,150,254,112,179,241,24,53,222,137,27,225,254,145,184,58,130,33,194,16,224,214,107,205,248,79,172,161,252,13,71}, {229,41,92,233,45,66,160,87,110,237,18,42,82,124,21,98,44,248,109,216,146,0,163,11,51,96,143,5,84,44,213,222}, {103,57,17,135,79,208,128,34,201,157,182,148,214,70,10,159,221,76,4,95,156,123,72,177,135,185,234,216,112,192,96,122}, {164,179,119,204,191,10,173,219,3,75,56,104,199,115,205,88,176,139,189,235,210,61,230,86,198,14,127,65,31,138,54,153}, {1,149,223,28,52,245,118,61,146,132,251,15,244,37,172,232,18,125,30,53,113,20,36,246,24,155,43,246,166,207,178,22}, {197,70,251,115,102,156,19,209,46,190,30,162,91,49,135,73,186,64,241,165,136,221,193,145,116,75,105,192,93,8,72,217}, {98,129,40,84,183,140,198,95,233,114,227,125,218,154,238,106,0,146,200,12,87,101,169,57,238,224,211,34,134,242,149,104}, {190,12,208,169,230,71,37,83,167,12,78,65,178,4,85,212,120,227,108,48,154,252,31,77,5,139,176,51,255,117,41,167}, {236,59,154,24,215,4,126,255,181,138,242,201,99,20,171,55,42,92,217,183,74,204,128,186,158,97,22,82,155,66,188,56}, {45,143,89,133,244,56,100,148,26,65,106,39,147,250,234,194,160,82,173,34,117,231,18,69,226,243,109,200,216,128,26,239}, {103,254,195,77,111,158,174,228,51,215,157,187,75,61,113,130,23,247,138,6,60,146,105,40,197,121,60,165,6,230,89,210}, {168,7,224,29,43,203,16,190,118,6,90,232,44,141,180,11,67,199,123,237,164,250,213,52,177,47,8,246,141,103,30,120}, {203,64,120,181,94,238,136,35,78,247,167,127,32,220,89,207,151,38,98,187,84,27,131,92,220,148,76,205,72,180,15,161} },
	{ {137,241,39,152,214,191,88,224,22,182,87,150,27,173,71,123,184,25,65,178,8,238,34,209,100,176,66,194,213,43,117,245}, {23,75,127,103,14,65,141,118,47,244,128,212,237,103,46,239,205,223,135,92,114,153,58,121,252,145,19,91,237,108,11,96}, {231,180,203,56,177,34,239,206,164,75,105,8,195,159,17,139,54,88,164,253,219,195,110,166,12,49,130,161,35,175,150,220}, {116,159,7,220,246,102,78,5,54,173,39,115,57,225,188,76,6,149,37,47,84,18,24,75,187,225,201,5,81,189,25,63}, {42,91,107,70,197,166,134,152,192,227,255,146,67,131,249,170,120,232,199,104,129,172,211,244,40,79,118,248,140,238,200,128}, {193,254,29,141,48,19,248,37,110,89,29,98,178,22,41,92,247,186,58,123,179,230,62,155,137,217,172,60,21,76,112,43}, {132,217,168,122,227,94,62,209,124,15,216,163,203,105,51,129,70,28,212,144,33,3,102,127,28,11,99,149,52,181,158,249}, {80,17,61,204,85,215,131,170,235,49,139,77,4,220,172,194,155,11,96,79,160,53,229,202,89,253,190,209,125,222,65,86}, {233,187,113,44,156,183,74,9,103,154,239,186,59,250,107,15,226,141,240,206,221,115,167,182,111,44,73,93,236,7,211,167}, {35,101,145,243,5,112,27,252,197,69,34,118,132,149,32,122,185,42,109,22,0,148,41,17,197,142,164,30,131,38,81,142}, {53,224,199,24,175,231,57,145,85,245,20,160,45,199,66,73,211,164,60,136,189,70,242,86,55,244,214,102,154,193,106,14}, {178,162,71,88,97,136,191,120,222,169,93,207,218,87,233,244,99,3,238,91,175,130,9,155,97,179,2,23,203,69,216,251}, {140,122,8,207,218,13,160,40,4,107,143,9,180,0,157,51,19,147,194,49,214,230,204,25,120,222,66,117,228,174,54,26}, {33,239,99,151,61,249,83,184,52,229,74,192,127,81,136,113,179,221,69,119,33,108,82,169,39,186,140,78,42,126,188,112}, {211,77,47,189,128,32,201,132,152,255,24,101,170,247,30,163,91,80,251,17,185,143,61,236,195,51,250,165,147,6,93,233}, {117,157,171,228,109,53,237,114,90,209,36,60,215,49,225,205,236,38,134,68,159,247,2,133,74,12,210,88,242,219,52,154}, 
	  {66,18,253,5,73,222,20,175,11,166,198,121,153,3,64,106,124,9,168,200,96,31,117,193,100,157,111,32,99,177,72,202}, {196,137,90,178,147,162,98,68,46,139,80,234,183,132,78,191,146,57,226,210,45,217,173,153,228,40,255,62,198,16,133,36}, {238,38,218,80,28,197,214,245,188,241,108,18,95,35,212,201,25,174,110,73,21,129,86,59,21,124,180,143,235,118,84,249}, {101,56,123,251,105,63,135,87,156,30,54,196,168,253,116,100,43,82,241,149,0,204,235,10,206,79,215,1,57,221,169,147}, {208,7,166,144,229,12,40,119,4,254,144,223,89,10,55,179,213,137,2,183,142,161,101,243,165,135,106,94,158,46,191,22}, {185,156,72,47,205,169,224,184,55,176,126,70,46,207,151,19,122,231,94,36,60,50,15,113,28,64,232,172,31,92,69,110}, {32,231,114,23,95,128,150,76,235,100,20,159,246,134,79,162,29,67,190,223,125,232,196,76,155,201,16,188,250,127,243,206}, {85,130,240,195,213,68,26,111,208,138,63,33,186,109,228,200,146,185,50,243,167,85,174,219,254,44,84,145,217,8,138,45}, {226,41,173,4,82,248,189,162,10,241,193,219,120,41,71,97,0,113,10,151,24,104,37,67,130,98,119,37,104,181,78,163}, {148,105,56,158,141,102,48,124,38,81,94,174,144,2,215,236,171,240,106,72,248,190,138,1,182,208,165,229,59,198,20,232}, {13,245,190,31,218,16,199,230,153,205,50,18,207,87,157,16,82,131,198,43,210,56,89,237,150,72,26,192,135,154,48,92}, {203,96,125,63,115,171,90,255,68,133,116,167,251,77,183,123,27,220,177,95,158,14,116,223,194,52,12,242,115,64,212,126}, {138,27,252,182,234,55,146,26,3,181,225,29,58,196,136,48,233,61,1,142,121,227,171,31,107,125,95,176,1,252,30,170}, {74,165,6,83,134,224,75,109,210,59,247,126,104,7,226,148,71,187,111,36,254,200,64,6,161,250,216,148,67,202,108,227}, {221,112,208,152,42,13,168,192,156,97,143,44,161,242,35,93,163,246,51,213,77,103,129,45,204,140,39,119,83,160,15,53}, {176,50,65,187,98,246,121,34,234,14,84,177,216,62,114,202,13,133,83,184,151,23,234,180,86,58,230,21,181,240,90,139} }	
	};

	// = = = = = processing = = = = = //
	// define map_m (to save the index of tiled dither array)
	int	height_map_m	=	cvCeil(((float)src1b.rows)/((float)sizeDitherArray));
	int	width_map_m		=	cvCeil(((float)src1b.cols)/((float)sizeDitherArray));
	Mat	map_m1b(Size(width_map_m+1,height_map_m+1),CV_8UC1);
	map_m1b.setTo(255);	// 255 denotes nothing
	// get entire dither array
	srand(3);
	bool	doneflag;
	for(int i=0; i<src1b.rows; i+=sizeDitherArray){
		for(int j=0; j<src1b.cols; j+=sizeDitherArray){

			uchar	&current_DA_index	=	map_m1b.ptr<uchar>(i/sizeDitherArray+1)[j/sizeDitherArray+1];	// current position is (+1,+1) rather than (0,0)
			uchar	left_DA_index		=	map_m1b.ptr<uchar>(i/sizeDitherArray+1)[j/sizeDitherArray];
			uchar	upper_DA_index		=	map_m1b.ptr<uchar>(i/sizeDitherArray)[j/sizeDitherArray+1];

			doneflag=false;	// check whether the process is done
			while(doneflag != true){	// 不可與之前的DA label相同

				// get DA's label
				current_DA_index = rand() % nDitherArray;	// 0 to 3

				// copy DA to entireDA
				if((current_DA_index != left_DA_index) && (current_DA_index != upper_DA_index)){	// then do halftoning

					doneflag = true;

					// halftoning
					for(int m=0; m<sizeDitherArray; m++){ 
						for(int n=0; n<sizeDitherArray; n++){
							if(i+m>=0 && i+m<src1b.rows && j+n>=0 && j+n<src1b.cols){
								if(src1b.ptr<uchar>(i+m)[j+n] < DAs[current_DA_index][m][n]){
									dst1b.ptr<uchar>(i+m)[j+n] = 0;
								}else{
									dst1b.ptr<uchar>(i+m)[j+n] = 255;
								}
							}else{
								// do nothing
							}
						}
					}
				}else{
					doneflag=false;
				}
			}
		}
	}

	return true;
}


//////////////////////////////////////////////////////////////////////////
//	dot diffusion
//////////////////////////////////////////////////////////////////////////

//	Dot diffusion proposed by Knuth
bool pixkit::halftoning::dotdiffusion::Knuth1987(const cv::Mat &src, cv::Mat &dst)
{
//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::ordereddithering::Knuth1987] accepts only grayscale image");
		return false;
	}
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[halftoning::ordereddithering::Knuth1987] image is empty");
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst1f	=	src.clone();
	tdst1f.convertTo(tdst1f, CV_32FC1); 

	const int sizeOfClassMatrix = 8;
	std::vector< std::vector< int > >CM_reg(3, std::vector< int >(3));

	const int ClassMatrix[8][8] = {
		34,	48,	40,	32,	29,	15,	23,	31,
		42,	58,	56,	53,	21,	5,	7,	10,
		50,	62,	61,	45,	13,	1,	2,	18,
		38,	46,	54,	37,	25,	17,	9,	26,
		28,	14,	22,	30,	35,	49,	41,	33,
		20,	4,	6,	11,	43,	59,	57,	52,
		12,	0,	3,	19,	51,	63,	60,	44,
		24,	16,	8,	27,	39,	47,	55,	36	
	};

	// diffusion weight in Dot Diffusion ---------------------------------------
	const float DiffusionWeight[3][3] = {
		1,	2,	1,
		2,	0,	2,
		1,	2,	1	
	};

	for(int k=0; k<sizeOfClassMatrix*sizeOfClassMatrix; k++){

		for(int m=0; m<sizeOfClassMatrix; m++){
			for(int n=0; n<sizeOfClassMatrix; n++){
				if(ClassMatrix[m][n] == k){

					for(int i=0; i<src.rows; i+=sizeOfClassMatrix){
						if (i+m >= src.rows)	continue;
		
						for(int j=0; j<src.cols; j+=sizeOfClassMatrix){
							if (j+n >= src.cols)	continue;

							// ED part
							float error;
							if(tdst1f.ptr<float>(i+m)[j+n] >= 128){	
								error = tdst1f.ptr<float>(i+m)[j+n]-255;	//error value
								tdst1f.ptr<float>(i+m)[j+n] = 255;
							}
							else{				
								error = tdst1f.ptr<float>(i+m)[j+n];	//error value
								tdst1f.ptr<float>(i+m)[j+n] = 0;
							}

							int X_index,Y_index;
							// ClassMatrix_reg (3*3 for diffusion_weight_reg) initialization
							for(int x = -1;x <= 1;x++){
								for(int y = -1;y <= 1;y++){
									if((m+x) >= sizeOfClassMatrix)	{X_index = m+x-sizeOfClassMatrix;}
									else if((m+x) < 0)				{X_index = m+x+sizeOfClassMatrix;}
									else							{X_index = m+x;}

									if((n+y) >= sizeOfClassMatrix)	{Y_index = n+y-sizeOfClassMatrix;}
									else if((n+y) < 0)				{Y_index = n+y+sizeOfClassMatrix;}
									else							{Y_index = n+y;}

									CM_reg[x+1][y+1] = ClassMatrix[X_index][Y_index];
								}
							}

							// make sure that error diffusion processing whether it is "over" the "size range" of the image
							float sum = 0.0;

							for(int x=-1; x<=1; x++){
								if( (m+i+x) >= src.rows || (m+i+x) < 0)		continue;
								for(int y=-1; y<=1; y++){
									if((n+j+y) >= src.cols || (n+j+y) < 0 )		continue;
									if(CM_reg[x+1][y+1] > CM_reg[1][1]){
										sum += DiffusionWeight[x+1][y+1];
									}	
								}
							}

							// error diffusion processing
							for(int x=-1; x<=1; x++){
								if( (m+i+x) >= src.rows || (m+i+x) < 0)		continue;
								for(int y=-1; y<=1; y++){
									if( (n+j+y) >= src.cols || (n+j+y) < 0 )		continue;
									if (x!=0 || y!=0){
										if(CM_reg[x+1][y+1] > CM_reg[1][1]){
											tdst1f.ptr<float>(m+i+x)[n+j+y]	+=	(error * DiffusionWeight[x+1][y+1] / sum);
										}
									}	
								}
							}				
						}
					}
				}
			}
		}
	}
	tdst1f.convertTo(dst,CV_8UC1);
	return true;
}

//	Dot diffusion proposed by Mese and Vaidyanathan
bool pixkit::halftoning::dotdiffusion::MeseVaidyanathan2000(const cv::Mat &src, cv::Mat &dst, int ClassMatrixSize)
{
//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::ordereddithering::MeseVaidyanathan2000] accepts only grayscale image");
		return false;
	}
	if(src.empty()){
		CV_Error(CV_HeaderIsNull,"[halftoning::ordereddithering::MeseVaidyanathan2000] image is empty");
		return false;
	}
	if(ClassMatrixSize!=8 && ClassMatrixSize!=16){
		CV_Error(CV_StsBadArg,"[halftoning::ordereddithering::MeseVaidyanathan2000] BlockSize should be 8 or 16.");
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	cv::Mat	tdst1f	=	src.clone();
	tdst1f.convertTo(tdst1f,CV_32FC1); 
	const int sizeOfClassMatrix = ClassMatrixSize;
	std::vector< std::vector< int > >CM_reg(3, std::vector< int >(3));
	std::vector< std::vector<int> > ClassMatrix( sizeOfClassMatrix, std::vector<int>(sizeOfClassMatrix) );

	// diffusion weight in Dot Diffusion ---------------------------------------
	const float DiffusionWeight[3][3] = {
		1,	2,	1,
		2,	0,	2,
		1,	2,	1	};

	switch(ClassMatrixSize){
		case 8:{
			const int coe[8][8] = {
				47,	31,	51,	24,	27,	45,	5,	21,
				37,	63,	53,	11,	22,	4,	1,	33,
				61,	0,	57,	16,	26,	29,	46,	8,
				20,	14,	9,	62,	18,	41,	38,	6,
				17,	13,	25,	15,	55,	48,	52,	58,
				3,	7,	2,	32,	30,	34,	56,	60,
				28,	40,	36,	39,	49,	43,	35,	10,
				54,	23,	50,	12,	42,	59,	44,	19	};

				for(int i=0;i<sizeOfClassMatrix;i++){
					for(int j=0;j<sizeOfClassMatrix;j++){
						ClassMatrix[i][j]=coe[i][j];
					}
				}
				break;
		}

		case 16:{
			const int coe[16][16] = {
				207,	0,		13,		17,		28,		55,		18,		102,	81,		97,		74,		144,	149,	169,	170,	172,
				3,		6,		23,		36,		56,		50,		65,		87,		145,	130,	137,	158,	182,	184,	195,	221,
				7,		14,		24,		37,		67,		69,		86,		5,		106,	152,	150,	165,	183,	192,	224,	1,
				15,		26,		43,		53,		51,		101,	115,	131,	139,	136,	166,	119,	208,	223,	226,	4,
				22,		39,		52,		71,		84,		103,	164,	135,	157,	173,	113,	190,	222,	225,	227,	16,
				40,		85,		72,		83,		104,	117,	167,	133,	168,	180,	200,	219,	231,	228,	12,		21,
				47,		120,	54,		105,	123,	132,	146,	176,	179,	202,	220,	230,	245,	2,		20,		41,
				76,		73,		127,	109,	138,	134,	178,	181,	206,	196,	229,	244,	246,	19,		42,		49,
				80,		99,		112,	147,	142,	171,	177,	203,	218,	232,	243,	248,	247,	33,		48,		68,
				108,	107,	140,	143,	185,	163,	204,	217,	233,	242,	249,	255,	44,		45,		70,		79,
				110,	141,	88,		75,		175,	205,	214,	234,	241,	250,	254,	38,		46,		77,		116,	100,	
				111,	148,	160,	174,	201,	215,	235,	240,	251,	252,	253,	61,		62,		93,		94,		125,
				151,	159,	189,	199,	197,	216,	236,	239,	25,		31,		60,		82,		92,		95,		124,	114,
				156,	188,	191,	209,	213,	237,	238,	29,		32,		59,		64,		91,		118,	78,		128,	155,	
				187,	194,	198,	212,	9,		10,		30,		35,		58,		63,		90,		96,		122,	129,	154,	161,
				193,	210,	211,	8,		11,		27,		34,		57,		66,		89,		98,		121,	126,	153,	162,	186		};

			for(int i=0;i<sizeOfClassMatrix;i++){
				for(int j=0;j<sizeOfClassMatrix;j++){
					ClassMatrix[i][j]=coe[i][j];
				}
			}
			break;
		}
	}

	for(int k=0; k<sizeOfClassMatrix*sizeOfClassMatrix; k++){

		for(int m=0; m<sizeOfClassMatrix; m++){
			for(int n=0; n<sizeOfClassMatrix; n++){
				if(ClassMatrix[m][n] == k){

					for(int i=0; i<src.rows; i+=sizeOfClassMatrix){
						if (i+m >= src.rows)	continue;
						for(int j=0; j<src.cols; j+=sizeOfClassMatrix){
							if (j+n >= src.cols)	continue;

							//ED part--------------------	
							float error;
							if(tdst1f.ptr<float>(i+m)[j+n] >= 128){	
								error = tdst1f.ptr<float>(i+m)[j+n]-255;	//error value
								tdst1f.ptr<float>(i+m)[j+n] = 255;
							}
							else{				
								error = tdst1f.ptr<float>(i+m)[j+n];	//error value
								tdst1f.ptr<float>(i+m)[j+n] = 0;
							}

							int X_index,Y_index;

							//ClassMatrix_reg(3*3 for diffusion_weight_reg) initialization-------
							for(int x = -1;x <= 1;x++){
								for(int y = -1;y <= 1;y++){
									if((m+x) >= sizeOfClassMatrix)	{X_index = m+x-sizeOfClassMatrix;}
									else if((m+x) < 0)				{X_index = m+x+sizeOfClassMatrix;}
									else							{X_index = m+x;}

									if((n+y) >= sizeOfClassMatrix)	{Y_index = n+y-sizeOfClassMatrix;}
									else if((n+y) < 0)				{Y_index = n+y+sizeOfClassMatrix;}
									else							{Y_index = n+y;}

									CM_reg[x+1][y+1] = ClassMatrix[X_index][Y_index];
								}
							}

							//make sure that error diffusion processing whether it is "over" the "size range" of the image
							float sum = 0.0;

							for(int x=-1; x<=1; x++){
								if( (m+i+x) >= src.rows || (m+i+x) < 0)		continue;
								for(int y=-1; y<=1; y++){
									if((n+j+y) >= src.cols || (n+j+y) < 0 )		continue;
									if(CM_reg[x+1][y+1] > CM_reg[1][1]){
										sum += DiffusionWeight[x+1][y+1];
									}	
								}
							}

							//error diffusion processing
							for(int x=-1; x<=1; x++){
								if( (m+i+x) >= src.rows || (m+i+x) < 0)		continue;
								for(int y=-1; y<=1; y++){
									if( (n+j+y) >= src.cols || (n+j+y) < 0 )		continue;
									if (x!=0 || y!=0){
										if(CM_reg[x+1][y+1] > CM_reg[1][1]){
											tdst1f.ptr<float>(m+i+x)[n+j+y]	+=	(error * DiffusionWeight[x+1][y+1] / sum);
										}
									}	
								}
							}				
						}
					}
				}
			}
		}
	}
	tdst1f.convertTo(dst,CV_8UC1);
	return true;
}

//	Dot diffusion proposed by Guo and Liu
bool pixkit::halftoning::dotdiffusion::GuoLiu2009(const cv::Mat &src, cv::Mat &dst, const int ClassMatrixSize){
	
	//////////////////////////////////////////////////////////////////////////
	// exception
	if(ClassMatrixSize!=8&&ClassMatrixSize!=16){
//		CV_Error(CV_StsBadArg,"[halftoning::dotdiffusion::GuoLiu2009] accepts only 8 and 16 these two class matrix sizes");
		return false;
	}
	if(src.type()!=CV_8U){
//		CV_Error(CV_BadNumChannels,"[halftoning::dotdiffusion::GuoLiu2009] accepts only grayscale image");
		return false;
	}
	if(src.empty()){
		return false;
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
	cv::Mat	tdst1d	=	src.clone();
	tdst1d.convertTo(tdst1d,CV_64FC1);
	// get point list
	std::vector<cv::Point>	pointList(ClassMatrixSize*ClassMatrixSize);
	for(int m=0;m<ClassMatrixSize;m++){
		for(int n=0;n<ClassMatrixSize;n++){
			pointList[CM[m][n]]	=	cv::Point(n,m);
		}
	}
	// 進行dot_diffusion
	int		OSCW=DiffusionMaskSize/2;
	int		OSCL=DiffusionMaskSize/2;
	int		number=0;
	while(number!=ClassMatrixSize*ClassMatrixSize){
		for(int i=pointList[number].y;i<src.rows;i+=ClassMatrixSize){
			for(int j=pointList[number].x;j<src.cols;j+=ClassMatrixSize){
				// 取得error
				double	error;
				if(tdst1d.ptr<double>(i)[j]<(float)(nColors-1.)/2.){
					error=tdst1d.ptr<double>(i)[j];
					tdst1d.ptr<double>(i)[j]=0.;
				}else{
					error=tdst1d.ptr<double>(i)[j]-(nColors-1.);
					tdst1d.ptr<double>(i)[j]=(nColors-1.);
				}
				// 取得分母
				double	fm=0.;
				for(int m=-OSCW;m<=OSCW;m++){
					for(int n=-OSCL;n<=OSCL;n++){
						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// 在影像範圍內
							if(CM[(i+m)%ClassMatrixSize][(j+n)%ClassMatrixSize]>number){		// 可以擴散的區域
								fm+=DW[m+OSCW][n+OSCL];
							}
						}
					}
				}
				// 進行擴散
				for(int m=-OSCW;m<=OSCW;m++){
					for(int n=-OSCL;n<=OSCL;n++){
						if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// 在影像範圍內
							if(CM[(i+m)%ClassMatrixSize][(j+n)%ClassMatrixSize]>number){		// 可以擴散的區域								
								tdst1d.ptr<double>(i+m)[j+n]+=error*DW[m+OSCW][n+OSCL]/fm;
							}
						}
					}
				}
			}
		}
		number++;
	}

	//////////////////////////////////////////////////////////////////////////
	dst	=	tdst1d;
	dst.convertTo(dst,src.type());
#if defined(_DEBUG)
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			CV_Assert(dst.ptr<uchar>(i)[j]==0||dst.ptr<uchar>(i)[j]==(nColors-1));
		}
	}
#endif

	return true;
}

//	Dot diffusion proposed by Lippens and Philips
bool pixkit::halftoning::dotdiffusion::LippensPhilips2007(const cv::Mat &src, cv::Mat &dst){

	//////////////////////////////////////////////////////////////////////////
	// exception
	if(src.type()!=CV_8U){
		CV_Error(CV_BadNumChannels,"[halftoning::ErrorDiffusion::Hft_EDF_Ostromoukhov] accepts only grayscale image");
	}
	dst.create(src.size(),src.type());

	float	hysteresis=0.;	// 仍在尋找問題, 此為非0之結果不同於該論文, 故先行設定為0

	// = = = = = Get entire CM = = = = = //
	int	order_global[8][8] = {{0,1,3,0,1,3,0,1},{3,3,2,0,1,2,3,3},{2,2,3,0,1,3,2,2},{0,1,2,0,1,2,0,1},{3,0,1,3,3,0,1,3},{2,0,1,2,2,0,1,2},{0,1,0,1,0,1,0,1},{3,3,3,3,3,3,3,3}};
	int	oriCM[16][16] = {	
	{0,		1,	14,	15,	16,		19,		20,		21,		234,	235,	236,	239,	240,	241,	254,	255},
	{3,		2,	13,	12,	17,		18,		23,		22,		233,	232,	237,	238,	243,	242,	253,	252},
	{4,		7,	8,	11,	30,		29,		24,		25,		230,	231,	226,	225,	244,	247,	248,	251},
	{5,		6,	9,	10,	31,		28,		27,		26,		229,	228,	227,	224,	245,	246,	249,	250},
	{58,	57,	54,	53,	32,		35,		36,		37,		218,	219,	220,	223,	202,	201,	198,	197},
	{59,	56,	55,	52,	33,		34,		39,		38,		217,	216,	221,	222,	203,	200,	199,	196},
	{60,	61,	50,	51,	46,		45,		40,		41,		214,	215,	210,	209,	204,	205,	194,	195},
	{63,	62,	49,	48,	47,		44,		43,		42,		213,	212,	211,	208,	207,	206,	193,	192},
	{64,	67,	68,	69,	122,	123,	124,	127,	128,	131,	132,	133,	186,	187,	188,	191},
	{65,	66,	71,	70,	121,	120,	125,	126,	129,	130,	135,	134,	185,	184,	189,	190},
	{78,	77,	72,	73,	118,	119,	114,	113,	142,	141,	136,	137,	182,	183,	178,	177},
	{79,	76,	75,	74,	117,	116,	115,	112,	143,	140,	139,	138,	181,	180,	179,	176},
	{80,	81,	94,	95,	96,		97,		110,	111,	144,	145,	158,	159,	160,	161,	174,	175},
	{83,	82,	93,	92,	99,		98,		109,	108,	147,	146,	157,	156,	163,	162,	173,	172},
	{84,	87,	88,	91,	100,	103,	104,	107,	148,	151,	152,	155,	164,	167,	168,	171},
	{85,	86,	89,	90,	101,	102,	105,	106,	149,	150,	153,	154,	165,	166,	169,	170}};
	// set initial CM ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int	CMs[4][16][16];
	// get type 3	
	for(int i=0;i<16;i++){
		for(int j=0;j<16;j++){
			CMs[3][i][j]=oriCM[i][j];
		}
	}
	// get type 2	
	for(int i=0;i<16;i++){
		for(int j=0;j<16;j++){
			CMs[2][i][j]=oriCM[15-i][15-j];
		}
	}
	// get type 1	
	for(int i=0;i<16;i++){
		for(int j=0;j<16;j++){
			CMs[1][i][j]=oriCM[15-j][15-i];
		}
	}
	// get type 0	
	for(int i=0;i<16;i++){
		for(int j=0;j<16;j++){
			CMs[0][i][j]=oriCM[j][i];
		}
	}
	// get 128 CM
	int	CM128[128][128];
	for(int i=0;i<128;i+=16){
		for(int j=0;j<128;j+=16){
			for(int m=0;m<16;m++){
				for(int n=0;n<16;n++){
					CM128[i+m][j+n]=CMs[order_global[i/16][j/16]][m][n];
				}
			}
		}
	}

	// get entire CM
	std::vector< std::vector< int > > entireCM (src.rows, std::vector< int >(src.cols));

	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			entireCM[i][j]=CM128[i%128][j%128];
		}
	}
	// get src(temp) image
	std::vector< std::vector<double> > RegImage(src.rows, std::vector<double>(src.cols) );	
	//copy Input image to RegImage
	for (int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			RegImage[i][j] = static_cast<double>(src.data[i* src.cols + j]);
		}
	}

	// = = = = = set diffused weighting = = = = = //
	float	coe_sum=32;
	float	coe[5][5]={{1,2,3,2,1},{2,0,0,0,2},{3,0,0,0,3},{2,0,0,0,2},{1,2,3,2,1}};
	int		OSCW=2;

	// = = = = = 進行dot_diffusion = = = = = //
	int		number=0;
	while(number!=256){	// 256=CMsize*CMsize
		for(int i=0;i<src.rows;i++){
			for(int j=0;j<src.cols;j++){
				if(entireCM[i][j]==number){

					// get 分母
					float	b_fm=0.,g_fm=0.;	// binary的分母 and grayscale的分母, sum(coe)=b_fm+g_fm;			
					for(int m=-OSCW;m<=OSCW;m++){
						for(int n=-OSCW;n<=OSCW;n++){
							if(i+m>=0 && i+m<src.rows && j+n>=0 && j+n<src.cols){	// 在影像範圍內
								if(entireCM[i+m][j+n]<number){		// 範圍內擴散過的區域
									b_fm += coe[m+OSCW][n+OSCW];
								}
							}
						}
					}
					g_fm=coe_sum-b_fm;

					// 取得周圍的擴散補償
					float	comp=0.;	// 補償
					if(b_fm!=0){
						for(int m=-OSCW;m<=OSCW;m++){
							for(int n=-OSCW;n<=OSCW;n++){
								if(i+m>=0 && i+m<src.rows && j+n>=0 && j+n<src.cols){	// 在影像範圍內
									if(entireCM[i+m][j+n]<number){		// 範圍內擴散過的區域
										comp += coe[m+OSCW][n+OSCW] / b_fm * (RegImage[i+m][j+n] < 128 ? -1 : 1);	// 待確認
									}
								}
							}
						}
						comp*=128.;	// 待確認
					}else{
						comp=0.;
					}

					// 取得error, 並取得halftone 輸出.
					double	error;
					if(RegImage[i][j] + comp*hysteresis < 128){
						error = RegImage[i][j];
						//src.data[i*src.cols+j] = 0.;
						dst.data[i*dst.cols+j] = 0;
					}else{
						error = RegImage[i][j]-255.;
						//src.data[i*src.cols+j] = 255.;
						dst.data[i*dst.cols+j] = 255;
					}
					// 進行擴散
					for(int m=-OSCW;m<=OSCW;m++){
						for(int n=-OSCW;n<=OSCW;n++){
							if(i+m>=0&&i+m<src.rows&&j+n>=0&&j+n<src.cols){	// 在影像範圍內
								if(entireCM[i+m][j+n]>number){		// 可以擴散的區域								
									RegImage[i+m][j+n] += error * coe[m+OSCW][n+OSCW] / g_fm;
								}
							}
						}
					}
				}
			}
		}
		number++;
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
//	ungrouped
//////////////////////////////////////////////////////////////////////////
bool pixkit::halftoning::ungrouped::generateTwoComponentGaussianModel(cv::Mat &dst1d,float k1,float k2,float sd1,float sd2){

	const	float	R	=	9.5;
	const	float	D	=	300;
	const	float	S	=	R*D;
	const	float	pi	=	3.141592653589793;
	const	float	fm	=	180.*180./((pi*D)*(pi*D));
	const	int		size=	21;
	const	int		h_size=	size/2;

	//////////////////////////////////////////////////////////////////////////
	///// get cpp
	dst1d.create(Size(size,size),CV_64FC1);
	for(int m=-h_size;m<=h_size;m++){
		for(int n=-h_size;n<=h_size;n++){
			float	x		=	(180.*(float)m)/(pi*S);
			float	y		=	(180.*(float)n)/(pi*S);
			float	chh		=	k1*std::expf(-(x*x+y*y)/(2.*sd1*sd1))	+	
								k2*std::expf(-(x*x+y*y)/(2.*sd2*sd2));
			dst1d.ptr<double>(m+h_size)[n+h_size]	=	fm*chh;
		}
	}
	// normalize
	dst1d	=	dst1d/sum(dst1d)[0];
	float	sumv	=	std::fabsf(sum(dst1d)[0]-1.);
	CV_DbgAssert(std::fabsf(sum(dst1d)[0]-1.)<0.000001);

	return true;
}
