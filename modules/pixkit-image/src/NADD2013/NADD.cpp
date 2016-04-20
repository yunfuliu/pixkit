//========================================================================
//
// NADD.cpp
// Authors: Yun-Fu Liu (1), Jing-Ming Guo (2)
// Institutions: National Taiwan University of Science and Technology
// Date: Dec. 27, 2013
// Email: yunfuliu@gmail.com, jmguo@seed.net.tw
// Paper: Yun-Fu Liu and Jing-Ming Guo, "New class tiling design for 
//        dot-diffused halftoning," IEEE Trans. Image Processing, 
//        vol. 22, no. 3, pp. 1199-1208, March 2013.
//
// NADD Halftoning Copyright (c) 2013, Yun-Fu Liu and Jing-Ming Guo, 
// all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * The name of the copyright holders may not be used to endorse or promote products
//   derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF 
// THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//========================================================================

#include <ctime>
#include "../../include/pixkit-image.hpp"

/************************************************************************/
/* CLOAD                                                                */
/************************************************************************/
class	CLOAD{
public:
	double	**	m_Image;
	int			m_Height;
	int			m_Width;
	CLOAD(int Height=0,int Width=0);	
	virtual		~CLOAD(void);
private:	
	double	*	m_Data;
	bool		m_ImageNewSpaceFlag;
	int			m_ImageHeadLength;
	bool		NewSpace(void);
	bool		ReleaseSpace(void);	// delete the spaces of m_Data and m_Image
};
CLOAD::CLOAD(int Height,int Width){
	m_Data				=	NULL;
	m_Image				=	NULL;
	m_Height			=	Height;
	m_Width				=	Width;
	m_ImageHeadLength	=	0;
	m_ImageNewSpaceFlag	=	false;

	if((m_Height!=0)&&(m_Width!=0))
		NewSpace();
}
CLOAD::~CLOAD(){
	ReleaseSpace();
}
bool CLOAD::NewSpace(){
	m_ImageNewSpaceFlag	=	true;
	m_Data				=	new	double		[m_Height*m_Width];
	m_Image				=	new	double	*	[m_Height];
	for(int i=0;i<m_Height;i++)
		m_Image[i]=&m_Data[i*m_Width];	
	return	true;
}
bool CLOAD::ReleaseSpace(){

	m_ImageNewSpaceFlag	=	false;

	if(m_Data != NULL){
		delete	[]	m_Data;
	}
	if(m_Image != NULL){
		delete	[]	m_Image;
	}

	return	true;
}


//////////////////////////////////////////////////////////////////////////
///// cm transformation
void iTrans_CM(int *src,int *dst,int &CM_Size){
	for(int i=0;i<CM_Size;i++){
		for(int j=0;j<CM_Size;j++){
			dst[j*CM_Size+i]=src[i*CM_Size+j];
		}
	}
}
void Trans_CM(int *src,int *dst,int &CM_Size){
	for(int i=0;i<CM_Size;i++){
		for(int j=0;j<CM_Size;j++){
			dst[(CM_Size-1-j)*CM_Size+(CM_Size-1-i)]=src[i*CM_Size+j];
		}
	}
}
void Copy_CM(int *src,int *dst,int &CM_Size){
	for(int i=0;i<CM_Size;i++){
		for(int j=0;j<CM_Size;j++){
			dst[i*CM_Size+j]=src[i*CM_Size+j];
		}
	}
}
void LfRt_CM(int *src,int *dst,int &CM_Size){

	for(int i=0;i<CM_Size;i++){
		for(int j=0;j<CM_Size;j++){
			dst[i*CM_Size+(CM_Size-1-j)]=src[i*CM_Size+j];
		}
	}

}
void UpDn_CM(int *src,int *dst,int &CM_Size){

	for(int i=0;i<CM_Size;i++){
		for(int j=0;j<CM_Size;j++){
			dst[(CM_Size-1-i)*CM_Size+j]=src[i*CM_Size+j];
		}
	}
}


//////////////////////////////////////////////////////////////////////////
///// cm process
bool change_cm_type(int type_num,int *src,int *dst,int &CM_Size){

	switch(type_num){
	case 0:		// keep the same
		break;
	case 1:		// upside down
		UpDn_CM(&src[0],&dst[0],CM_Size);
		Copy_CM(&dst[0],&src[0],CM_Size);
		break;
	case 2:		// left-right mirror
		LfRt_CM(&src[0],&dst[0],CM_Size);
		Copy_CM(&dst[0],&src[0],CM_Size);
		break;
	case 3:		// transpose
		Trans_CM(&src[0],&dst[0],CM_Size);
		Copy_CM(&dst[0],&src[0],CM_Size);
		break;
	case 4:		// inverse transpose
		iTrans_CM(&src[0],&dst[0],CM_Size);
		Copy_CM(&dst[0],&src[0],CM_Size);
		break;
	case 5:		// transpose with inverse transpose
		Trans_CM(&src[0],&dst[0],CM_Size);
		iTrans_CM(&dst[0],&src[0],CM_Size);
		break;
	case 6:		// transpose with left-right mirror
		Trans_CM(&src[0],&dst[0],CM_Size);
		LfRt_CM(&dst[0],&src[0],CM_Size);
		break;
	case 7:		// inverse transpose with left-right mirror
		iTrans_CM(&src[0],&dst[0],CM_Size);
		LfRt_CM(&dst[0],&src[0],CM_Size);
		break;
	}

	return true;
}
bool same_cm_point(int *src,int src_corner_num/*1�k�W3�k�U5���U7���W*/,int order_type/*7���W6��5���U4�U3�k�U*/,int order_corner_num/*1�k�W3�k�U5���U7���W*/,CLOAD &order,int x,int y,int &CM_Size,bool check_diagonal=false){

	//////////////////////////////////////////////////////////////////////////get src corner position
	int	src_x,src_y;
	int	src_num;
	switch(src_corner_num){
	case 1:
		src_x=CM_Size-1;
		src_y=CM_Size-1;
		break;
	case 3:
		src_x=CM_Size-1;
		src_y=0;
		break;
	case 5:
		src_x=0;
		src_y=0;
		break;
	case 7:
		src_x=0;
		src_y=CM_Size-1;
		break;
	default:
		printf("system error!\n");
		system("pause");
		break;
	}
	src_num=src[src_y*CM_Size+src_x];

	//////////////////////////////////////////////////////////////////////////get order corner position
	switch(order_corner_num){
	case 1:
		src_x=CM_Size-1;
		src_y=CM_Size-1;
		break;
	case 3:
		src_x=CM_Size-1;
		src_y=0;
		break;
	case 5:
		src_x=0;
		src_y=0;
		break;
	case 7:
		src_x=0;
		src_y=CM_Size-1;
		break;
	default:
		printf("system error!\n");
		system("pause");
		break;
	}

	//////////////////////////////////////////////////////////////////////////get order corner number
	int	order_num;
	int	order_x,order_y;
	switch(order_type){
	case 3:
		if(x+CM_Size+src_x>=order.m_Width||y-CM_Size+src_y<0){
			return false;	// �k�U�Lcm
		}else{
			order_x=x+CM_Size+src_x;
			order_y=y-CM_Size+src_y;
		}
		break;
	case 4:
		if(y-CM_Size+src_y<0){
			return false;	// �U�Lcm
		}else{
			order_x=x+src_x;
			order_y=y-CM_Size+src_y;
		}
		break;
	case 5:
		if(x-CM_Size+src_x<0||y-CM_Size+src_y<0){
			return false;	// ���U�Lcm
		}else{
			order_x=x-CM_Size+src_x;
			order_y=y-CM_Size+src_y;
		}
		break;
	case 6:
		if(x-CM_Size+src_x<0){
			return false;	// ���Lcm
		}else{
			order_x=x-CM_Size+src_x;
			order_y=y+src_y;
		}
		break;
	case 7:
		if(x-CM_Size+src_x<0||y+CM_Size+src_y>=order.m_Height){
			return false;	// ���W�Lcm
		}else{
			order_x=x-CM_Size+src_x;
			order_y=y+CM_Size+src_y;
		}
		break;
	default:
		printf("system error!\n");
		system("pause");
		break;
	}
	order_num=(int)order.m_Image[order_y][order_x];

	// compare
	if(check_diagonal){	
		//////////////////////////////////////////////////////////////////////////check �﨤��
		int	dia_pair1_1=src[0]			,dia_pair1_2=src[(CM_Size-1)*CM_Size+(CM_Size-1)];
		int	dia_pair2_1=src[CM_Size-1]	,dia_pair2_2=src[(CM_Size-1)*CM_Size];

		if((src_num==dia_pair1_1&&order_num==dia_pair1_2)||(src_num==dia_pair1_2&&order_num==dia_pair1_1)){
			return true;
		}else if((src_num==dia_pair2_1&&order_num==dia_pair2_2)||(src_num==dia_pair2_2&&order_num==dia_pair2_1)){
			return true;
		}else{
			return false;
		}

	}else{
		//////////////////////////////////////////////////////////////////////////check �ۦP��
		if(src_num==order_num){
			return true;
		}else{
			return false;
		}
	}
}
bool check_cm_type(int *src,CLOAD &order,int x,int y,int &CM_Size){

	//////////////////////////////////////////////////////////////////////////common
	// check �P��cm���F��
	if(same_cm_point(&src[0],7,6,1,order,x,y,CM_Size)){
		//		printf("E1 ");
		return false;
	}
	if(same_cm_point(&src[0],5,6,3,order,x,y,CM_Size)){
		//		printf("E2 ");
		return false;
	}
	// check �M����cm���ۦ�
	if(same_cm_point(&src[0],5,6,5,order,x,y,CM_Size)&&same_cm_point(&src[0],7,6,7,order,x,y,CM_Size)){
		//		printf("E3 ");
		return false;
	}
	// check �M����cm���i�P���۳s
	if(same_cm_point(&src[0],5,6,1,order,x,y,CM_Size)&&same_cm_point(&src[0],7,6,3,order,x,y,CM_Size)){
		//		printf("E4 ");
		return false;
	}

	// �קK�P�۾F���O�x�}�����P�򦳬ۦP������(���U)
	if(y-1>=0&&((x+CM_Size/2-1)<order.m_Width)&&(y-CM_Size>=0)){
		if((src[0]==order.m_Image[y-1][x+CM_Size/2-1])&&(src[(CM_Size-1)*CM_Size]==order.m_Image[y-CM_Size][x+CM_Size/2-1])){
			//			printf("E5 ");
			return false;
		}
	}

	// �קK�P�۾F���O�x�}�����P�򦳬ۦP������(�k�U)
	if(y-1>=0&&((x+CM_Size/2)<order.m_Width)&&(y-CM_Size>=0)){
		if((src[CM_Size-1]==order.m_Image[y-1][x+CM_Size/2])&&(src[(CM_Size-1)*CM_Size+(CM_Size-1)]==order.m_Image[y-CM_Size][x+CM_Size/2])){
			//			printf("E6 ");
			return false;
		}
	}

	return true;
}


//////////////////////////////////////////////////////////////////////////
///// data compression
// for load ()
bool char3tovalue8(unsigned char *csrc,int clength,int *idst,int ilength){

	// check the length of src
	if(ilength!=8||clength!=3){
		printf("CCT::char3tovalue8() error!");
		return false;
	}

	idst[0]=csrc[0]>>5;
	idst[1]=(csrc[0]&31)>>2;
	idst[2]=((csrc[0]&3)<<1) | (csrc[1]>>7);
	idst[3]=(csrc[1]&127)>>4;
	idst[4]=(csrc[1]&15)>>1;
	idst[5]=((csrc[1]&1)<<2) | (csrc[2]>>6);
	idst[6]=(csrc[2]&63)>>3;
	idst[7]=csrc[2]&7;
	return true;
}
bool trans_ctmap_to_ct(int &m_CT_Height,int &m_CT_Width,unsigned char **m_ct,unsigned char *m_ctData,int &m_numberofcm,unsigned char *m_ctmap,int &m_CM_Size,int **m_cm){
	// input m_ctmap; output m_ct

	// exception
	if(m_ctmap==NULL){
		printf("trans_ctmap_to_ct(): m_ctmap is empty.\n");
		return false;
	}
	if(m_numberofcm==0){
		printf("trans_ctmap_to_ct(): numberofcm should not be zero.\n");
		return false;
	}

	// new space
	if(m_ct!=NULL){
		delete	[]	m_ct;
		delete	[]	m_ctData;
	}
	m_ctData=	new unsigned char	[m_CT_Height*m_CT_Width];
	m_ct	=	new unsigned char *	[m_CT_Height];
	for(int i=0;i<m_CT_Height;i++){
		m_ct[i]=&m_ctData[i*m_CT_Width];
	}

	//////////////////////////////////////////////////////////////////////////
	// get class matrix
	//////////////////////////////////////////////////////////////////////////
	bool	seq_cm[8];
	int		i,j,m,n;
	int		*	buffer_CM	=	new	int			[m_CM_Size*m_CM_Size];
	int		*	temp_CM		=	new	int			[m_CM_Size*m_CM_Size];
	int	ctmap_index=0;
	for(i=0;i<m_CT_Height;i+=m_CM_Size){
		for(j=((i/m_CM_Size)%2==0?0:-m_CM_Size/2);j<m_CT_Width;j+=m_CM_Size){

			Copy_CM(&m_cm[0][0],&buffer_CM[0],m_CM_Size);
			change_cm_type((int)m_ctmap[ctmap_index],&buffer_CM[0],&temp_CM[0],m_CM_Size);
			ctmap_index++;
			// determine class matrix
			for(m=0;m<m_CM_Size;m++){
				for(n=0;n<m_CM_Size;n++){
					if(i+m>=0&&i+m<m_CT_Height&&j+n>=0&&j+n<m_CT_Width){
						m_ct[i+m][j+n]=buffer_CM[m*m_CM_Size+n];
					}
				}
			}
		}
	}	

	delete	[]	buffer_CM;
	delete	[]	temp_CM;

	return true;
}
// for save ()
bool value8tochar3(int *isrc,int ilength,unsigned char *cdst,int clength){	// length: src's length

	// check the length of src
	if(ilength!=8||clength!=3){
		printf("CCT::value8tochar3() error.\n");
		return false;
	}

	// check all the values of src whether within the reasonable range.
	for(int i=0;i<ilength;i++){	
		if(isrc[i]>=8||isrc[i]<0){
			printf("CCT::value8tochar3() error.\n");
			return false;
		}
	}

	cdst[0]=(((isrc[0]<<1)|(isrc[1]>>2))		<<4)	|	(((isrc[1]&3)<<2)|(isrc[2]>>1));
	cdst[1]=((((isrc[2]&1)<<3)|(isrc[3]))		<<4)	|	((isrc[4]<<1)|(isrc[5]>>2));
	cdst[2]=((((isrc[5]&11)<<2)|(isrc[6]>>1))	<<4)	|	(((isrc[6]&1)<<3)|(isrc[7]));

	return true;
}


/************************************************************************/
/* CCT                                                                  */
/************************************************************************/
pixkit::halftoning::dotdiffusion::CNADDCT::CNADDCT(){	
	m_CT_height=0;
	m_CT_width=0;
	m_CTmap_height=0;
	m_CTmap_width=0;
	m_CM_size=8;
	m_numberOfCM=0;
	m_ctmap=NULL;
	m_ct=NULL;
	m_ctData=NULL;
	m_cm=NULL;
	m_cmData=NULL;
	pointList.resize(0);
	imgSize_pointList	=	cv::Size(0,0);

	m_cmData	=	new int	[m_CM_size*m_CM_size];
	m_cm		=	new int *[m_CM_size];
	for(int i=0;i<m_CM_size;i++){
		m_cm[i]	=	&m_cmData[i*m_CM_size];
	}

	int		temp_cm[8][8]={{26,15,11,42,41,37,12,7},{27,32,33,5,10,16,23,4},{13,17,18,19,20,21,14,40},{6,45,44,8,0,29,39,47},{49,34,25,43,38,28,24,48},{51,9,36,35,31,22,1,50},{53,46,3,2,30,56,58,52},{55,57,59,60,62,63,61,54}};
	for(int i=0;i<m_CM_size;i++){
		for(int j=0;j<m_CM_size;j++){
			m_cm[i][j]=temp_cm[i][j];
		}
	}

}
pixkit::halftoning::dotdiffusion::CNADDCT::~CNADDCT(){
	if(m_ct!=NULL){
		delete	[]	m_ct;
		delete	[]	m_ctData;
	}
	if(m_ctmap!=NULL){
		delete	[]	m_ctmap;
	}
}
bool pixkit::halftoning::dotdiffusion::CNADDCT::generation(cv::Size ctSize){

	// exceptions
	if(ctSize.height%m_CM_size!=0){
		printf("CCT::generation(): the height [%d] of CT should be fully divided by 8.\n",ctSize.height);
		return false;
	}
	if(ctSize.width%m_CM_size!=0){
		printf("CCT::generation(): the width [%d] of CT should be fully divided by 8.\n",ctSize.width);
		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	// assign parameter
	m_CT_height=ctSize.height;
	m_CT_width=ctSize.width;
	// assign space
	m_ctData	=	new unsigned char	[m_CT_height*m_CT_width];
	m_ct		=	new unsigned char *	[m_CT_height];
	for(int i=0;i<m_CT_height;i++){
		m_ct[i]=&m_ctData[i*m_CT_width];
	}
	//////////////////////////////////////////////////////////////////////////
	// assign space
	CLOAD	order(m_CT_height,m_CT_width);	// temp_ct for transfer
	m_numberOfCM	=	m_CT_height*m_CT_width/64+m_CT_height/16;	// # CM in a CT
	m_ctmap	=	new	unsigned char	[m_numberOfCM];
	for(int i=0;i<m_numberOfCM;i++){
		m_ctmap[i]=0;
	}
	//////////////////////////////////////////////////////////////////////////get class matrix
	bool	seq_cm[8];
	bool	select_ok;
	int		randv;
	int		i,j,m,n;
	int		*	buffer_CM	=	new	int			[m_CM_size*m_CM_size];
	int		*	temp_CM		=	new	int			[m_CM_size*m_CM_size];
	srand(65487);	// the seed for [TIP]
	int	ctmap_index=0;
	for(i=0;i<m_CT_height;i+=m_CM_size){
		for(j=((i/m_CM_size)%2==0?0:-m_CM_size/2);j<m_CT_width;j+=m_CM_size){

			// check class matrix
			for(m=0;m<8;m++){
				seq_cm[m]=false;
			}
			select_ok=false;
			while(!select_ok){
				randv=rand()%8;
				if(seq_cm[randv]==false){
					seq_cm[randv]=true;
					Copy_CM(&m_cm[0][0],&buffer_CM[0],m_CM_size);
					change_cm_type(randv,&buffer_CM[0],&temp_CM[0],m_CM_size);
					select_ok=check_cm_type(&buffer_CM[0],order,j,i,m_CM_size);
				}
			}
			m_ctmap[ctmap_index]=((char)randv);
			//			printf("%d\t",randv);
			ctmap_index++;
			// determine class matrix
			for(m=0;m<m_CM_size;m++){
				for(n=0;n<m_CM_size;n++){
					if(i+m>=0&&i+m<m_CT_height&&j+n>=0&&j+n<m_CT_width){
						order.m_Image[i+m][j+n]=buffer_CM[m*m_CM_size+n];
					}
				}
			}
		}
	}	

	delete	[]	buffer_CM;
	delete	[]	temp_CM;

	for(int i=0;i<m_CT_height;i++){
		for(int j=0;j<m_CT_width;j++){
			m_ct[i][j]=order.m_Image[i][j];
		}
	}

	srand((unsigned char)time(NULL));

	return true;
}
bool pixkit::halftoning::dotdiffusion::CNADDCT::save(char name[]){
	// save the ct to .map file

	// exception
	if(m_CT_height<=0||m_CT_width<=0){
		printf("CCT::save(): ct's height or width should not be less than or equals to zero.\n");
		return false;
	}

	FILE	*fn=fopen(name,"wb");
	if(fn==NULL){
		CV_Error(CV_StsNullPtr,"[pixkit::halftoning::dotdiffusion::CNADDCT::save] fn is NULL.");
		return false;
	}
	//////////////////////////////////////////////////////////////////////////
	// save the height and width of ct (format: short int (2B) for each of height and width)
	fwrite(&m_CT_height,sizeof(short int),1,fn);
	fwrite(&m_CT_width,sizeof(short int),1,fn);

	// save numberofcm	
	fwrite(&m_numberOfCM,sizeof(short int),1,fn);

	// compress the ct map (format: 3bit for each value, thus 3B for 8 values)
	// and save ct map
	int				temp_value[8];
	unsigned char	temp_char[3];
	int i=8;
	if(m_numberOfCM>=8){
		for(i=0;i<m_numberOfCM;i+=8){
			if((i+8-1)<m_numberOfCM){	// do jump, if the next is still in the current range
				// initialize
				for(int j=0;j<8;j++){
					temp_value[j]=0;
				}

				// copy
				for(int j=0;j<8;j++){
					temp_value[j]=m_ctmap[i+j];
				}
				value8tochar3(temp_value,8,temp_char,3);
				fwrite(temp_char,3*sizeof(unsigned char),1,fn);
			}
		}
	}
	if(m_numberOfCM%8!=0){
		for(int j=0;j<8;j++){
			temp_value[j]=0;
		}
		for(int j=i-8;j<m_numberOfCM;j++){
			temp_value[j-i+8]=m_ctmap[j];
		}
		value8tochar3(temp_value,8,temp_char,3);
		fwrite(temp_char,3*sizeof(unsigned char),1,fn);
	}

	//////////////////////////////////////////////////////////////////////////
	fclose(fn);

	return true;
}
bool pixkit::halftoning::dotdiffusion::CNADDCT::load(char name[]){
	// load the .map file and make an entire CT from CT map

	FILE	*fn=fopen(name,"rb");
	if(fn==NULL){
		CV_Error(CV_StsNullPtr,"[pixkit::halftoning::dotdiffusion::CNADDCT::load] fn is NULL.");
		return false;
	}
	//////////////////////////////////////////////////////////////////////////
	// load the size of ct
	fread(&m_CT_height,sizeof(short int),1,fn);
	fread(&m_CT_width,sizeof(short int),1,fn);

	// load numberofcm
	fread(&m_numberOfCM,sizeof(short int),1,fn);

	// assign space
	if(m_ctmap!=NULL){
		delete	[]	m_ctmap;
	}
	m_ctmap	=	new unsigned char	[m_numberOfCM];

	// load ct map
	int				temp_value[8];
	unsigned char	temp_char[3];
	int	i=8;
	if(m_numberOfCM>=8){
		for(i=0;i<m_numberOfCM;i+=8){
			if((i+8-1)<m_numberOfCM){	// do jump, if the next is still in the current range
				fread(temp_char,3*sizeof(unsigned char),1,fn);
				char3tovalue8(temp_char,3,temp_value,8);
				for(int j=0;j<8;j++){
					m_ctmap[i+j]=temp_value[j];
				}
			}
		}
	}
	if(m_numberOfCM%8!=0){
		fread(temp_char,3*sizeof(unsigned char),1,fn);
		char3tovalue8(temp_char,3,temp_value,8);
		for(int j=i-8;j<m_numberOfCM;j++){
			m_ctmap[j]=temp_value[j-i+8];
		}
	}

	// transfer the ct map to ct
	trans_ctmap_to_ct(m_CT_height,m_CT_width,m_ct,m_ctData,m_numberOfCM,m_ctmap,m_CM_size,m_cm);

	//////////////////////////////////////////////////////////////////////////
	fclose(fn);		

	return true;
}


/************************************************************************/
/* NADD2013                                                             */
/************************************************************************/
void getPointList1(const cv::Size imgSize, int *cct, std::vector<std::vector<cv::Point>> &point_list,const int cm_elememt_size){
	point_list.resize(cm_elememt_size);
	for(int i=0;i<imgSize.height;i++){
		for(int j=0;j<imgSize.width;j++){			
			point_list[cct[i*imgSize.width+j]].push_back(cv::Point(j,i));
		}
	}
}
bool pixkit::halftoning::dotdiffusion::NADD2013(cv::Mat &src,cv::Mat &dst,pixkit::halftoning::dotdiffusion::CNADDCT &cct){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if(src.type()!=CV_8U){
//		CV_Error(CV_BadNumChannels,"");
		return false;
	}
	if(src.empty()){
//		CV_Error(CV_StsBadArg,"src is empty.");
		return false;
	}

	// get src clone to dst
	dst=src.clone();
	int	&m_Height	=	dst.rows;
	int	&m_Width	=	dst.cols;

	// create CT for image processing
	int	*	orderData	=	new int		[m_Height*m_Width];
	int	**	order		=	new int	*	[m_Height];
	for(int i=0;i<m_Height;i++){
		order[i]=&orderData[i*m_Width];
	}
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			order[i][j]=(int)cct.m_ct[i%cct.m_CT_height][j%cct.m_CT_width];
		}
	}

	// create temp space for dst
	float	*	temp_dstData	=	new float		[m_Height*m_Width];
	float	**	temp_dst		=	new float	*	[m_Height];
	for(int i=0;i<m_Height;i++){
		temp_dst[i]=&temp_dstData[i*m_Width];
	}
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			temp_dst[m_Height-1-i][j]=(float)dst.data[i*m_Width+j];
		}
	}

	// setting
	double	DW[9]={0.34549,1,0.34549,1,0,1,0.34549,1,0.34549};
	int		CM_Size=8;	// size of class matrix
	int		DW_Size=3;	// size of diffused matrix

	// get pointlist, it stores the point locations of each processing order
	if(cct.pointList.size()==0){
		cct.imgSize_pointList	=	src.size();	// stores the size of pointList
		getPointList1(src.size(),orderData,cct.pointList,CM_Size*CM_Size);
	}else if(src.size()==cct.imgSize_pointList){
		// do nothing
	}else{
		cct.imgSize_pointList	=	src.size();	// assign a new size for pointList
		getPointList1(src.size(),orderData,cct.pointList,CM_Size*CM_Size);
	}	

	// perform dot diffusion
	int	hDW_Size=DW_Size/2;
	for(int k=0;k<CM_Size*CM_Size;k++){ // proc order
		for(int p=0;p<cct.pointList[k].size();p++){
			// get i and j;
			int i=cct.pointList[k][p].y;
			int j=cct.pointList[k][p].x;

			// get error
			double	error;
			if(temp_dst[i][j]<128){
				error=temp_dst[i][j];
				temp_dst[i][j]=0.;
			}else{
				error=temp_dst[i][j]-255.;
				temp_dst[i][j]=255.;
			}
			// get fm
			double	fm=0.;					
			for(int m=-hDW_Size;m<=hDW_Size;m++){
				for(int n=-hDW_Size;n<=hDW_Size;n++){
					if(i+m>=0&&i+m<m_Height&&j+n>=0&&j+n<m_Width){	// in the image region
						if(order[i+m][j+n]>k){	// diffusable region
							fm+=DW[(m+hDW_Size)*DW_Size+(n+hDW_Size)];
						}
					}
				}
			}
			// diffuse
			for(int m=-hDW_Size;m<=hDW_Size;m++){
				for(int n=-hDW_Size;n<=hDW_Size;n++){
					if(i+m>=0&&i+m<m_Height&&j+n>=0&&j+n<m_Width){	// in the image region
						if(order[i+m][j+n]>k){	// diffusable region						
							temp_dst[i+m][j+n]+=error*DW[(m+hDW_Size)*DW_Size+(n+hDW_Size)]/fm;
						}
					}
				}
			}
		}
	}

	// copy from temp_dst to dst
	for(int i=0;i<m_Height;i++){
		for(int j=0;j<m_Width;j++){
			dst.data[i*m_Width+j]=(uchar)(temp_dst[m_Height-1-i][j]+0.5);
		}
	}

	delete	[]	temp_dstData;
	delete	[]	temp_dst;
	delete	[]	orderData;
	delete	[]	order;

	return true;
}
