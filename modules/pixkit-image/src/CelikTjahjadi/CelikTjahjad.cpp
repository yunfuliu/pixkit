#include "../../include/pixkit-image.hpp"

bool EM_Alg(cv::Mat src,std::vector<float> &Mean, std::vector<float> &Sd, std::vector<float> &W){

	const int K = Mean.size();  //高斯模型的數目

	double nColors = 256.0, e = 0.01;

	std::vector<double> new_m(K,0),new_sd(K,0),new_w(K,0);
	
	cv::Mat Img_data;
	src.convertTo(Img_data ,CV_64FC1);  //將資料轉換到浮點運算

	int Count = 0;
	std::vector<std::vector <double>> p(K,std::vector <double> (src.rows*src.cols,0));  
	while(1){
		//計算每個數據,它由第k哥高斯分量所生成的機率
		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++){

				double x = Img_data.at<double> (i,j), Sum = 0;

				for(int k=0;k<K;k++){
					if(Sd[k]==0){
						if(Mean[k] == x)
							p[k][i*src.cols+j] = W[k]*1.0;
						else
							p[k][i*src.cols+j] = 0;
					}
					else
						p[k][i*src.cols+j] = W[k]*exp(-(x-Mean[k])*(x-Mean[k])/(2*Sd[k]*Sd[k]))/(Sd[k])/sqrt(2.0*3.141589);
						
					Sum += p[k][i*src.cols+j];
				}

				for(int k=0;k<K;k++)
					p[k][i*src.cols+j] /= Sum;
			}
			//更新每個分量
			for(int k=0;k<K;k++){
				double m = Mean[k];
				new_w[k] = 0, new_m[k] =0, new_sd[k] = 0;

				for(int i=0;i<src.rows;i++)
					for(int j=0;j<src.cols;j++){
						double x = Img_data.at<double> (i,j);

						new_w[k] += p[k][i*src.cols+j];
						new_m[k] += p[k][i*src.cols+j]*x;
						new_sd[k] += p[k][i*src.cols+j]*(x-m)*(x-m);
					}
	
					new_m[k] /= new_w[k];
					new_sd[k] = sqrt(new_sd[k]/new_w[k]);
					new_w[k] /= src.rows*src.cols;
			}



			bool con = false;

			for(int k=0;k<K;k++){
				if(abs(Mean[k]-new_m[k])<e && abs(Sd[k]-new_sd[k])<e && abs(W[k]-new_w[k])<e)
					con = true;
				else{
					con = false;
					break;
				}
			}
			
			for(int k=0;k<K;k++){
				Mean[k] = new_m[k];
				Sd[k] = new_sd[k];
				W[k] = new_w[k];
			}

			if(con ==true)
				break;
	}

	return true;
}

double calcCDF(double src,double mean,double sd){
	// exception when sd=0
	if(sd==0&&mean<=src){
		return 1.;
	}

	if(sd==0&&mean>=src){
		return 0.;
	}
	double x=(src-mean)/(sqrt(2.)*sd);
	double t=1./(1.+0.3275911*fabs(x));
	double erf=		0.254829592	*t
		-	0.284496736	*t*t
		+	1.421413741	*t*t*t
		-	1.453152027	*t*t*t*t
		+	1.061405429	*t*t*t*t*t;
	erf=1.-erf*exp(-(x*x));
	return 0.5*(1+(x<0?-erf:erf));	
}

bool pixkit::enhancement::global::CelikTjahjadi2012(cv::Mat &src,cv::Mat &dst,int N){
	std::vector<float> Mean(N,0),Sd(N,0),W(N,0);

	double nColors = 256.0;
	//給予初值
	for(int i=0;i<N;i++){
		Mean[i] = nColors/N*i+ nColors/N/2;
		Sd[i] = 8;
		W[i] = 1.0/N;
	}

	bool con = false;
	//-----------------------------------------------------------------
	//建立高斯模型
	//-------------------------------------------------------------------
	while(con==false){

		int K = 0;  //高斯模型的數目

		for(int i=0;i<Sd.size();i++){
			if(Sd[i]!=0)
				K++;
		}

		std::vector<float> t_m(K,0),t_sd(K,0),t_w(K,0);

		for(int i=0;i<Sd.size();i++){
			if(Sd[i] != 0)
				t_m[i] = Mean[i], t_sd[i] = Sd[i], t_w[i] = W[i];
		}

		EM_Alg(src,t_m,t_sd,t_w);

		int k=0;
		for(int i=0;i<K;i++){
			if(t_sd[i]!=0){
				Mean[k] = t_m[i], Sd[k] = t_sd[i], W[k] = t_w[i];
				k++;
			}
		}

		if(k<K){
			for(int i=k;i<K;i++){
				Mean[k] = 0, Sd[k] = 0, W[k] = 0;
			}
		}else
			con = true;
	}
	//----------------------------------------------------------------
	//強化運算
	//------------------------------------------------------------------
	int K = 0;  //高斯模型的數目

	for(int i=0;i<Sd.size();i++){
		if(Sd[i]!=0)
			K++;
	}

	dst.create(src.size(),src.type());

	std::vector<double> new_m(K,0),new_sd(K,0);

	std::vector<double> x(K+1,0);//存放交點
	std::vector<double> y(K+1,0);//存放每個區域的輸出動態範圍

	std::vector<float> pdf(nColors,0), cdf(nColors,0);

	//決定交點的最右邊和最左邊
	for(int i=0;i<nColors;i++)
		for(int k=0;k<K;k++){
			cdf[i] += W[k]*calcCDF((double)i,Mean[k],Sd[k]);
		}

		for(int i=0;i<256;i++){
			if(cdf[i]>(1.0/(src.rows*src.cols))){
				x[0] = i;
				break;
			}
		}

		for(int i=0;i<nColors;i++){
			if(cdf[i]>(1-1.0/(src.rows*src.cols))){
				x[K] = i;
				break;
			}

			if(i=nColors-1)
				x[K] = nColors-1;
		}
		//計算其他高斯分量的交點
		for(int k=1;k<K;k++){
			double a = (Sd[k-1]*Sd[k-1]) - (Sd[k]*Sd[k]);
			double b = 2*((Mean[k-1]*Sd[k]*Sd[k]) - (Mean[k]*Sd[k-1]*Sd[k-1]));
			double c = (Mean[k]*Mean[k]*Sd[k-1]*Sd[k-1]-Mean[k-1]*Mean[k-1]*Sd[k]*Sd[k])-2*Sd[k-1]*Sd[k-1]*Sd[k]*Sd[k]*log((W[k]*Sd[k-1]+0.0000001)/(W[k-1]*Sd[k]+0.0000001));
			double x1,x2;

			x1 = (-b+sqrt(b*b-4*a*c))/(2*a);
			x2 = (-b-sqrt(b*b-4*a*c))/(2*a);
			//------------------------------------
			if(x1>(nColors-1) || x1<0)
				x1 = x2;

			if(x2>(nColors-1) || x2<0)
				x2 = x1;
			//-------------------------------------------
			if(k!=K-1){
				if(x1>x2)
					x[k] = x1;
				else
					x[k] = x2;
			} else{
				if(x1<x2)
					x[k] = x1;
				else
					x[k] = x2;
			}
		}
		//計算每個劃分區段的權重
		std::vector<double> aphi(K,0);

		double sum_sd=0,sum_cdf=0;
		for(int k=0;k<K;k++){
			aphi[k] = pow((double)Sd[k],0.5)*(cdf[x[k+1]]-cdf[x[k]]);
			sum_sd += pow((double)Sd[k],0.5);
			sum_cdf += (cdf[x[k+1]]-cdf[x[k]]);
		}

		for(int k=0;k<K;k++){
			aphi[k] = aphi[k]/sum_cdf/sum_sd;
		}

		double nor=0;
		for(int k=0;k<K;k++)
			nor += aphi[k];

		for(int k=0;k<K;k++)
			aphi[k] /= nor;
		//計算輸出的分配範圍
		y[0] = 0;
		for(int k=0;k<K;k++){
			y[k+1] = y[k]+(nColors-1)*aphi[k];
		}

		for(int k=0;k<K;k++){
			new_m[k] = (((x[k]-Mean[k])/(x[k+1]-Mean[k])*y[k+1])-y[k])/((x[k]-Mean[k])/(x[k+1]-Mean[k])-1);
			new_sd[k] = (y[k]-new_m[k])/(x[k]-Mean[k])*Sd[k];
		}

		for(int i=0;i<src.rows;i++) 
			for(int j=0;j<src.cols;j++){
				double t=0, s=(double)src.data[i*src.cols+j];
				for(int k=0;k<K;k++){
					t += (((s-Mean[k])/Sd[k])*new_sd[k]+new_m[k])*W[k];
				}
				if(t>255.0)
					t = 255;
				if(t<0)
					t = 0;

				dst.data[i*dst.cols+j] = (unsigned char) (t+0.5);
			}

			return true;
}