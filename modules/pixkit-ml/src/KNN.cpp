#include "../include/pixkit-ml.hpp"

using namespace std;

double Euclidean_Distance(const vector<double>& v1, const vector<double>& v2){
	assert(v1.size() == v2.size());
	double ret = 0.0;
	for (vector<double>::size_type i = 0; i != v1.size(); ++i){
		ret += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return sqrt(ret);
}
void DistanceMatrix(std::vector<vector<double> >& dm, const std::vector<pixkit::classification::SSample> &dataset,const std::vector<pixkit::classification::SSample> &sample){
	for ( vector<pixkit::classification::SSample>::size_type i = 0; i != sample.size(); ++i){
		vector<double> DM_data;
		for (vector<pixkit::classification::SSample>::size_type j = 0; j != dataset.size(); ++j){
			DM_data.push_back(Euclidean_Distance(sample[i].features, dataset[j].features));
		}
		dm.push_back(DM_data);
	}
}
vector<double> kappa_Matrix(const vector<double>& v1, const vector<double>& v2, vector<double>& kappa)
{	
	assert(v1.size() == v2.size()); 

	for (vector<double>::size_type i = 0; i != v1.size(); ++i)
	{
		kappa[i] +=sqrt((v1[i] - v2[i]) * (v1[i] - v2[i])) ;
	}

	return (kappa);

}

void BandwidthMatrix(vector<vector<double> >& Bw, const vector<pixkit::classification::SSample>& sample, const vector<pixkit::classification::SSample>& dataset)
{

	for (vector<pixkit::classification::SSample>::size_type i = 0; i !=sample.size(); ++i)
	{
		vector<double> kappa_data,kappa(sample[i].features.size());

		for (vector<pixkit::classification::SSample>::size_type j = 0; j != dataset.size(); ++j)
		{


			kappa_data=kappa_Matrix(sample[i].features, dataset[j].features,kappa);

		}

		Bw.push_back(kappa_data);

	}


}

void Tradition_KNN_Process(vector<pixkit::classification::SSample>& sample, const vector<pixkit::classification::SSample>& dataset, const vector<vector<double> >& dm, unsigned int k){
	for (vector<pixkit::classification::SSample>::size_type i = 0; i != sample.size(); ++i){
		multimap<double, string> dts;
		for (vector<double>::size_type j = 0; j != dm[i].size(); ++j){
			if (dts.size() < k){
				dts.insert(make_pair(dm[i][j], dataset[j].classnumber)); 
			}else{
				multimap<double, string>::iterator it = dts.end();
				--it;
				if (dm[i][j] < (it->first)){
					dts.erase(it);
					dts.insert(make_pair(dm[i][j], dataset[j].classnumber));
				}
			}
		}
		map<string, double> tds;
		string type = "";
		double weight = 0.0;
		for (multimap<double, string>::const_iterator cit = dts.begin(); cit != dts.end(); ++cit){
			++tds[cit->second];
			if (tds[cit->second] > weight){
				weight = tds[cit->second];
				type = cit->second;
			} 
		}
		for (map<string, double>::const_iterator cit = tds.begin(); cit != tds.end(); ++cit){
			cout<<cit->first <<"="<<cit->second<<'\n'<<endl;
		}
		sample[i].classnumber = type;
		cout<<"Result Class = "<<sample[i].classnumber<<'\n'<<endl;
		cout<<"=============================================="<<endl;
	}
}

void FKNN_Process(vector<pixkit::classification::SSample>& sample, const vector<pixkit::classification::SSample>& dataset, const vector<vector<double> >& dm, unsigned int k)
{
	for (vector<pixkit::classification::SSample>::size_type i = 0; i != sample.size(); ++i)
	{
		multimap<double, string> dts;
		for (vector<double>::size_type j = 0; j != dm[i].size(); ++j)
		{
			if (dts.size() < k ) 
			{
				dts.insert(make_pair(dm[i][j], dataset[j].classnumber)); 
			}
			else
			{
				multimap<double, string>::iterator it = dts.end();
				--it;
				if (dm[i][j] < (it->first))
				{
					dts.erase(it);
					dts.insert(make_pair(dm[i][j], dataset[j].classnumber));

				}
			}
		}
		map<string, double> tds,fin;
		string type = "";
		double weight = 0.0;
		double now_weight = 0.0;
		double total_weight= 0.0;
		for (multimap<double, string>::const_iterator cit = dts.begin(); cit != dts.end(); ++cit)
		{

			if (cit->first==0)
			{
				tds[cit->second] += 1.0 / (cit->first+1);
				total_weight=total_weight+(1.0 / (cit->first+1));

			}else{

				tds[cit->second] += 1.0 / cit->first;
				total_weight=total_weight+(1.0 / (cit->first));
			}




		}

		for (map<string, double>::const_iterator cit = tds.begin(); cit != tds.end(); ++cit)
		{

			now_weight=(cit->second/total_weight);

			if (now_weight > weight)
			{
				weight = now_weight;
				type = cit->first;

			}

		}

		sample[i].classnumber = type;
		cout<<"Result Class = "<<sample[i].classnumber<<'\n'<<endl;
		cout<<"=============================================="<<endl;
	}

}

void FRNN_Process(vector<pixkit::classification::SSample>& sample, const vector<pixkit::classification::SSample>& dataset, const vector<vector<double> >& dm,const vector<vector<double> >& Bw)
{
	for (vector<pixkit::classification::SSample>::size_type i = 0; i != sample.size(); ++i)
	{
		multimap<double, string> dts,dist,dist1;
		map<string, double> tds,fin,merbership,countnmber;
		

		for (vector<double>::size_type k = 0; k != dm[i].size(); ++k)
		{
			dist.insert(make_pair(dm[i][k], dataset[k].classnumber));

		}


	
		for (vector<double>::size_type j = 0; j != dataset.size(); ++j)
		{
			double d=0.0;
			for (vector<double>::size_type x = 0; x != Bw[i].size(); ++x)
			{
				if (Bw[i][x]==0)
				{
					d+=(dataset.size()/(2))*(sample[i].features[x]-dataset[j].features[x])*(sample[i].features[x]-dataset[j].features[x]);
					
				}else{
					
					d+=(dataset.size()/(2*Bw[i][x]))*(sample[i].features[x]-dataset[j].features[x])*(sample[i].features[x]-dataset[j].features[x]);
					
				}
			}
			dts.insert(make_pair(dm[i][j], dataset[j].classnumber));
			d=d+(1/dataset.size())*exp(-d);

			for (multimap<double, string>::const_iterator cit = dts.begin(); cit != dts.end(); ++cit)
			{

				if (cit->first==0)
				{
					tds[cit->second] += 1.0 / (cit->first+1);

				}else{

					tds[cit->second] += 1.0 / cit->first;
				}

			}
		}

		string type = "";
		double weight = 0.0;

		for (multimap<double, string>::const_iterator cit = dts.begin(); cit != dts.end(); ++cit)
		{

			
			tds[cit->second] += cit->first;

			if (1/tds[cit->second] > weight)
			{
				weight = 1/tds[cit->second];
				type = cit->second;

			}
			
		}


		sample[i].classnumber = type;
		cout<<"Result Class = "<<sample[i].classnumber<<'\n'<<endl;
		cout<<"=============================================="<<endl;
	}

}

bool pixkit::classification::KNN(std::vector<SSample> &sample,const std::vector<SSample> &dataset,int k){

	//////////////////////////////////////////////////////////////////////////
	/////
	if(k<0){
		CV_Error(CV_StsBadArg,"[pixkit::classification::KNN] k should >= 1.");
	}

	vector<vector<double> > DM;
    DistanceMatrix(DM, dataset, sample);
	Tradition_KNN_Process(sample, dataset, DM, k);

	return	true;
}

bool pixkit::classification::FKNN(std::vector<SSample> &sample,const std::vector<SSample> &dataset,int k){

	//////////////////////////////////////////////////////////////////////////
	/////
	if(k<0){
		CV_Error(CV_StsBadArg,"[pixkit::classification::KNN] k should >= 1.");
	}

	vector<vector<double> > DM;
	DistanceMatrix(DM, dataset, sample);
	FKNN_Process(sample, dataset, DM, k);

	return	true;
}

bool pixkit::classification::FRNN(std::vector<SSample> &sample,const std::vector<SSample> &dataset){

	//////////////////////////////////////////////////////////////////////////
	/////

	vector<vector<double> > DM,Bw;
	DistanceMatrix(DM, dataset, sample);
	BandwidthMatrix(Bw, dataset, sample);
	FRNN_Process(sample, dataset, DM,Bw);

	return	true;
}