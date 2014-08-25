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