#include "../include/pixkit-ml.hpp"

using namespace std;

void pixkit::mldata::readTrain(std::vector<pixkit::classification::SSample>& data, const std::string file){

	ifstream fin(file.c_str()); 
	if (!fin){
		cout << "File error!" << endl;
		exit(1);
	}

	string line;
	double d = 0.0;

	while (getline(fin, line)){
		istringstream in(line); 
		pixkit::classification::SSample data1;
		in >> data1.classnumber;
		while (in >> d){
			data1.features.push_back(d);
		}
		data.push_back(data1);
	}

	fin.close();
}
void pixkit::mldata::readTest(std::vector<pixkit::classification::SSample>& data, const std::string file){
	ifstream fin(file.c_str());

	if (!fin){
		cout << "File error!" << endl;
		exit(1);
	}

	double d = 0.0;
	string line;
	while (getline(fin, line)){
		istringstream in(line);
		pixkit::classification::SSample data1;
		while (in >> d){
			data1.features.push_back(d);
		}
		data.push_back(data1);
	}

	fin.close();
}
void pixkit::mldata::write(std::vector<pixkit::classification::SSample>& data, const std::string file){
	ofstream fout(file.c_str());
	if (!fout){
		fout << "File error!" << endl;
		exit(1);
	}
	for (vector<pixkit::classification::SSample>::size_type i = 0; i != data.size(); ++i){
		fout << data[i].classnumber << '\t';
		for (vector<double>::size_type j = 0; j != data[i].features.size(); ++j){
			fout << data[i].features[j] << ' ';
		}
		fout << endl;		
	}
}
