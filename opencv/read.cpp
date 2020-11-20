#include "read.h"
using namespace std;
double *get_para(string &filename)
{
	string file = "/home/maxwell/DJY/code/clion_pro/CudaOpencvDisp/xml/";
	file += filename;
	ifstream ifs(file);
	string str;
	int count = 0;
	static double readmat[9];
	while (ifs >> str)
	{
		//cout << str << endl;
		//cout << atof(str.c_str())<<endl;
		readmat[count] = stod(str);
		count++;
	}
	ifs.close();
	return readmat;
}