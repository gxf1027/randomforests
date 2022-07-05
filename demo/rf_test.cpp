/*
Author: Gu Xingfang
Contact: gxf1027@126.com
*/

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
using namespace std;

#include "../src/RandomCLoquatForests.h"
#include "../src/RandomRLoquatForests.h"
#include "../src/UserInteraction2.h"

int command(const char* cd)
{
	if( 0 == strcmp(cd, "-C") || 0 == strcmp(cd,"-c") )
		return 0;
	if( 0 == strcmp(cd, "-D") || 0 == strcmp(cd,"-d") )
		return 1;
	if( 0 == strcmp(cd, "-O") || 0 == strcmp(cd,"-o") )
		return 2;
	if( 0 == strcmp(cd, "-p") || 0 == strcmp(cd,"-P") )
		return 3;
	return -1;
}

void printRFConfigC(RandomCForests_info& config)
{
	cout << "**********************************************************" << endl;
	cout << "Parameters:" << endl;
	cout << "MaxDepth: " << config.maxdepth << endl;
	cout << "TreesNum: " << config.ntrees << endl;
	cout << "SplitVariables: " << config.mvariables << endl;
	cout << "MinSamplesSplit: " << config.minsamplessplit << endl;
	cout << "Randomness: " << config.randomness << endl;
	cout << "**********************************************************" << endl;
}

void printRFConfigR(RandomRForests_info& config)
{
	cout << "**********************************************************" << endl;
	cout << "Parameters:" << endl;
	cout << "MaxDepth: " << config.maxdepth << endl;
	cout << "TreesNum: " << config.ntrees << endl;
	cout << "SplitVariables: " << config.mvariables << endl;
	cout << "MinSamplesSplit: " << config.minsamplessplit << endl;
	cout << "Randomness: " << config.randomness << endl;
	cout << "**********************************************************" << endl;
}

int main(int argc, char** argv)
{
	// (0) analysis command line
	enum { CLASSIFICATION = 0, REGRESSION = 1 };
	int prob = CLASSIFICATION;
	char* chCommand[3] = { NULL };
	int index, i;

	for (i = 1; i < argc; i += 2) {
		if ((index = command(argv[i])) >= 0) {
			if (index == 3) {
				if (1 == atoi(argv[i + 1]))
					prob = REGRESSION;
				else
					prob = CLASSIFICATION;
				continue;
			}

			if (chCommand[index] != NULL) {
				cout << "One command is assigned more than once." << endl;
				return -1;
			}

			chCommand[index] = new char[strlen(argv[i + 1]) + 1];
			memset(chCommand[index], 0, strlen(argv[i + 1]) + 1);
			memcpy(chCommand[index], argv[i + 1], strlen(argv[i + 1]));
		}
	}

	// (1) Read RF Model
	string datapath;

	LoquatCForest* loquatCForest = NULL;
	LoquatRForest* loquatRForest = NULL;
	switch (prob)
	{
	case 1:
		if (chCommand[0] != NULL && chCommand[1] != NULL) // -c RF_Model.xml
		{

			int rv = BuildRandomRegressionForestModelFromXML2(chCommand[0], loquatRForest);
			if (0 > rv)
			{
				cout << "Reading RF model: " << chCommand[0] << " failed!" << endl;
				for (i = 0; i < 3; i++)
					if (chCommand[i])
						delete[] chCommand[i];
				return -1;
			}
		}
		else
		{
			// TODO: 无法下一步
		}
		break;
	case 0:
	default:
		if (chCommand[0] != NULL && chCommand[1] != NULL) // -c config.xml
		{
			int rv = BuildRandomClassificationForestModelFromXML2(chCommand[0], loquatCForest);
			if (0 > rv)
			{
				cout << "Reading RF model: " << chCommand[0] << " failed!" << endl;
				for (i = 0; i < 3; i++)
					if (chCommand[i])
						delete[] chCommand[i];
				return -1;
			}
		}
		else
		{
			// TODO: 无法下一步
		}
	}

	// (2) Read test data
	if (chCommand[1] == NULL)
	{
		cout << "Test data is not assigned." << endl;
		for (i = 0; i < 3; i++)
			if (chCommand[i])
				delete[] chCommand[i];
		return -1;
	}

	float** data = NULL;
	int* label = NULL;
	float** target = NULL;
	Dataset_info_C datainfo_c;
	Dataset_info_R datainfo_r;
	int rd = 1;
	switch (prob)
	{
	case 1:
		rd = InitalRegressionDataMatrixFormFile2(chCommand[1], data, target, datainfo_r);
		if (1 != rd)
		{
			if (-2 == rd)
				cout << "Reading file: " << chCommand[1] << " failed!" << endl;
			if (-1 == rd)
				cout << "the format of data or extra information isn't correctly compiled" << endl;

			for (i = 0; i < 3; i++)
				if (chCommand[i])
					delete[] chCommand[i];
			return -1;
		}

		break;
	case 0:
	default:
		rd = InitalClassificationDataMatrixFormFile2(chCommand[1], data, label, datainfo_c);
		if (1 != rd)
		{
			if (-2 == rd)
				cout << "Reading file: " << chCommand[1] << " failed!" << endl;
			if (-1 == rd)
				cout << "the format of data or extra information isn't correctly compiled" << endl;

			for (i = 0; i < 3; i++)
				if (chCommand[i])
					delete[] chCommand[i];
			return -1;
		}
	}


	// (3) testing
	float* mean_squared_error = NULL;
	float** target_predict = NULL;

	int rv=1, error_num = 0;
	int* label_predict = NULL;
	timeIt(1);

	switch (prob)
	{
	case 1:
		printRFConfigR(loquatRForest->RFinfo);
		mean_squared_error = new float[datainfo_r.variables_num_y];
		memset(mean_squared_error, 0, sizeof(float) * datainfo_r.variables_num_y);
		target_predict = new float* [datainfo_r.samples_num];
		for (int i = 0; i < datainfo_r.samples_num; i++)
		{
			//TODO: 对每个样本进行输出
			target_predict[i] = NULL;
			EvaluateOneSample(data[i], loquatRForest, target_predict[i]);
			for (int j = 0; j < datainfo_r.variables_num_y; j++)
				mean_squared_error[j] += (target[i][j] - target_predict[i][j])*(target[i][j] - target_predict[i][j]);
		}

		cout << "mse: ";
		for (int j = 0; j < datainfo_r.variables_num_y; j++)
		{
			mean_squared_error[j] /= datainfo_r.samples_num;
			cout << mean_squared_error[j] << ", ";
		}
		cout << endl;
		if (chCommand[2] != NULL) // -o result.out
		{
			ofstream res(chCommand[2]);
			res << "target\ttarget(predicted)" << endl;
			for (int i = 0; i < datainfo_r.samples_num; i++)
			{
				res << "[";
				for (int j=0; j<datainfo_r.variables_num_y; j++)
				{
					res << target[i][j];
					if (j!=datainfo_r.variables_num_y-1)
						res <<',';
				}
				res<< "]"<<"\t"<<"[";
				for (int j=0; j<datainfo_r.variables_num_y; j++)
				{
					res << target_predict[i][j];
					if (j!=datainfo_r.variables_num_y-1)
						res <<',';
				}
				res<< "]" <<endl;
			}
			
			res.close();
		}
		break;
	case 0:
	default:
		printRFConfigC(loquatCForest->RFinfo);
		error_num = 0;
		label_predict = new int[datainfo_c.samples_num];
		int* error_c = new int[datainfo_c.classes_num];
		int* num_c = new int[datainfo_c.classes_num];
		memset(error_c, 0, sizeof(int) * datainfo_c.classes_num);
		memset(num_c, 0, sizeof(int) * datainfo_c.classes_num);

		for (int i = 0; i < datainfo_c.samples_num; i++)
		{
			EvaluateOneSample(data[i], loquatCForest, label_predict[i], 0);
			num_c[label[i]]++;
			if (label_predict[i] != label[i])
			{
				error_num++;
				error_c[label[i]]++;
			}
		}

		for (int i = 0; i < datainfo_c.classes_num; i++)
		{
			if (num_c[i] == 0)
				continue;
			cout << setfill(' ') << "class " << i << ": " << error_c[i] << "/" << setw(10) << left << num_c[i];
			cout << " \terror rate: " << setprecision(5) << error_c[i] * 1.0 / num_c[i] * 100 << "%" << endl;
		}
		cout << "________________________________________________" << endl;
		cout << "       " << error_num << "/" << datainfo_c.samples_num << setprecision(5) << "\t\terror rate: " << error_num * 1.0 / datainfo_c.samples_num * 100 << "%" << endl;
		delete[] error_c;
		delete[] num_c;
		if (chCommand[2] != NULL) // -o result.out
		{
			ofstream res(chCommand[2]);
			res << "label\tlabel(predicted)" << endl;
			for (int i = 0; i < datainfo_c.samples_num; i++)
			{
				res << label[i] << "\t" << label_predict[i] << endl;
			}
			res.close();
		}
	}

	if (1 != rv)
		goto GAME_OVER;

	// (4) clearing work
	switch (prob)
	{
	case 1:
		ReleaseRegressionForest(&loquatRForest);
		break;
	case 0:
	default:
		ReleaseClassificationForest(&loquatCForest);
	}

	getchar();

GAME_OVER:
	int samples_num = prob == 1 ? datainfo_r.samples_num : datainfo_c.samples_num;
	for (int i = 0; i < samples_num; i++)
		delete[]data[i];
	delete[] data;
	switch (prob)
	{
	case 1:
		for (i = 0; i < samples_num; i++)
			delete[] target[i];
		delete[] target;
		delete[] mean_squared_error;
		for (int i = 0; i < datainfo_r.samples_num; i++)
			delete[] target_predict[i];
		delete[] target_predict;
		break;
	case 0:
	default:
		delete[] label;
		delete[] label_predict;
	}

	for (i = 0; i < 3; i++)
		delete[] chCommand[i];

	return 1;
}
