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
	// (0) parse command line
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

	// (1) Read parameters for Random Forests
	RandomCForests_info RFinfo_C;
	RandomRForests_info RFinfo_R;
	string datapath;
	switch (prob)
	{
	case 1:
		if (chCommand[0] != NULL && chCommand[1] != NULL) // -c config.xml
		{
			int rv = ReadRegressionForestConfigFile2(chCommand[0], RFinfo_R);
			if (0 > rv)
			{
				cout << "Reading configuration file: " << chCommand[0] << " failed!" << endl;
				for (i = 0; i < 3; i++)
					if (chCommand[i])
						delete[] chCommand[i];
				return -1;
			}
			else {
				// 其他参数
				RFinfo_R.predictionModel = PredictionModel::constant; // 不对外开放
			}
		}
		else // -d data.txt
		{

			// RFinfo_R.maxdepth = 40;
			// RFinfo_R.mvariables = -1;
			// RFinfo_R.ntrees = 200;
			// RFinfo_R.selratio = 1.0f;
			// RFinfo_R.minsamplessplit = 5;
			// RFinfo_R.predictionModel = PredictionModel::constant;
			// RFinfo_R.randomness = RF_TREE_RANDOMNESS::TREE_RANDOMNESS_WEAK;
			//UseDefaultSettingsForRFs(RFinfo_R);
		}
		break;
	case 0:
	default:
		if (chCommand[0] != NULL && chCommand[1] != NULL) // -c config.xml
		{
			int rv = ReadClassificationForestConfigFile2(chCommand[0], RFinfo_C);
			if (0 > rv)
			{
				cout << "Reading configuration file: " << chCommand[0] << " failed!" << endl;
				for (i = 0; i < 3; i++)
					if (chCommand[i])
						delete[] chCommand[i];
				return -1;
			}
			else
			{
				// 其他参数
			}
		}
		else // -d data.txt
		{
			// RFinfo_C.maxdepth = 40;
			// RFinfo_C.mvariables = -1;
			// RFinfo_C.ntrees = 100;
			// RFinfo_C.selratio = 1.0f;
			// RFinfo_C.minsamplessplit = 5;
			// RFinfo_C.randomness = RF_TREE_RANDOMNESS::TREE_RANDOMNESS_WEAK;
			//UseDefaultSettingsForRFs(RFinfo_C); // 在读取完训练样本后才能设置默认参数，应为m=sqrt(datainfo.variable_x)
		}
	}

	// (2) Read training data
	if (chCommand[1] == NULL)
	{
		cout << "Training data is not assigned." << endl;
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
			if (-2==rd)
				cout << "Reading file: " << chCommand[1] << " failed!" << endl;
			if (-1==rd)
				cout << "the format of data or extra information isn't correctly compiled" << endl;

			for (i = 0; i < 3; i++)
				if (chCommand[i])
					delete[] chCommand[i];
			return -1;
		}

		RFinfo_R.datainfo = datainfo_r;
		if (chCommand[0] == NULL)
		{
			cout << "Default parameters for Random Forests are used." << endl;
			UseDefaultSettingsForRFs(RFinfo_R);
		}
		//RFinfo_R.mvariables = RFinfo_R.mvariables<=0 ? datainfo_r.variables_num_x/3.0 : RFinfo_R.mvariables;
		printRFConfigR(RFinfo_R);
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

		RFinfo_C.datainfo = datainfo_c;
		//RFinfo_C.mvariables = RFinfo_C.mvariables<=0 ? (int)sqrtf((float)datainfo_c.variables_num) : RFinfo_C.mvariables;
		if (chCommand[0] == NULL)
		{
			cout << "Default parameters for Random Forests are used." << endl;
			UseDefaultSettingsForRFs(RFinfo_C);
		}
		printRFConfigC(RFinfo_C);
	}


	LoquatCForest* loquatCForest = NULL;
	LoquatRForest* loquatRForest = NULL;

		// (3) Train
	float* mean_squared_error = NULL;
	timeIt(1);
	int rv;
	int trace = 20;
	switch (prob)
	{
	case 1:
		trace = RFinfo_R.ntrees <= 100 ? 10 : (RFinfo_R.ntrees < 500 ? 20 : 50);
		rv = TrainRandomForestRegressor(data, target, RFinfo_R, loquatRForest, false, trace);
		cout << "Train Regression Forests Successfully." << endl;
		cout << "time consumption:" << timeIt(0) << endl;
		MSEOnOutOfBagSamples(data, target, loquatRForest, mean_squared_error);
		cout << "mean squared error on oob data:" << endl;
		for (int k = 0; k < datainfo_r.variables_num_y; k++)
			cout << mean_squared_error[k] << " ";
		cout << endl;
		delete[] mean_squared_error;
		if (chCommand[2] != NULL)
			SaveRandomRegressionForestModelToXML2(chCommand[2], loquatRForest);
		break;
	case 0:
	default:
		trace = RFinfo_C.ntrees <= 100 ? 10 : (RFinfo_C.ntrees < 500 ? 20 : 50);
		rv = TrainRandomForestClassifier(data, label, RFinfo_C, loquatCForest, trace);
		cout << "Train Classification Forests Successfully." << endl;
		cout << "time consumption:" << timeIt(0) << endl;
		float error_rate = -1.f;
		OOBErrorEstimate(data, label, loquatCForest, error_rate, 1);
		cout << "error rate oob:    " << error_rate * 100 << "%" << endl;
		ErrorOnInbagTrainSamples(data, label, loquatCForest, error_rate, 1);
		cout << "error rate in-bag: " << error_rate * 100 << "%" << endl;
		// Save in xml format
		if (chCommand[2] != NULL)
			SaveRandomClassificationForestModelToXML2(chCommand[2], loquatCForest);
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
		break;
	case 0:
	default:
		delete[] label;
	}

	for (i = 0; i < 3; i++)
		delete[] chCommand[i];

	return 1;
}