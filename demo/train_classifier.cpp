/*
Author: GXF
Contact: gxf1027@126.com
Train random forests from codes
*/

#include "../src/RandomCLoquatForests.h"
#include "../src/UserInteraction2.h"
#include <cmath>
using namespace std;


int main(int argc, char** argv)
{
	// read training samples if necessary
	char filename[500] = "../dataset/classification/spambase.gxf.data";
	float** data = NULL;
	int* label = NULL;
	Dataset_info_C datainfo;
	int rv = InitalClassificationDataMatrixFormFile2(filename, data/*OUT*/, label/*OUT*/, datainfo/*OUT*/);
	// check the return value
	// 	... ...
	
	// setting random forests parameters
	RandomCForests_info rfinfo;
	rfinfo.datainfo = datainfo;
	rfinfo.maxdepth = 40;
	rfinfo.ntrees = 200;
	rfinfo.mvariables = (int)sqrtf(datainfo.variables_num);
	rfinfo.minsamplessplit = 5;
	rfinfo.randomness = 1;
	// train forest
	LoquatCForest* loquatCForest = NULL;
	rv = TrainRandomForestClassifier(data, label, rfinfo, loquatCForest /*OUT*/, -1/*random_state*/, 20);
	// check the return value
	// 	... ...

	float error_rate = 1.f;
	OOBErrorEstimate(data, label, loquatCForest, error_rate /*OUT*/);
	// save RF model, 0:xml, 1:plain text
	SaveRandomClassificationForestModel("Modelfile.xml", loquatCForest, 0);
	// clear the memory allocated for the entire forest
	ReleaseClassificationForest(&loquatCForest);
	// release money: data, label
	for (int i = 0; i < datainfo.samples_num; i++)
		delete[] data[i];
	delete[] data;
	delete[] label;
	return 0;
}
