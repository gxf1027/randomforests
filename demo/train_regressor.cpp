/*
Author: GXF
Contact: gxf1027@126.com
Train random forests from codes
*/

#include "../src/RandomRLoquatForests.h"
#include "../src/UserInteraction2.h"
using namespace std;


int main(int argc, char** argv)
{
	// read training samples if necessary 
	char filename[500] = "../dataset/regression/Combined_Cycle_Power_Plant.gxf.data";
	float** data = NULL;
	float* target = NULL;
	Dataset_info_R datainfo;
	int rv = InitalRegressionDataMatrixFormFile2(filename, data /*OUT*/, target /*OUT*/, datainfo /*OUT*/);
	// check the return value
	// 	... ...
	
	// setting random forests parameters
	RandomRForests_info rfinfo;
	rfinfo.datainfo = datainfo;
	rfinfo.maxdepth = 40;
	rfinfo.ntrees = 200;
	rfinfo.mvariables = (int)(datainfo.variables_num_x / 3.0 + 0.5);
	rfinfo.minsamplessplit = 5;
	rfinfo.randomness = 1;
	rfinfo.predictionModel = PredictionModel::constant;
	// train forest
	LoquatRForest* loquatRForest = NULL;
	rv = TrainRandomForestRegressor(data, target, rfinfo, loquatRForest /*OUT*/, false, 20);
	// check the return value
	// 	... ...

	float* mean_squared_error = NULL;
	MSEOnOutOfBagSamples(data, target, loquatRForest, mean_squared_error /*OUT*/);
	delete[] mean_squared_error;
	// save RF model, 0:xml, 1:plain text
	SaveRandomRegressionForestModel("testModelfile-R.xml", loquatRForest, 0);
	// clear the memory
	ReleaseRegressionForest(&loquatRForest);
	// release money: data, target
	for (int i = 0; i < datainfo.samples_num; i++)
		delete[] data[i];
	delete[] data;
	delete[] target;
	return 0;
}
