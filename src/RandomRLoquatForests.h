/*
Some instructions:
For regression, the general rule is to set leaf size to 5 and select one third of input features for decision splits at random. Matlab(R2010b) Help document:'Regression and Classification by Bagging Decision Trees'.
*/

#pragma once
#ifndef _GXF_PLANTING_REGRESSION_LOQUAT_FORESTS_
#define _Gxf_PLANTING_REGRESSION_LOQUAT_FORESTS_

#include <stdlib.h>
#include "SharedRoutines.h"

typedef struct Dataset_info_R
{
	int samples_num;
	int variables_num_x;
	int variables_num_y;
}Dataset_info_R;

typedef enum PredictionModel
{
	constant = 1,
	linear
}PredictionModel;

typedef struct
{
	Dataset_info_R datainfo;
	int maxdepth;
	int ntrees;
	int mvariables;
	int minsamplessplit;
	int randomness;
	PredictionModel predictionModel;
}RandomRForests_info;

typedef struct _LeafNodeInfo
{
	float *MeanOfArrived;
	double **CovMatOfArrived;
	double *linearPredictor;
	float arrivedRatio;
	int dimension;
}LeafNodeInfo;

struct LoquatRTreeNode
{
	enum TreeNodeTpye nodetype;
	int depth;

	int arrival_samples_num;
	const int *samples_index;

	int split_variable_index;
	float split_value;

	struct LoquatRTreeNode *pParentNode;
	struct LoquatRTreeNode **pSubNode;	
	int subnodes_num;

	float train_impurity;

	LeafNodeInfo *pLeafNodeInfo;
};

struct LoquatRTreeStruct
{
	int *inbag_samples_index;
	int inbag_samples_num;
	int *outofbag_samples_index;
	int outofbag_samples_num;
	int depth;						// the largest depth index, from 0 
	int leaf_node_num;
	struct LoquatRTreeNode *rootNode;
};

typedef struct LoquatRForestSturct
{
	struct LoquatRTreeStruct **loquatTrees;
	RandomRForests_info RFinfo;
	bool bTargetNormalize;
	float *scale;
	float *offset;
}LoquatRForest;

void UseDefaultSettingsForRFs(RandomRForests_info &RF_info);

int CheckRegressionForestParameters(RandomRForests_info &RF_info);

/*
Description:	Train a Random Regression Forests model

[in]	1.data:		two dimension array [N][M], containing the total training data with their variable
		2.target:	two dimension array [N][K], the target(output) value of the training data, multi-dimensional output is supported
 		3.RFinfo:	the struct contains necessary information of Random Regression Forests, namely the number of trees, and the number 
					of variable split at each node.
		4.bTargetNormalize: whether the target is normalized

[out]	1.loquatForest: the trained RF model, containing N trees.

return:
		1. -3: The data_info structure is assigned with incorrect values.
		2. -2: 'loquatForest' may be allocated with memory or isn't assigned with NULL.
		3. -1: Error happened in function 'GrowRandomizedRLoquatTree'.
		4.  1: A RFs model is build successfully.

NOTE: The user MUSTN'T allocate memory for loquatForest before this function is called, and SHOULD assign NULL to 'loquatForest' structure.
      Memory management is handled by the function automatically.
*/
int TrainRandomForestRegressor(float **data, float *target, RandomRForests_info RFinfo, LoquatRForest *&loquatForest, bool bTargetNormalize=true, int trace=0);

/*-----------------------------------------------CLEAR MEMORY-----------------------------------------------*/

/*
Description:	Release all the memory allocated for a forest.
[in/out]
		1.	loquatForest:		A forest to be deleted. And the address of this pointer pointing to the forest structure is assigned with NULL.
*/
int ReleaseRegressionForest(LoquatRForest** loquatForest);


/*-----------------------------------------------EVALUATE and TEST-----------------------------------------------*/
int EvaluateOneSample(float *data, LoquatRForest *loquatForest, float *&target_predicted, int nMethod=0);

int MSEOnTestSamples(float **data_test, float *target, int nTestSamplesNum, LoquatRForest *loquatForest, float *&mean_squared_error, int nMethod=0, char *RecordName =NULL);

int MSEOnOutOfBagSamples(float **data, float *target_test, LoquatRForest *loquatForest, float *&mean_squared_error);


#endif
