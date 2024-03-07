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

typedef enum SplitCriterion
{
	mse = 1,
	covar
}SplitCriterion;

typedef struct
{
	Dataset_info_R datainfo;
	int maxdepth;
	int ntrees;
	int mvariables;
	int minsamplessplit;
	int randomness;
	PredictionModel predictionModel;
	SplitCriterion splitCrierion;
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
	int *samples_index;

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
		2.target:	one dimension array [N*K], the target(output) value of the training data, multi-dimensional output is supported
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

/*
Description:     Compute raw/z-score variables Importance.
Method:         "In every tree grown in the forest, put down the oob cases and count the number of votes cast for the correct class.
				 Now randomly permute the values of variable m in the oob cases and put these cases down the tree.
				 Subtract the number of votes for the correct class in the variable-m-permuted oob data from the number of votes for the correct class in the untouched oob data.
				 The average of this number over all trees in the forest is the raw importance score for variable m.

				 If the values of this score from tree to tree are independent, then the standard error can be computed by a standard computation.
				 The correlations of these scores between trees have been computed for a number of data sets and proved to be quite low,
				 therefore we compute standard errors in the classical way, divide the raw score by its standard error to get a z-score,
				 ands assign a significance level to the z-score assuming normality."
				 Leo Breiman and Adele Cutler:(https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#varimp)

[in]:			1.data:			two dimension array [N][M], containing the total training data with their variable
				2.label:		the labels of training data
				3.loquatForest:	the trained Random Forests model, which also includes data information and model information
				4.nType:		0: raw variable importance score
								1: z-score
[out]:			1.varImportance:	normalized raw/z-score importance score.
*/
int RawVariableImportanceScore(float** data, float* target, LoquatRForest* loquatForest, int nType, float* varImportance, bool bNormalize, char* filename);

/*
Description:	calculate proximities between the i-th sample and every other sample with algorithm proposed by
				'Jake S.Rhodes, Adele Cutler, Kevin R. Moon. Geometry- and Accuracy-Preserving Random Forest Proximities. TPAMI,2023.'
[in]
[out]			proximities:  a pointer to the 1D array, with the dimension samples_num*1.
return:			1  -- success
				-1 -- i-th sample is not a out-of-bag sample for every tree in forest. Possible when the number of trees is small.

*/
int RegressionForestGAPProximity(LoquatRForest* forest, float** data, const int index_i, float*& proximities /*OUT*/);

#endif
