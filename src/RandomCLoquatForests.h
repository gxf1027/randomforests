/*
GuXF, 9.2011, @NJUST, lastest modification 7.2022
Contact: gxf1027@126.com.
Try my best to implement random forests.
Reference 	[1]. Leo Breiman. Random Forests. Machine Learning 45 (1), 5¨C32, 2001.
			[2]. Random Forests classifier description (Official site of Leo Breiman's RF): http://stat-www.berkeley.edu/users/breiman/RandomForests/cc_home.htm
			[3]. Ho, Tin. Random Decision Forest. 3rd International Conference on Document Analysis and Recognition, 1995: 278¨C282.
			[4]. ALGLIB , Implementation of modified random forest algorithm: http://www.alglib.net/dataanalysis/decisionforest.php
			[5]. Matlab 2010b Help document: "Regression and Classification by Bagging Decision Trees".
			[6]. Robert E.Banfield. A comparison of decision tree ensemble creation techniques. IEEE trans. on Pattern Analysis and Machine Intelligence, 2007
			[7]. Antonio Criminisi, Ender Konukoglu, Jamie Shotton. Decision Forests for Classification, Regression, Density Estimation, Manifold Learning and Semi-Supervised Learning. MSR-TR-2011-114. (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/decisionForests_MSR_TR_2011_114.pdf)
			[8]. Geurts, P., Ernst, D. & Wehenkel, L. Extremely randomized trees. Machine Learning 63, 3¨C42, 2006.
*/

#pragma once
#ifndef _GXF_PLANTING_CLASSIFICATION_LOQUAT_FORESTS_
#define _GXF_PLANTING_CLASSIFICATION_LOQUAT_FORESTS_

#include <string>
#include "SharedRoutines.h"

/*-----------------------------------------------STRUCTURE DIFINITION-----------------------------------------------*/
typedef struct Dataset_info_C
{
	int samples_num;
	int variables_num;
	int classes_num;
}Dataset_info_C;

typedef struct
{
	Dataset_info_C datainfo;
	int maxdepth;
	int ntrees;
	int mvariables;
	int minsamplessplit;
	int randomness;
}RandomCForests_info;

typedef struct
{
	int SlideSize;
	int SlideWindowSize;
	int BuildSize;
}PlantStopCriterion;

struct LoquatCTreeNode
{
	enum TreeNodeTpye nodetype;
	int depth;

	int arrival_samples_num;
	int *samples_index;

	int split_variable_index;
	float split_value;

	struct LoquatCTreeNode *pParentNode;
	struct LoquatCTreeNode **pSubNode;	
	int subnodes_num;

	float train_impurity;
	float *class_distribution;

	int leaf_node_label;
	float leaf_confidence;
};


struct LoquatCTreeStruct
{
	int *inbag_samples_index;
	int inbag_samples_num;
	int *outofbag_samples_index;
	int outofbag_samples_num;
	int depth;						// the largest depth index, from 0 
	int leaf_node_num;
	struct LoquatCTreeNode *rootNode;
};

typedef struct LoquatCForestSturct
{
	struct LoquatCTreeStruct **loquatTrees;
	RandomCForests_info RFinfo;
}LoquatCForest;


/*-----------------------------------------------TRAIN-----------------------------------------------*/

void UseDefaultSettingsForRFs(RandomCForests_info &RF_info);

/*
Description: Make sure that the elements of struct RandomCForests_info are correctly assigned and so do the labels.
return:	1. -1  : the data_info structure is assigned with incorrect values.
		2.  0  : other incorrectly assigned values are found, and default or recommended values are assigned to them.
		3.  1  : values of all parameters are of correct range.
*/
int CheckClassificationForestParameters(RandomCForests_info &RF_info);

/*
Description:	Train a Random Classification Forests model

[in]	1.data:    two dimension array [N][M], containing the total training data with their variable
		2.label:   the labels of the training data
		3.RFinfo:  the struct contains necessary information of Random Classification Forests, namely the number of trees, and the number 
		   		 of variable split at each node.
		4.trace:   if >0, print oob error rate every 'trace' trees during training

[out]	1.loquatForest: the trained RF model, containing N trees.

return:
		1. -3: The data_info structure is assigned with incorrect values.
		2. -2: 'loquatForest' may be allocated with memory or isn't assigned with NULL.
		3. -1: Error happened in function 'GrowRandomizedDLoquatTree'.
		4.  1: A RFs model is build successfully.

NOTE: The user MUSTN'T allocate memory for loquatForest before this function is called, and SHOULD assign NULL to 'loquatForest' structure.
	  Memory management is handled by the function automatically.
*/
int TrainRandomForestClassifier(float** data, int* label, RandomCForests_info RFinfo, LoquatCForest*& loquatForest, int trace = 0);

/*
Description:	Train a Random Forests model using adaptive stopping criterion,
NOTE:           The maximum number of trees is bound to 'RFinfo.nTrees', and 'stopCriterion' includes parameters corresponding to the behaviour of evaluating sequential oob error.
Reference:      A comparison of decision tree ensemble creation techniques. TPAMI07
return:
		1. -4: Error happened in evaluating oob error.
		2. -3: The data_info structure is assigned with incorrect values.
		3. -2: 'loquatForest' may be allocated with memory or isn't assigned with NULL.
		4. -1: Error happened in function 'GrowRandomizedDLoquatTree'.
		5.  1: A RFs model is build successfully.
*/
int TrainRandomForestClassifierWithStopCriterion(float **data, int *label, RandomCForests_info RFinfo, LoquatCForest *&loquatForest, 
											  PlantStopCriterion stopCriterion, int &nPlantedTreeNum, float *&error_rate_sequent);

/*
Description:	Output detail of Random Forest model
[in]  1.forest:  a trained random forest model

	  2.out:    std::out, file(an object of ofstream) or others
*/
void PrintForestInfo(const LoquatCForest *forest, std::ostream &out);

/*-----------------------------------------------CLEAR MEMORY-----------------------------------------------*/

/*
Description:	Release all the memory allocated for a forest.
[in/out]
		1.	loquatForest:		A forest to be deleted. And the address of this pointer pointing to the forest structure is assigned with NULL.
*/
int ReleaseClassificationForest(LoquatCForest **loquatForest);

/*-----------------------------------------------EVALUATE and TEST-----------------------------------------------*/

/*
Description:	Predict class label of one testing sample.
[in]:		1.data:				one dimension array [N], containing one testing sample
			2.loquatForest:		the trained Random Forests model, which also includes data information and model information
			3.nType:			1: hard voting decision		 Count(c|data) = ¡Æ(t)delt{leaf_t(data).label==c}
								0: confidence voting decision   P(c|data) = ¡Æ(t:leaf_t(data).label==c){Pt(c|data)}
								otherwise: confidence 'soft' decision   P(c|data) = ¡Æ(t){Pt(c|data)}, 
																		C = argmax(c){P(c|data)}
																		t=1...T(the number of trees)
[out]:		1.label_index 	Predicted class label, starting from zero.
return:		1: Successfully
			0: Errors have occured when getting the arrived leafnode of the samples on certain trees. Under this condition, value of the argument 'label_index' indicates less reliability.
			-1:Errors have occured when getting the arrived leafnode of the samples on all of the trees.
                 
*/
int EvaluateOneSample(float *data, LoquatCForest *loquatForest, int &label_index, const int isHardDecision=1);

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
int RawVariableImportanceScore(float **data, int *label, LoquatCForest *loquatForest, int nType, float *varImportance, bool bNormalize, char *filename=0);
int RawVariableImportanceScore2(float** data, int* label, LoquatCForest* loquatForest, int nType, float* varImportance, bool bNormalize, char* filename = 0);


/*
Description: Using in-bag samples to estimate training error.
*/
int ErrorOnInbagTrainSamples(float **data, const int *label, const LoquatCForest *loquatForest, float &error_rate, int isHardDecision=1);

/*
Description: Using out-of-bag(oob) samples to estimate generalization error.
Method:		"Put each case left out in the construction of the kth tree down the kth tree to get a classification. 
			 In this way, a test set classification is obtained for each case in about one-third of the trees. 
			 At the end of the run, take j to be the class that got most of the votes every time case n was oob. 
			 The proportion of times that j is not equal to the true class of n averaged over all cases is the oob error estimate. 
			 This has proven to be unbiased in many tests."
			 Leo Breiman and Adele Cutler:(http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr)

[in]	1.data:			two dimension array [N][M], containing the total training data with their variable.
		2.label:		the labels of training data.
		3.loquatForest:	A trained random forest containing trees.
[out]	1.error_rate:	Estimated generalization error.

return	1. >0			done successfully
		2. <=0			some errors happened(0:GetArrivedLeafNode returns NULL, -1: one of at least one trees is NULL, -2: oob samples of at least one trees are missing)
*/
int OOBErrorEstimate(float** data, const int* label, const LoquatCForest* loquatForest, float& error_rate, int isHardDecision = 1);



/*
Description:  Compute generalization performance of trained RF model with test dataset.

[in]	1.data_test:		test dataset, the number of variables and classes must be identical with the training dataset.
		2.label_test:		class label for each samples in test dataset.
		3.nTestSamplesNum:	the number of samples in test dataset.
		4.loquatForest:		the trained RF model.
[out]  1.error_rate: 		error_rate on test dataset.
*/
int ErrorOnTestSamples(float **data_test, const int *label_test, const int nTestSamplesNum, const LoquatCForest *loquatForest, float &error_rate, int isHardDecision=1);


/*
Description:	calculate proximities between the i-th sample and every other sample with algorithm proposed by
				'Jake S.Rhodes, Adele Cutler, Kevin R. Moon. Geometry- and Accuracy-Preserving Random Forest Proximities. TPAMI,2023.'
[in]
[out]			proximities:  a pointer to the 1D array, with the dimension samples_num*1.
return:			1  -- success
				-1 -- i-th sample is not a out-of-bag sample for every tree in forest. Possible when the number of trees is small.

*/
int ClassificationForestGAPProximity(LoquatCForest* forest, float** data, const int index_i, float*& proximities);

/*
Description:    Compute raw outlier measurement using RF-GAP.
Method:			"Outliers are generally defined as cases that are removed from the main body of the data. Translate this as:
				 outliers are cases whose proximities to all other cases in the data are generally small.
				 A useful revision is to define outliers relative to their class.
				 Thus, an outlier in class j is a case whose proximities to all other class j cases are small."
				 Leo Breiman and Adele Cutler:(https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#outliers)
				 raw_score = (samples_num/P_hat(n)-median)/dev, where P_hat(n) = ¡ÆProx^2(n,k),subject to the k that cl(k)=cl(n) and n¡Ùk.

[in]:            1.loquatForest
				 2.data
				 3.label
				 4.raw_score: MUST be assigned with 'NULL'

[out]:           1.raw_score: a pointer to the 1D array, with the dimension samples_num*1.
*/
int RawOutlierMeasure(LoquatCForest* loquatForest, float** data, int* label, float*& raw_score);

#endif
