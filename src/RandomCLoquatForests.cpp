/*
Author: GuXF
Contact: gxf1027@126.com
*/

#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <vector>
#include <deque>
#include <cassert>
#include <cstring> // for memset...
#include <float.h>
#include <algorithm>
#include <iomanip>
#include <map>
#include <unordered_map>

using namespace std;

#include "RandomCLoquatForests.h"
#include "SharedRoutines.h"

#ifdef OPENMP_SPEEDUP
#include "omp.h"
#endif


#define FLOAT_MAX							3.0e38f
#define VERY_SMALL_VALUE					1e-10f
#define INTERVAL_STEPS_NUM					50
#define STOP_CRITERION_MIN_GINI_IMPURITY	0.01    // 这个参数变小(0.1->0.01)可以显著提高分类准确率
#define DEFAULT_MAX_TREE_DEPTH_C			40
#define DEFAULT_MIN_SAMPLES_C				5
#define GROW_DEEPER_FOR_PROXIMITY			8124
#define NODE_NUM_TO_GROW_DEEPER				50
//#define new  new(_CLIENT_BLOCK, __FILE__, __LINE__)

struct _GrowNodeInput
{
	int total_samples_num;
	int total_variables_num;
	int total_classes_num;
	int mvariables;

	int leafMinSamples;
	int parent_depth;
	int maxDepth;
	int randomness;
};
typedef struct _GrowNodeInput GrowNodeInput;


LoquatCTreeNode::~LoquatCTreeNode()
{
	delete[] samples_index;
	delete[] class_distribution;
		
	if (pSubNode != NULL)
	{
		delete pSubNode[0];
		delete pSubNode[1];
	}
	delete pSubNode;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NOT Ready to Publish
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
Description: Using out-of-bag(oob) samples to estimate generalization error sequentially.

[in]   1.data:                   two dimension array [N][M], containing the total training data with their variable.
	   2.label:                  the labels of training data.
	   3.loquatForest:           A trained random forest containing trees.
[out]  1.error_rate_sequent:     Estimated generalization error according to growing tree number.
*/
int OOBErrorEstimateSequential(float** data, int* label, LoquatCForest* loquatForest, float*& error_rate_sequent, int isHardDecision = 1, char* filename = NULL);

/*
Description:       Compute RF margin on one sample, using the method by Amir Saffari's ORF paper
				   mg(X,y) = P(y|X) - max(p(k|X)), k≠y, y is the true label
[in]:  1. data:            one sample
	   2. label:           the true label
	   3. loquatForest:    Random Forests Model
[out]  1. margin:          margin computed

return:
		1: computing margin successfully
		0: Errors happen at getting the pointer of leaf node;
*/
int ComputeWeightedMargin(float** data, int* label, int samples_num, LoquatCForest* loquatForest, float& margin);

/*
Description:       Compute RF margin on one sample, using Breiman’s Random Forests paper
				   mg(X,y) = avkI(hk(X)=y) - max avkI(hk(X)=j), j≠y, y is the true label
[in]:  1. data:            one sample
	   2. label:           the true label
	   3. loquatForest:    Random Forests Model
[out]  1. margin:          margin computed

return:
		1: computing margin successfully
		0: Errors happen at getting the pointer of leaf node;
*/
int ComputeVotingMargin(float** data, int* label, int samples_num, LoquatCForest* loquatForest, float& margin);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/*
Description:	 To build an entire randomized tree without pruning

[in]  1.data:    two dimension array [N][M], containing the total training data with their variable

	  2.label:   the labels of training data

	  3.RFinfo:  the struct contains necessary information of Random Forests, namely the number of trees, and the number
				 of variable splitted at each node.

[out] 1.loquatTree: the grown tree, containing each nodes, the input tree must be NULL without memory has been allocated.

[return]
	  1.   -2:   the input value of the pointer loquatTree is not NULL, maybe the memory has been allocated before calling this function
	  2.    1:   A randomized tree is grown successfully.
NOTE: The user MUSTN'T allocate memory for loquatTree, and should assign NULL to 'loquatForest' structure.
	  Memory management is handled by the function.
*/
int GrowRandomizedCLoquatTreeRecursively(float** data, int* label, RandomCForests_info RFinfo, struct LoquatCTreeStruct*& loquatTree);    

/*
Description:    evaluate an observation(sample) on one tree
return:         1--success
*/
int PredictAnTestSampleOnOneTree(float* data, int variables_num, struct LoquatCTreeStruct* loquatTree, int& predicted_class_index, float& confidence);
/*
Description:    Get a leaf node a specified sample falling into.
return:         the pointer to the leaf node. So all of the attributes that leaf node possesses is available
*/
const struct LoquatCTreeNode* GetArrivedLeafNode(const LoquatCForest* RF, const int tree_index, float* data);

int HarvestOneLeafNode(struct LoquatCTreeNode** treeNode);

void UseDefaultSettingsForRFs(RandomCForests_info &RF_info)
{
	RF_info.ntrees = 200;
	RF_info.mvariables = (int)(sqrtf((float)(RF_info.datainfo.variables_num)) + 0.5);
	if( RF_info.mvariables < 1 )
		RF_info.mvariables = 1;
	RF_info.minsamplessplit = DEFAULT_MIN_SAMPLES_C;
	RF_info.randomness = TREE_RANDOMNESS_WEAK; // RF_TREE_RANDOMNESS::TREE_RANDOMNESS_WEAK;
	RF_info.maxdepth = DEFAULT_MAX_TREE_DEPTH_C;
}

static float split_count = 0;
static time_t tm1 = 0;

int CheckClassificationForestParameters(RandomCForests_info &RF_info)
{
	if( RF_info.datainfo.classes_num<=1 || RF_info.datainfo.samples_num<=0 || RF_info.datainfo.variables_num<= 0 )
	{
		cout<<">>>>>>>>>>>>>>>>>>>>>>>>>Parameters Check Information>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl;
		cout<<"Data Information is not properly assigned to."<<endl;
		if(RF_info.datainfo.classes_num <= 1)
			cout<<"    Classes_num can't be less than 1"<<endl;
		if( RF_info.datainfo.samples_num <= 0 )
			cout<<"    Samples_num can't be less than 0"<<endl;
		if( RF_info.datainfo.variables_num <= 0 )
			cout<<"    Variables_num can't be less than 0"<<endl;
		cout<<"You MUST check out your dataset file."<<endl;
		cout<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl;
		return -1;
	}

	int rv = 1;
	if( RF_info.maxdepth <= 1 )
	{
		cout<<">>>>'maxdepth' must be more than 1,"<<endl;
		cout<<">>>>thus, "<<DEFAULT_MAX_TREE_DEPTH_C<<" is assigned to 'maxdepth'."<<endl;
		RF_info.maxdepth = DEFAULT_MAX_TREE_DEPTH_C;
		rv = 0;
	}

	if( RF_info.mvariables <= 0 )
	{
		int m = (int)sqrtf((float)(RF_info.datainfo.variables_num));
		cout<<">>>>'mvariables' must be no less than 1,"<<endl;
		cout<<">>>>thus, "<<m<<" is assigned to 'mvariables'."<<endl;
		RF_info.mvariables = m;
		rv = 0;
	}
	if( RF_info.mvariables > RF_info.datainfo.variables_num )
	{
		int m = (int)sqrtf((float)(RF_info.datainfo.variables_num));
		cout<<">>>>'mvariables' should be no more than number of the variables of the data set,"<<endl;
		cout<<">>>>thus, the recommended value: "<<m<<" is assigned to."<<endl;
		RF_info.mvariables = m;
		rv = 0;
	}
	if( RF_info.ntrees <=0 )
	{
		cout<<">>>>'ntrees' must be no less than 1,"<<endl;
		RF_info.ntrees = 200;
		cout<<">>>>thus,"<< RF_info.ntrees <<" is assigned to 'ntrees'."<<endl;
		rv = 0;
	}
	if (RF_info.minsamplessplit <= 0)
	{
		cout << ">>>>'minsamplessplit' must be no less than 1," << endl;
		RF_info.minsamplessplit = 5;
		cout << ">>>>thus, the default value:" << RF_info.minsamplessplit << " is assigned to 'minsamplessplit'." << endl;
		rv = 0;
	}
	
	return rv;
}

int chekcDataSetLabel(int* label, int sample_num, int class_num)
{
	int* count = new int[class_num];
	memset(count, 0, sizeof(int) * class_num);

	for (int k = 0; k < sample_num; k++)
	{
		if (label[k] >= class_num || label[k] < 0)
		{
			cout << ">>>>Values of label index must be no more than classes_num-1," << endl;
			cout << ">>>>That is, if the classes_num=N,the label index should be 0,1,...,N-1" << endl;
			delete[]count;
			return -1;
		}
		count[label[k]]++;
	}

	for (int c = 0; c < class_num; c++)
	{
		if (count[c] == 0)
		{
			cout << ">>>>NO sample has the label index " << c << "." << endl;
			delete[] count;
			return -2;
		}
	}

	delete[] count;
	return 1;
}


int TrainRandomForestClassifierWithStopCriterion(float** data, int* label, RandomCForests_info RFinfo, LoquatCForest*& loquatForest,
										PlantStopCriterion stopCriterion, int random_state, int& nPlantedTreeNum, float*& error_rate_sequent)
{
	if (random_state < 0)
	{
		srand_freebsd(GenerateSeedFromSysTime());
	}
	else
	{
		srand_freebsd((unsigned int)random_state);
	}

	if (loquatForest != NULL)
		return -2;

	int rv = CheckClassificationForestParameters(RFinfo);
	switch (rv)
	{
	case -1:
		cout << "--------------ERROR:'PlantRandomDLoquatForestsWithStopCriterion'--------------" << endl;
		cout << ">>>>The data_info structure is assigned with incorrect values." << endl;
		cout << "-----------------------------------------------------------------------------" << endl;
		return -3;
	case 0:
		cout << "--------------WARNING:'PlantRandomDLoquatForestsWithStopCriterion'--------------" << endl;
		cout << ">>>>Some incorrectly assigned values are found, and default or recommended values are assigned to them." << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		break;
	case 1:
		break;
	}

	if (stopCriterion.BuildSize <= stopCriterion.SlideSize)
	{
		cout << "--------------WARNING:'PlantRandomDLoquatForestsWithStopCriterion'--------------" << endl;
		cout << ">>>>Stopping criterion is not properly designated." << endl;
		cout << ">>>>'BuildSize' is set to 20, and 'SlideSize' to 5." << endl;
		cout << "-------------------------------------------------------------------------------" << endl;
		stopCriterion.BuildSize = 20;
		stopCriterion.SlideSize = 5;
	}

	int i, j, k, Ntrees, samples_num, classes_num, variables_num, indx, predicted_class_index;
	loquatForest = new LoquatCForest;
	assert(loquatForest != NULL);
	loquatForest->loquatTrees = NULL;
	loquatForest->RFinfo = RFinfo;
	Ntrees = loquatForest->RFinfo.ntrees;
	samples_num = RFinfo.datainfo.samples_num;
	classes_num = RFinfo.datainfo.classes_num;
	variables_num = RFinfo.datainfo.variables_num;


	float* Acc = new float[Ntrees];
	float* Smooth = new float[Ntrees];
	float* MaxSmValue = new float[Ntrees];
	memset(Acc, 0, sizeof(float) * Ntrees);
	memset(Smooth, 0, sizeof(float) * Ntrees);
	memset(MaxSmValue, 0, sizeof(float) * Ntrees);
	//struct LoquatCTreeStruct* ploquatTree = NULL;
	const int* pIndex = NULL;
	int** data_class_count = new int* [samples_num];
	int* predicted_labels = new int[samples_num];
	int oobnum, numofseen = 0, error_num = 0;
	float* error_sequent = new float[Ntrees];
	float confidence;

	for (i = 0; i < samples_num; i++)
	{
		data_class_count[i] = new int[classes_num];
		assert(NULL != data_class_count[i]);
		memset(data_class_count[i], 0, sizeof(int) * classes_num);
		predicted_labels[i] = -1;
	}

	vector<struct LoquatCTreeStruct*> trees;

	// Grow Trees Sequentially
	for (i = 0; i < Ntrees; i++)
	{
		struct LoquatCTreeStruct* ploquatTree = NULL;

		rv = GrowRandomizedCLoquatTreeRecursively(data, label, RFinfo, ploquatTree);
		if (1 != rv)
			return -1;
		
		trees.push_back(ploquatTree);

		// Check Stop Criterion
		oobnum = ploquatTree->outofbag_samples_num;
		pIndex = ploquatTree->outofbag_samples_index;

		for (j = 0; j < oobnum; j++)
		{
			indx = pIndex[j];
			rv = PredictAnTestSampleOnOneTree(data[indx], variables_num, ploquatTree, predicted_class_index, confidence);
			if (rv != 1)
			{
				for (int k = 0; k < samples_num; k++)
				{
					delete[] data_class_count[k];
					data_class_count[k] = NULL;
				}
				delete[] data_class_count;
				data_class_count = NULL;
				delete[] predicted_labels;
				predicted_labels = NULL;
				delete[] error_sequent;
				error_sequent = NULL;
				delete[] Acc;
				delete[] Smooth;
				delete[] MaxSmValue;
				return -4;
			}

			data_class_count[indx][predicted_class_index] += 1;
			if (predicted_labels[indx] == -1)
			{
				predicted_labels[indx] = predicted_class_index;
				numofseen++;
				if (predicted_class_index != label[indx])
					error_num++;
			}
			else
			{
				if (data_class_count[indx][predicted_class_index] > data_class_count[indx][predicted_labels[indx]])
				{
					if (predicted_class_index != label[indx] && predicted_labels[indx] == label[indx])
						error_num++;
					else if (predicted_class_index == label[indx] && predicted_labels[indx] != label[indx])
						error_num--;
					predicted_labels[indx] = predicted_class_index;
				}
			}
		}

		error_sequent[i] = error_num / (float)numofseen;
		Acc[i] = error_sequent[i];
		int st = i - stopCriterion.SlideSize < 0 ? 0 : i - stopCriterion.SlideSize;
		int iid;
		for (k = st; k <= i; k++)
		{
			Smooth[i] += Acc[k];
		}
		Smooth[i] /= (i - st + 1);

		if (i > 0 && i % stopCriterion.BuildSize == 0)
		{
			iid = i / stopCriterion.BuildSize - 1;
			st = i - stopCriterion.BuildSize;
			MaxSmValue[iid] = Smooth[st];
			for (k = st + 1; k <= i; k++)
			{
				if (Smooth[st] > MaxSmValue[iid])
					MaxSmValue[iid] = Smooth[st];
			}

			if (iid - 1 >= 0)
			{
				if (MaxSmValue[iid] >= MaxSmValue[iid - 1]) // satisfy the stopping criterion
					break;
			}

		}
	}

	
	nPlantedTreeNum = trees.size(); // number of planted tree
	//0604
	loquatForest->loquatTrees = new struct LoquatCTreeStruct* [nPlantedTreeNum];
	loquatForest->RFinfo.ntrees = nPlantedTreeNum;
	for (i = 0; i < nPlantedTreeNum; i++)
		loquatForest->loquatTrees[i] = trees[i];

	if (error_rate_sequent)
		delete[] error_rate_sequent;
	error_rate_sequent = new float[nPlantedTreeNum];
	for (j = 0; j < nPlantedTreeNum; j++)
		error_rate_sequent[j] = error_sequent[j];

	// Release allocated memory
	for (j = 0; j < samples_num; j++)
	{
		delete[] data_class_count[j];
		data_class_count[j] = NULL;
	}
	delete[] data_class_count;
	delete[] predicted_labels;
	delete[] error_sequent;
	delete[] Acc;
	delete[] Smooth;
	delete[] MaxSmValue;

	return 1;
}

int TrainRandomForestClassifier(float **data, int *label, RandomCForests_info RFinfo, LoquatCForest *&loquatForest, int random_state, int trace)
{
	if (random_state < 0)
	{
		srand_freebsd(GenerateSeedFromSysTime());
	}
	else
	{
		srand_freebsd((unsigned int)random_state);
	}
	

	if( loquatForest != NULL )
		return -2;

	int rv = CheckClassificationForestParameters(RFinfo);
	switch (rv)
	{
	case 0:
		cout<<">>>>Some incorrectly assigned values are found, and default or recommended values are assigned to them."<<endl;
		break;
	case 1:
		break;
	}

	rv = chekcDataSetLabel(label, RFinfo.datainfo.samples_num, RFinfo.datainfo.classes_num);
	if (rv < 0)
	{
		return -3;
	}

	const int Ntrees = RFinfo.ntrees;
	loquatForest = new LoquatCForest;
	assert( loquatForest != NULL );
	loquatForest->loquatTrees = NULL;
	loquatForest->RFinfo = RFinfo;
	loquatForest->loquatTrees = new struct LoquatCTreeStruct *[Ntrees];

	for( int i=0; i<Ntrees; i++ )
	{
		loquatForest->loquatTrees[i] = NULL;
	}

	for ( int i=0; i<Ntrees; i++ )
	{
		rv = GrowRandomizedCLoquatTreeRecursively( data, label, RFinfo, loquatForest->loquatTrees[i]);
		if( 1 != rv )	
			return -1;
		
		if (trace > 0 && (i + 1) % trace == 0)
		{
			float ooberror = 0;
			loquatForest->RFinfo.ntrees = i + 1;
			OOBErrorEstimate(data, label, loquatForest, ooberror, 1);
			cout << "Tree: " << i + 1 << "\tOOB error rate: " << ooberror * 100 << "%" << endl;
			loquatForest->RFinfo.ntrees = Ntrees;
		}
		
	}
	
	return 1;
}

#ifdef OPENMP_SPEEDUP
int TrainRandomForestClassifierOMP(float** data, int* label, RandomCForests_info RFinfo, LoquatCForest*& loquatForest, int random_state, int jobs, int trace)
{
	if (random_state < 0)
	{
		srand_freebsd(GenerateSeedFromSysTime());
	}
	else
	{
		srand_freebsd((unsigned int)random_state);
	}

	if (loquatForest != NULL)
		return -2;

	int rv = CheckClassificationForestParameters(RFinfo);
	switch (rv)
	{
	case 0:
		cout << ">>>>Some incorrectly assigned values are found, and default or recommended values are assigned to them." << endl;
		break;
	case 1:
		break;
	}

	rv = chekcDataSetLabel(label, RFinfo.datainfo.samples_num, RFinfo.datainfo.classes_num);
	if (rv < 0)
	{
		return -3;
	}

	cout << "max_threads: " << omp_get_max_threads() << endl;
	jobs = jobs > omp_get_max_threads() ? omp_get_max_threads() : jobs;
	omp_set_num_threads(jobs);
	cout << "jobs: " << jobs << endl;

	const int Ntrees = RFinfo.ntrees;
	loquatForest = new LoquatCForest;
	assert(loquatForest != NULL);
	loquatForest->loquatTrees = NULL;
	loquatForest->RFinfo = RFinfo;
	loquatForest->loquatTrees = new struct LoquatCTreeStruct* [Ntrees];

	for (int i = 0; i < Ntrees; i++)
	{
		loquatForest->loquatTrees[i] = NULL;
	}

	int growed_num = 0;
	
#pragma omp parallel for 
	for (int i = 0; i < Ntrees; i++)
	{
		rv = GrowRandomizedCLoquatTreeRecursively(data, label, RFinfo, loquatForest->loquatTrees[i]);
		// if( 1 != rv )	
		// 	return -1;
		//cout<<"Tree "<<i+1<<"is grown."<<endl;
		#pragma omp critical  
		{
			growed_num++;
			if (trace > 0 && growed_num % trace == 0)
			{
				//cout << "Tree: " << growed_num << endl;
			}
		}


	}

	//cout << "growed: " << growed_num << endl;


	return 1;
}

#endif  // OPENMP_SPEEDUP


void MaximumConfienceClassLabel(const float* const class_distribution, const int class_num, int &label, float *max_confidence)
{
	float max_conf = class_distribution[0];
	label = 0;
	for( int i=1; i<class_num; i++ )
	{
		if( class_distribution[i] > max_conf )
		{
			max_conf = class_distribution[i];
			label = i;
		}
	}

	if( max_confidence != NULL )
		*max_confidence = max_conf;
}

// 返回分布最大的前两个类别及对应置信度
void MaximumConfienceClassLabelTop2(const float* const class_distribution, const int class_num, int& label, float* max_confidence, int& secondary_label, float* second_max_conf)
{
	float max_conf = class_distribution[0];
	label = 0;
	for (int i = 1; i < class_num; i++)
	{
		if (class_distribution[i] > max_conf)
		{
			max_conf = class_distribution[i];
			label = i;
		}
	}
	if (max_confidence != NULL)
		*max_confidence = max_conf;

	// 分布第二的类别
	float sec_conf = (label == 0 ? class_distribution[1] : class_distribution[0]);
	secondary_label = (label == 0 ? 1 : 0);
	for (int i = 0; i < class_num; i++)
	{
		if (i == label)
			continue;

		if (class_distribution[i] > sec_conf)
		{
			sec_conf = class_distribution[i];
			secondary_label = i;
		}
	}
	if (NULL != second_max_conf)
	{
		*second_max_conf = sec_conf;
	}
}

/*
Description: Extremely random method is used to choose the split value without using Information Gain.
             This function is called by 'SplitOnDLoquatNode' only when the candidate variables to split are identical.
*/
int ExtremeRandomlySplitOnDLoquatNode(float **data, int variables_num, const int *innode_samples_index, int innode_num, int &split_variable_index, float &split_value)
{
	int j, index;
	float maxv=0.f, minv=0.f;
	int var_index, totalTryNum = variables_num*2;

	if( totalTryNum < 10 )
		totalTryNum = 20;

	// randomly select the variables(attribute) candidate choosing to split on the node

	while( --totalTryNum )
	{
		var_index = rand_freebsd()%variables_num;

		index = innode_samples_index[0];
		maxv = data[index][var_index];
		minv = data[index][var_index];

		for( j=1; j<innode_num; j++ )
		{
			index = innode_samples_index[j];
			if ( data[index][var_index] > maxv )
				maxv = data[index][var_index];
			else if( data[index][var_index] < minv )
				minv = data[index][var_index];
		}

		if( maxv - minv > VERY_SMALL_VALUE * 1e6 )
		{
			split_variable_index = var_index;
			int s = rand_freebsd() % 100;
			split_value = minv + s/100.f * (maxv-minv);// (maxv + minv) /2.f;
			break;
		}
	}
	//cout<<"total: "<<2*variables_num-totalTryNum<<" max:"<<maxv<<" min:"<<minv<<endl;
	if( totalTryNum == 0 )
		return -1;
	else
		return 1;
}


int SplitOnDLoquatNodeWithEveryAttempt(float** data, int* label, const int samples_num, const int variables_num, const int classes_num,
		const int* innode_samples_index, const int innode_num, const int Mvariable, int& split_variable_index, float& split_value)
{
	int selSplitIndex = -1;
	int rv = 1;
	int index = 0;
	int lcount = 0,	rcount = 0;
	int lc_best = 0, rc_best = 0;
	double lgini, rgini, gini, mingini = 1e38;

	vector<int> arrayindx;
	for (int i = 0; i < variables_num; i++)
		arrayindx.push_back(i);

	int* lsubNodeClassnum = new int[classes_num];
	int* rsubNodeClassnum = new int[classes_num];
	bool bfindSplitV = false;
	split_variable_index = -1;

	for (int i = 0; i < Mvariable; i++)
	{
		int iid = rand_freebsd() % (variables_num - i);
		selSplitIndex = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);

		for (int j = 0; j < innode_num; j++)
		{
			float splitv = data[innode_samples_index[j]][selSplitIndex];
			lcount = 0;	rcount = 0;
			memset(lsubNodeClassnum, 0, sizeof(int) * classes_num);
			memset(rsubNodeClassnum, 0, sizeof(int) * classes_num);

			for (int k = 0; k < innode_num; k++)
			{
				index = innode_samples_index[k];
				if (data[index][selSplitIndex] <= splitv)
				{
					lcount++;
					lsubNodeClassnum[label[index]]++;
				}
				else
				{
					rcount++;
					rsubNodeClassnum[label[index]]++;
				}
			}

			lgini = 0;	rgini = 0;
			const int lc = lcount == 0 ? 1 : lcount;
			const int rc = rcount == 0 ? 1 : rcount;
			for (int n = 0; n < classes_num; n++)
			{
				lgini += (lsubNodeClassnum[n] / (double)lc) * (lsubNodeClassnum[n] / (double)lc);
				rgini += (rsubNodeClassnum[n] / (double)rc) * (rsubNodeClassnum[n] / (double)rc);
			}
			lgini = 0.5 * (1.0 - lgini);
			rgini = 0.5 * (1.0 - rgini);
			//gini = lcount/(float)innode_num*lgini + rcount/(float)innode_num*rgini;
			gini = (lcount * lgini + rcount * rgini) / innode_num;
			if (gini < mingini)
			{
				bfindSplitV = true;
				mingini = gini;
				split_variable_index = selSplitIndex;
				split_value = splitv;
				lc_best = lcount;
				rc_best = rcount;
			}

		}
	}


	if (bfindSplitV == false) // 如果所有被选择分量的maxv==minv
	{
		if (-1 == ExtremeRandomlySplitOnDLoquatNode(data, variables_num, innode_samples_index, innode_num, split_variable_index, split_value))
		{
			split_variable_index = -1;
			split_value = 0;
			rv = -1;
		}
		else
			rv = 0;
	}
	else
	{
		if (lc_best == 0 || rc_best == 0)
		{
			//split_value = (maxv[order] + minv[order]) * 0.5f;
			split_variable_index = rand_freebsd() % variables_num;
			int index1 = rand_freebsd() % innode_num;
			int index2 = rand_freebsd() % innode_num;
			split_value = 0.5f * (data[innode_samples_index[index1]][split_variable_index] + data[innode_samples_index[index2]][split_variable_index]);
		}
	}


	delete[] lsubNodeClassnum;
	delete[] rsubNodeClassnum;
	return rv;
}


/*
Description:	Split the data at each node in the tree.

[in]  1.data:				 two dimension array [N][M], containing the total training data with their variable
	  2.label:				 the labels of training data
	  3.variables_num:		 the total number of variables
	  4.classes_num:		 the number of classed
	  5.innode_samples_index:the index array of the original training data, indicating the training samples arrival at the node
	  6.innode_num:          the number of training samples at the node
	  7.Mvariable:           the number of candidate variables choosing to split the node

[out] 1. split_variable_index:the index of variable chosen to split at the node
	  2.the value to split at the node

return:
	  -1: split failed, all arrival samples may be splitted to the same subnode.
	   0: split the batch of arrival samples using extremely random selection method.
	   1: split successfully
*/
int SplitOnDLoquatNode(float** data, int* label, const int variables_num, const int classes_num,
				const int* innode_samples_index, const int innode_num, const int Mvariable, int& split_variable_index, float& split_value)
{

	/*if (innode_num <= 20)
	{
		return SplitOnDLoquatNodeWithEveryAttempt(data, label, samples_num, variables_num, classes_num, innode_samples_index, innode_num, Mvariable, split_variable_index, split_value);
	}*/

	int i, j, index, rv = 1;
	int lc_best = 0, rc_best = 0;
	int selSplitIndex;
	float maxv, minv, step;
	float splitv = 0;
	double lgini, rgini, gini, mingini = 1e38;

	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for (i = 0; i < variables_num; i++)
		arrayindx.push_back(i);

	int* lsubNodeClassnum = new int[classes_num];
	int* rsubNodeClassnum = new int[classes_num];
	memset(lsubNodeClassnum, 0, sizeof(int) * classes_num);
	memset(rsubNodeClassnum, 0, sizeof(int) * classes_num);

	bool bfindSplitV = false;
	split_variable_index = -1;
	int lcount = 0, rcount = 0;

	for (i = 0; i < Mvariable; i++)
	{
		int iid = rand_freebsd() % (variables_num - i);
		selSplitIndex = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);

		index = innode_samples_index[0];
		maxv = data[index][selSplitIndex];
		minv = data[index][selSplitIndex];
		for (j = 1; j < innode_num; j++)
		{
			index = innode_samples_index[j];
			if (data[index][selSplitIndex] > maxv)
				maxv = data[index][selSplitIndex];
			else if (data[index][selSplitIndex] < minv)
				minv = data[index][selSplitIndex];
		}
		step = (maxv - minv) / INTERVAL_STEPS_NUM;		
		
		if (step < FLT_EPSILON)
			continue; 

		int counter = 0;

		splitv = minv;

		while (counter <= INTERVAL_STEPS_NUM + 1)
		{
			if (counter == INTERVAL_STEPS_NUM + 1)
			{
				// 当minv 与 maxv差距很小时 splitv+= step可能会因为float精度问题不变化，
				// 此问题在Superconductivty_train数据集出现
				// 因此使用一次均值来避免这个问题
				splitv = (minv + maxv) * 0.5f;
			}

			lcount = 0;	rcount = 0;
			memset(lsubNodeClassnum, 0, sizeof(int) * classes_num);
			memset(rsubNodeClassnum, 0, sizeof(int) * classes_num);

			for (j = 0; j < innode_num; j++)
			{
				index = innode_samples_index[j];
				if (data[index][selSplitIndex] <= splitv)
				{
					lcount++;
					lsubNodeClassnum[label[index]]++;
				}
				else
				{
					rcount++;
					rsubNodeClassnum[label[index]]++;
				}

			}


			lgini = 0;	rgini = 0;
			const int lc = lcount == 0 ? 1 : lcount;
			const int rc = rcount == 0 ? 1 : rcount;
			for (j = 0; j < classes_num; j++)
			{
				lgini += (lsubNodeClassnum[j] / (double)lc) * (lsubNodeClassnum[j] / (double)lc);
				rgini += (rsubNodeClassnum[j] / (double)rc) * (rsubNodeClassnum[j] / (double)rc);
			}
			lgini = 0.5 * (1.0 - lgini);
			rgini = 0.5 * (1.0 - rgini);
			//gini = lcount/(float)innode_num*lgini + rcount/(float)innode_num*rgini;
			gini = (lcount * lgini + rcount * rgini) / innode_num;
			if (gini < mingini)
			{
				bfindSplitV = true;
				mingini = gini;
				split_variable_index = selSplitIndex;
				split_value = splitv;
				lc_best = lcount;
				rc_best = rcount;
			}

			++counter;
			splitv += step;

		}
	}

	if (bfindSplitV == false) // 如果所有被选择分量的maxv==minv
	{
		if (-1 == ExtremeRandomlySplitOnDLoquatNode(data, variables_num, innode_samples_index, innode_num, split_variable_index, split_value))
		{
			/*int r = rand() % Mvariable;
			split_variable_index = selSplitIndex[r];
			split_value = (maxv[r] + minv[r]) / 2.f;*/
			split_variable_index = -1;
			split_value = 0;
			rv = -1;
		}
		else
			rv = 0;
	}
	else
	{
		if (lc_best == 0 || rc_best == 0)
		{
			//split_value = (maxv[order] + minv[order]) * 0.5f;
			split_variable_index = rand_freebsd() % variables_num;
			int index1 = rand_freebsd() % innode_num;
			int index2 = rand_freebsd() % innode_num;
			split_value = 0.5f*(data[innode_samples_index[index1]][split_variable_index] + data[innode_samples_index[index2]][split_variable_index]);
		}
	}

	delete[] lsubNodeClassnum;
	delete[] rsubNodeClassnum;
	return rv;
}

typedef struct var_label {
	float var;
	int label;
}var_label;

/*
Description:	Split the data at each node in the tree by completely search method.
				2021-03-11 by GXF

[in]  1.data:				 two dimension array [N][M], containing the total training data with their variable
	  2.label:				 the labels of training data
	  3.samples_num:		 the total number of samples
	  4.variables_num:		 the total number of variables
	  5.classes_num:		 the number of classed
	  6.innode_samples_index:the index array of the original training data, indicating the training samples arrival at the node
	  7.innode_num:          the number of training samples at the node
	  8.Mvariable:           the number of candidate variables choosing to split the node

[out] 1. split_variable_index:the index of variable chosen to split at the node
	  2.the value to split at the node

return:
	  -1: split failed, all arrival samples may be splitted to the same subnode.
	   0: split the batch of arrival samples using extremely random selection method.
	   1: split successfully
*/
int SplitOnDLoquatNodeCompletelySearch(float** data, int* label, int samples_num, int variables_num, int classes_num,
	const int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{
	int i, j, k, index, rv = 1;
	float splitv = 0;
	double lgini, rgini, gini, gini_best = -1e38;
	int lcount = 0, rcount = 0;

#ifdef TEST_CHECK
	int* labelstat = new int[classes_num];
	memset(labelstat, 0, sizeof(int) * classes_num);
	for (k = 0; k < innode_num; k++)
	{
		labelstat[label[innode_samples_index[k]]]++;
	}
#endif
	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for (i = 0; i < variables_num; i++)
		arrayindx.push_back(i);

	float** labelsCum = new float* [classes_num]; // 类别累计直方图
	for (k = 0; k < classes_num; k++)
	{
		labelsCum[k] = new float[innode_num];
		memset(labelsCum[k], 0, sizeof(float) * innode_num);
	}

	var_label* vls = new var_label[innode_num];
	bool bfindSplitV = false;
	split_variable_index = -1;

	for (j = 0; j < Mvariable; j++)
	{

		const int iid = rand_freebsd() % (variables_num - j);
		const int selSplitIndex = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);
		assert(selSplitIndex >= 0 && selSplitIndex < variables_num);

		for (k = 0; k < classes_num; k++)
		{
			memset(labelsCum[k], 0, sizeof(float) * innode_num);
		}


		for (k = 0; k < innode_num; k++)
		{
			index = innode_samples_index[k];
			vls[k].var = data[index][selSplitIndex];
			vls[k].label = label[index];
		}
		// 用自定义函数对象排序
		struct {
			bool operator()(const var_label& a, const var_label& b) const
			{
				return a.var < b.var;
			}
		} customComp;

		std::sort(vls, vls+ innode_num, customComp);

		// 计算累计直方图(需排序后)
		for (k = 0; k < innode_num; k++)
		{
			labelsCum[vls[k].label][k] = 1.f;
		}

		for (i = 0; i < classes_num; i++)
		{
			for (k = 1; k < innode_num; k++)
			{
				labelsCum[i][k] += labelsCum[i][k - 1];
			}
		}

		/*labelsCum[vls[0].label][0] = 1.f;
		for (k = 1; k < innode_num; k++)
		{
			for (i = 0; i < classes_num; i++)
			{
				labelsCum[i][k] = labelsCum[i][k - 1] + (vls[k].label == i);
			}
		}*/

#ifdef TEST_CHECK
		for (int c = 0; c < classes_num; c++)
		{
			assert(labelsCum[c][innode_num - 1] == labelstat[c]);
		}
#endif

		// 找最佳split
		for (k = 1; k < innode_num; k++)
		{
			assert(vls[k].var >= vls[k - 1].var);
			if (vls[k].var - vls[k-1].var < FLT_EPSILON)
				continue;

			splitv = 0.5f * (vls[k].var + vls[k - 1].var);
			// calc gini
			lcount = k;
			rcount = innode_num - k;
			lgini = 0.f;
			rgini = 0.f;
			float tmpv = 0.f;
			for (i = 0; i < classes_num; i++)
			{
				lgini += 1.0 * labelsCum[i][k - 1] * labelsCum[i][k - 1];
				tmpv = labelsCum[i][innode_num - 1] - labelsCum[i][k - 1];
				rgini += 1.0 * tmpv * tmpv;
			}
			/*
			lgini = (1.0 - lgini / (1.0 * lcount * lcount)) * 0.5;
			rgini = (1.0 - rgini / (1.0 * rcount * rcount)) * 0.5;
			// gini = lcount/(float)innode_num*lgini + rcount/(float)innode_num*rgini;
			gini = (lcount * lgini + rcount * rgini) / innode_num; //+ (innode_num<500 ? 0 : 0.05) * (lcount * 1.0 / innode_num - 0.5) * (lcount * 1.0 / innode_num - 0.5);
			*/
			gini = lgini / lcount + rgini / rcount;
			if (gini > gini_best)
			{
				bfindSplitV = true;
				gini_best = gini;
				split_variable_index = selSplitIndex;
				split_value = splitv;
			}
		}

	}
	if (bfindSplitV == false)
	{
		if (-1 == ExtremeRandomlySplitOnDLoquatNode(data, variables_num, innode_samples_index, innode_num, split_variable_index, split_value))
		{
			split_variable_index = -1;
			split_value = 0;
			rv = -1;
		}
		else
			rv = 0;
	}
	delete[] vls;
	for (k = 0; k < classes_num; k++)
	{
		delete[] labelsCum[k];
	}
	delete[] labelsCum;
#ifdef TEST_CHECK
	delete[] labelstat;
#endif
	return rv;
}

// 修改了申请内存方式，以及其他细节（见for循环中的ptr）
int SplitOnDLoquatNodeCompletelySearch2(float** data, int* label, int variables_num, int classes_num,
    const int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{
    int i, j, k, index, rv = 1;
    float splitv = 0;
    float lgini, rgini, gini, gini_best = -1e20f;
    int lcount = 0, rcount = 0;

#ifdef TEST_CHECK
    int* labelstat = new int[classes_num];
    memset(labelstat, 0, sizeof(int) * classes_num);
    for (k = 0; k < innode_num; k++)
    {
        labelstat[label[innode_samples_index[k]]]++;
    }
#endif
    // randomly select the variables(attribute) candidate choosing to split on the node
    vector<int> arrayindx;
    for (i = 0; i < variables_num; i++)
        arrayindx.push_back(i);

    unsigned char *memblock = new unsigned char[classes_num*innode_num*sizeof(float)+innode_num*sizeof(var_label)+classes_num*sizeof(float*)];

    float** labelsCum = new float* [classes_num]; // 类别累计直方图
    labelsCum[0] = (float *)memblock;
    for (k = 1; k < classes_num; k++)
    {
        labelsCum[k] = labelsCum[k-1]+innode_num;
        //memset(labelsCum[k], 0, sizeof(float) * innode_num);
    }

    var_label* vls = (var_label *)(memblock+classes_num*innode_num*sizeof(float));
    bool bfindSplitV = false;
    split_variable_index = -1;
    for (j = 0; j < Mvariable; j++)
    {

        const int iid = rand_freebsd() % (variables_num - j);
        const int selSplitIndex = arrayindx[iid];
        arrayindx.erase(arrayindx.begin() + iid);
        assert(selSplitIndex >= 0 && selSplitIndex < variables_num);
        for (k = 0; k < classes_num; k++)
        {
            memset(labelsCum[k], 0, sizeof(float) * innode_num);
        }


        for (k = 0; k < innode_num; k++)
        {
            index = innode_samples_index[k];
            vls[k].var = data[index][selSplitIndex];
            vls[k].label = label[index];
        }
        // 用自定义函数对象排序
        struct {
            bool operator()(const var_label& a, const var_label& b) const
            {
                return a.var < b.var;
            }
        } customComp;

        std::sort(vls, vls+ innode_num, customComp);

        // 计算累计直方图(需排序后)
        for (k = 0; k < innode_num; k++)
        {
            labelsCum[vls[k].label][k] = 1.f;
        }

        for (i = 0; i < classes_num; i++)
        {
            float *ptr = labelsCum[i];
            for (k = 1; k < innode_num; k++)
            {
                //labelsCum[i][k] += labelsCum[i][k - 1];
                ptr[k] += ptr[k-1];
            }
        }

        /*labelsCum[vls[0].label][0] = 1.f;
        for (k = 1; k < innode_num; k++)
        {
            for (i = 0; i < classes_num; i++)
            {
                labelsCum[i][k] = labelsCum[i][k - 1] + (vls[k].label == i);
            }
        }*/

#ifdef TEST_CHECK
        for (int c = 0; c < classes_num; c++)
        {
            assert(labelsCum[c][innode_num - 1] == labelstat[c]);
        }
#endif

        // 找最佳split
        for (k = 1; k < innode_num; k++)
        {
            assert(vls[k].var >= vls[k - 1].var);
            if (vls[k].var - vls[k-1].var < FLT_EPSILON)
                continue;

            splitv = 0.5f * (vls[k].var + vls[k - 1].var);
            // calc gini
            lcount = k;
            rcount = innode_num - k;
            lgini = 0.f;
            rgini = 0.f;
            float tmpv = 0.f;
            float *ptr;
            for (i = 0; i < classes_num; i++)
            {
                // lgini += labelsCum[i][k - 1] * labelsCum[i][k - 1];
                // tmpv = labelsCum[i][innode_num - 1] - labelsCum[i][k - 1];
                // rgini += tmpv * tmpv;

                ptr = labelsCum[i];
                lgini += ptr[k - 1] *ptr[k - 1];
                tmpv = ptr[innode_num - 1] - ptr[k - 1];
                rgini += tmpv * tmpv;
            }
            
            gini = lgini / lcount + rgini / rcount;
            if (gini > gini_best)
            {
                bfindSplitV = true;
                gini_best = gini;
                split_variable_index = selSplitIndex;
                split_value = splitv;
            }
        }

    }
    if (bfindSplitV == false)
    {
        if (-1 == ExtremeRandomlySplitOnDLoquatNode(data, variables_num, innode_samples_index, innode_num, split_variable_index, split_value))
        {
            split_variable_index = -1;
            split_value = 0;
            rv = -1;
        }
        else
            rv = 0;
    }
    
    delete[] memblock;
    delete[] labelsCum;

#ifdef TEST_CHECK
    delete[] labelstat;
#endif
    return rv;
}

/*
* Implementation of the following work: 
* P. Geurts, D. Ernst, and L. Wehenkel. Extremely randomized trees. Machine Learning, 63(1):3–42, 2006a.
*/
int SplitOnDLoquateNodeExtremeRandomly(float** data, int* label, int variables_num, int classes_num,
	const int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{
	int i, j, index, rv = 1;
	int selSplitIndex;
	float maxv, minv, splitv_cand, feat_v;
	float lgini, rgini, gini, mingini = FLOAT_MAX;

	int lcount = 0, rcount = 0, lcount_split, rcount_split;
	int* lsubNodeClassnum = new int[classes_num];
	int* rsubNodeClassnum = new int[classes_num];

	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for (i = 0; i < variables_num; i++)
		arrayindx.push_back(i);

	bool bfindSplitV = false;
	split_variable_index = -1;

	for (i = 0; i < Mvariable; i++)
	{
		//int iid = rand()%(variables_num-i);
		int iid = rand_freebsd() % (variables_num - i);
		selSplitIndex = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);

		index = innode_samples_index[0];
		maxv = data[index][selSplitIndex];
		minv = data[index][selSplitIndex];

		for (j = 1; j < innode_num; j++)
		{
			index = innode_samples_index[j];
			feat_v = data[index][selSplitIndex];
			if (feat_v > maxv)
				maxv = feat_v;
			else if (feat_v < minv)
				minv = feat_v;
		}

		if (maxv - minv < FLT_EPSILON)
			continue;

		splitv_cand = ((float)rand_freebsd()) / RAND_MAX_RF * (maxv - minv) + minv;
		
		lcount = 0;	rcount = 0;
		memset(lsubNodeClassnum, 0, sizeof(int) * classes_num);
		memset(rsubNodeClassnum, 0, sizeof(int) * classes_num);

		for (j = 0; j < innode_num; j++)
		{
			index = innode_samples_index[j];
			if (data[index][selSplitIndex] <= splitv_cand)
			{
				lcount++;
				lsubNodeClassnum[label[index]]++;
			}
			else
			{
				rcount++;
				rsubNodeClassnum[label[index]]++;
			}

		}

		lgini = 0;	rgini = 0;
		const int lc = lcount == 0 ? 1 : lcount;
		const int rc = rcount == 0 ? 1 : rcount;
		for (j = 0; j < classes_num; j++)
		{
			lgini += (lsubNodeClassnum[j] / (float)lc) * (lsubNodeClassnum[j] / (float)lc);
			rgini += (rsubNodeClassnum[j] / (float)rc) * (rsubNodeClassnum[j] / (float)rc);
		}
		lgini = 0.5f * (1.f - lgini);
		rgini = 0.5f * (1.f - rgini);
		gini = (lcount * lgini + rcount * rgini) / (float)innode_num;
		if (gini < mingini)
		{
			bfindSplitV = true;
			mingini = gini;
			split_variable_index = selSplitIndex;
			split_value = splitv_cand;
			lcount_split = lcount;
			rcount_split = rcount;

		}
	}

	if (bfindSplitV == false)
	{
		if (-1 == ExtremeRandomlySplitOnDLoquatNode(data, variables_num, innode_samples_index, innode_num, split_variable_index, split_value))
		{
			split_variable_index = -1;
			split_value = 0;
			rv = -1;
		}
		else
			rv = 0;
	}

	delete[] lsubNodeClassnum;
	delete[] rsubNodeClassnum;
	return rv;
}


typedef struct value_index {
	float value;
	int index;
}value_index;
int _cmp_val(const void* a, const void* b)
{
	return ((value_index*)a)->value > ((value_index*)b)->value ? 1 : -1;
}

// data SHOULD be normalized or standardization in advance
int SplitNodeUnsupervisedly(float** data, int variables_num, const int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{
	int i, j, index, rv = 1;
	int* selSplitIndex = new int[Mvariable];

	float splitv = 0;
	double gini_like, mingini = 1e38;
	int var_index;

	vector<int> arrayindx;
	for (i = 0; i < variables_num; i++)
		arrayindx.push_back(i);
	for (i = 0; i < Mvariable; i++)
	{
		int iid = rand_freebsd() % (variables_num - i);
		selSplitIndex[i] = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);
	}

	int lcount = 0, rcount = 0;
	double lMeanSqr, rMeanSqr;
	double lVar, rVar;

	double** targetCum = new double* [variables_num];
	for (int y = 0; y < variables_num; y++)
	{
		targetCum[y] = new double[innode_num];
		memset(targetCum[y], 0, sizeof(double) * innode_num);
	}
	double* targetSqrCum = new double[innode_num];

	value_index* vts = new value_index[innode_num];

	bool bfindSplitV = false;
	split_variable_index = -1;
	for (j = 0; j < Mvariable; j++)
	{
		var_index = selSplitIndex[j];

		for (int k = 0; k < innode_num; k++)
		{
			index = innode_samples_index[k];
			vts[k].value = data[index][var_index];
			vts[k].index = index;
		}

		qsort(vts, innode_num, sizeof(value_index), _cmp_val);

		memset(targetSqrCum, 0, sizeof(double) * innode_num);
		for (int y = 0; y < variables_num; y++)
		{
			//const double t = target[vts[0].index * variables_num_y + y];
			const double t = data[vts[0].index][y];
			targetCum[y][0] = t;
			targetSqrCum[0] += t * t;
		}
		for (int k = 1; k < innode_num; k++)
		{
			targetSqrCum[k] = targetSqrCum[k - 1];
			for (int y = 0; y < variables_num; y++)
			{
				//const double t = target[vts[k].index * variables_num_y + y];
				const double t = data[vts[k].index][y];
				targetCum[y][k] = targetCum[y][k - 1] + t;
				targetSqrCum[k] += t * t;
			}
		}

		for (int k = 1; k < innode_num; k++)
		{
			if (abs(vts[k - 1].value - vts[k].value) < FLT_EPSILON)
			{
				continue;
			}

			splitv = 0.5f * (vts[k - 1].value + vts[k].value);
			lcount = k;
			rcount = innode_num - k;

			lVar = targetSqrCum[k - 1] / lcount;
			rVar = (targetSqrCum[innode_num - 1] - targetSqrCum[k - 1]) / rcount;
			lMeanSqr = rMeanSqr = 0.0;
			double mean_y;
			for (int y = 0; y < variables_num; y++)
			{
				mean_y = targetCum[y][k - 1] / lcount;
				lMeanSqr += mean_y * mean_y;
				mean_y = (targetCum[y][innode_num - 1] - targetCum[y][k - 1]) / rcount;
				rMeanSqr += mean_y * mean_y;
			}
			
			//assert(lVar - lMeanSqr >= 0 && rVar - rMeanSqr >= 0);
			gini_like = (lcount * (lVar - lMeanSqr) + rcount * (rVar - rMeanSqr)) / innode_num;

			if (gini_like < mingini)
			{
				bfindSplitV = true;
				mingini = gini_like;
				split_variable_index = var_index;
				split_value = splitv;
			}
		}

	}

	if (false == bfindSplitV)
	{
		split_variable_index = -1;
		split_value = 0;
		rv = -1;
	}

	delete[] selSplitIndex;
	delete[] vts;
	for (int y = 0; y < variables_num; y++)
		delete[] targetCum[y];
	delete[] targetCum;
	delete[] targetSqrCum;

	return rv;
}

//int ClearAllocatedMemoryOnNodeDuringCTraining(struct LoquatCTreeNode* treeNode)
//{
//	if (NULL == treeNode)
//	{
//		return 0;
//	}
//	// clear left subnode
//	ClearAllocatedMemoryOnNodeDuringCTraining(treeNode->pSubNode[0]);
//	// clear right subnode
//	ClearAllocatedMemoryOnNodeDuringCTraining(treeNode->pSubNode[1]);
//	// clear  this node
//	delete[] treeNode->samples_index;
//	treeNode->samples_index = NULL;
//	treeNode->arrival_samples_num = 0;
//	return 1;
//}
//int ClearAllocatedMemoryDuringCTraining(struct LoquatCTreeStruct* loquatTree)
//{
//	if (NULL == loquatTree)
//	{
//		return 0;
//	}
//	return ClearAllocatedMemoryOnNodeDuringCTraining(loquatTree->rootNode);
//}

//int ClearAllocatedMemoryDuringCTraining(struct LoquatCTreeStruct *loquatTree)
//{
//	int depth = loquatTree->depth;
//	int i;
//	unsigned int j, maxNodeNumThisDepth=0;
//	struct LoquatCTreeNode **pPreNode = NULL, **pCurNode = NULL;
//	pPreNode = new struct LoquatCTreeNode *[1];
//	pPreNode[0] = loquatTree->rootNode;
//
//	// (1) 
//	if( pPreNode[0] == NULL )
//	{
//		delete [] pPreNode;
//		return 1;
//	}
//	else if( pPreNode[0]->nodetype == enLeafNode )
//	{
//		if( pPreNode[0]->samples_index !=NULL )
//		{
//			delete [] pPreNode[0]->samples_index;
//			pPreNode[0]->samples_index = NULL;
//			pPreNode[0]->arrival_samples_num = 0;
//		}
//		delete [] pPreNode; // 释放指针 
//		return 1;
//	}else
//	{
//		if( pPreNode[0]->samples_index !=NULL )
//		{
//			delete [] pPreNode[0]->samples_index;
//			pPreNode[0]->samples_index = NULL;
//			pPreNode[0]->arrival_samples_num = 0;
//		}
//	}
//
//	for( i=1; i <= depth; i++ )
//	{
//		maxNodeNumThisDepth = (int)powf(2.f, (float)i);
//		pCurNode = new struct LoquatCTreeNode *[maxNodeNumThisDepth];
//		// 20210309
//		for (j = 0; j < maxNodeNumThisDepth; j++) 
//		{
//			pCurNode[j] = NULL;
//		}
//
//		for ( j=0; j<maxNodeNumThisDepth/2; j++ ) 
//		{
//			if( pPreNode[j] == NULL )
//			{
//				pCurNode[j*2] = NULL;
//				pCurNode[j*2+1] = NULL;
//			}
//			else if( pPreNode[j]->nodetype == enLeafNode )
//			{
//				pCurNode[j*2] = NULL;
//				pCurNode[j*2+1] = NULL;
//			}
//			else{
//				pCurNode[j*2] = pPreNode[j]->pSubNode[0];
//				pCurNode[j*2+1] = pPreNode[j]->pSubNode[1];
//			}
//		}
//
//		for( j=0; j<maxNodeNumThisDepth; j++ )
//		{
//			if( pCurNode[j] != NULL )
//			{
//				if( pCurNode[j]->samples_index != NULL )
//				{
//					delete [] pCurNode[j]->samples_index;
//					pCurNode[j]->samples_index = NULL;
//					pCurNode[j]->arrival_samples_num = 0;
//				}
//			}
//		}
//
//		delete []pPreNode;
//		pPreNode = new struct LoquatCTreeNode *[maxNodeNumThisDepth];
//		for( j=0; j<maxNodeNumThisDepth; j++ )
//		{
//			pPreNode[j] = pCurNode[j];
//		}
//		delete []pCurNode;
//		pCurNode = NULL;
//	}
//
//	delete [] pPreNode;	
//
//	return 1;
//}

int AnalyzeTrainingSamplesArrivedAtOneNode(const int* label, const int classes_num, const int* innode_samples_index, const int innode_num, float& impurity, float*& class_distribution)
{
	if (innode_num == 0)
	{
		impurity = 0.f;
		return 0;
	}
	if (class_distribution != NULL)
		delete[] class_distribution;
	class_distribution = new float[classes_num];
	memset(class_distribution, 0, sizeof(float) * classes_num);
	int i, index, label_index;
	for (i = 0; i < innode_num; i++)
	{
		index = innode_samples_index[i];
		label_index = label[index];
		class_distribution[label_index] += 1.f;
	}
	impurity = 0.0f;
	for (i = 0; i < classes_num; i++)
	{
		class_distribution[i] /= innode_num;
		impurity += class_distribution[i] * class_distribution[i];
	}
	impurity = 0.5f * (1 - impurity);
	return 1;
}


struct LoquatCTreeNode* GrowLoquatCTreeNodeRecursively(float** data, int* label, int* sample_arrival_index, const int arrival_num, const GrowNodeInput* pInputParam, struct LoquatCTreeStruct* loquatTree)
{
	int total_samples_num = pInputParam->total_samples_num;
	int total_variables_num = pInputParam->total_variables_num;
	int total_classes_num = pInputParam->total_classes_num;
	struct LoquatCTreeNode* treeNode = new struct LoquatCTreeNode;
	assert(NULL != treeNode);
	treeNode->class_distribution = NULL;
	treeNode->pParentNode = NULL;
	treeNode->pSubNode = NULL;
	treeNode->samples_index = NULL;

	treeNode->depth = pInputParam->parent_depth + 1;

	if (treeNode->depth == 0)
		treeNode->nodetype = TreeNodeTpye::enRootNode;
	else
		treeNode->nodetype = TreeNodeTpye::enLinkNode;

	treeNode->subnodes_num = 2;
	treeNode->pParentNode = NULL;
	treeNode->pSubNode = new struct LoquatCTreeNode* [2];
	treeNode->pSubNode[0] = NULL;
	treeNode->pSubNode[1] = NULL;
	treeNode->split_value = 0.0f;
	treeNode->split_variable_index = -1;
	treeNode->train_impurity = 0;
	treeNode->class_distribution = NULL;
	treeNode->leaf_node_label = -1;
	treeNode->leaf_confidence = 0;
	treeNode->arrival_samples_num = arrival_num;
	treeNode->samples_index = sample_arrival_index;

	/*if (0 == treeNode->depth)
	{
		treeNode->samples_index = new int [arrival_num];
		memcpy(treeNode->samples_index, sample_arrival_index, arrival_num * sizeof(int));
	}*/

	// 以上用到达样本生成一个新节点，以下开始判断这个节点是否可以再分裂
	AnalyzeTrainingSamplesArrivedAtOneNode(label, total_classes_num, treeNode->samples_index, treeNode->arrival_samples_num,
		treeNode->train_impurity, treeNode->class_distribution);

	bool isFewSamples = (treeNode->arrival_samples_num <= pInputParam->leafMinSamples);
	bool isLowImpurity = (treeNode->train_impurity < STOP_CRITERION_MIN_GINI_IMPURITY);
	bool isMaxDepth = (treeNode->depth == pInputParam->maxDepth - 1);  // depth is the index of the level, which is from 0.
	
	bool isEqualConf = false;
	float max_conf, second_max_conf;
	int second_label;
	if (isFewSamples)
	{
		MaximumConfienceClassLabelTop2(treeNode->class_distribution, total_classes_num, treeNode->leaf_node_label, &max_conf, second_label, &second_max_conf);
		isEqualConf = max_conf - second_max_conf <= FLT_MIN ? true : false;
	}

	bool isleaf = ((isLowImpurity || isMaxDepth) || (isFewSamples && !isEqualConf));
	bool bGrowToGreaterDepth = pInputParam->maxDepth == GROW_DEEPER_FOR_PROXIMITY && 
								isLowImpurity && arrival_num > NODE_NUM_TO_GROW_DEEPER;

	if ( isleaf && !bGrowToGreaterDepth)
	{
		treeNode->nodetype = TreeNodeTpye::enLeafNode;
		treeNode->pSubNode[0] = NULL;
		treeNode->pSubNode[1] = NULL;
		loquatTree->leaf_node_num++;
		MaximumConfienceClassLabel(treeNode->class_distribution, total_classes_num, treeNode->leaf_node_label, NULL);
		treeNode->leaf_confidence = treeNode->class_distribution[treeNode->leaf_node_label];
		treeNode->samples_index = new int[treeNode->arrival_samples_num]; // 20230516
		memcpy(treeNode->samples_index, sample_arrival_index, sizeof(int)*treeNode->arrival_samples_num);
	}
	else // linknode
	{
		treeNode->leaf_node_label = -1;
		treeNode->leaf_confidence = 0.f;

		if (bGrowToGreaterDepth)
		{
			assert(isleaf);
			SplitNodeUnsupervisedly(data, total_variables_num, treeNode->samples_index, treeNode->arrival_samples_num,
				pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
			//cout<<"go deeper: "<<treeNode->arrival_samples_num<<endl;
		}
		else {
			switch (pInputParam->randomness)
			{
			case TREE_RANDOMNESS_WEAK:
				SplitOnDLoquatNodeCompletelySearch2(data, label, total_variables_num, total_classes_num,
					treeNode->samples_index, treeNode->arrival_samples_num,
					pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
				break;
			case TREE_RANDOMNESS_MODERATE:
				SplitOnDLoquatNode(data, label, total_variables_num, total_classes_num,
					treeNode->samples_index, treeNode->arrival_samples_num,
					pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
				break;
			case TREE_RANDOMNESS_STRONG:
				SplitOnDLoquateNodeExtremeRandomly(data, label, total_variables_num, total_classes_num,
					treeNode->samples_index, treeNode->arrival_samples_num,
					pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
				break;


			default:
				SplitOnDLoquatNode(data, label, total_variables_num, total_classes_num,
					treeNode->samples_index, treeNode->arrival_samples_num,
					pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
				break;
			}
		}

		int leftsubnode_samples_num = 0, rightsubnode_samples_num = 0;
		//int* subnode_samples_queue = NULL;

		//0607
		if (treeNode->split_variable_index >= 0)
		{
			const int split_varid = treeNode->split_variable_index;
			const float split_v = treeNode->split_value;
			int* samples_index = treeNode->samples_index; 

			int* samples_index_tmp = new int[arrival_num];
			for (int n = 0; n < arrival_num; n++)
			{
				if (data[samples_index[n]][split_varid] <= split_v)
				{
					samples_index_tmp[leftsubnode_samples_num] = samples_index[n];
					leftsubnode_samples_num++;
				}
				else
				{
					samples_index_tmp[arrival_num-1-rightsubnode_samples_num] = samples_index[n];
					rightsubnode_samples_num++;
				}
			}
			assert(leftsubnode_samples_num + rightsubnode_samples_num == arrival_num);
			assert(arrival_num == treeNode->arrival_samples_num);
			memcpy(samples_index, samples_index_tmp, sizeof(int) * arrival_num);
			delete[] samples_index_tmp;
			
#if 0
			for (int nn = 0; nn < treeNode->arrival_samples_num; nn++)
			{
				if (nn < st_pos)
					assert(data[samples_index[nn]][split_varid] <= split_v);
				else
					assert(data[samples_index[nn]][split_varid] > split_v);
			}
#endif
		}
		
		// 在非常少的情况下分裂一枝的样本个数为0,
		// 很有可能是因为到达这个节点的样本的属性都一样
		// 连ExtremelRandomSplit都没成功
		if (0 == leftsubnode_samples_num || 0 == rightsubnode_samples_num)
		{
			treeNode->nodetype = TreeNodeTpye::enLeafNode;
			loquatTree->leaf_node_num++;
			treeNode->split_value = 0;
			treeNode->split_variable_index = -1;
			treeNode->pSubNode[0] = NULL;
			treeNode->pSubNode[1] = NULL;
			MaximumConfienceClassLabel(treeNode->class_distribution, total_classes_num, treeNode->leaf_node_label, NULL);
			treeNode->leaf_confidence = treeNode->class_distribution[treeNode->leaf_node_label];
			//if (NULL != subnode_samples_queue)
			//	delete[] subnode_samples_queue;
			treeNode->samples_index = new int[treeNode->arrival_samples_num]; // 20230516
			memcpy(treeNode->samples_index, sample_arrival_index, sizeof(int)*treeNode->arrival_samples_num);
			return treeNode;
		}


		/*int* leftsubnode_index = subnode_samples_queue;
		int* rightsubnode_index = subnode_samples_queue + leftsubnode_samples_num;
		assert(leftsubnode_samples_num + rightsubnode_samples_num == treeNode->arrival_samples_num);*/

		GrowNodeInput input = *pInputParam;
		input.parent_depth = treeNode->depth;// error happened here 调试了很久!
		if (treeNode->depth + 1 > loquatTree->depth) //到了这一步后，肯定会往下生长两个节点，所以要判断树的总深度是否增加
			loquatTree->depth = treeNode->depth + 1; // 更新整棵树的当前最大深度

		// recursively call the function 
		treeNode->pSubNode[0] = GrowLoquatCTreeNodeRecursively(data, label,
			treeNode->samples_index,
			leftsubnode_samples_num,
			&input,
			loquatTree);

		treeNode->pSubNode[1] = GrowLoquatCTreeNodeRecursively(data, label,
			treeNode->samples_index+leftsubnode_samples_num,
			rightsubnode_samples_num,
			&input,
			loquatTree);

		bool is_pruning = false;
		if (is_pruning && treeNode->nodetype != enRootNode)
		{
			if (treeNode->pSubNode[0]->nodetype == enLeafNode && treeNode->pSubNode[1]->nodetype == enLeafNode)
			{
				bool toprune = leftsubnode_samples_num == 1 || rightsubnode_samples_num == 1;
				float ratio = leftsubnode_samples_num * 1.f / rightsubnode_samples_num;
				toprune = toprune && (ratio >= 10.f || ratio <= 0.1f);
				if (toprune)
				{
					treeNode->nodetype = TreeNodeTpye::enLeafNode;
					HarvestOneLeafNode(&treeNode->pSubNode[0]);
					HarvestOneLeafNode(&treeNode->pSubNode[1]);
					loquatTree->leaf_node_num = loquatTree->leaf_node_num-1;
					treeNode->split_value = 0;
					treeNode->split_variable_index = -1;
					MaximumConfienceClassLabel(treeNode->class_distribution, total_classes_num, treeNode->leaf_node_label, NULL);
					treeNode->leaf_confidence = treeNode->class_distribution[treeNode->leaf_node_label];
					treeNode->samples_index = new int[treeNode->arrival_samples_num];
					memcpy(treeNode->samples_index, sample_arrival_index, sizeof(int)* treeNode->arrival_samples_num);
					return treeNode;
				}
			}
		}
		

		treeNode->pSubNode[0]->pParentNode = treeNode;
		treeNode->pSubNode[1]->pParentNode = treeNode;
		
		if (treeNode->pSubNode[0]->nodetype != TreeNodeTpye::enLeafNode)
		{
			treeNode->pSubNode[0]->samples_index = NULL;
			treeNode->pSubNode[0]->arrival_samples_num = 0;
		}
		if (treeNode->pSubNode[1]->nodetype != TreeNodeTpye::enLeafNode)
		{
			treeNode->pSubNode[1]->samples_index = NULL;
			treeNode->pSubNode[1]->arrival_samples_num = 0;
		}


	} /*end of else*/

	return treeNode;
}

int GrowRandomizedCLoquatTreeRecursively(float **data, int *label, RandomCForests_info RFinfo, struct LoquatCTreeStruct *&loquatTree)
{
	if( loquatTree != NULL )
		return -2;

	int i, j, index=0;
	float ratio = 1.0f;
	int selnum = (int)(RFinfo.datainfo.samples_num * ratio +0.5);
	const int total_samples_num   = RFinfo.datainfo.samples_num;

	loquatTree = new struct LoquatCTreeStruct; 
	assert( loquatTree != NULL );
	loquatTree->inbag_samples_index = NULL;
	loquatTree->outofbag_samples_index = NULL;
	loquatTree->rootNode = NULL;

	loquatTree->depth = 0;
	loquatTree->leaf_node_num = 0;
	loquatTree->inbag_samples_num = selnum;
	loquatTree->inbag_samples_index = new int [selnum]; // 有重复的！
	assert( NULL != loquatTree->inbag_samples_index );

	// (1) Resampling training samples (bootstrap training samples)
	bool *inbagmask = new bool[total_samples_num];
	assert( NULL != inbagmask );
	int inbagcount = 0;   // inbagcount 不包括重叠的计数
	memset(inbagmask, false, total_samples_num*sizeof(bool));

	for( i=0; i<selnum; i++ )
	{
		index = (int)((rand_freebsd() * 1.0 / RAND_MAX_RF) * total_samples_num);
		if( index<0 )
			index = 0;
		else if( index >= total_samples_num )
			index = total_samples_num-1;
		//min_index = min_index > index ? index : min_index;
		//max_index = max_index < index ? index : max_index;
		loquatTree->inbag_samples_index[i] = index;  // resampling from original data with replacement
		if( inbagmask[index] == false )
		{
			inbagmask[index] = true;
			inbagcount++;
		}
	}
	//cout << "min: " << min_index << "max: " << max_index << endl;
	loquatTree->outofbag_samples_num = total_samples_num - inbagcount;
	loquatTree->outofbag_samples_index = new int[loquatTree->outofbag_samples_num];
	assert( NULL != loquatTree->outofbag_samples_index );
	for( i=0,j=0; i<total_samples_num; i++ )
	{
		if( inbagmask[i] == false )
			loquatTree->outofbag_samples_index[j++] = i;
	}
	delete []inbagmask;

	// (2) Build the entire tree from the root node.
	GrowNodeInput inputParam={RFinfo.datainfo.samples_num,
							  RFinfo.datainfo.variables_num, 
							  RFinfo.datainfo.classes_num, 
							  RFinfo.mvariables,
	                          RFinfo.minsamplessplit,
							  -1/*parent depth of the root node*/, 
							  RFinfo.maxdepth, 
							  RFinfo.randomness};
	loquatTree->rootNode = GrowLoquatCTreeNodeRecursively(data, label,
														loquatTree->inbag_samples_index, 
														loquatTree->inbag_samples_num,
														&inputParam, loquatTree);
	loquatTree->rootNode->pParentNode = NULL;
	loquatTree->rootNode->arrival_samples_num = 0;
	loquatTree->rootNode->samples_index = NULL;
	//ClearAllocatedMemoryDuringCTraining(loquatTree); // Clear some memory allocated for growing tree.

	return 1;
}

int EvaluateOneSample(float* data, LoquatCForest* loquatForest, int& label_index, const int isHardDecision)
{
	int classes_num = loquatForest->RFinfo.datainfo.classes_num;
	int tree_num = loquatForest->RFinfo.ntrees;
	const struct LoquatCTreeNode* pLeafNode = NULL;

	float* data_class_confidence = NULL;
	data_class_confidence = new float[classes_num];
	memset(data_class_confidence, 0, sizeof(float) * classes_num); // 初始化都为0.f


	int t, k, effect_trees, rv = 1;
	for (effect_trees = 0, t = 0; t < tree_num; t++)
	{
		pLeafNode = GetArrivedLeafNode(loquatForest, t, data);
		if (pLeafNode == NULL)
		{
			rv = 0;
			continue;
		}
		effect_trees++;
		if (isHardDecision > 0 || NULL == pLeafNode->class_distribution) // 'class_distribution' is nullptr when RF was loaded from a pain text file
			data_class_confidence[pLeafNode->leaf_node_label] += 1.0f;
		else if (isHardDecision == 0)
			data_class_confidence[pLeafNode->leaf_node_label] += pLeafNode->leaf_confidence;
		else
		{
			for (k = 0; k < classes_num; k++)
				data_class_confidence[k] += pLeafNode->class_distribution[k];
		}
	}

	if (effect_trees == 0)
	{
		delete[] data_class_confidence;
		rv = -1;
		return rv;
	}

	
	MaximumConfienceClassLabel(data_class_confidence, classes_num, label_index, NULL);

	delete[] data_class_confidence;

	return rv;
}

int RawVariableImportanceScore2(float** data, int* label, LoquatCForest* loquatForest, int nType, float* varImportance, bool bNormalize, int random_state, char* filename)
{
	if (!(nType == 0 || nType == 1))
	{
		cout << endl;
		cout << "------------------ERROR:'RawVariableImportanceScore'------------------" << endl;
		cout << "The input parameter 'nType' must be 0 or 1." << endl;
		cout << " 'nType'     0: raw variable importance score" << endl;
		cout << "             1: z-score" << endl;
		cout << "----------------------------------------------------------------------" << endl;
		cout << endl;
		return -1;
	}

	if (random_state < 0)
	{
		srand_freebsd(GenerateSeedFromSysTime());
	}
	else
	{
		srand_freebsd((unsigned int)random_state);
	}

	int var, tr, i, j, oobnum, index, rv=1;
	const int variables_num = loquatForest->RFinfo.datainfo.variables_num;
	const int Ntrees = loquatForest->RFinfo.ntrees;
	struct LoquatCTreeStruct* pTree = NULL;
	int* pIndex = NULL, predicted_class_index;
	float confidence;
	int correct_num = 0, correct_num_premute = 0;
	float* tmp_data = new float[variables_num];

	int** DeltMatrix = new int* [Ntrees];
	assert(DeltMatrix != NULL);
	for (i = 0; i < Ntrees; i++)
	{
		DeltMatrix[i] = new int[variables_num];
		memset(DeltMatrix[i], 0, sizeof(int) * variables_num);
	}

	float* mean_var = new float[variables_num];
	memset(mean_var, 0, sizeof(float) * variables_num);
	float* std2_var = new float[variables_num];
	memset(std2_var, 0, sizeof(float) * variables_num);
	int* oobOfTrees = new int[Ntrees];
	memset(oobOfTrees, 0, sizeof(int) * Ntrees);

	bool oobFound = false;
	for (tr = 0; tr < Ntrees; tr++)
	{
		pTree = loquatForest->loquatTrees[tr];
		pIndex = pTree->outofbag_samples_index;
		oobnum = pTree->outofbag_samples_num;

		if (pIndex == NULL || oobnum == 0)
			continue;
		oobFound = true;
		oobOfTrees[tr] = oobnum;
		int* permuted_order = new int[oobnum];
		
		for (correct_num = 0, i = 0; i < oobnum; i++)
		{
			index = pIndex[i];
			PredictAnTestSampleOnOneTree(data[index], variables_num, pTree, predicted_class_index, confidence);
			if (predicted_class_index == label[index])
				correct_num++;
		}

		for (var = 0; var < variables_num; var++)
		{
			// permuting (var)th variables
			// permute [0, oobnum-1]
			permute(oobnum, permuted_order);

			for (correct_num_premute = 0, i = 0; i < oobnum; i++)
			{
				index = pIndex[permuted_order[i]];
				//memcpy_s(tmp_data, variables_num*sizeof(float), data[pIndex[i]], variables_num*sizeof(float));
				memcpy(tmp_data, data[pIndex[i]], variables_num * sizeof(float));
				tmp_data[var] = data[index][var];
				PredictAnTestSampleOnOneTree(tmp_data, variables_num, pTree, predicted_class_index, confidence);
				if (predicted_class_index == label[pIndex[i]])
					correct_num_premute++;
			}

			DeltMatrix[tr][var] = correct_num - correct_num_premute;
			mean_var[var] += (correct_num - correct_num_premute) / (float)oobOfTrees[tr];
		}

		delete[]permuted_order;
	}

	if (oobFound == false)
	{
		cout << endl;
		cout << "-----------------  ERROR:'RawVariableImportanceScore'-----------------" << endl;
		cout << "Error: Out-of-bag samples are not found in RF. The results mean nothing." << endl;
		cout << "----------------------------------------------------------------------" << endl;
		cout << endl;
		rv = 0;
	}

	for (i = 0; i < variables_num; i++)
		mean_var[i] = mean_var[i] / Ntrees;

	for (i = 0; i < variables_num; i++)
	{
		for (j = 0; j < Ntrees; j++)
		{ 
			if (oobOfTrees[j] > 0)
				std2_var[i] += (DeltMatrix[j][i] / (float)oobOfTrees[j] - mean_var[i]) * (DeltMatrix[j][i] / (float)oobOfTrees[j] - mean_var[i]);
		}
		std2_var[i] /= Ntrees;
	}

	float* raw_score = new float[variables_num];
	float* z_score = new float[variables_num];
	memset(raw_score, 0, sizeof(float) * variables_num);
	memset(z_score, 0, sizeof(float) * variables_num);

	float fsum = 0.f;
	// raw score
	for (i = 0; i < variables_num; i++)
	{
		raw_score[i] = mean_var[i];
		fsum += raw_score[i];
	}

	// Normalization
	if (bNormalize) {
		for (i = 0; i < variables_num; i++)
			raw_score[i] /= fsum;
	}

	// z-score
	fsum = 0.f;
	for (i = 0; i < variables_num; i++)
	{
		//varImportance[i] = mean_var[i] / (sqrtf(std2_var[i] + VERY_SMALL_VALUE) / sqrtn);
		z_score[i] = mean_var[i] / (sqrtf(std2_var[i]) + FLT_EPSILON); //0530
		fsum += z_score[i];
	}

	// Normalization
	if (bNormalize) {
		for (i = 0; i < variables_num; i++)
			z_score[i] /= fsum;
	}

	if (nType == 0) // raw_score
	{
		memcpy(varImportance, raw_score, sizeof(float) * variables_num);
	}
	else  // z-score
	{
		memcpy(varImportance, z_score, sizeof(float) * variables_num);
	}

	
	if (filename != NULL)
	{
		fstream vieFile;
		vieFile.open(filename, ios_base::out);
		if (!vieFile.is_open())
		{
			cout << endl;
			cout << "-----------------  ERROR:'RawVariableImportanceScore'-----------------" << endl;
			cout << "Warning: file is not created." << endl;
			cout << "----------------------------------------------------------------------" << endl;
			cout << endl;
			rv = 0;
		}
		else
		{
			vieFile << "variable index" << "\t" << "raw score" << "\t\t" << "z-score" << endl;
			for (i = 0; i < variables_num; i++)
				vieFile << i << "\t\t" << raw_score[i] << "\t\t" << z_score[i] << endl;
			vieFile.close();
		}

	}

	delete[] tmp_data;

	for (i = 0; i < Ntrees; i++)
	{
		delete[] DeltMatrix[i];
		DeltMatrix[i] = NULL;
	}
	delete[] DeltMatrix;
	delete[] mean_var;
	delete[] std2_var;
	delete[] oobOfTrees;
	delete[] raw_score;
	delete[] z_score;

	return rv;
}

#ifdef  OPENMP_SPEEDUP
int RawVariableImportanceScore2OMP(float** data, int* label, LoquatCForest* loquatForest, int nType, float* varImportance, bool bNormalize, int random_state, char* filename, int jobs)
{
	if (!(nType == 0 || nType == 1))
	{
		cout << endl;
		cout << "------------------ERROR:'RawVariableImportanceScore'------------------" << endl;
		cout << "The input parameter 'nType' must be 0 or 1." << endl;
		cout << " 'nType'     0: raw variable importance score" << endl;
		cout << "             1: z-score" << endl;
		cout << "----------------------------------------------------------------------" << endl;
		cout << endl;
		return -1;
	}

	if (random_state < 0)
	{
		srand_freebsd(GenerateSeedFromSysTime());
	}
	else
	{
		srand_freebsd((unsigned int)random_state);
	}

	int tr, rv = 1;
	const int variables_num = loquatForest->RFinfo.datainfo.variables_num;
	const int Ntrees = loquatForest->RFinfo.ntrees;

	int** DeltMatrix = new int* [Ntrees];
	assert(DeltMatrix != NULL);
	for (int i = 0; i < Ntrees; i++)
	{
		DeltMatrix[i] = new int[variables_num];
		memset(DeltMatrix[i], 0, sizeof(int) * variables_num);
	}

	float* mean_var = new float[variables_num];
	memset(mean_var, 0, sizeof(float) * variables_num);
	float* std2_var = new float[variables_num];
	memset(std2_var, 0, sizeof(float) * variables_num);
	int* oobOfTrees = new int[Ntrees];
	memset(oobOfTrees, 0, sizeof(int) * Ntrees);

	omp_set_num_threads(jobs);

	bool oobFound = false;

	#pragma omp parallel for 
	for (tr = 0; tr < Ntrees; tr++)
	{
		struct LoquatCTreeStruct* pTree = loquatForest->loquatTrees[tr];
		int *pIndex = pTree->outofbag_samples_index;
		int oobnum = pTree->outofbag_samples_num;

		if (pIndex == NULL || oobnum == 0)
			continue;
		oobFound = true;
		oobOfTrees[tr] = oobnum;
		int* permuted_order = new int[oobnum];

		int index, predicted_class_index;
		float confidence = 0.f;
		int correct_num = 0, correct_num_premute = 0;
		int i = 0;
		for (; i < oobnum; i++)
		{
			index = pIndex[i];
			PredictAnTestSampleOnOneTree(data[index], variables_num, pTree, predicted_class_index, confidence);
			if (predicted_class_index == label[index])
				correct_num++;
		}

		float* tmp_data = new float[variables_num];
		for (int var = 0; var < variables_num; var++)
		{
			// permuting (var)th variables
			// permute [0, oobnum-1]
			permute(oobnum, permuted_order);

			for (correct_num_premute = 0, i = 0; i < oobnum; i++)
			{
				index = pIndex[permuted_order[i]];
				//memcpy_s(tmp_data, variables_num*sizeof(float), data[pIndex[i]], variables_num*sizeof(float));
				memcpy(tmp_data, data[pIndex[i]], variables_num * sizeof(float));
				tmp_data[var] = data[index][var];
				PredictAnTestSampleOnOneTree(tmp_data, variables_num, pTree, predicted_class_index, confidence);
				if (predicted_class_index == label[pIndex[i]])
					correct_num_premute++;
			}

			DeltMatrix[tr][var] = correct_num - correct_num_premute;
		}

		delete [] permuted_order;
		delete [] tmp_data;
	}

	for (int v=0; v<variables_num; v++)
		for (int t = 0; t < Ntrees; t++)
		{
			mean_var[v] += DeltMatrix[t][v] / (float)oobOfTrees[t];
		}

	if (oobFound == false)
	{
		cout << endl;
		cout << "-----------------  ERROR:'RawVariableImportanceScore'-----------------" << endl;
		cout << "Error: Out-of-bag samples are not found in RF. The results mean nothing." << endl;
		cout << "----------------------------------------------------------------------" << endl;
		cout << endl;
		rv = 0;
	}

	for (int i = 0; i < variables_num; i++)
		mean_var[i] = mean_var[i] / Ntrees;

	for (int i = 0; i < variables_num; i++)
	{
		for (int j = 0; j < Ntrees; j++)
		{
			if (oobOfTrees[j] > 0)
				std2_var[i] += (DeltMatrix[j][i] / (float)oobOfTrees[j] - mean_var[i]) * (DeltMatrix[j][i] / (float)oobOfTrees[j] - mean_var[i]);
		}
		std2_var[i] /= Ntrees;
	}

	float* raw_score = new float[variables_num];
	float* z_score = new float[variables_num];
	memset(raw_score, 0, sizeof(float) * variables_num);
	memset(z_score, 0, sizeof(float) * variables_num);

	float fsum = 0.f;
	// raw score
	for (int i = 0; i < variables_num; i++)
	{
		raw_score[i] = mean_var[i];
		fsum += raw_score[i];
	}

	// Normalization
	if (bNormalize) {
		for (int i = 0; i < variables_num; i++)
			raw_score[i] /= fsum;
	}

	// z-score
	fsum = 0.f;
	for (int i = 0; i < variables_num; i++)
	{
		//varImportance[i] = mean_var[i] / (sqrtf(std2_var[i] + VERY_SMALL_VALUE) / sqrtn);
		z_score[i] = mean_var[i] / (sqrtf(std2_var[i]) + FLT_EPSILON); //0530
		fsum += z_score[i];
	}

	// Normalization
	if (bNormalize) {
		for (int i = 0; i < variables_num; i++)
			z_score[i] /= fsum;
	}

	if (nType == 0) // raw_score
	{
		memcpy(varImportance, raw_score, sizeof(float) * variables_num);
	}
	else  // z-score
	{
		memcpy(varImportance, z_score, sizeof(float) * variables_num);
	}


	if (filename != NULL)
	{
		fstream vieFile;
		vieFile.open(filename, ios_base::out);
		if (!vieFile.is_open())
		{
			cout << endl;
			cout << "-----------------  ERROR:'RawVariableImportanceScore'-----------------" << endl;
			cout << "Warning: file is not created." << endl;
			cout << "----------------------------------------------------------------------" << endl;
			cout << endl;
			rv = 0;
		}
		else
		{
			vieFile << "variable index" << "\t" << "raw score" << "\t\t" << "z-score" << endl;
			for (int i = 0; i < variables_num; i++)
				vieFile << i << "\t\t" << raw_score[i] << "\t\t" << z_score[i] << endl;
			vieFile.close();
		}

	}


	for (int i = 0; i < Ntrees; i++)
	{
		delete[] DeltMatrix[i];
		DeltMatrix[i] = NULL;
	}
	delete[] DeltMatrix;
	delete[] mean_var;
	delete[] std2_var;
	delete[] oobOfTrees;
	delete[] raw_score;
	delete[] z_score;

	return rv;
}
#endif  // OPENMP_SPEEDUP

LoquatCTreeNode*** createLeafNodeMatrix(LoquatCForest* forest, float** data)
{
	const int samples_num = forest->RFinfo.datainfo.samples_num;
	const int trees_num = forest->RFinfo.ntrees;

	LoquatCTreeNode*** leafNodeMat = new LoquatCTreeNode **[samples_num];
	for (int n = 0; n < samples_num; n++)
	{
		leafNodeMat[n] = new LoquatCTreeNode * [trees_num];
		for (int t = 0; t < trees_num; t++)
		{
			leafNodeMat[n][t] = (LoquatCTreeNode*)GetArrivedLeafNode(forest, t, data[n]);
			assert(leafNodeMat[n][t] != NULL);
		}
	}

	return leafNodeMat;
}

//int ClassificationForestOrigProximity(LoquatCForest* forest, float* data, LoquatCTreeNode*** leafNodeMatrix, float*& proximities)
//{
//
//	const int samples_num = forest->RFinfo.datainfo.samples_num;
//	const int trees_num = forest->RFinfo.ntrees;
//	int rv = 1;
//
//	if (NULL != proximities)
//		delete[] proximities;
//	proximities = new float[samples_num];
//	memset(proximities, 0, sizeof(float) * samples_num);
//
//	for (int t = 0; t < trees_num; t++)
//	{
//		LoquatCTreeNode* leaf_i = (LoquatCTreeNode*)GetArrivedLeafNode(forest, t, data);
//		if (leaf_i == NULL)
//		{
//			rv = -1;
//			continue;
//		}
//		for (int n = 0; n < samples_num; n++)
//		{
//			if (leafNodeMatrix[n][t] == leaf_i)
//			{
//				proximities[n] += 1.f;
//			}
//		}
//	}
//
//	for (int n = 0; n < samples_num; n++)
//	{
//		proximities[n] /= trees_num;
//		assert(proximities[n] >= 0 && proximities[n] <= 1.f);
//	}
//
//	return rv;
//}

int _ClassificationForestOrigProximity(LoquatCForest* forest, float* data, LoquatCTreeNode*** leafNodeMatrix, float*& proximities)
{

	const int samples_num = forest->RFinfo.datainfo.samples_num;
	const int trees_num = forest->RFinfo.ntrees;

	if (NULL != proximities)
		delete[] proximities;
	proximities = new float[samples_num];
	memset(proximities, 0, sizeof(float) * samples_num);

	LoquatCTreeNode** leaves_i = new LoquatCTreeNode * [trees_num];
	for (int t = 0; t < trees_num; t++)
	{
		leaves_i[t] = (LoquatCTreeNode*)GetArrivedLeafNode(forest, t, data);
		assert(leaves_i[t] != NULL);
		if (leaves_i[t] == NULL)
		{
			delete[] leaves_i;
			return -1;
		}
	}

	for (int n = 0; n < samples_num; n++)
	{
		for (int t = 0; t < trees_num; t++)
		{
			if (leafNodeMatrix[n][t] == leaves_i[t])
			{
				proximities[n] += 1.f;
			}
		}
	}

	for (int n = 0; n < samples_num; n++)
	{
		proximities[n] /= trees_num;
		assert(proximities[n] >= 0 && proximities[n] <= 1.f);
	}

	delete[] leaves_i;

	return 1;
}

int ClassificationForestGAPProximity(LoquatCForest* forest, float** data, const int index_i, float*& proximities)
{
	if (NULL != proximities)
		delete[] proximities;


	proximities = new float[forest->RFinfo.datainfo.samples_num];
	memset(proximities, 0, sizeof(float) * forest->RFinfo.datainfo.samples_num);

	int* arrivals = NULL;

	const int ntrees = forest->RFinfo.ntrees;
	const int samples_num = forest->RFinfo.datainfo.samples_num;
	int oobtree_num = 0;
	for (int t = 0; t < ntrees; t++)
	{
		//where the i-th sample is oob
		const struct LoquatCTreeStruct* tree = forest->loquatTrees[t];
		bool i_oob = false;
		for (int n = 0; n < tree->outofbag_samples_num; n++)
		{
			if (index_i == tree->outofbag_samples_index[n])
			{
				i_oob = true;
				break;
			}
		}


		if (false == i_oob)
			continue;

		oobtree_num++;

		const struct LoquatCTreeNode* leaf_i = GetArrivedLeafNode(forest, t, data[index_i]);

		if (leaf_i->samples_index != NULL)
		{
			const int arrival_num = leaf_i->arrival_samples_num;
			//cout << "arrial_num: " << arrival_num<<"conf: "<<leaf_i->leaf_confidence << endl;
			const float w = 1.0f / arrival_num;
			for (int n = 0; n < arrival_num; n++)
			{
				proximities[leaf_i->samples_index[n]] += w;
			}
		}
		else
		{
			// because forest did not store sample index arrrived at the leaf node, each in bag sample has to be tested
			int arrival_num = 0;
			if (NULL == arrivals)
				arrivals = new int[samples_num];
			memset(arrivals, 0, sizeof(int) * samples_num);
			for (int n = 0; n < tree->inbag_samples_num; n++)
			{
				const int j = tree->inbag_samples_index[n];
				const struct LoquatCTreeNode* leaf_j = GetArrivedLeafNode(forest, t, data[j]);
				if (leaf_i == leaf_j)
				{
					arrival_num++;
					arrivals[j]++;
				}
			}

			for (int n = 0; n < samples_num; n++)
			{
				if (arrivals[n])
					proximities[n] += arrivals[n] * 1.f / arrival_num;
			}

		}

	}

	delete[] arrivals;
	if (0 == oobtree_num)
		return -1;

	for (int j = 0; j < forest->RFinfo.datainfo.samples_num; j++)
		proximities[j] = proximities[j] / oobtree_num;

	return 1;
}

int ClassificationForestOrigProximity(LoquatCForest* forest, float** data, const int index_i, float*& proximities)
{
	LoquatCTreeNode*** leafMatrix = createLeafNodeMatrix(forest, data);
	
	int  rv = _ClassificationForestOrigProximity(forest, data[index_i], leafMatrix, proximities);
	
	const int samples_num = forest->RFinfo.ntrees;
	for (int n = 0; n < samples_num; n++)
		delete[] leafMatrix[n];
	delete[] leafMatrix;

	return rv;
}

int ClassificationForestProximity(LoquatCForest* forest, float** data, const int index_i, PROXIMITY_TYPE prox_type, float*& proximities)
{
	int rv = -1;
	switch (prox_type)
	{
	case PROXIMITY_TYPE::PROX_ORIGINAL:
		rv = ClassificationForestOrigProximity(forest, data, index_i, proximities);
		break;
	case PROXIMITY_TYPE::PROX_GEO_ACC:
		rv = ClassificationForestGAPProximity(forest, data, index_i, proximities);
		break;
	default:
		rv = ClassificationForestOrigProximity(forest, data, index_i, proximities);
		break;
	}

	return rv;
}

int RawOutlierMeasure(LoquatCForest* loquatForest, float** data, int* label, float*& raw_score)
{
	if (NULL != raw_score)
		delete[] raw_score;

	int rv = 1;
	const  int samples_num = loquatForest->RFinfo.datainfo.samples_num;
	const int class_num = loquatForest->RFinfo.datainfo.classes_num;

	raw_score = new float[samples_num];
	memset(raw_score, 0, sizeof(float) * samples_num);

	float* proximities = NULL;
	for (int i = 0; i < samples_num; i++)
	{
		ClassificationForestGAPProximity(loquatForest, data, i, proximities /*OUT*/);

		float  proximity2_sum = 0.f;
		for (int j = 0; j < samples_num; j++)
		{
			if (label[j] != label[i] || j == i)
				continue;

			// within class
			proximity2_sum += proximities[j] * proximities[j];
		}

		raw_score[i] = samples_num / (proximity2_sum + 1e-5);

		delete[] proximities;
		proximities = NULL;
	}

	float* dev = new float[class_num];
	float* median = new float[class_num];
	memset(dev, 0, sizeof(float) * class_num);
	memset(median, 0, sizeof(float) * class_num);

	for (int j = 0; j < class_num; j++)
	{
		vector<float>  raw_score_class_j;

		for (int i = 0; i < samples_num; i++)
		{
			if (label[i] == j)
				raw_score_class_j.push_back(raw_score[i]);
		}

		std::sort(raw_score_class_j.begin(), raw_score_class_j.end());

		const int sample_num_j = raw_score_class_j.size();
		if (0 == sample_num_j)
		{
			rv = 0;
			dev[j] = 1.f;
			median[j] = 1.f;
			continue;
		}

		if (sample_num_j % 2 == 1)
			median[j] = raw_score_class_j[sample_num_j / 2];
		else
			median[j] = 0.5f * (raw_score_class_j[sample_num_j / 2] + raw_score_class_j[sample_num_j / 2 - 1]);


		for (auto it = raw_score_class_j.begin(); it != raw_score_class_j.end(); it++)
			dev[j] += abs(*it - median[j]);
		dev[j] = dev[j] / sample_num_j;
	}

	for (int i = 0, lb = 0; i < samples_num; i++)
	{
		lb = label[i];
		raw_score[i] = RF_MAX((raw_score[i] - median[lb]) / (dev[lb] + 1e-5), 0.0);
	}

	delete[] dev;
	delete[] median;

	return rv;
}
 
int PredictAnTestSampleOnOneTree(float *data, const int variables_num, struct LoquatCTreeStruct *loquatTree, int &predicted_class_index, float &confidence)
{
	int max_depth_index = loquatTree->depth, cc=0;
	struct LoquatCTreeNode *pNode = loquatTree->rootNode;
	int test_variables_index;
	float test_splitv;

	while(1)
	{
		if( pNode == NULL )
			return -3;

		if( pNode->nodetype == TreeNodeTpye::enLeafNode )
		{
			predicted_class_index = pNode->leaf_node_label;
			confidence = pNode->leaf_confidence;
			return 1;
		}

		test_variables_index = pNode->split_variable_index;
		test_splitv = pNode->split_value;
		if( test_variables_index >= variables_num )
		{
			predicted_class_index = -1;
			confidence = 0.f;
			return -2;
		}

		if( data[test_variables_index] <= test_splitv )
		{
			pNode = pNode->pSubNode[0]; // 左枝
		}else
			pNode = pNode->pSubNode[1]; // 右枝

		cc++;
		if( cc>max_depth_index )
			break;
	}

	return -1;
}

const struct LoquatCTreeNode *GetArrivedLeafNode(const LoquatCForest *RF, const int tree_index, float *data)
{
	if( tree_index < 0 || tree_index >= RF->RFinfo.ntrees)
		return NULL;

	const int max_depth_index = RF->loquatTrees[tree_index]->depth;
	int depth = 0;
	struct LoquatCTreeNode *pNode = RF->loquatTrees[tree_index]->rootNode;

	while(1)
	{
		if( pNode == NULL )
			return NULL;

		if( pNode->nodetype == TreeNodeTpye::enLeafNode )
			return pNode;

		if( data[pNode->split_variable_index] <= pNode->split_value )
			pNode = pNode->pSubNode[0]; // 左枝
		else
			pNode = pNode->pSubNode[1]; // 右枝

		if( (++depth) > max_depth_index )
			break;
	}

	return NULL;
}

int ErrorOnInbagTrainSamples(float** data, const int* label, const LoquatCForest* loquatForest, float& error_rate, int isHardDecision)
{
	int Ntrees = loquatForest->RFinfo.ntrees;
	int classes_num = loquatForest->RFinfo.datainfo.classes_num;
	int samples_num = loquatForest->RFinfo.datainfo.samples_num;
	int i, j, k, inbagnum, indx, rv = 1, * pIndex = NULL;
	struct LoquatCTreeStruct* ploquatTree = NULL;
	const struct LoquatCTreeNode* leafNode = NULL;
	int predicted_class_index = -1;

	float** data_class_confidence = NULL;

	
	data_class_confidence = new float* [samples_num];
	assert(NULL != data_class_confidence);
	for (i = 0; i < samples_num; i++)
	{
		data_class_confidence[i] = new float[classes_num];
		assert(NULL != data_class_confidence[i]);
		memset(data_class_confidence[i], 0, sizeof(float) * classes_num); // 初始化都为0.f
	}

	bool* bInbagnum_norep = new bool[samples_num];
	assert(bInbagnum_norep != NULL);
	bool* bHaveSeenInATree = new bool[samples_num];
	assert(bHaveSeenInATree != NULL);
	memset(bInbagnum_norep, false, sizeof(bool) * samples_num);
	int inbagnum_norep = 0;

	for (i = 0; i < Ntrees; i++)
	{
		ploquatTree = loquatForest->loquatTrees[i];

		if (ploquatTree == NULL)
		{
			rv = -1;
			continue;
		}
		if (ploquatTree->inbag_samples_index == NULL)
		{
			rv = -2;
			continue;
		}

		memset(bHaveSeenInATree, false, sizeof(bool) * samples_num);

		inbagnum = ploquatTree->inbag_samples_num;
		pIndex = ploquatTree->inbag_samples_index;

		for (j = 0; j < inbagnum; j++) // in-bag samples 会有重复
		{
			indx = pIndex[j];

			if (true == bHaveSeenInATree[indx]) // 在这颗树上已经计算过这个sample
				continue;
			else
				bHaveSeenInATree[indx] = true;

			if (false == bInbagnum_norep[indx])
			{
				bInbagnum_norep[indx] = true;
				inbagnum_norep++;
			}

			leafNode = GetArrivedLeafNode(loquatForest, i, data[indx]);
			if (NULL == leafNode)
			{
				rv = 0;   // 有错误产生，但不至于退出函数
				continue; // 下一棵树
			}

			if (isHardDecision > 0) // HARD
			{
				predicted_class_index = leafNode->leaf_node_label;
				data_class_confidence[indx][predicted_class_index] += 1.f;
			}
			else // Probabilistic
			{
				for (k = 0; k < classes_num; k++)
					data_class_confidence[indx][k] += leafNode->class_distribution[k];
			}

		}
	}

	int error_num = 0;
	for (i = 0; i < samples_num; i++)
	{
		if (false == bInbagnum_norep[i]) //可能有些sample从来没有进入到in-bag
			continue;

		MaximumConfienceClassLabel(data_class_confidence[i], classes_num, predicted_class_index, NULL);

		if (predicted_class_index != label[i])
			error_num++;
	}

	error_rate = inbagnum_norep > 0 ? error_num / (float)inbagnum_norep : 1.0f;

	// Release allocated memory
	for (j = 0; j < samples_num; j++)
	{
		delete[] data_class_confidence[j];
		data_class_confidence[j] = NULL;
	}
	delete[] data_class_confidence;
	data_class_confidence = NULL;
	

	delete[] bInbagnum_norep;
	delete[] bHaveSeenInATree;

	return 1;
}

int OOBErrorEstimate(float** data, const int* label, const LoquatCForest* loquatForest, float& error_rate, int isHardDecision)
{
	const int ntrees = loquatForest->RFinfo.ntrees;
	const int classes_num = loquatForest->RFinfo.datainfo.classes_num;
	const int samples_num = loquatForest->RFinfo.datainfo.samples_num;
	int i, j, k, oobnum, indx, rv = 1, * pIndex = NULL;
	const  struct LoquatCTreeStruct* ploquatTree = NULL;
	const struct LoquatCTreeNode* leafNode = NULL;
	//float confience = 0.0f;
	int predicted_class_index = -1;

	float** data_class_confidence = NULL;
	
	data_class_confidence = new float* [samples_num];
	assert(NULL != data_class_confidence);
	for (i = 0; i < samples_num; i++)
	{
		data_class_confidence[i] = new float[classes_num];
		assert(NULL != data_class_confidence[i]);
		memset(data_class_confidence[i], 0, sizeof(float) * classes_num); // 初始化都为0.f
	}

	bool* bEffective = new bool[samples_num]; // 可能有些sample是所有树的inbag
	memset(bEffective, 0, sizeof(bool) * samples_num);
	int effectiveNum = 0;

	for (i = 0; i < ntrees; i++)
	{
		ploquatTree = loquatForest->loquatTrees[i];
		if (ploquatTree == NULL)
		{
			rv = -1;
			continue;
		}
		if (ploquatTree->outofbag_samples_index == NULL)
		{
			rv = -2;
			continue;
		}

		oobnum = ploquatTree->outofbag_samples_num;
		pIndex = ploquatTree->outofbag_samples_index;

		for (j = 0; j < oobnum; j++)
		{
			indx = pIndex[j];
			if (bEffective[indx] == false)
			{
				bEffective[indx] = true;
				effectiveNum++;
			}

			leafNode = GetArrivedLeafNode(loquatForest, i, data[indx]);
			if (NULL == leafNode)
			{
				rv = 0;   // 有错误产生，但不至于退出函数
				continue; // 下一个OOB sample
			}

			if (isHardDecision > 0 || NULL == leafNode->class_distribution) // HARD
			{
				predicted_class_index = leafNode->leaf_node_label;
				data_class_confidence[indx][predicted_class_index] += 1;
			}
			else // Probabilistic
			{
				for (k = 0; k < classes_num; k++)
					data_class_confidence[indx][k] += leafNode->class_distribution[k];
			}
		}
	}

	if (effectiveNum == 0)
	{
		for (j = 0; j < samples_num; j++)
		{
			delete[] data_class_confidence[j];
			data_class_confidence[j] = NULL;
		}
		delete[] data_class_confidence;
		data_class_confidence = NULL;

		delete[] bEffective;

		error_rate = 1.0f;
		return rv;
	}

	int error_num = 0;
	for (i = 0; i < samples_num; i++)
	{
		if (false == bEffective[i])
			continue;

		MaximumConfienceClassLabel(data_class_confidence[i], classes_num, predicted_class_index, NULL);
		if (predicted_class_index != label[i])
			error_num++;
	}

	error_rate = error_num / (float)effectiveNum;

	// Release allocated memory
	
	for (j = 0; j < samples_num; j++)
	{
		delete[] data_class_confidence[j];
		data_class_confidence[j] = NULL;
	}
	delete[] data_class_confidence;
	data_class_confidence = NULL;
	
	delete[] bEffective;

	return rv;
}

int OOBErrorEstimateSequential(float **data, int *label, LoquatCForest *loquatForest, float *&error_rate_sequent,  int isHardDecision, char *filename)
{
	int Ntrees = loquatForest->RFinfo.ntrees;
	//int variables_num = loquatForest->RFinfo.datainfo.variables_num;
	int classes_num = loquatForest->RFinfo.datainfo.classes_num;
	int samples_num = loquatForest->RFinfo.datainfo.samples_num;
	int i, j, k, oobnum, indx, rv=1, *pIndex = NULL;
	struct LoquatCTreeStruct *ploquatTree = NULL;
	const struct LoquatCTreeNode *leafNode = NULL;
	//float confience = 0.0f;
	int predicted_class_index = -1;
	int *predicted_labels = new int[samples_num];
	for ( i=0; i<samples_num; i++ )
		predicted_labels[i] = -1;
	int numofseen = 0, error_num = 0;

	if( error_rate_sequent )
		delete [] error_rate_sequent;
	error_rate_sequent = new float [Ntrees];

	int **data_class_count = NULL;
	float **data_class_confidence = NULL;

	if( isHardDecision > 0 )
	{
		data_class_count = new int *[samples_num];
		assert( NULL != data_class_count );
		for ( i=0; i<samples_num; i++ )
		{
			data_class_count[i] = new int [classes_num];
			assert( NULL != data_class_count[i] );
			memset(data_class_count[i], 0, sizeof(int)*classes_num ); // 初始化都为0
		}
	}
	else
	{
		data_class_confidence = new float *[samples_num];
		assert( NULL != data_class_confidence );
		for ( i=0; i<samples_num; i++ )
		{
			data_class_confidence[i] = new float [classes_num];
			assert( NULL != data_class_confidence[i] );
			memset(data_class_confidence[i], 0, sizeof(float)*classes_num ); // 初始化都为0.f
		}
	}

	for( i=0; i<Ntrees; i++ )
	{
		ploquatTree = loquatForest->loquatTrees[i];
		if( ploquatTree == NULL )
			continue;
		if( ploquatTree->outofbag_samples_index == NULL )
			continue;

		oobnum = ploquatTree->outofbag_samples_num;
		pIndex = ploquatTree->outofbag_samples_index;


		for( j=0; j<oobnum; j++ )
		{
			indx = pIndex[j];
			//rv = PredictAnTestSampleOnOneTree(data[indx], variables_num, ploquatTree, predicted_class_index, confience);
			leafNode = GetArrivedLeafNode(loquatForest, i, data[indx]);
			if( NULL == leafNode )
			{
				rv = 0;
				continue;
			}

			if( isHardDecision > 0 ) // HARD
			{
				predicted_class_index = leafNode->leaf_node_label;
				data_class_count[indx][predicted_class_index] += 1;
			}else // Probabilistic
			{
				for( k=0; k < classes_num; k++ )
					data_class_confidence[indx][k] += leafNode->class_distribution[k];
			}

			//data_class_count[indx][predicted_class_index] += 1;
			if( isHardDecision > 0 ) // HARD
			{
				if( predicted_labels[indx] == -1 )
				{
					predicted_labels[indx] = predicted_class_index;
					numofseen++;
					if( predicted_class_index != label[indx] )
						error_num++;
				}
				else
				{
					if( data_class_count[indx][predicted_class_index] > data_class_count[indx][predicted_labels[indx]] )
					{
						if( predicted_class_index != label[indx] && predicted_labels[indx] == label[indx] )
							error_num++;
						else if( predicted_class_index == label[indx] && predicted_labels[indx] != label[indx] )
							error_num--;
						predicted_labels[indx] = predicted_class_index;
					}
				}
			}else  // Probabilistic
			{
				// 当前累积预测
				MaximumConfienceClassLabel(data_class_confidence[indx], classes_num, predicted_class_index, NULL);
				if( predicted_labels[indx] == -1 ) // 这个数据 unseen so far
				{
					predicted_labels[indx] = predicted_class_index;
					numofseen++;
					if( predicted_class_index != label[indx] )
						error_num++;
				}
				else
				{
					if( predicted_class_index != predicted_labels[indx] ) // 如果当前预测与之前预测不同
					{
						if( predicted_labels[indx] == label[indx] ) // 之前预测与真实相同，则错数量误加1
							error_num++;
						else if( predicted_class_index == label[indx] ) // 当前预测与真实相同，则错误数量减1
							error_num--;
						predicted_labels[indx] = predicted_class_index; // 更新累积的预测
					}
					// else // 如果当前预测与之前预测相同，错误数量保持不变
				}
			}
			
		}

		error_rate_sequent[i] = error_num/(float)numofseen;
	}

	if( filename != NULL )
	{
		fstream oobEsFile;
		oobEsFile.open(filename, ios_base::out);
		if( !oobEsFile.is_open() )
		{
			cout<<endl;
			cout<<"-----------------  ERROR:' OOBErrorEstimateSequential'-----------------"<<endl;
			cout<<"Warning: file is not created."<<endl;
			cout<<"----------------------------------------------------------------------"<<endl;
			cout<<endl;
			return 1;
		}
		for( i=0; i<Ntrees; i++ )
			oobEsFile<<error_rate_sequent[i]<<endl;
		oobEsFile.close();
	}

	// Release allocated memory
	if( isHardDecision>0 )
	{
		for ( j=0; j<samples_num; j++ )
		{
			delete [] data_class_count[j];
			data_class_count[j] = NULL;
		}
		delete [] data_class_count;
		data_class_count = NULL;
	}else
	{
		for ( j=0; j<samples_num; j++ )
		{
			delete [] data_class_confidence[j];
			data_class_confidence[j] = NULL;
		}
		delete [] data_class_confidence;
		data_class_confidence = NULL;
	}

	delete []predicted_labels;

	return rv;
}

int ErrorOnTestSamples(float** data_test, const int* label_test, const int nTestSamplesNum, const LoquatCForest* loquatForest, float& error_rate, int isHardDecision)
{
	int Ntrees = loquatForest->RFinfo.ntrees;
	int classes_num = loquatForest->RFinfo.datainfo.classes_num;
	int i, j, k, rv = 1;
	const struct LoquatCTreeStruct* ploquatTree = NULL;
	const struct LoquatCTreeNode* leafNode = NULL;
	//float confience = 0.0f;
	int predicted_class_index = -1;

	float** data_class_confidence = NULL;

	data_class_confidence = new float* [nTestSamplesNum];
	assert(NULL != data_class_confidence);
	for (i = 0; i < nTestSamplesNum; i++)
	{
		data_class_confidence[i] = new float[classes_num];
		assert(NULL != data_class_confidence[i]);
		memset(data_class_confidence[i], 0, sizeof(float) * classes_num); // 初始化都为0.f
	}
	

	for (i = 0; i < Ntrees; i++)
	{
		ploquatTree = loquatForest->loquatTrees[i];
		if (ploquatTree == NULL)
			continue;

		for (j = 0; j < nTestSamplesNum; j++)
		{
			leafNode = GetArrivedLeafNode(loquatForest, i, data_test[j]);
			if (NULL == leafNode)
			{
				rv = 0;   // 有错误产生，但不至于退出函数
				continue; // 下一棵树
			}

			if (isHardDecision > 0 || NULL == leafNode->class_distribution) // HARD
			{
				predicted_class_index = leafNode->leaf_node_label;
				data_class_confidence[j][predicted_class_index] += 1;
			}
			else // Probabilistic
			{
				for (k = 0; k < classes_num; k++)
					data_class_confidence[j][k] += leafNode->class_distribution[k];
			}

		}
	}

	int error_num = 0;
	for (i = 0; i < nTestSamplesNum; i++)
	{
		MaximumConfienceClassLabel(data_class_confidence[i], classes_num, predicted_class_index, NULL);
		if (predicted_class_index != label_test[i])
			error_num++;
	}

	error_rate = error_num / (float)nTestSamplesNum;

	// Release allocated memory

	for (j = 0; j < nTestSamplesNum; j++)
	{
		delete[] data_class_confidence[j];
		data_class_confidence[j] = NULL;
	}
	delete[] data_class_confidence;
	data_class_confidence = NULL;

	return rv;
}

int ComputeWeightedMarginOnOneSample(float *data, int label, LoquatCForest *loquatForest, float &margin)
{
	const struct LoquatCTreeNode *leafNode = NULL;
	int ntrees = loquatForest->RFinfo.ntrees;
	int classes_num = loquatForest->RFinfo.datainfo.classes_num;
	int i, j, rv = 1;
	float *accumul_class_distri = new float [classes_num];
	memset( accumul_class_distri, 0, sizeof(float)*classes_num );

	for ( i=0; i<ntrees; i++ )
	{
		leafNode = GetArrivedLeafNode(loquatForest, i, data);
		if( NULL == leafNode )
		{
			rv = 0;   // 有错误产生，但不至于退出函数
			continue; // 下一棵树
		}
		for( j=0; j<classes_num; j++ )
			accumul_class_distri[j] += leafNode->class_distribution[j];
	}

	float max_distri = -1e10;
	for( j=0; j<classes_num; j++ )
	{
		if( j == label )
			continue;
		if( accumul_class_distri[j] > max_distri )
			max_distri = accumul_class_distri[j];
	}

	margin = 1.0f/ntrees * (accumul_class_distri[label] - max_distri);

	delete [] accumul_class_distri;

	return rv;
}

int ComputeWeightedMargin(float **data, int *label, int samples_num, LoquatCForest *loquatForest, float &margin)
{
	float tmp = 0.f, m = 0.f;
	int rv = 0;
	for( int k=0; k<samples_num; k++ )
	{
		rv = ComputeWeightedMarginOnOneSample(data[k], label[k], loquatForest, m);
		tmp += m;
		assert(rv > 0);
	}
	margin = tmp/samples_num;
	return 1;
}


int ComputeVotingMarginOnOneSample(float *data, int label, LoquatCForest *loquatForest, float &margin)
{
	const struct LoquatCTreeNode *leafNode = NULL;
	int ntrees = loquatForest->RFinfo.ntrees;
	int classes_num = loquatForest->RFinfo.datainfo.classes_num;
	int i, j, rv=1;
	int *accumul_class_votes = new int [classes_num];
	memset( accumul_class_votes, 0, sizeof(int)*classes_num );

	for ( i=0; i<ntrees; i++ )
	{
		leafNode = GetArrivedLeafNode(loquatForest, i, data);
		if( NULL == leafNode )
		{
			rv = 0;   // 有错误产生，但不至于退出函数
			continue; // 下一棵树
		}
		accumul_class_votes[leafNode->leaf_node_label]++;
	}

	int max_votes = -1;
	for( j=0; j<classes_num; j++ )
	{
		if( j == label )
			continue;
		if( accumul_class_votes[j] > max_votes )
			max_votes = accumul_class_votes[j];
	}

	margin = 1.0f/ntrees * (accumul_class_votes[label] - max_votes);

	delete [] accumul_class_votes;

	return rv;
}

int ComputeVotingOOBMarginOnOneSample(float* data, int label, int index, LoquatCForest* loquatForest, float& margin)
{
	const struct LoquatCTreeNode* leafNode = NULL;
	int ntrees = loquatForest->RFinfo.ntrees;
	int classes_num = loquatForest->RFinfo.datainfo.classes_num;
	int i, j, rv = 1;
	int* accumul_class_votes = new int[classes_num];
	memset(accumul_class_votes, 0, sizeof(int) * classes_num);

	int oob_trees = 0;

	for (i = 0; i < ntrees; i++)
	{
		const LoquatCTreeStruct* tree = loquatForest->loquatTrees[i];
		bool isOOB = false;
		for (int n = 0; n < tree->outofbag_samples_num; n++)
		{
			if (tree->outofbag_samples_index[n] == index)
			{
				isOOB = true;
				break;
			}
		}

		if (false == isOOB)
			continue;
		oob_trees++;
		leafNode = GetArrivedLeafNode(loquatForest, i, data);
		if (NULL == leafNode)
		{
			rv = 0;   // 有错误产生，但不至于退出函数
			continue; // 下一棵树
		}
		accumul_class_votes[leafNode->leaf_node_label]++;
	}

	if (0 == oob_trees)
		return -1;

	int max_votes = -1;
	for (j = 0; j < classes_num; j++)
	{
		if (j == label)
			continue;
		if (accumul_class_votes[j] > max_votes)
			max_votes = accumul_class_votes[j];
	}

	margin = (accumul_class_votes[label] - 1.0f*max_votes)/oob_trees;
	
	delete[] accumul_class_votes;

	return rv;
}

int ComputeVotingMargin(float **data, int *label, int samples_num, LoquatCForest *loquatForest, float &margin)
{
	float tmp = 0.f, m = 0.f;
	int rv = 0;
	for( int k=0; k<samples_num; k++ )
	{
		rv = ComputeVotingMarginOnOneSample(data[k], label[k], loquatForest, m);
		tmp += m;
		assert(rv > 0);
	}
	margin = tmp/samples_num;
	return 1;
}

int ComputeVotingOOBMargin(float** data, int* label, LoquatCForest* loquatForest, float& margin)
{
	float tmp = 0.f, m = 0.f;
	int rv = 0, count = 0;
	const int samples_num = loquatForest->RFinfo.datainfo.samples_num;

	for (int k = 0; k < samples_num; k++)
	{
		rv = ComputeVotingOOBMarginOnOneSample(data[k], label[k], k, loquatForest, m);
		tmp += m *(rv>0 ? 1:0);
		count += (rv > 0 ? 1 : 0);
	}
	if (count > 0)
	{
		margin = tmp / count;
		return 1;
	}
	margin = 0.f;
	return 0;
}

int HarvestOneLeafNode(struct LoquatCTreeNode **treeNode)
{
	if( *treeNode == NULL )
		return 1;

	if( (*treeNode)->samples_index != NULL )
	{
		delete [] (*treeNode)->samples_index;
		(*treeNode)->samples_index = NULL;
		(*treeNode)->arrival_samples_num = 0;
	}

	if( (*treeNode)->class_distribution != NULL )
	{
		delete [] (*treeNode)->class_distribution;
		(*treeNode)->class_distribution = NULL;
	}

	if( (*treeNode)->pSubNode != NULL )
	{
		delete [] (*treeNode)->pSubNode;
		(*treeNode)->pSubNode = NULL;
	}

	delete *treeNode;
	*treeNode = NULL;

	return 1;
}

int HarvestOneCLoquatTreeRecursively(struct LoquatCTreeStruct **loquatTree)
{
	if( (*loquatTree) == NULL )
		return 1;

	delete (*loquatTree)->rootNode;

	if( (*loquatTree)->inbag_samples_index != NULL )
	{
		delete [] (*loquatTree)->inbag_samples_index;
		(*loquatTree)->inbag_samples_index = NULL;
		(*loquatTree)->inbag_samples_num = 0;
	}

	if( (*loquatTree)->outofbag_samples_index != NULL )
	{
		delete [] (*loquatTree)->outofbag_samples_index;
		(*loquatTree)->outofbag_samples_index = NULL;
		(*loquatTree)->outofbag_samples_num = 0;
	}

	delete *loquatTree;
	*loquatTree = NULL;

	return 1;
}

int ReleaseClassificationForest(LoquatCForest **loquatForest)
{
	if( (*loquatForest) == NULL )
		return 1;

	const int Ntrees = (*loquatForest)->RFinfo.ntrees;
	for ( int i=0; i<Ntrees; i++ )
	{
		if( (*loquatForest)->loquatTrees[i] == NULL )
			continue;

		HarvestOneCLoquatTreeRecursively(&(*loquatForest)->loquatTrees[i]);	
	}

	delete [] ((*loquatForest)->loquatTrees); // 二级指针
	delete (*loquatForest);
	(*loquatForest) = NULL;

	return 1;
}

// int HarvestEntireCLoquatForestRecursively(LoquatCForest *loquatForest)
// {
// 	int Ntrees = loquatForest->RFinfo.ntrees;
// 	for ( int i=0; i<Ntrees; i++ )
// 	{
// 		if( loquatForest->loquatTrees[i] == NULL )
// 			continue;
// 		HarvestOneDLoquatTreeRecursively(loquatForest->loquatTrees[i]);
// 		delete loquatForest->loquatTrees[i];
// 		loquatForest->loquatTrees[i] = NULL;
// 	}
// 	delete []loquatForest->loquatTrees; // 二级指针
// 	return 1;
// }

// 2021-03-15
void DisplayLoquatTreeInfo(struct LoquatCTreeStruct* loquatTree, RandomCForests_info RFinfo)
{
	if (NULL == loquatTree || NULL == loquatTree->rootNode)
	{
		cout << "NULL pointer error" << endl;
		return;
	}

	cout << endl << "------------------------Tree Info------------------------" << endl;
	//cout <<  "depth of the tree: " << loquatTree->depth << endl;
	vector< LoquatCTreeNode* > treeNodes;
	treeNodes.push_back(loquatTree->rootNode);

	int depth = 0;

	while (treeNodes.size()>0)
	{
		vector<LoquatCTreeNode*> nextDepthNodes;
		vector<LoquatCTreeNode*>::iterator it = treeNodes.begin();
		cout << "^^^^^^^^^^^^^^^^^^^ depth: " << depth << "^^^^^^^^^^^^^^^^^^^^^^^" << endl;
		for (; it != treeNodes.end(); it++)
		{
			// display node info
			struct LoquatCTreeNode* pNode = (*it);
			switch (pNode->nodetype)
			{
			case TreeNodeTpye::enLeafNode:
				cout << "***Leaf Node:" << endl;
				cout << "   the variable to split:" << "no" << "  split value:" << "no" << endl;
				cout << "   arrival number:" << pNode->arrival_samples_num << "  impurity on training data:" << pNode->train_impurity << endl;
				for (int k = 0; k < RFinfo.datainfo.classes_num; k++)
					cout << "   class" << k << ": " << pNode->class_distribution[k] << ", " << endl;
				cout << "   assigned class label: " << pNode->leaf_node_label << endl;
				break;
			case TreeNodeTpye::enLinkNode:
				cout << "***Link Node:" << endl;
				break;
			case TreeNodeTpye::enRootNode:
				cout << "***Root Node:" << endl;
				break;
			}

			if (pNode->nodetype == TreeNodeTpye::enLinkNode || pNode->nodetype == TreeNodeTpye::enRootNode)
			{
				cout << "   the variable to split:" << pNode->split_variable_index << "  split value:" << pNode->split_value << endl;
				cout << "   arrival number:" << pNode->arrival_samples_num << "  impurity on training data:" << pNode->train_impurity << endl;
				for (int k = 0; k < RFinfo.datainfo.classes_num; k++)
					cout << "   class" << k << ": " << pNode->class_distribution[k] << ", ";
				cout << endl;
			}

			// 叶子节点
			if (TreeNodeTpye::enLeafNode == (*it)->nodetype)
			{
				continue;
			}

			nextDepthNodes.push_back((*it)->pSubNode[0]);
			nextDepthNodes.push_back((*it)->pSubNode[1]);
			
		}

		depth++;
		treeNodes.assign(nextDepthNodes.begin(), nextDepthNodes.end()); // 清空原vector，赋予新数据

	}

}


void PrintForestInfo(const LoquatCForest* forest, ostream &out)
{
	if (NULL == forest)
	{
		return;
	}

	int max_depth = 0, max_nodes = 0, max_leaf_nodes = 0;
	int max_depth_tree = -1, max_nodes_tree = -1, max_leaf_nodes_tree = -1;
	float aver_depth = 0.f, aver_nodes = 0.f, aver_leaf_nodes = 0.f;
	for (int t = 0; t < forest->RFinfo.ntrees; t++)
	{
		const struct LoquatCTreeStruct* pTree = forest->loquatTrees[t];
		if (NULL == pTree)
		{
			continue;
		}

		vector< LoquatCTreeNode* > treeNodes;
		treeNodes.push_back(pTree->rootNode);

		int node_num = 0, leaf_node_num = 0;

		while (treeNodes.size() > 0)
		{
			vector<LoquatCTreeNode*> nextDepthNodes;
			vector<LoquatCTreeNode*>::iterator it = treeNodes.begin();
			
			for (; it != treeNodes.end(); it++)
			{
				
				const struct LoquatCTreeNode* const pNode = (*it);
				switch (pNode->nodetype)
				{
				case TreeNodeTpye::enLeafNode:
					leaf_node_num++;
					
				case TreeNodeTpye::enLinkNode:
				case TreeNodeTpye::enRootNode:
					node_num++;
					break;
				}

				// 叶子节点
				if (TreeNodeTpye::enLeafNode == pNode->nodetype)
				{
					continue;
				}

				nextDepthNodes.push_back(pNode->pSubNode[0]);
				nextDepthNodes.push_back(pNode->pSubNode[1]);

			}

			treeNodes.assign(nextDepthNodes.begin(), nextDepthNodes.end()); // 清空原vector，赋予新数据

		}

		assert(leaf_node_num == pTree->leaf_node_num);
		out << "Tree " << t+1 << ": depth "<<pTree->depth<<",\t nodes " << node_num << ",\t leaf_nodes " << leaf_node_num << endl;

		if (pTree->depth > max_depth)
		{
			max_depth = pTree->depth;
			max_depth_tree = t+1;
		}
		if (node_num > max_nodes)
		{
			max_nodes = node_num;
			max_nodes_tree = t+1;
		}
		if (leaf_node_num > max_leaf_nodes)
		{
			max_leaf_nodes = leaf_node_num;
			max_leaf_nodes_tree = t+1;
		}

		aver_depth += pTree->depth;
		aver_nodes += node_num;
		aver_leaf_nodes += leaf_node_num;
	}
	aver_depth /= forest->RFinfo.ntrees;
	aver_nodes /= forest->RFinfo.ntrees;
	aver_leaf_nodes /= forest->RFinfo.ntrees;
	out << "-------------------------------------------------------------------------" << endl;
	out << "max depth: " <<max_depth<<"("<<max_depth_tree<<"),\t max_nodes: "<<max_nodes<<"("<<max_nodes_tree<<"),\t max_leaf: "<<max_leaf_nodes<<"("<<max_leaf_nodes_tree<<")"<< endl;
    streamsize ss = std::cout.precision();
	out << setiosflags(ios::fixed)<< std::setprecision(1) << "aver depth: " << aver_depth << ", \t aver nodes: " << aver_nodes << ", \t aver leaf nodes: " << aver_leaf_nodes << endl;
    out << resetiosflags(ios::fixed)<<std::setprecision(ss); // set back
}
