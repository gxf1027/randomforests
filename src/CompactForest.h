/*
* Random Forest with more compact structures
* About 50% less memory occupation, and 30% faster for testing
*/ 

#pragma once
#ifndef _COMPACT_FOREST_
#define _COMPACT_FOREST_

#include "RandomCLoquatForests.h"

typedef struct CompactTreeNode {
	int split_index;
	float split_value;
	int left_index;
	int right_index;
}CompactTreeNode;

typedef struct CompactForest {
	CompactTreeNode** trees;
	int* nodes_of_tree; // non-leaf-num
	RandomCForests_info rfinfo;
}CompactForest;

/*
* Convert from origin forest to forest with compact structures
*/
CompactForest* toCompactForest(LoquatCForest* forest);
int ErrorOnTestSamples(float** data_test, const int* label_test, const int nTestSamplesNum, const CompactForest* forest, float& error_rate);
int ErrorOnTestSamples2(float** data_test, const int* label_test, const int nTestSamplesNum, const CompactForest* forest, float& error_rate);
void ReleaseCompactForest(CompactForest** forest);

#endif