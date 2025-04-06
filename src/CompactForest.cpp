#include "CompactForest.h"
#include <vector>
#include <assert.h>
#include <string.h>

CompactForest* toCompactForest(LoquatCForest* forest)
{
	const int ntrees = forest->RFinfo.ntrees;
	CompactForest* compactForest = new CompactForest;
	compactForest->rfinfo = forest->RFinfo;
	compactForest->trees = new CompactTreeNode * [ntrees];
	compactForest->nodes_of_tree = new int[ntrees];
	memset(compactForest->nodes_of_tree, 0, sizeof(int) * ntrees);

	// number of non-leaf nodes
	for (int t = 0; t < ntrees; t++)
	{
		const struct LoquatCTreeStruct* pTree = forest->loquatTrees[t];
		if (NULL == pTree)
		{
			continue;
		}

		std::vector< LoquatCTreeNode* > treeNodes;
		treeNodes.push_back(pTree->rootNode);


		while (treeNodes.size() > 0)
		{
			std::vector<LoquatCTreeNode*> nextDepthNodes;
			std::vector<LoquatCTreeNode*>::iterator it = treeNodes.begin();

			for (; it != treeNodes.end(); it++)
			{

				const struct LoquatCTreeNode* const pNode = (*it);
				switch (pNode->nodetype)
				{
				case TreeNodeType::LINK_NODE:
				case TreeNodeType::ROOT_NODE:
					compactForest->nodes_of_tree[t]++;
					break;
				}

				if (TreeNodeType::LEAF_NODE == pNode->nodetype)
				{
					continue;
				}

				nextDepthNodes.push_back(pNode->pSubNode[0]);
				nextDepthNodes.push_back(pNode->pSubNode[1]);

			}

			treeNodes.assign(nextDepthNodes.begin(), nextDepthNodes.end());
		}
	}

	const int class_num = forest->RFinfo.datainfo.classes_num;
	for (int t = 0; t < ntrees; t++)
	{
		const int node_num = compactForest->nodes_of_tree[t];
		compactForest->trees[t] = new CompactTreeNode[node_num + class_num];
		CompactTreeNode* this_tree = compactForest->trees[t];

		for (int c = 0; c < class_num; c++)
		{
			compactForest->trees[t][node_num + c].left_index = -1;
			compactForest->trees[t][node_num + c].right_index = -1;
			compactForest->trees[t][node_num + c].split_index = -1; // indicate leaf node
			compactForest->trees[t][node_num + c].split_value = c; // class index
		}

		LoquatCTreeNode* root = forest->loquatTrees[t]->rootNode;
		std::vector<LoquatCTreeNode*> nodes_this_level;
		std::vector<LoquatCTreeNode*> nodes_next_level;
		nodes_this_level.push_back(root);
		int index = 0;
		while (nodes_this_level.size() > 0)
		{
			LoquatCTreeNode* pNode;
			nodes_next_level.clear();

			const int start = index;
			const int this_level_size = nodes_this_level.size();
			for (int k = 0; k < this_level_size; k++)
			{
				pNode = nodes_this_level[k];

				if (pNode->nodetype == TreeNodeType::LEAF_NODE)
					assert(0);

				if (pNode->pSubNode[0]->nodetype == TreeNodeType::LEAF_NODE)
				{
					this_tree[index].left_index = node_num + pNode->pSubNode[0]->leaf_node_label;
				}
				else {
					nodes_next_level.push_back(pNode->pSubNode[0]);
					this_tree[index].left_index = start + this_level_size + (nodes_next_level.size() - 1);
				}

				if (pNode->pSubNode[1]->nodetype == TreeNodeType::LEAF_NODE)
				{
					this_tree[index].right_index = node_num + pNode->pSubNode[1]->leaf_node_label;
				}
				else {
					nodes_next_level.push_back(pNode->pSubNode[1]);
					this_tree[index].right_index = start + this_level_size + (nodes_next_level.size() - 1);
				}

				this_tree[index].split_value = pNode->split_value;
				this_tree[index].split_index = pNode->split_variable_index;


				index++;
			}

			nodes_this_level.swap(nodes_next_level);
			nodes_next_level.clear();
		}

		assert(index == node_num);
	}



	return compactForest;
}

void ReleaseCompactForest(CompactForest** forest)
{
	const int ntrees = (*forest)->rfinfo.ntrees;

	delete[](*forest)->nodes_of_tree;

	for (int t = 0; t < ntrees; t++)
	{
		delete[](*forest)->trees[t];
	}

	delete[](*forest)->trees;
	delete (*forest);

	*forest = NULL;
}

// tree-firest
int ErrorOnTestSamples(float** data_test, const int* label_test, const int nTestSamplesNum, const CompactForest* forest, float& error_rate)
{
	const int class_num = forest->rfinfo.datainfo.classes_num;
	int* label_count = new int[nTestSamplesNum * class_num];
	memset(label_count, 0, sizeof(int) * nTestSamplesNum * class_num);

	int index = 0;

	for (int t = 0; t < forest->rfinfo.ntrees; t++)
	{
		const CompactTreeNode* const tree = forest->trees[t];

		for (int n = 0; n < nTestSamplesNum; n++)
		{
			index = 0;
			while (tree[index].split_index >= 0) {

				if (data_test[n][tree[index].split_index] <= tree[index].split_value)
				{
					index = tree[index].left_index;
				}
				else {
					index = tree[index].right_index;
				}
			}

			assert(tree[index].split_value >= 0 && tree[index].split_value < class_num);

			label_count[n * class_num + (int)(tree[index].split_value)]++;
		}
	}

	error_rate = 0;
	for (int n = 0; n < nTestSamplesNum; n++)
	{
		int max_class = label_count[n * class_num + 0];
		int pred_label = 0;
		for (int c = 1; c < class_num; c++)
		{
			if (label_count[n * class_num + c] > max_class)
			{
				max_class = label_count[n * class_num + c];
				pred_label = c;
			}
		}

		if (pred_label != label_test[n])
			error_rate += 1.f;

	}

	error_rate = error_rate / nTestSamplesNum;

	delete[] label_count;

	return 1;
}

// sample-first
int ErrorOnTestSamples2(float** data_test, const int* label_test, const int nTestSamplesNum, const CompactForest* forest, float& error_rate)
{
	const int class_num = forest->rfinfo.datainfo.classes_num;
	int* label_count = new int[class_num];

	int index = 0;
	error_rate = 0.f;

	for (int n = 0; n < nTestSamplesNum; n++)
	{
		memset(label_count, 0, sizeof(int) * class_num);

		for (int t = 0; t < forest->rfinfo.ntrees; t++)
		{
			const CompactTreeNode* const tree = forest->trees[t];
			index = 0;
			while (tree[index].split_index >= 0) {

				if (data_test[n][tree[index].split_index] <= tree[index].split_value)
				{
					index = tree[index].left_index;
				}
				else {
					index = tree[index].right_index;
				}
			}

			label_count[(int)(tree[index].split_value)]++;
		}

		int max_c = label_count[0];
		int pred_c = 0;
		for (int c = 1; c < class_num; c++)
		{
			if (label_count[c] > max_c)
			{
				max_c = label_count[c];
				pred_c = c;
			}
		}
		if (pred_c != label_test[n])
			error_rate += 1.0f;
	}

	error_rate = error_rate / nTestSamplesNum;

	delete[] label_count;

	return 1;
}