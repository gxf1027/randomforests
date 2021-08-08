#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <vector>
#include <deque>
#include <cassert>
#include <string>
#include <stdint.h>
using namespace std;

#include "UserInteraction2.h"
#include "SharedRoutines.h"

#include "tinyxml2/tinyxml2.h"
using namespace tinyxml2;

/*
#define CHECKISNULL(POINTER, CHDISPLAY, CHERROR_INFO) \
	if( (POINTER)==NULL ) { strcpy_s((CHDISPLAY), _countof((CHDISPLAY)), (CHERROR_INFO)); goto QUIT_XML; }
*/

#define CHECKISNULL(POINTER, CHDISPLAY, CHERROR_INFO) \
	if( (POINTER)==NULL ) { strcpy((CHDISPLAY), (CHERROR_INFO)); return -1; }


int ReadClassificationForestConfigFile2(const char *configXMLPath, RandomCForests_info &RF_info)
{
	int rv = 0;

	XMLDocument doc;
	doc.LoadFile(configXMLPath);
	// TODO check return 

	XMLElement* pElem = doc.FirstChildElement("RandomForestConfig");
	if (NULL == pElem)
		return -3;

	XMLElement* pConfigElem = pElem->FirstChildElement("MaxDepth");
	if (NULL == pConfigElem)
	{
		cout << "'MaxDepth' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.maxdepth);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'MaxDepth'" << endl;
		return -1;
	}


	pConfigElem = pElem->FirstChildElement("TreesNum");
	if (NULL == pConfigElem)
	{
		cout << "'TreesNum' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.ntrees);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'TreesNum'" << endl;
		return -1;
	}

	pConfigElem = pElem->FirstChildElement("SplitVariables");
	if (NULL == pConfigElem)
	{
		cout << "'SplitVariables' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.mvariables);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'SplitVariables'" << endl;
		return -1;
	}

	pConfigElem = pElem->FirstChildElement("MinSamplesSplit");
	if (NULL == pConfigElem)
	{
		cout << "'MinSamplesSplit' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.minsamplessplit);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'MinSamplesSplit'" << endl;
		return -1;
	}

	pConfigElem = pElem->FirstChildElement("Randomness");
	if (NULL == pConfigElem)
	{
		cout << "'Randomness' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.randomness);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'Randomness'" << endl;
		return -1;
	}

	return 1;
}

int ReadRegressionForestConfigFile2(const char *configXMLPath, RandomRForests_info &RF_info)
{
	int rv = 0;

	XMLDocument doc;
	doc.LoadFile(configXMLPath);

	XMLElement* pElem = doc.FirstChildElement("RandomForestConfig");
	if (0 == pElem)
	{
		return -3;
	}

	XMLElement* pConfigElem = pElem->FirstChildElement("MaxDepth");
	if (NULL == pConfigElem)
	{
		cout << "'MaxDepth' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.maxdepth);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'MaxDepth'" << endl;
		return -1;
	}

	pConfigElem = pElem->FirstChildElement("TreesNum");
	if (NULL == pConfigElem)
	{
		cout << "'TreesNum' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.ntrees);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'TreesNum'" << endl;
		return -1;
	}

	pConfigElem = pElem->FirstChildElement("SplitVariables");
	if (NULL == pConfigElem)
	{
		cout << "'SplitVariables' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.mvariables);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'SplitVariables'" << endl;
		return -1;
	}

	pConfigElem = pElem->FirstChildElement("MinSamplesSplit");
	if (NULL == pConfigElem)
	{
		cout << "'MinSamplesSplit' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.minsamplessplit);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'MinSamplesSplit'" << endl;
		return -1;
	}

	pConfigElem = pElem->FirstChildElement("Randomness");
	if (NULL == pConfigElem)
	{
		cout << "'Randomness' cann't be found." << endl;
		return -2;
	}
	rv = sscanf(pConfigElem->GetText(), "%d", &RF_info.randomness);
	if (0 == rv || EOF == rv)
	{
		cout << "exception with reading 'Randomness'" << endl;
		return -1;
	}

	RF_info.predictionModel = PredictionModel::constant;

	return 1;
}


int InitalClassificationDataMatrixFormFile2(const char *fileName, float **&data, int *&label, Dataset_info_C &data_info)
{
	int i, j;
	ifstream file(fileName, ifstream::in);
	if (file.fail())
	{
		file.close();
		return -2;
	}

	int N, M, C;
	/*char *extraInfo=new char[100];
	file>>extraInfo>>N;
	file>>extraInfo>>M;
	file>>extraInfo>>C;
	delete [] extraInfo;*/

	std::string line, subline;
	// @totoal_sample_num= N
	std::getline(file, line);
	subline = line.substr(line.find("=") + 1);
	subline.erase(0, subline.find_first_not_of(" "));
	subline.erase(subline.find_last_not_of(" ") + 1);
	try
	{
		N = stoi(subline);
	}
	catch (const std::exception& ex)
	{
		cout << "exception happened when read 'totoal_sample_num'" << endl;
		return -1;
	}

	// @variable_num= M
	std::getline(file, line);
	subline = line.substr(line.find("=") + 1);
	subline.erase(0, subline.find_first_not_of(" "));
	subline.erase(subline.find_last_not_of(" ") + 1);
	try
	{
		M = stoi(subline);
	}
	catch (const std::exception& ex)
	{
		cout << "exception happened when read 'variable_num'" << endl;
		return -1;
	}

	// @class_num= M
	std::getline(file, line);
	subline = line.substr(line.find("=") + 1);
	subline.erase(0, subline.find_first_not_of(" "));
	subline.erase(subline.find_last_not_of(" ") + 1);
	try
	{
		C = stoi(subline);
	}
	catch (const std::exception& ex)
	{
		cout << "exception happened when read 'class_num'" << endl;
		return -1;
	}


	if (N <= 0 || M <= 0 || C <= 0)
	{
		cout << "**********************************************************" << endl;
		cout << "Make sure that the File format is correct:" << endl;
		cout << "@totoal_sample_num= N" << endl;
		cout << "@variable_num= M" << endl;
		cout << "@class_num= C" << endl;
		cout << "label1   x11 x12......x1M" << endl;
		cout << "label2   x21 x22......x2M" << endl;
		cout << "... ...  ... ..." << endl;
		cout << "labelN   xN1 xN2......xNM" << endl;
		cout << "**********************************************************" << endl;
		return -1;
	}

	cout << "**********************************************************" << endl;
	cout << "Data set information:" << endl;
	cout << "Number of training samples =" << " " << N << endl;
	cout << "Number of variables =       " << " " << M << endl;
	cout << "Number of classes =         " << " " << C << endl;
	cout << "**********************************************************" << endl;

	data_info.samples_num = N;
	data_info.variables_num = M;
	data_info.classes_num = C;

	data = new float* [N];
	for (i = 0; i < N; i++)
		data[i] = new float[M];

	label = new int[N];
	float flabel;

	for (i = 0; i < N; i++)
	{
		file >> flabel;
		label[i] = (int)flabel;
		// 		if( label0 == -1 )
		// 			label[i] = -1;
		// 		else
		// 			label[i] = 1;

		for (j = 0; j < M; j++)
			file >> data[i][j]; // variable value for ith sample's (j+1)th variable
	}

	file.close();

	return 1;
}

int InitalRegressionDataMatrixFormFile2(const char *fileName, float **&data, float **&target, Dataset_info_R &data_info)
{
	int i, j;
	ifstream file(fileName, ifstream::in);
	if (file.fail())
	{
		file.close();
		return -2;
	}

	int N, M, K;
	/*char *extraInfo=new char[100];
	file>>extraInfo>>N;
	file>>extraInfo>>M;
	file>>extraInfo>>K;
	delete [] extraInfo;*/

	std::string line, subline;
	// @totoal_sample_num= N
	std::getline(file, line);
	subline = line.substr(line.find("=") + 1);
	subline.erase(0, subline.find_first_not_of(" "));
	subline.erase(subline.find_last_not_of(" ") + 1);
	try
	{
		N = stoi(subline);
	}
	catch (const std::exception& ex)
	{
		cout << "exception happened when read 'totoal_sample_num'" << endl;
		return -1;
	}

	// @variable_num_x= M
	std::getline(file, line);
	subline = line.substr(line.find("=") + 1);
	subline.erase(0, subline.find_first_not_of(" "));
	subline.erase(subline.find_last_not_of(" ") + 1);
	try
	{
		M = stoi(subline);
	}
	catch (const std::exception& ex)
	{
		cout << "exception happened when read 'variable_num_x'" << endl;
		return -1;
	}

	// @variable_num_y= K
	std::getline(file, line);
	subline = line.substr(line.find("=") + 1);
	subline.erase(0, subline.find_first_not_of(" "));
	subline.erase(subline.find_last_not_of(" ") + 1);
	try
	{
		K = stoi(subline);
	}
	catch (const std::exception& ex)
	{
		cout << "exception happened when read 'variable_num_y'" << endl;
		return -1;
	}

	if (N <= 0 || M <= 0 || K <= 0)
	{
		cout << "**********************************************************" << endl;
		cout << "Make sure that the File format is correct:" << endl;
		cout << "@totoal_sample_num= N" << endl;
		cout << "@variable_num_x= M" << endl;
		cout << "@variable_num_y= K" << endl;
		cout << "target11 target12......target1K || x11 x12......x1M" << endl;
		cout << "target21 target22......target2K || x21 x22......x2M" << endl;
		cout << "... ...  ... ..." << endl;
		cout << "targetN1 targetN2......targetNK || xN1 xN2......xNM" << endl;
		cout << "**********************************************************" << endl;
		return -1;
	}

	cout << "**********************************************************" << endl;
	cout << "Data set information:" << endl;
	cout << "Number of training samples =   " << " " << N << endl;
	cout << "Number of variables(x) =       " << " " << M << endl;
	cout << "Number of target(y) variables =" << " " << K << endl;
	cout << "**********************************************************" << endl;

	data_info.samples_num = N;
	data_info.variables_num_x = M;
	data_info.variables_num_y = K;

	data = new float* [N];
	target = new float* [N];
	for (i = 0; i < N; i++)
	{
		data[i] = new float[M];
		target[i] = new float[K];
	}

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < K; j++)
			file >> target[i][j];

		for (j = 0; j < M; j++)
			file >> data[i][j];
	}

	file.close();

	return 1;
}

/*
The overall hierarchy:
 <RandomForestModel>
	<Messages>
	<RF_model_parameters>
	<Train_data_info>
	<TREES>
		<Tree 1>
        <Tree 2>
        ... ...
        <Tree N>
*/

// Using deque (STL structure) to facilitate level traversal of a single tree
void SaveRandomClassificationForestModelToXML2(const char *pFilePath, LoquatCForest *loquatForest)
{
	RandomCForests_info *pRF_info = &(loquatForest->RFinfo);
	int NTrees = pRF_info->ntrees, tree_depth;
	int i;
	struct LoquatCTreeStruct *pTree = NULL;

	XMLDocument* doc = new XMLDocument();
	doc->LinkEndChild(doc->NewDeclaration()); 

	XMLElement *root_RF = doc->NewElement("RandomDecisionForestModel");
	doc->LinkEndChild(root_RF);

	// 2021-03-26
	time_t rawtime;
	tm* timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);
	//
	XMLElement *elem_msg = doc->NewElement("Messages");
	root_RF->LinkEndChild(elem_msg);
	XMLElement *elem_msg_author = doc->NewElement("Author");
	elem_msg_author->LinkEndChild(doc->NewText("Gu XF"));
	elem_msg->LinkEndChild(elem_msg_author);
	XMLElement *elem_msg_email = doc->NewElement("Email");
	elem_msg_email->LinkEndChild(doc->NewText("gxf1027@126.com"));
	elem_msg->LinkEndChild(elem_msg_email);
	// 2021-03-26
	XMLElement* elem_msg_createtime = doc->NewElement("CreateTime");
	elem_msg_createtime->LinkEndChild(doc->NewText(buffer));
	elem_msg->LinkEndChild(elem_msg_createtime);

	// Random Forests papermeters
	XMLElement *elem_RF_info = doc->NewElement("RF_model_parameters");
	root_RF->LinkEndChild(elem_RF_info);
	XMLElement *elem_ntrees = doc->NewElement("Trees_in_forest");
	elem_ntrees->SetAttribute("N", pRF_info->ntrees);
	elem_RF_info->LinkEndChild(elem_ntrees);
	XMLElement *elem_nvar = doc->NewElement("Variables_to_split");
	elem_nvar->SetAttribute("M", pRF_info->mvariables);
	elem_RF_info->LinkEndChild(elem_nvar);
	XMLElement *elem_maxdepth = doc->NewElement("MaxDepth");
	elem_maxdepth->SetAttribute("maxdepth",pRF_info->maxdepth);
	elem_RF_info->LinkEndChild(elem_maxdepth);
	XMLElement *elem_minsamplessplit = doc->NewElement("MinSamplesSplit");
	elem_minsamplessplit->SetAttribute("min_samples_split", pRF_info->minsamplessplit);
	elem_RF_info->LinkEndChild(elem_minsamplessplit);
	XMLElement *elem_randomness = doc->NewElement("Randomness");
	elem_randomness->SetAttribute("randomness", pRF_info->randomness);
	elem_RF_info->LinkEndChild(elem_randomness);


	// Train data information
	XMLElement *elem_train_data = doc->NewElement("Train_data_info");
	root_RF->LinkEndChild(elem_train_data);
	XMLElement *elem_samples = doc->NewElement("Samples");
	elem_samples->SetAttribute("samples_num", pRF_info->datainfo.samples_num);
	elem_train_data->LinkEndChild(elem_samples);
	XMLElement *elem_vars = doc->NewElement("Variables");
	elem_vars->SetAttribute("variables_num", pRF_info->datainfo.variables_num);
	elem_train_data->LinkEndChild(elem_vars);
	XMLElement *elem_class = doc->NewElement("Classes");
	elem_class->SetAttribute("classes_num", pRF_info->datainfo.classes_num);
	elem_train_data->LinkEndChild(elem_class);

	XMLElement *elem_trees = doc->NewElement("TREES");
	root_RF->LinkEndChild(elem_trees);

	deque<LoquatCTreeNode *> dqNodes;
	deque<int64_t> dqNodeIndex;
	vector<XMLElement *> xmldepths;
	vector<int > nodes_in_depth;
	LoquatCTreeNode *tmpNode = NULL; // 接受从dqNodes弹栈的斩首元素
	int64_t tmpIndex; // 用于保存节点在所在层次的序号
	
	for ( i=0; i<NTrees; i++ )
	{
		pTree = loquatForest->loquatTrees[i];
		tree_depth = pTree->depth; // 树的深度，根节点为0

		// 先写入i-th树的基本信息
		// XMLElement *elem_tree = new XMLElement("Tree");
		XMLElement *elem_tree = doc->NewElement("Tree");
		elem_tree->SetAttribute("tree_index", i);
		elem_tree->SetAttribute("depth", tree_depth); // 树的深度(从0开始,root=0)
		elem_tree->SetAttribute("leaf_nodes_num", pTree->leaf_node_num); // 叶子节点总个数
		elem_trees->LinkEndChild(elem_tree); // 将tree结构连接到XML结构上层trees

		dqNodes.clear();
		dqNodeIndex.clear();
		xmldepths.clear();
		nodes_in_depth.clear();

		// 首先将root节点放到队列中
		dqNodes.push_back(pTree->rootNode);
		dqNodeIndex.push_back(0L);
		
		// XMLElement *elem_depth = new XMLElement("Depth");
		XMLElement *elem_depth = doc->NewElement("Depth");
		elem_depth->SetAttribute("dIndex", 0);
		elem_tree->LinkEndChild(elem_depth);
		xmldepths.push_back(elem_depth);
		nodes_in_depth.push_back(0);
		//cout<<"tree: "<<i<<endl;
		while(!dqNodes.empty())
		{
			tmpNode = dqNodes.front(); // 从队列中弹出一个节点
			dqNodes.pop_front(); 
			tmpIndex = dqNodeIndex.front(); // 这个节点对应的序号(在它所在层次的序号)
			dqNodeIndex.pop_front(); 
			int current_depth = tmpNode->depth;
			if( current_depth == xmldepths.size() ) // 层次序号从0开始，相等表示当前弹出的节点是下一个层次的节点，要创建新的xml“depth”结构
			{
				// XMLElement *elem_depth = new XMLElement("Depth"); // 新建一个层次结构
				XMLElement *elem_depth = doc->NewElement("Depth"); // 新建一个层次结构
				elem_depth->SetAttribute("dIndex", current_depth);
				elem_tree->LinkEndChild(elem_depth); // 连接到xml的上层结构
				xmldepths.push_back(elem_depth); // 将这个层次的指针保存下来
				nodes_in_depth.push_back(0); // 新一个层次的nodes个数开始为0
			}
			// 将节点信息写入
			if( tmpNode->nodetype == enLeafNode )
			{
				// XMLElement *elem_node = new XMLElement("elem_node");
				XMLElement *elem_node = doc->NewElement("elem_node");
				elem_node->SetAttribute("type", 2);
				elem_node->SetAttribute("index", tmpIndex);
				elem_node->SetAttribute("split_variable_index", tmpNode->split_variable_index);
				elem_node->SetAttribute("split_value", tmpNode->split_value);
				elem_node->SetAttribute("leaf_node_label",tmpNode->leaf_node_label);
				elem_node->SetAttribute("leaf_confidence", tmpNode->leaf_confidence);
				if( tmpNode->class_distribution != NULL )
				{
					char class_attri_name[20];
					for( int c=0; c<loquatForest->RFinfo.datainfo.classes_num; c++)
					{
						memset(class_attri_name, 0, sizeof(char)*20);
						sprintf(class_attri_name, "class%d",c);
						elem_node->SetAttribute(class_attri_name, tmpNode->class_distribution[c]);
					}
				}
				xmldepths[current_depth]->LinkEndChild(elem_node);
				nodes_in_depth[current_depth]++;
			}else
			{
				// XMLElement *elem_node = new XMLElement("elem_node");
				XMLElement *elem_node = doc->NewElement("elem_node");
				if ( tmpNode->nodetype == enLinkNode )
					elem_node->SetAttribute("type",1);
				else if( tmpNode->nodetype == enRootNode )
					elem_node->SetAttribute("type",0);
				elem_node->SetAttribute("index", tmpIndex);
				elem_node->SetAttribute("split_variable_index",tmpNode->split_variable_index);
				elem_node->SetAttribute("split_value", tmpNode->split_value);
				xmldepths[current_depth]->LinkEndChild(elem_node);
				nodes_in_depth[current_depth]++;
			}

			if( tmpNode->nodetype != enLeafNode ) // 如果不是叶子节点(enLinkNode或者enRootNode)，就将子节点入队列
			{
				dqNodes.push_back(tmpNode->pSubNode[0]);
				dqNodeIndex.push_back(tmpIndex*2L);
				dqNodes.push_back(tmpNode->pSubNode[1]);
				dqNodeIndex.push_back(tmpIndex*2L+1L);
			}
		}

		assert( (pTree->depth+1) == xmldepths.size() );
		// saving work has been done for i-th tree, following is trivial work.
		for( int d = 0; d<=pTree->depth; d++ )
			xmldepths[d]->SetAttribute("Nodes", nodes_in_depth[d]); // set the number of nodes in each depth of i-th tree
	}

	doc->SaveFile( pFilePath ); 
	delete doc;
}

void SaveRandomRegressionForestModelToXML2(const char *pFilePath, LoquatRForest *loquatForest)
{
	RandomRForests_info *pRF_info = &(loquatForest->RFinfo);
	int NTrees = pRF_info->ntrees, tree_depth;
	int i;
	struct LoquatRTreeStruct *pTree = NULL;

	XMLDocument* doc = new XMLDocument();
	doc->LinkEndChild( doc->NewDeclaration() );

	XMLElement *root_RF = doc->NewElement("RandomRegressionForestModel");
	doc->LinkEndChild(root_RF);

	// 2021-03-26
	time_t rawtime;
	tm* timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);
	//
	XMLElement *elem_msg = doc->NewElement("Messages");
	root_RF->LinkEndChild(elem_msg);
	XMLElement *elem_msg_author = doc->NewElement("Author");
	elem_msg_author->LinkEndChild(doc->NewText("Gu XF"));
	elem_msg->LinkEndChild(elem_msg_author);
	XMLElement *elem_msg_email = doc->NewElement("Email");
	elem_msg_email->LinkEndChild(doc->NewText("gxf1027@126.com"));
	elem_msg->LinkEndChild(elem_msg_email);
	// 2021-03-26
	XMLElement* elem_msg_createtime = doc->NewElement("CreateTime");
	elem_msg_createtime->LinkEndChild(doc->NewText(buffer));
	elem_msg->LinkEndChild(elem_msg_createtime);

	// RF\B5\C4\D0\C5Ϣ
	XMLElement *elem_RF_info = doc->NewElement("RF_model_parameters");
	root_RF->LinkEndChild(elem_RF_info);
	XMLElement *elem_ntrees = doc->NewElement("Trees_in_forest");
	elem_ntrees->SetAttribute("N", pRF_info->ntrees);
	elem_RF_info->LinkEndChild(elem_ntrees);
	XMLElement *elem_nvar = doc->NewElement("Variables_to_split");
	elem_nvar->SetAttribute("M", pRF_info->mvariables);
	elem_RF_info->LinkEndChild(elem_nvar);
	XMLElement *elem_maxdepth = doc->NewElement("MaxDepth");
	elem_maxdepth->SetAttribute("maxdepth",pRF_info->maxdepth);
	elem_RF_info->LinkEndChild(elem_maxdepth);
	XMLElement *elem_minsamplessplit = doc->NewElement("MinSamplesSplit");
	elem_minsamplessplit->SetAttribute("min_samples_split", pRF_info->minsamplessplit);
	elem_RF_info->LinkEndChild(elem_minsamplessplit);
	XMLElement *elem_randomness = doc->NewElement("Randomness");
	elem_randomness->SetAttribute("randomness", pRF_info->randomness);
	elem_RF_info->LinkEndChild(elem_randomness);
	XMLElement *elem_predictionmodel = doc->NewElement("PredictionModel");
	elem_predictionmodel->SetAttribute("prediction_model", pRF_info->predictionModel);
	elem_RF_info->LinkEndChild(elem_predictionmodel);
	
	// ѵ\C1\B7\CA\FD\BEݵ\C4\D0\C5Ϣ
	XMLElement *elem_train_data = doc->NewElement("Train_data_info");
	root_RF->LinkEndChild(elem_train_data);
	XMLElement *elem_samples = doc->NewElement("Samples");
	elem_samples->SetAttribute("samples_num", pRF_info->datainfo.samples_num);
	elem_train_data->LinkEndChild(elem_samples);
	XMLElement *elem_vars_y = doc->NewElement("Variables_y");
	elem_vars_y->SetAttribute("variables_num", pRF_info->datainfo.variables_num_y);
	elem_train_data->LinkEndChild(elem_vars_y);
	XMLElement *elem_vars_x = doc->NewElement("Variables_x");
	elem_vars_x->SetAttribute("variables_num", pRF_info->datainfo.variables_num_x);
	elem_train_data->LinkEndChild(elem_vars_x);

	// Extra information for forest
	if( loquatForest->RFinfo.datainfo.variables_num_y > 1 && NULL != loquatForest->offset && NULL != loquatForest->scale)
	{
		ostringstream oss_sc, oss_os;

		XMLElement *extra_RF_info = doc->NewElement("RF_extra_information");
		root_RF->LinkEndChild(extra_RF_info);

		XMLElement *elem_scale = doc->NewElement("scale");
		oss_sc<<loquatForest->scale[0];
		for( int d=1; d<loquatForest->RFinfo.datainfo.variables_num_y; d++ )
			oss_sc<<"_"<<loquatForest->scale[d];
		elem_scale->SetAttribute("sc", oss_sc.str().c_str());
		extra_RF_info->LinkEndChild(elem_scale);

		XMLElement *elem_offset = doc->NewElement("offset");
		oss_os<<loquatForest->offset[0];
		for( int d=1; d<loquatForest->RFinfo.datainfo.variables_num_y; d++ )
			oss_os<<"_"<<loquatForest->offset[d];
		elem_offset->SetAttribute("os", oss_os.str().c_str());
		extra_RF_info->LinkEndChild(elem_offset);
	}

	XMLElement *elem_trees = doc->NewElement("TREES");
	root_RF->LinkEndChild(elem_trees);

	deque<LoquatRTreeNode*> dqNodes;
	deque<int64_t> dqNodeIndex;
	vector<XMLElement *> xmldepths;
	vector<int > nodes_in_depth;
	LoquatRTreeNode *tmpNode = NULL; // 接受从dqNodes弹栈的斩首元素
	int64_t tmpIndex; // 用于保存节点在所在层次的序号

	for ( i=0; i<NTrees; i++ )
	{
		pTree = loquatForest->loquatTrees[i];
		tree_depth = pTree->depth;

		XMLElement *elem_tree = doc->NewElement("Tree");
		elem_tree->SetAttribute("tree_index", i);
		elem_tree->SetAttribute("depth", tree_depth); // 树的深度(从0开始,root=0)
		elem_tree->SetAttribute("leaf_nodes_num", pTree->leaf_node_num); // 叶子节点总个数
		elem_trees->LinkEndChild(elem_tree);

		dqNodes.clear();
		dqNodeIndex.clear();
		xmldepths.clear();
		nodes_in_depth.clear();

		// 首先将root节点放到队列中
		dqNodes.push_back(pTree->rootNode);
		dqNodeIndex.push_back(0L);

		XMLElement *elem_depth = doc->NewElement("Depth");
		elem_depth->SetAttribute("dIndex", 0);
		elem_tree->LinkEndChild(elem_depth);
		xmldepths.push_back(elem_depth);
		nodes_in_depth.push_back(0);

		while(!dqNodes.empty())
		{
			tmpNode = dqNodes.front(); // 从队列中弹出一个节点
			dqNodes.pop_front(); 
			tmpIndex = dqNodeIndex.front(); // 这个节点对应的序号(在它所在层次的序号)
			dqNodeIndex.pop_front(); 
			int current_depth = tmpNode->depth;
			if( current_depth == xmldepths.size() ) // 层次序号从0开始，相等表示当前弹出的节点是下一个层次的节点，要创建新的xml“depth”结构
			{
				// XMLElement *elem_depth = new XMLElement("Depth"); // 新建一个层次结构
				XMLElement *elem_depth = doc->NewElement("Depth"); // 新建一个层次结构
				elem_depth->SetAttribute("dIndex", current_depth);
				elem_tree->LinkEndChild(elem_depth); // 连接到xml的上层结构
				xmldepths.push_back(elem_depth); // 将这个层次的指针保存下来
				nodes_in_depth.push_back(0); // 新一个层次的nodes个数开始为0
			}
			// 将节点信息写入
			if( tmpNode->nodetype == enLeafNode )
			{
				XMLElement *elem_node = doc->NewElement("elem_node");
				elem_node->SetAttribute("type", 2);
				elem_node->SetAttribute("index", tmpIndex);
				elem_node->SetAttribute("split_variable_index", tmpNode->split_variable_index);
				elem_node->SetAttribute("split_value", tmpNode->split_value);
				elem_node->SetAttribute("arriveRatio", tmpNode->pLeafNodeInfo->arrivedRatio);
				ostringstream oss;
				oss<<tmpNode->pLeafNodeInfo->MeanOfArrived[0];
				for(int d=1; d<tmpNode->pLeafNodeInfo->dimension; d++)
					oss<<"_"<<tmpNode->pLeafNodeInfo->MeanOfArrived[d];
				elem_node->SetAttribute("MeanOfArrived", oss.str().c_str());
				//elem_depth->LinkEndChild(elem_node);
				// 0526
				if (NULL != tmpNode->pLeafNodeInfo->linearPredictor)
				{
					// variable_num_x+1, variable_num_y
					ostringstream osslm;
					const double* pLinearModel = tmpNode->pLeafNodeInfo->linearPredictor;
					const int limodel_param = (loquatForest->RFinfo.datainfo.variables_num_x + 1) * loquatForest->RFinfo.datainfo.variables_num_y;
					osslm << pLinearModel[0];
					for (int d = 1; d < limodel_param; d++)
						osslm << "_" << pLinearModel[d];
					elem_node->SetAttribute("LinearModel", osslm.str().c_str());
				}

				xmldepths[current_depth]->LinkEndChild(elem_node);
				nodes_in_depth[current_depth]++;
			}else
			{
				XMLElement *elem_node = doc->NewElement("elem_node");
				if ( tmpNode->nodetype == enLinkNode )
					elem_node->SetAttribute("type",1);
				else if( tmpNode->nodetype == enRootNode )
					elem_node->SetAttribute("type",0);
				elem_node->SetAttribute("index", tmpIndex);
				elem_node->SetAttribute("split_variable_index", tmpNode->split_variable_index);
				elem_node->SetAttribute("split_value", tmpNode->split_value);
				//elem_depth->LinkEndChild(elem_node);
				xmldepths[current_depth]->LinkEndChild(elem_node);
				nodes_in_depth[current_depth]++;
			}

			if( tmpNode->nodetype != enLeafNode ) // 如果不是叶子节点(enLinkNode或者enRootNode)，就将子节点入队列
			{
				dqNodes.push_back(tmpNode->pSubNode[0]);
				dqNodeIndex.push_back(tmpIndex*2L);
				dqNodes.push_back(tmpNode->pSubNode[1]);
				dqNodeIndex.push_back(tmpIndex*2L+1L);
			}
		}

		assert( (pTree->depth+1) == xmldepths.size() );
		// saving work has been done for i-th tree, following is trivial work.
		for( int d = 0; d<=pTree->depth; d++ )
			xmldepths[d]->SetAttribute("Nodes", nodes_in_depth[d]); // set the number of nodes in each depth of i-th tree
	}
	
	doc->SaveFile( pFilePath ); 
	delete doc;
}

//void SaveRandomRForestModelToXMLFileObsolete(char *pFilePath, LoquatRForest *loquatForest)
//{
//	RandomRForests_info *pRF_info = &(loquatForest->RFinfo);
//	int NTrees = pRF_info->ntrees, tree_depth;
//	int i,j, k, maxNodeNumThisDepth, nodes_num=0;
//	struct LoquatRTreeStruct *pTree = NULL;
//	struct LoquatRTreeNode **pNode = NULL, **pNextNode = NULL;
//
//	XMLDocument* doc = new XMLDocument();
//	doc->LinkEndChild( doc->NewDeclaration() );
//
//	XMLElement *root_RF = doc->NewElement("RandomRegressionForestModel");
//	doc->LinkEndChild(root_RF);
//
//	//
//	XMLElement *elem_msg = doc->NewElement("Messages");
//	root_RF->LinkEndChild(elem_msg);
//	XMLElement *elem_msg_author = doc->NewElement("Author");
//	elem_msg_author->LinkEndChild(doc->NewText("Gu Xingfang"));
//	elem_msg->LinkEndChild(elem_msg_author);
//	XMLElement *elem_msg_email = doc->NewElement("Email");
//	elem_msg_email->LinkEndChild(doc->NewText("gxf1027@gmail.com"));
//	elem_msg->LinkEndChild(elem_msg_email);
//
//	// RF\B5\C4\D0\C5Ϣ
//	XMLElement *elem_RF_info = doc->NewElement("RF_model_parameters");
//	root_RF->LinkEndChild(elem_RF_info);
//	XMLElement *elem_ntrees = doc->NewElement("Trees_in_forest");
//	elem_ntrees->SetAttribute("N", pRF_info->ntrees);
//	elem_RF_info->LinkEndChild(elem_ntrees);
//	XMLElement *elem_nvar = doc->NewElement("Variables_to_split");
//	elem_nvar->SetAttribute("M", pRF_info->mvariables);
//	elem_RF_info->LinkEndChild(elem_nvar);
//	XMLElement *elem_maxdepth = doc->NewElement("MaxDepth");
//	elem_maxdepth->SetAttribute("maxdepth",pRF_info->maxdepth);
//	elem_RF_info->LinkEndChild(elem_maxdepth);
//	XMLElement *elem_selratio = doc->NewElement("InBagRate");
//	elem_selratio->SetAttribute("inbagrate", (double)(pRF_info->selratio));
//	elem_RF_info->LinkEndChild(elem_selratio);
//	
//	// ѵ\C1\B7\CA\FD\BEݵ\C4\D0\C5Ϣ
//	XMLElement *elem_train_data = doc->NewElement("Train_data_info");
//	root_RF->LinkEndChild(elem_train_data);
//	XMLElement *elem_samples = doc->NewElement("Samples");
//	elem_samples->SetAttribute("samples_num", pRF_info->datainfo.samples_num);
//	elem_train_data->LinkEndChild(elem_samples);
//	XMLElement *elem_vars_y = doc->NewElement("Variables_y");
//	elem_vars_y->SetAttribute("variables_num", pRF_info->datainfo.variables_num_y);
//	elem_train_data->LinkEndChild(elem_vars_y);
//	XMLElement *elem_vars_x = doc->NewElement("Variables_x");
//	elem_vars_x->SetAttribute("variables_num", pRF_info->datainfo.variables_num_x);
//	elem_train_data->LinkEndChild(elem_vars_x);
//
//	// Extra information for forest
//	if( loquatForest->RFinfo.datainfo.variables_num_y > 1 )
//	{
//		ostringstream oss_sc, oss_os;
//
//		XMLElement *extra_RF_info = doc->NewElement("RF_extra_information");
//		root_RF->LinkEndChild(extra_RF_info);
//
//		XMLElement *elem_scale = doc->NewElement("scale");
//		oss_sc<<loquatForest->scale[0];
//		for( int d=1; d<loquatForest->RFinfo.datainfo.variables_num_y; d++ )
//			oss_sc<<"_"<<loquatForest->scale[d];
//		elem_scale->SetAttribute("sc", oss_sc.str().c_str());
//		extra_RF_info->LinkEndChild(elem_scale);
//
//		XMLElement *elem_offset = doc->NewElement("offset");
//		oss_os<<loquatForest->offset[0];
//		for( int d=1; d<loquatForest->RFinfo.datainfo.variables_num_y; d++ )
//			oss_os<<"_"<<loquatForest->offset[d];
//		elem_offset->SetAttribute("os", oss_os.str().c_str());
//		extra_RF_info->LinkEndChild(elem_offset);
//	}
//
//	XMLElement *elem_trees = doc->NewElement("TREES");
//	root_RF->LinkEndChild(elem_trees);
//	for ( i=0; i<NTrees; i++ )
//	{
//		pTree = loquatForest->loquatTrees[i];
//		tree_depth = pTree->depth;
//
//		XMLElement *elem_tree = doc->NewElement("Tree");
//		elem_tree->SetAttribute("tree_index", i);
//		elem_trees->LinkEndChild(elem_tree);
//
//		if( pNode != NULL )
//			delete [] pNode;
//		if( pNextNode != NULL )
//			delete [] pNextNode;
//
//		pNode = new struct LoquatRTreeNode*[1];
//		pNode[0] = pTree->rootNode; 
//		pNextNode = NULL;
//
//		for ( j=0; j<=tree_depth; j++ )
//		{
//			nodes_num = 0;
//
//			maxNodeNumThisDepth = (int)powf(2.f, (float)j);
//			pNextNode = new struct LoquatRTreeNode *[maxNodeNumThisDepth*2];
//
//			XMLElement *elem_depth = doc->NewElement("Depth");
//			elem_depth->SetAttribute("dIndex", j);
//			elem_tree->LinkEndChild(elem_depth);
//
//			for ( k=0; k<maxNodeNumThisDepth; k++ )
//			{
//				if( pNode[k]==NULL )
//				{
//					pNextNode[k*2]   = NULL;
//					pNextNode[k*2+1] = NULL;
//					continue;
//				}
//
//				if( pNode[k]->nodetype == enLeafNode )
//				{
//					XMLElement *elem_node = doc->NewElement("elem_node");
//					elem_node->SetAttribute("type", 2);
//					elem_node->SetAttribute("index", k);
//					elem_node->SetAttribute("split_variable_index", pNode[k]->split_variable_index);
//					elem_node->SetAttribute("split_value", (double)(pNode[k]->split_value));
//					elem_node->SetAttribute("arriveRatio", (double)(pNode[k]->pLeafNodeInfo->arrivedRatio));
//					ostringstream oss;
//					oss<<pNode[k]->pLeafNodeInfo->MeanOfArrived[0];
//					for(int d=1; d<pNode[k]->pLeafNodeInfo->dimension; d++)
//						oss<<"_"<<pNode[k]->pLeafNodeInfo->MeanOfArrived[d];
//					elem_node->SetAttribute("MeanOfArrived", oss.str().c_str());
//					elem_depth->LinkEndChild(elem_node);
//					nodes_num++;
//					pNextNode[k*2]   = NULL;
//					pNextNode[k*2+1] = NULL;
//				}else
//				{
//					XMLElement *elem_node = doc->NewElement("elem_node");
//					if ( pNode[k]->nodetype == enLinkNode )
//						elem_node->SetAttribute("type",1);
//					else if( pNode[k]->nodetype == enRootNode )
//						elem_node->SetAttribute("type",0);
//					elem_node->SetAttribute("index", k);
//					elem_node->SetAttribute("split_variable_index", pNode[k]->split_variable_index);
//					elem_node->SetAttribute("split_value", double(pNode[k]->split_value));
//					elem_depth->LinkEndChild(elem_node);
//					nodes_num++;
//					pNextNode[k*2]   = pNode[k]->pSubNode[0];
//					pNextNode[k*2+1] = pNode[k]->pSubNode[1];
//				}
//			}
//
//			elem_depth->SetAttribute("Nodes", nodes_num);
//
//			delete [] pNode;
//			pNode = new struct LoquatRTreeNode *[maxNodeNumThisDepth*2];
//			for ( k=0; k<maxNodeNumThisDepth*2; k++ )
//				pNode[k] = pNextNode[k];
//			delete [] pNextNode;
//			pNextNode = NULL;
//		}
//
//		elem_tree->SetAttribute("depth", tree_depth);
//		elem_tree->SetAttribute("leaf_nodes_num", pTree->leaf_node_num);
//	}
//
//	if( pNextNode != NULL )
//		delete [] pNextNode;
//	if( pNode != NULL )
//		delete [] pNode;
//
//	doc->SaveFile( pFilePath ); 
//	delete doc;
//}


int BuildRandomClassificationForestModelFromXML2(const char* pFilePath, LoquatCForest*& loquatForest)
{
	if (loquatForest != NULL)
		return -3;

	loquatForest = new LoquatCForest;
	loquatForest->loquatTrees = NULL;
	assert(loquatForest != NULL);

	XMLDocument doc;
	int loadOkay = doc.LoadFile(pFilePath);
	if (XML_SUCCESS != loadOkay)
	{
		cout << "-----------------  ERROR:'BuildDecisionForestModelFromXMLFile' ------------------" << endl;
		cout << "Failed to load file " << pFilePath << endl;
		cout << "------------------------------------------------------------------------------------------------------" << endl;
		return -2;
	}

	char error_disp[100];
	int i, j, k, Ntrees, tree_depth, leaf_nodes_num, maxNodesThisDepth = 0;

	XMLElement* pRootElem = doc.FirstChildElement();
	XMLElement* pElem = pRootElem->FirstChildElement(); // <Messages>
	XMLElement* elem_RF_model = pRootElem->FirstChildElement("RF_model_parameters"); // <RF_model_parameters>
	CHECKISNULL(elem_RF_model, error_disp, "RF_model_parameters");

	XMLElement* elem_RF_info = elem_RF_model->FirstChildElement("Trees_in_forest");
	CHECKISNULL(elem_RF_info, error_disp, "Trees_in_forest");
	Ntrees = elem_RF_info->FirstAttribute()->IntValue(); // trees
	loquatForest->RFinfo.ntrees = Ntrees;

	elem_RF_info = elem_RF_model->FirstChildElement("Variables_to_split");
	CHECKISNULL(elem_RF_info, error_disp, "Variables_to_split");
	loquatForest->RFinfo.mvariables = elem_RF_info->FirstAttribute()->IntValue();

	elem_RF_info = elem_RF_model->FirstChildElement("MaxDepth");
	CHECKISNULL(elem_RF_info, error_disp, "MaxDepth");
	loquatForest->RFinfo.maxdepth = elem_RF_info->FirstAttribute()->IntValue();

	elem_RF_info = elem_RF_model->FirstChildElement("MinSamplesSplit");
	CHECKISNULL(elem_RF_info, error_disp, "MinSamplesSplit");
	loquatForest->RFinfo.minsamplessplit = elem_RF_info->FirstAttribute()->IntValue();

	elem_RF_info = elem_RF_model->FirstChildElement("Randomness");
	CHECKISNULL(elem_RF_info, error_disp, "Randomness");
	loquatForest->RFinfo.randomness = elem_RF_info->FirstAttribute()->IntValue();

	loquatForest->loquatTrees = new struct LoquatCTreeStruct* [Ntrees];
	assert(loquatForest->loquatTrees != NULL);
	for (i = 0; i < Ntrees; i++)
	{
		loquatForest->loquatTrees[i] = NULL;
	}

	XMLElement* elem_Train_data_info = pRootElem->FirstChildElement("Train_data_info");
	CHECKISNULL(elem_Train_data_info, error_disp, "Train_data_info");

	XMLElement* elem_data_info = elem_Train_data_info->FirstChildElement("Samples");
	CHECKISNULL(elem_data_info, error_disp, "Samples");
	loquatForest->RFinfo.datainfo.samples_num = elem_data_info->FirstAttribute()->IntValue(); // samples_num

	elem_data_info = elem_Train_data_info->FirstChildElement("Variables");
	CHECKISNULL(elem_data_info, error_disp, "Variables");
	loquatForest->RFinfo.datainfo.variables_num = elem_data_info->FirstAttribute()->IntValue(); // variables_num

	elem_data_info = elem_Train_data_info->FirstChildElement("Classes");
	CHECKISNULL(elem_data_info, error_disp, "Classes");
	loquatForest->RFinfo.datainfo.classes_num = elem_data_info->FirstAttribute()->IntValue(); // classes_num

	XMLElement* elem_TREES = pRootElem->FirstChildElement("TREES");
	CHECKISNULL(elem_TREES, error_disp, "TREES");

	XMLElement* elem_Tree = elem_TREES->FirstChildElement(); // 指向第一棵树

	for (i = 0; i < Ntrees; i++)
	{
		loquatForest->loquatTrees[i] = new struct LoquatCTreeStruct;
		assert(loquatForest->loquatTrees[i] != NULL);

		// information for i-th tree
		tree_depth = elem_Tree->FirstAttribute()->Next()->IntValue();
		leaf_nodes_num = elem_Tree->FirstAttribute()->Next()->Next()->IntValue();
		loquatForest->loquatTrees[i]->depth = tree_depth;
		loquatForest->loquatTrees[i]->leaf_node_num = leaf_nodes_num;
		loquatForest->loquatTrees[i]->inbag_samples_index = NULL;
		loquatForest->loquatTrees[i]->inbag_samples_num = 0;
		loquatForest->loquatTrees[i]->outofbag_samples_index = NULL;
		loquatForest->loquatTrees[i]->outofbag_samples_num = 0;
		// loquatForest->loquatTrees[i]->rootNode 

		vector<LoquatCTreeNode*> nodes_above_level; // 上一层的节点指针!
		XMLElement* elem_Depth = elem_Tree->FirstChildElement(); // 指向第0层所在的xml结构

		// 新建根节点
		loquatForest->loquatTrees[i]->rootNode = new struct LoquatCTreeNode;
		nodes_above_level.push_back(loquatForest->loquatTrees[i]->rootNode); // 指针放入栈中
		// 从xml中获得节点信息
		XMLElement* elem_node = elem_Depth->FirstChildElement(); // 指向根节点所在的xml结构
		const XMLAttribute* attribute = elem_node->FirstAttribute();
		int type = attribute->IntValue();		attribute = attribute->Next();
		int index = attribute->IntValue(); 		attribute = attribute->Next();
		int split_index = attribute->IntValue();		attribute = attribute->Next();
		float split_value = attribute->FloatValue();

		// int type = elem_node->FindAttribute("type")->IntValue();
		// int index = elem_node->FindAttribute("index")->IntValue(); 
		// int split_index = elem_node->FindAttribute("split_variable_index")->IntValue();
		// float split_value = elem_node->FindAttribute("split_value")->FloatValue();
		// 将节点信息储存到实际树节点中
		LoquatCTreeNode* root = loquatForest->loquatTrees[i]->rootNode;
		root->depth = 0;
		root->split_variable_index = split_index;
		root->split_value = split_value;
		root->samples_index = NULL;
		root->arrival_samples_num = -1;
		root->train_impurity = 0.f;
		root->class_distribution = NULL;
		switch (type)
		{
		case enRootNode:
			root->nodetype = enRootNode;
			// 0527
			root->subnodes_num = 2;
			root->pSubNode = new struct LoquatCTreeNode* [2];
			root->pSubNode[0] = NULL;
			root->pSubNode[1] = NULL;
			root->leaf_node_label = -1;
			root->leaf_confidence = 0.f;
			root->class_distribution = NULL;
			root->pParentNode = NULL;
			break;
		case enLinkNode:
			root->nodetype = enLinkNode;
			root->subnodes_num = 2;
			root->pSubNode = new struct LoquatCTreeNode* [2];
			root->pSubNode[0] = NULL;
			root->pSubNode[1] = NULL;
			root->leaf_node_label = -1;
			root->leaf_confidence = 0.f;
			root->class_distribution = NULL; // need to be tested
			root->pParentNode = NULL;
			break;
		case enLeafNode:
			root->nodetype = enLeafNode;
			root->subnodes_num = 0;
			root->pSubNode = new struct LoquatCTreeNode* [2];
			root->pSubNode[0] = NULL;
			root->pSubNode[1] = NULL; // 叶子节点没有子节点
			attribute = attribute->Next();
			root->leaf_node_label = attribute->IntValue();
			attribute = attribute->Next();
			root->leaf_confidence = attribute->FloatValue();
			// need to be tested
			if (NULL != attribute->Next())
			{
				attribute = attribute->Next(); // class_distribution
				root->class_distribution = new float[loquatForest->RFinfo.datainfo.classes_num];
				memset(root->class_distribution, 0, sizeof(float) * loquatForest->RFinfo.datainfo.classes_num);
				for (int c = 0; c < loquatForest->RFinfo.datainfo.classes_num; c++)
				{
					if (attribute == NULL)
					{
						cout << "exception: class distribution unfound in leaf node." << endl;
						root->class_distribution[c] = 0.f;
					}
					else
						root->class_distribution[c] = attribute->FloatValue();
					attribute = attribute->Next();
				}
			}
			root->pParentNode = NULL;
			break;
		default:
			assert(0);
		}


		for (j = 1; j <= tree_depth; j++) // 从序号为1的层次开始(即根节点下面第一层)
		{
			elem_Depth = elem_Depth->NextSiblingElement(); // 指向下一层
			//int nodes_num_acctual = elem_Depth->FirstAttribute()->Next()->IntValue(); // 该层次的节点个数
			int nodes_num_acctual = elem_Depth->FindAttribute("Nodes")->IntValue(); // 该层次的节点个数
			XMLElement* elem_node = elem_Depth->FirstChildElement(); //
			int pp = 0;
			vector<LoquatCTreeNode* > nodes_current_level;
			for (k = 0; k < nodes_num_acctual; k++)
			{
				int type = elem_node->FindAttribute("type")->IntValue();
				int index = elem_node->FindAttribute("index")->IntValue();
				int split_index = elem_node->FindAttribute("split_variable_index")->IntValue();
				float split_value = elem_node->FindAttribute("split_value")->FloatValue();

				LoquatCTreeNode* tmpNode = new LoquatCTreeNode; // 新建一个节点
				nodes_current_level.push_back(tmpNode); // 放入当前层次的节点栈
				tmpNode->depth = j;
				tmpNode->split_variable_index = split_index;
				tmpNode->split_value = split_value;
				tmpNode->samples_index = NULL;
				tmpNode->arrival_samples_num = -1;
				tmpNode->train_impurity = 0.f;
				tmpNode->class_distribution = NULL;

				switch (type)
				{
				case enRootNode: // 不可能再是root了
					assert(0);
				case enLinkNode:
					tmpNode->nodetype = enLinkNode;
					tmpNode->subnodes_num = 2;
					tmpNode->pSubNode = new struct LoquatCTreeNode* [2];
					tmpNode->pSubNode[0] = NULL;
					tmpNode->pSubNode[1] = NULL;
					tmpNode->leaf_node_label = -1;
					tmpNode->leaf_confidence = 0.f;
					tmpNode->class_distribution = NULL; // need to be tested
					while (1)
					{
						if (nodes_above_level[pp]->nodetype != enLeafNode) // 是一个连接节点
						{
							if (nodes_above_level[pp]->pSubNode[0] == NULL)
							{
								nodes_above_level[pp]->pSubNode[0] = tmpNode;
								tmpNode->pParentNode = nodes_above_level[pp];
								break;
							}
							else if (nodes_above_level[pp]->pSubNode[1] == NULL)
							{
								nodes_above_level[pp]->pSubNode[1] = tmpNode;
								tmpNode->pParentNode = nodes_above_level[pp];
								pp++; // 这个上层节点的两个子节点都已经连接了!
								break;
							}
							assert(0); // 不可能到这里
						}
						else
							pp++;
					}
					break;
				case enLeafNode:
				{ // start of case enLeafNode
					tmpNode->nodetype = enLeafNode;
					tmpNode->subnodes_num = 0;
					tmpNode->pSubNode = new struct LoquatCTreeNode* [2]; //0527
					tmpNode->pSubNode[0] = NULL; //0527
					tmpNode->pSubNode[1] = NULL; //0527
					tmpNode->leaf_node_label = elem_node->FindAttribute("leaf_node_label")->IntValue();
					tmpNode->leaf_confidence = elem_node->FindAttribute("leaf_confidence")->FloatValue();
					const XMLAttribute* attribute = elem_node->FindAttribute("leaf_confidence");
					// need to be tested
					if (NULL != attribute->Next())
					{
						attribute = attribute->Next(); // class_distribution
						tmpNode->class_distribution = new float[loquatForest->RFinfo.datainfo.classes_num];
						memset(tmpNode->class_distribution, 0, sizeof(float) * loquatForest->RFinfo.datainfo.classes_num);
						for (int c = 0; c < loquatForest->RFinfo.datainfo.classes_num; c++)
						{
							if (attribute == NULL)
							{
								tmpNode->class_distribution[c] = 0.f;
								cout << "exception: class distribution unfound in leaf node." << endl;
								break; // 20210524
							}
							else
								tmpNode->class_distribution[c] = attribute->FloatValue();
							attribute = attribute->Next();
						}
					}
					while (1)
					{
						if (nodes_above_level[pp]->nodetype != enLeafNode) // 是一个连接节点
						{
							if (nodes_above_level[pp]->pSubNode[0] == NULL)
							{
								nodes_above_level[pp]->pSubNode[0] = tmpNode;
								tmpNode->pParentNode = nodes_above_level[pp];
								break;
							}
							else if (nodes_above_level[pp]->pSubNode[1] == NULL)
							{
								nodes_above_level[pp]->pSubNode[1] = tmpNode;
								tmpNode->pParentNode = nodes_above_level[pp];
								pp++; // 这个上层节点的两个子节点都已经连接了!
								break;
							}
							assert(0); // 不可能到这里
						}
						else
							pp++;
					}

				} // end of case enLeafNode
				break;
				default:
					assert(0);
				}

				elem_node = elem_node->NextSiblingElement(); // !!!
			} // for( k=0; k<nodes_num_acctual; k++ ) 

			nodes_above_level = nodes_current_level;
		}

		elem_Tree = elem_Tree->NextSiblingElement(); // 指向下一个xml结构表示的tree
	}

	return 1;

QUIT_XML:
	cout << "-----------------  ERROR:'BuildDecisionForestModelFromXMLFile' ------------------" << endl;
	cout << "'" << error_disp << "' can not be found." << endl;
	cout << "----------------------------------------------------------------------" << endl;
	if (loquatForest->loquatTrees != NULL)
	{
		delete[] loquatForest->loquatTrees;
		loquatForest->loquatTrees = NULL;
	}
	delete loquatForest;
	loquatForest = NULL;
	return -1;
}


int BuildRandomRegressionForestModelFromXML2(const char* pFilePath, LoquatRForest*& loquatForest)
{
	if (loquatForest != NULL)
		return -3;

	loquatForest = new LoquatRForest;
	loquatForest->loquatTrees = NULL; // \B7\C0ֹҰָ\D5\EB
	assert(loquatForest != NULL);

	XMLDocument doc;
	XMLError loadOkay = doc.LoadFile(pFilePath);
	if (loadOkay != XML_SUCCESS)
	{
		cout << "-----------------  ERROR:'BuildRegressionForestModelFromXMLFile' ------------------" << endl;
		cout << "Failed to load file " << pFilePath << endl;
		cout << "----------------------------------------------------------------------" << endl;
		return -2;
	}

	char error_disp[100];
	int i, j, k, Ntrees, tree_depth, leaf_nodes_num, maxNodesThisDepth = 0;

	XMLElement* pRootElem = doc.FirstChildElement();
	XMLElement* pElem = pRootElem->FirstChildElement(); // <Messages>
	XMLElement* elem_RF_model = pRootElem->FirstChildElement("RF_model_parameters"); // <RF_model_parameters>
	CHECKISNULL(elem_RF_model, error_disp, "RF_model_papeameters");

	XMLElement* elem_RF_info = elem_RF_model->FirstChildElement("Trees_in_forest");
	CHECKISNULL(elem_RF_info, error_disp, "Trees_in_forest");
	Ntrees = elem_RF_info->FirstAttribute()->IntValue(); // trees
	loquatForest->RFinfo.ntrees = Ntrees;

	elem_RF_info = elem_RF_model->FirstChildElement("Variables_to_split");
	CHECKISNULL(elem_RF_info, error_disp, "Variables_to_split");
	loquatForest->RFinfo.mvariables = elem_RF_info->FirstAttribute()->IntValue();

	elem_RF_info = elem_RF_model->FirstChildElement("MaxDepth");
	CHECKISNULL(elem_RF_info, error_disp, "MaxDepth");
	loquatForest->RFinfo.maxdepth = elem_RF_info->FirstAttribute()->IntValue();

	elem_RF_info = elem_RF_model->FirstChildElement("MinSamplesSplit");
	CHECKISNULL(elem_RF_info, error_disp, "MinSamplesSplit");
	loquatForest->RFinfo.minsamplessplit = elem_RF_info->FirstAttribute()->IntValue();

	elem_RF_info = elem_RF_model->FirstChildElement("Randomness");
	CHECKISNULL(elem_RF_info, error_disp, "Randomness");
	loquatForest->RFinfo.randomness = elem_RF_info->FirstAttribute()->IntValue();

	elem_RF_info = elem_RF_model->FirstChildElement("PredictionModel");
	CHECKISNULL(elem_RF_info, error_disp, "PredictionModel");
	loquatForest->RFinfo.predictionModel = (PredictionModel)(elem_RF_info->FirstAttribute()->IntValue());
	

	loquatForest->loquatTrees = new struct LoquatRTreeStruct* [Ntrees];
	assert(loquatForest->loquatTrees != NULL);
	for (i = 0; i < Ntrees; i++)
	{
		loquatForest->loquatTrees[i] = NULL;
	}

	XMLElement* elem_Train_data_info = pRootElem->FirstChildElement("Train_data_info");
	CHECKISNULL(elem_Train_data_info, error_disp, "Train_data_info");

	XMLElement* elem_data_info = elem_Train_data_info->FirstChildElement("Samples");
	CHECKISNULL(elem_data_info, error_disp, "Samples");
	loquatForest->RFinfo.datainfo.samples_num = elem_data_info->FirstAttribute()->IntValue(); // samples_num

	elem_data_info = elem_Train_data_info->FirstChildElement("Variables_y");
	CHECKISNULL(elem_data_info, error_disp, "Variables_y");
	loquatForest->RFinfo.datainfo.variables_num_y = elem_data_info->FirstAttribute()->IntValue(); // variables_y

	elem_data_info = elem_Train_data_info->FirstChildElement("Variables_x");
	CHECKISNULL(elem_data_info, error_disp, "Variables_x");
	loquatForest->RFinfo.datainfo.variables_num_x = elem_data_info->FirstAttribute()->IntValue(); // variables_x

	// optional: extra_information
	loquatForest->scale = NULL;
	loquatForest->offset = NULL;
	loquatForest->bTargetNormalize = false;
	if ( /*loquatForest->RFinfo.datainfo.variables_num_y > 1 && */ NULL != pRootElem->FirstChildElement("RF_extra_information"))
	{
		int d;
		char ch;
		loquatForest->scale = new float[loquatForest->RFinfo.datainfo.variables_num_y];
		loquatForest->offset = new float[loquatForest->RFinfo.datainfo.variables_num_y];

		XMLElement* elem_extra_info = pRootElem->FirstChildElement("RF_extra_information");
		CHECKISNULL(elem_extra_info, error_disp, "Train_data_info");

		XMLElement* elem_info = elem_extra_info->FirstChildElement("scale");
		CHECKISNULL(elem_info, error_disp, "scale");
		stringstream iss_sc(elem_info->FirstAttribute()->Value());
		iss_sc >> loquatForest->scale[0];
		for (d = 1; d < loquatForest->RFinfo.datainfo.variables_num_y; d++)
			iss_sc >> ch >> loquatForest->scale[d];

		elem_info = elem_extra_info->FirstChildElement("offset");
		CHECKISNULL(elem_info, error_disp, "offset");
		stringstream iss_os(elem_info->FirstAttribute()->Value());
		iss_os >> loquatForest->offset[0];
		for (d = 1; d < loquatForest->RFinfo.datainfo.variables_num_y; d++)
			iss_os >> ch >> loquatForest->offset[d];
	}

	XMLElement* elem_TREES = pRootElem->FirstChildElement("TREES");
	CHECKISNULL(elem_TREES, error_disp, "TREES");

	XMLElement* elem_Tree = elem_TREES->FirstChildElement(); // 指向第一棵树

	for (i = 0; i < Ntrees; i++)
	{
		loquatForest->loquatTrees[i] = new struct LoquatRTreeStruct;
		assert(loquatForest->loquatTrees[i] != NULL);

		tree_depth = elem_Tree->FirstAttribute()->Next()->IntValue();
		leaf_nodes_num = elem_Tree->FirstAttribute()->Next()->Next()->IntValue();
		loquatForest->loquatTrees[i]->depth = tree_depth;
		loquatForest->loquatTrees[i]->leaf_node_num = leaf_nodes_num;
		loquatForest->loquatTrees[i]->inbag_samples_index = NULL;
		loquatForest->loquatTrees[i]->inbag_samples_num = 0;
		loquatForest->loquatTrees[i]->outofbag_samples_index = NULL;
		loquatForest->loquatTrees[i]->outofbag_samples_num = 0;
		// loquatForest->loquatTrees[i]->rootNode 

		vector<LoquatRTreeNode*> nodes_above_level; // 上一层的节点指针!
		XMLElement* elem_Depth = elem_Tree->FirstChildElement(); // 指向第0层所在的xml结构

		// 新建根节点
		loquatForest->loquatTrees[i]->rootNode = new struct LoquatRTreeNode;
		nodes_above_level.push_back(loquatForest->loquatTrees[i]->rootNode); // 指针放入栈中
		// 从xml中获得节点信息
		XMLElement* elem_node = elem_Depth->FirstChildElement(); // 指向根节点所在的xml结构
		const XMLAttribute* attribute = elem_node->FirstAttribute();
		int type = attribute->IntValue();		attribute = attribute->Next();
		int index = attribute->IntValue(); 		attribute = attribute->Next();
		int split_index = attribute->IntValue();		attribute = attribute->Next();
		float split_value = attribute->FloatValue();

		LoquatRTreeNode* root = loquatForest->loquatTrees[i]->rootNode;
		root->depth = 0;
		root->split_variable_index = split_index;
		root->split_value = split_value;
		root->samples_index = NULL;
		root->arrival_samples_num = -1;
		root->train_impurity = 0.f;
		root->pLeafNodeInfo = NULL;
		switch (type)
		{
		case enRootNode:
			root->nodetype = enRootNode;
			// 0527
			root->subnodes_num = 2;
			root->pSubNode = new struct LoquatRTreeNode* [2];
			root->pSubNode[0] = NULL;
			root->pSubNode[1] = NULL;
			root->pLeafNodeInfo = NULL;
			root->pParentNode = NULL;
			break;
		case enLinkNode:
			root->nodetype = enLinkNode;
			root->subnodes_num = 2;
			root->pSubNode = new struct LoquatRTreeNode* [2];
			root->pSubNode[0] = NULL;
			root->pSubNode[1] = NULL;
			root->pLeafNodeInfo = NULL;
			root->pParentNode = NULL;
			break;
		case enLeafNode:
		{
			root->nodetype = enLeafNode;
			root->subnodes_num = 0;
			root->pSubNode = new struct LoquatRTreeNode* [2];
			root->pSubNode[0] = NULL;
			root->pSubNode[1] = NULL; // 叶子节点没有子节点

			//attribute = attribute->Next();
			root->pLeafNodeInfo = new LeafNodeInfo;
			root->pLeafNodeInfo->dimension = loquatForest->RFinfo.datainfo.variables_num_y;
			root->pLeafNodeInfo->CovMatOfArrived = NULL;
			root->pLeafNodeInfo->MeanOfArrived = NULL;
			root->pLeafNodeInfo->linearPredictor = NULL;  //0527 否则在释放内存时出错
			attribute = attribute->Next(); // arriveRatio
			assert(attribute != NULL);
			root->pLeafNodeInfo->arrivedRatio = attribute->FloatValue();
			attribute = attribute->Next(); // MeanOfArrived
			assert(attribute != NULL);
			stringstream iss_moa(attribute->Value());
			root->pLeafNodeInfo->MeanOfArrived = new float[loquatForest->RFinfo.datainfo.variables_num_y];
			iss_moa >> root->pLeafNodeInfo->MeanOfArrived[0];
			char ch;
			for (int d = 1; d < loquatForest->RFinfo.datainfo.variables_num_y; d++)
				iss_moa >> ch >> root->pLeafNodeInfo->MeanOfArrived[d];
			
			const XMLAttribute* atrri_linearModel = elem_node->FindAttribute("LinearModel");
			if (NULL != atrri_linearModel)
			{
				const int lmdim = (loquatForest->RFinfo.datainfo.variables_num_x + 1) * loquatForest->RFinfo.datainfo.variables_num_y;
				root->pLeafNodeInfo->linearPredictor = new double[lmdim];
				// TODO:
				stringstream iss_lm(atrri_linearModel->Value());
				iss_lm >> root->pLeafNodeInfo->linearPredictor[0];
				for (int d = 1; d < lmdim; d++)
					iss_lm >> ch >> root->pLeafNodeInfo->linearPredictor[d];
			}

			root->pParentNode = NULL;
		}
		break;
		default:
			assert(0);
		}

		for (j = 1; j <= tree_depth; j++) // 从序号为1的层次开始(即根节点下面第一层)
		{
			elem_Depth = elem_Depth->NextSiblingElement(); // 指向下一层
			//int nodes_num_acctual = elem_Depth->FirstAttribute()->Next()->IntValue(); // 该层次的节点个数
			int nodes_num_acctual = elem_Depth->FindAttribute("Nodes")->IntValue(); // 该层次的节点个数
			XMLElement* elem_node = elem_Depth->FirstChildElement(); //
			int pp = 0;
			vector<LoquatRTreeNode* > nodes_current_level;
			for (k = 0; k < nodes_num_acctual; k++)
			{
				// const XMLAttribute *attribute = elem_node->FirstAttribute();
				// int type = attribute->IntValue();
				// attribute = attribute->Next();
				// int index = attribute->IntValue(); 
				// attribute =  attribute->Next();
				// int split_index = attribute->IntValue();
				// attribute = attribute->Next();
				// float split_value = attribute->FloatValue();

				int type = elem_node->FindAttribute("type")->IntValue();
				int index = elem_node->FindAttribute("index")->IntValue();
				int split_index = elem_node->FindAttribute("split_variable_index")->IntValue();
				float split_value = elem_node->FindAttribute("split_value")->FloatValue();

				LoquatRTreeNode* tmpNode = new LoquatRTreeNode; // 新建一个节点
				nodes_current_level.push_back(tmpNode); // 放入当前层次的节点栈
				tmpNode->depth = j;
				tmpNode->split_variable_index = split_index;
				tmpNode->split_value = split_value;
				tmpNode->samples_index = NULL;
				tmpNode->arrival_samples_num = -1;
				tmpNode->train_impurity = 0.f;
				tmpNode->pLeafNodeInfo = NULL;

				switch (type)
				{
				case enRootNode: // 不可能再是root了
					assert(0);
				case enLinkNode:
					tmpNode->nodetype = enLinkNode;
					tmpNode->subnodes_num = 2;
					tmpNode->pSubNode = new struct LoquatRTreeNode* [2];
					tmpNode->pSubNode[0] = NULL;
					tmpNode->pSubNode[1] = NULL;

					while (1)
					{
						if (nodes_above_level[pp]->nodetype != enLeafNode) // 是一个连接节点
						{
							if (nodes_above_level[pp]->pSubNode[0] == NULL)
							{
								nodes_above_level[pp]->pSubNode[0] = tmpNode;
								tmpNode->pParentNode = nodes_above_level[pp];
								break;
							}
							else if (nodes_above_level[pp]->pSubNode[1] == NULL)
							{
								nodes_above_level[pp]->pSubNode[1] = tmpNode;
								tmpNode->pParentNode = nodes_above_level[pp];
								pp++; // 这个上层节点的两个子节点都已经连接了!
								break;
							}
							assert(0); // 不可能到这里
						}
						else
							pp++;
					}
					break;
				case enLeafNode:
				{ // start of case enLeafNode
					tmpNode->nodetype = enLeafNode;
					tmpNode->subnodes_num = 0;
					tmpNode->pSubNode = new struct LoquatRTreeNode* [2];
					tmpNode->pSubNode[0] = NULL;
					tmpNode->pSubNode[1] = NULL;
					tmpNode->pLeafNodeInfo = new LeafNodeInfo;
					tmpNode->pLeafNodeInfo->dimension = loquatForest->RFinfo.datainfo.variables_num_y;
					tmpNode->pLeafNodeInfo->CovMatOfArrived = NULL;
					tmpNode->pLeafNodeInfo->MeanOfArrived = NULL;
					tmpNode->pLeafNodeInfo->linearPredictor = NULL; //0527 否则在释放内存时出错
					//attribute = attribute->Next(); // arriveRatio
					//assert(attribute != NULL);
					tmpNode->pLeafNodeInfo->arrivedRatio = elem_node->FindAttribute("arriveRatio")->FloatValue(); //attribute->FloatValue();
					//attribute = attribute->Next(); // MeanOfArrived
					//assert(attribute != NULL);
					//stringstream iss_moa(attribute->Value());
					stringstream iss_moa(elem_node->FindAttribute("MeanOfArrived")->Value());
					tmpNode->pLeafNodeInfo->MeanOfArrived = new float[loquatForest->RFinfo.datainfo.variables_num_y];
					iss_moa >> tmpNode->pLeafNodeInfo->MeanOfArrived[0];
					char ch;
					for (int d = 1; d < loquatForest->RFinfo.datainfo.variables_num_y; d++)
						iss_moa >> ch >> tmpNode->pLeafNodeInfo->MeanOfArrived[d];
					const XMLAttribute* atrri_linearModel = elem_node->FindAttribute("LinearModel");
					if (NULL != atrri_linearModel)
					{
						const int lmdim = (loquatForest->RFinfo.datainfo.variables_num_x + 1) * loquatForest->RFinfo.datainfo.variables_num_y;
						tmpNode->pLeafNodeInfo->linearPredictor = new double[lmdim];
						// TODO:
						stringstream iss_lm(atrri_linearModel->Value());
						iss_lm >> tmpNode->pLeafNodeInfo->linearPredictor[0];
						for (int d = 1; d < lmdim; d++)
							iss_lm >> ch >> tmpNode->pLeafNodeInfo->linearPredictor[d];
					}
					while (1)
					{
						if (nodes_above_level[pp]->nodetype != enLeafNode) // 是一个连接节点
						{
							if (nodes_above_level[pp]->pSubNode[0] == NULL)
							{
								nodes_above_level[pp]->pSubNode[0] = tmpNode;
								tmpNode->pParentNode = nodes_above_level[pp];
								break;
							}
							else if (nodes_above_level[pp]->pSubNode[1] == NULL)
							{
								nodes_above_level[pp]->pSubNode[1] = tmpNode;
								tmpNode->pParentNode = nodes_above_level[pp];
								pp++; // 这个上层节点的两个子节点都已经连接了!
								break;
							}
							assert(0); // 不可能到这里
						}
						else
							pp++;
					}

				} // end of case enLeafNode
				break;
				default:
					assert(0);
				}

				elem_node = elem_node->NextSiblingElement(); // !!!
			} // for( k=0; k<nodes_num_acctual; k++ ) 

			nodes_above_level = nodes_current_level;
		}

		elem_Tree = elem_Tree->NextSiblingElement(); // 指向下一个xml结构表示的tree
	}
	return 1;

QUIT_XML:
	cout << "-----------------  ERROR:'BuildRegressionForestModelFromXMLFile' ------------------" << endl;
	cout << "'" << error_disp << "' can not be found." << endl;
	cout << "----------------------------------------------------------------------" << endl;
	if (loquatForest->loquatTrees != NULL)
	{
		delete[] loquatForest->loquatTrees;
		loquatForest->loquatTrees = NULL;
	}
	delete loquatForest;
	loquatForest = NULL;
	return -1;
}