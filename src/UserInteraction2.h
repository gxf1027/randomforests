#ifndef _USER_INTERACTION_RF_2_H_
#define _USER_INTERACTION_RF_2_H_

#include "RandomCLoquatForests.h"
#include "RandomRLoquatForests.h"

/*
Description: Read configuration from a XML file. 
             if read successfully, the arguments except 'datainfo' in RF_info are initialized.
			 NOTE: Train data are not read in this function, but file path of train data is returned for reading these data afterwards.
[return] 
             1.  1:  read successfully
			 2.  -1: values in XML may be inappropriate
			 3.  -2: Element Name is not correct
			 4.  -3: Element structure is not correct
*/
int ReadClassificationForestConfigFile2(const char *configXMLPath, RandomCForests_info &RF_info);
int ReadRegressionForestConfigFile2(const char *configXMLPath, RandomRForests_info &RF_info);

/*
Description:  Read training data and their labels(for classification) or targets(for regression) to the matrix 'data' and the vector 'label'('target'),
			  the file must be formulated as the following format.

[in] 1.filename:     the direction of txt file containing training samples and their labels
					 the file format
					 (1) classification:
					 @totoal_sample_num=N (total number of samples)
					 @variable_num=M (number of variables)
					 @class_num=C (number of classes)
					 label1   x11 x12......x1M
					 label2   x21 x22......x2M
					 ... ...
					 labelN   xN1 xN2......xNM
					 (2) regression:
					 @totoal_sample_num=N (total number of samples)
					 @variable_num_x=M (number of variables)
					 @variable_num_y=K (target dimension, in most cases 1)
					 target11 target12...target1K x11 x12......x1M
					 ... ...
					 targetN1 targetN2...targetNK xN1 xN2......xNM

	  2.data         two dimension array [N][M], the memory will be allocated in this function,

	  3.label        the labels of the corresponding training samples
		target[]     two dimension array [N][K], the output of the corresponding training samples, supporting multi-dimensional output

	  NOTE: the user must make sure that above pointers (data,label,target) are not assigned with blocks of memory
					 (assign NULL to them before calling the function).

[out] 1.data_info    fill the struct when the function returns

[return]
	  1.  -2:        file can't be opened.
	  2.  -1:        the format of data or extra information isn't correctly compiled.
	  3.   1:        read data , labels and other information successfully.
*/
int InitalClassificationDataMatrixFormFile2(const char *fileName, float **&data, int *&label, Dataset_info_C &data_info);

int InitalRegressionDataMatrixFormFile2(const char *fileName, float **&data, float **&target, Dataset_info_R &data_info);
int InitalRegressionDataMatrixFormFile22(const char* fileName, float**& data, float*& target, Dataset_info_R& data_info);


/*
Description:    Save a trained Random Forest model to a XML/PlainText file
[in]:  pFilePath:      file to save
[in]:  loquatForest:   A successfully trained Random Forests model
[out]: outputType:     0--XML file; 1--PlainText file
*/
void SaveRandomClassificationForestModel(const char *pFilePath, LoquatCForest *loquatForest, int outputType=0);
void SaveRandomRegressionForestModel(const char *pFilePath, LoquatRForest *loquatForest, int outputType=0);
/*
Description:    Read a xml file and build a Random Forests model corresponding to the content of the xml file.
[in]:  pFilePath:      file to save
[in]:  fileType:       0--XML file; 1--PlainText file
[out]: loquatForest:   loquatForest NULL pointer(NULL MUST be assigned to it when calling, memory management is handled by the function)
*/
int BuildRandomClassificationForestModel(const char *pFilePath, int fileType, LoquatCForest *&loquatForest);
int BuildRandomRegressionForestModel(const char *pFilePath, int fileType, LoquatRForest *&loquatForest);

#endif /* _USER_INTERACTION_RF_2_H_ */