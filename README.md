# randomforests
C++ implementation of random forests

**使用**
包含头文件：
- 分类 <RandomCLoquatForests.h>  或  回归 <RandomRLoquatForests.h>
- 公共 <SharedRoutines.h>
- 交互 <UserInteraction2.h> （包括数据集读取、模型保存和读取）**可选**

各函数及其参数在头文件中有详细说明，容易上手。以下给出训练和预测的代码片段。

**训练**

从本地数据文件读入数据集进行训练，计算oob-error(oob-mse)，并保存forest到本地。
（1）分类森林
```cpp
#include <cmath>
using namespace std;

#include "RandomCLoquatForests.h"
#include "UserInteraction2.h"

int main()
{
	// read training samples if necessary
	char filename[500] = "./DataSet/Classification/pendigits.tra";
	float** data = NULL;
	int* label = NULL;
	Dataset_info_C datainfo;
	InitalClassificationDataMatrixFormFile2(filename, data/*OUT*/, label/*OUT*/, datainfo/*OUT*/);
	// setting random forests parameters
	RandomCForests_info rfinfo;
	rfinfo.datainfo = datainfo;
	rfinfo.maxdepth = 40;
	rfinfo.ntrees = 500;
	rfinfo.mvariables = (int)sqrtf(datainfo.variables_num);
	rfinfo.minsamplessplit = 5;
	rfinfo.randomness = 1;
	// train forest
	LoquatCForest* loquatCForest = NULL;
	TrainRandomForestClassifier(data, label, rfinfo, loquatCForest /*OUT*/, 50);
	float error_rate = 1.f;
	OOBErrorEstimate(data, label, loquatCForest, error_rate /*OUT*/);
	// save RF model
	SaveRandomClassificationForestModelToXML2("Modelfile.xml", loquatCForest);
	// clear the memory allocated for the entire forest
	ReleaseClassificationForest(&loquatCForest);
	// release money: data, label
	for (int i = 0; i < datainfo.samples_num; i++)
   		delete[] data[i];
	delete[] data;
	delete[] label;
	return 0;
}
```
（2）回归森林
```cpp
#include "RandomRLoquatForests.h"
#include "UserInteraction2.h"
using namespace std;

int main()
{
	// read training samples if necessary 
    char filename[500] = "./DataSet/Regression/Housing_Data_Set-R.txt"; 
	float** data = NULL;
	float** target = NULL;
	Dataset_info_R datainfo;
	InitalRegressionDataMatrixFormFile2(filename, data /*OUT*/, target /*OUT*/, datainfo /*OUT*/);
	// setting random forests parameters
	RandomRForests_info rfinfo;
	rfinfo.datainfo = datainfo;
	rfinfo.maxdepth = 40;
	rfinfo.ntrees = 200;
	rfinfo.mvariables = (int)(datainfo.variables_num_x / 3.0 + 0.5); 
	rfinfo.minsamplessplit = 5;
	rfinfo.randomness = 1; 
	// train forest
	LoquatRForest* loquatRForest = NULL;
	TrainRandomForestRegressor(data, target, rfinfo, loquatRForest /*OUT*/, false, 20);
	float* mean_squared_error = NULL;
	MSEOnOutOfBagSamples(data, target, loquatRForest, mean_squared_error /*OUT*/);
	delete[] mean_squared_error;
	// save RF model
	SaveRandomRegressionForestModelToXML2("Modelfile-R.xml", loquatRForest);
	// clear the memory
	ReleaseRegressionForest(&loquatRForest);
	// release money: data, target
	for (int i = 0; i < datainfo.samples_num; i++)
   	{
		   delete[] data[i];
		   delete[] target[i];
	}	
	delete[] data;
	delete[] target;
}
```

说明
- 以上代码仅为主干，实际使用需对函数返回值进行判断。
- RF结构体对象loquatForest的内存由*TrainRandomForestClassifier* /*TrainRandomForestRegressor* 负责分配，由*ReleaseClassificationForest* /*ReleaseRegressionForest* 释放内存，**用户无需对其分配或者释放**
- *OOBErrorEstimate*  计算out-of-bag分类错误率，输入参数data, label**必须**与训练时相同,*MSEOnOutOfBagSamples*类同
- *InitalClassificationDataMatrixFormFile2*/*InitalRegressionDataMatrixFormFile2* 从本地文件读取数据集。也可以自行准备训练数据，就可以不调用上述函数。
