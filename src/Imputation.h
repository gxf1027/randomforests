#pragma once
#ifndef _IMPUTATION_FOREST_
#define _IMPUTATION_FOREST_

#include "SharedRoutines.h"
#include "RandomCLoquatForests.h"
#include "RandomRLoquatForests.h"

/*
Description:	Missing values imputation for classification dataset by Random Forest Proximity
[in]			1.data:				samples with missing values, two dimension array [N][M]
				2.is_categorical:	optional, NULL or an array with length M. if NULL, meaning all features of 'data' are numerical; if not NULL, '0' for numerical,'1' for categorical
				3.data_orig:		optional, samples without missing values, with which NRMSE or RMSE can be accessed
				4.label:			sample labels
				5.RFinfo:			random forest parameters
				6.prox_type:		type of proximity, 'ProximityType::PROX_GEO_ACC' or 'ProximityType::PROX_ORIGINAL'
				7.max_iteration:	max rounds of iteration			
[out]			
return:			Imputated samples with the same dimension of 'data'
*/
float** MissingValuesImputaion(float** data, bool* is_categorical, const float** data_orig, int* label, RandomCForests_info RFinfo, ProximityType prox_type=ProximityType::PROX_GEO_ACC, int max_iteration=2, bool verbose=true, int random_state=0, int jobs=1);


/*
Description:	Missing values imputation for regression dataset by Random Forest Proximity
[in]			1.data:				samples with missing values, two dimension array [N][M]
				2.is_categorical:	optional, NULL or an array with length M. if NULL, meaning all features of 'data' are numerical; if not NULL, '0' for numerical,'1' for categorical
				3.data_orig:		optional, samples without missing values, with which NRMSE or RMSE can be accessed
				4.target:			target
				5.RFinfo:			random forest parameters
				6.prox_type:		type of proximity, 'ProximityType::PROX_GEO_ACC' or 'ProximityType::PROX_ORIGINAL'
				7.max_iteration:	max rounds of iteration
[out]
return:			Imputated samples with the same dimension of 'data'
*/
float** MissingValuesImputaion(float** data, bool* is_categorical, const float** data_orig, float* target, RandomRForests_info RFinfo, ProximityType prox_type, int max_iteration, bool verbose, int random_state, int jobs);

#endif