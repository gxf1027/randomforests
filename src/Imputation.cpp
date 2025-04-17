#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <vector>
#include <deque>
#include <cassert>
#include <cstring>
#include <float.h>
#include <algorithm>
#include <iomanip>
#include <map>
#include <unordered_map>
#include "Imputation.h"
#include "Proximity.h"
#include "omp.h"



int chekcDataSetLabel(int* label, int sample_num, int class_num);
LoquatCTreeNode*** createLeafNodeMatrix(LoquatCForest* forest, float** data);
LoquatRTreeNode*** createLeafNodeMatrix(LoquatRForest* forest, float** data);
int _ClassificationForestOrigProximity(LoquatCForest* forest, float* data, LoquatCTreeNode*** leafNodeMatrix, float*& proximities);
int RegressionForestOrigProximity2(LoquatRForest* forest, float* data, LoquatRTreeNode*** leafNodeMatrix, float*& proximities);
int GrowGreaterDepth(LoquatCForest*& forest, float** data, int* label);

int standardization(float** data, int samples_num, int variables_num, bool* is_categorical, unsigned char** nanmask, float** data_sd, float* mean, float* std)
{
	int *non_miss = new int [variables_num];
	memset(non_miss, 0, sizeof(int)*variables_num);
	memset(mean, 0, sizeof(float)*variables_num);
	memset(std, 0, sizeof(float)*variables_num);

	bool has_missing = (nanmask == NULL ? false : true);
	for (int v=0; v<variables_num; v++)
	{
		if (is_categorical != NULL && is_categorical[v])
			continue;

		for (int n=0; n<samples_num; n++)
		{
			if (has_missing && nanmask[n][v])
				continue;

			non_miss[v]++;
			mean[v] += data[n][v];
		}
		if (non_miss[v] == 0)
		{
			delete[] non_miss;
			return -1;
		}
		mean[v] = mean[v]/non_miss[v];
	}


	for (int v=0; v<variables_num; v++)
	{
		if (is_categorical != NULL && is_categorical[v])
			continue;
		const float m = mean[v];
		assert(non_miss[v] > 0);
		for (int n=0; n<samples_num; n++)
		{
			if (has_missing && nanmask[n][v])
				continue;

			std[v] += (data[n][v] - m)*(data[n][v] - m);
		}

		std[v] = sqrtf(std[v] / non_miss[v]);
		if (std[v] <= FLT_MIN)
			std::cout<<"[standardization] WARN: values along the "<<v<<"-th "<<v<<" are all equal."<<std::endl;	
	}


	for (int v=0; v<variables_num; v++)
	{
		if (is_categorical != NULL && is_categorical[v])
			continue;

		const float m = mean[v];
		const float s = std[v];
		if (s<=FLT_MIN)
			continue;

		for (int n=0; n<samples_num; n++)
		{
			if (has_missing && nanmask[n][v])
				data_sd[n][v] = data[n][v];
			else
				data_sd[n][v] = (data[n][v] - m)/s;
		}
	}

	delete[] non_miss;

	return 1;
}

bool** MedianFill(float** data, bool* is_categorical, int* label, RandomCForests_info RFinfo, float** data_imp/*OUT*/)
{
	const int samples_num = RFinfo.datainfo.samples_num;
	const int var_num = RFinfo.datainfo.variables_num;
	const int class_num = RFinfo.datainfo.classes_num;

	// roughly fill missing value with median of the every class
	bool** bzero = new bool* [var_num]; // if there exists samples without missing values, with the c-th class label along v-th variable 
	
	for (int v = 0; v < var_num; v++)
	{
		bzero[v] = new bool[class_num];
		if (!is_categorical[v])
		{
			std::vector<float> vvar;
			std::vector<float>* vt = new std::vector<float>[class_num];

			for (int n = 0; n < samples_num; n++)
			{
				if (!std::isnan(data[n][v]))
				{
					vt[label[n]].push_back(data[n][v]);
					vvar.push_back(data[n][v]);
				}
			}

			//cout<<vvar.size()<<endl;
			std::sort(vvar.begin(), vvar.end());
			float median_allcls = vvar[vvar.size() / 2];

			for (int c = 0; c < class_num; c++)
			{
				bzero[v][c] = vt[c].size() > 0 ? false : true;
				std::sort(vt[c].begin(), vt[c].end());
			}

			// filling
			for (int n = 0; n < samples_num; n++)
			{
				if (std::isnan(data[n][v]))
				{
					data_imp[n][v] = bzero[v][label[n]] == false ? vt[label[n]][vt[label[n]].size() / 2] : median_allcls;
				}
			}

			delete[] vt;
		}
		else
		{
			std::unordered_map<int, int>* freq = new std::unordered_map<int, int>[class_num];
			std::unordered_map<int, int> freqAllCls;
			for (int n = 0; n < samples_num; n++)
			{
				if (!std::isnan(data[n][v]))
				{
					int b = label[n];
					int k = int(data[n][v]);
					if (freq[b].end() == freq[b].find(k))
						freq[b].emplace(k, 1);
					else
						freq[b][k]++;

					if (freqAllCls.end() == freqAllCls.find(k))
						freqAllCls.emplace(k, 1);
					else
						freqAllCls[k]++;
				}
			}

			// find the category wit the most ocurred frequency over every class
			int* mostfreq = new int[class_num]; // over every class
			memset(mostfreq, 0, sizeof(int) * class_num);
			int mostfreq_allcls = -1;
			for (int c = 0; c < class_num; c++)
			{
				bzero[v][c] = freq[c].size() > 0 ? false : true;
				int q = -1;
				for (auto it = freq[c].begin(); it != freq[c].end(); ++it)
				{
					if (it->second > q)
					{
						q = it->second;
						mostfreq[c] = it->first;
					}
				}
			}

			int tmp = -1;
			for (auto it = freqAllCls.begin(); it != freqAllCls.end(); ++it)
			{
				if (it->second > tmp)
				{
					tmp = it->second;
					mostfreq_allcls = it->first; // categroy value of all samples with the most occurred frequency over v-th variable
				}
			}

			// filling
			for (int n = 0; n < samples_num; n++)
			{
				if (std::isnan(data[n][v]))
				{
					data_imp[n][v] = bzero[v][label[n]] == false ? mostfreq[label[n]] : mostfreq_allcls;
				}
			}

			delete[] freq;
			delete[] mostfreq;
		}

	}

	return bzero;
}

/*
* mean_true: mean over missing values
* var_true:  variance over missing values 
* cnt: the number of missing values
*/
void PreparePrecisionCalculation(int samples_num, int var_num, const float** data_orig, unsigned char** nanmask, int* cnt, float *mean_true, float* var_true)
{
	memset(cnt, 0, sizeof(int)*var_num);
	memset(mean_true, 0, sizeof(float)*var_num);

	for (int n = 0; n < samples_num; n++)
	{
		for (int v = 0; v < var_num; v++)
		{
			if (nanmask[n][v])
			{
				//var_true[v] += data_orig[n][v] * data_orig[n][v];
				mean_true[v] += data_orig[n][v];
				cnt[v]++;
			}
		}
	}

	//for (int v = 0; v < var_num; v++)
	//{
	//	if (cnt[v] == 0)
	//		continue;
	//	mean_true[v] /= cnt[v];
	//	var_true[v] = var_true[v] / cnt[v] - mean_true[v] * mean_true[v];
	//	//assert(var_true[v] >= 0.f);
	//	cout << var_true[v] << endl;
	//}


	for (int v = 0; v < var_num; v++)
	{
		if (cnt[v] == 0 )
		{
			var_true[v] = FLT_MAX;
			mean_true[v] = 0.f;
			continue;
		}

		var_true[v] = 0.f;
		mean_true[v] /= cnt[v];
		for (int n = 0; n < samples_num; n++)
		{
			if (nanmask[n][v])
			{
				var_true[v] += (data_orig[n][v] - mean_true[v]) * (data_orig[n][v] - mean_true[v]);
			}
		}
		var_true[v] = var_true[v] / cnt[v];
		assert(var_true[v] >= 0.f);
		//cout << var_true[v] << endl;
	}
}

/*
* 
*/
void PrintPrecision(int samples_num, int var_num, float **data_imp, const float **data_orig, unsigned char** nanmask, const bool* is_categorical, const int* cnt_mv, const float* var_true, int method='1')
{
	float d_allvar = 0.f, d_contin = 0.f, d_cate = 0.f;
	int total_nan = 0, contin_nan = 0, cate_nan = 0;
	float tmp = 0.f;
	float* dmean_imp = new float[var_num];
	memset(dmean_imp, 0, sizeof(float) * var_num);
	for (int n = 0; n < samples_num; n++)
	{
		for (int v = 0; v < var_num; v++)
		{
			if (nanmask[n][v])
			{
				total_nan++;
				if (false == is_categorical[v])
				{
					contin_nan++;
					tmp = (data_orig[n][v] - data_imp[n][v]) * (data_orig[n][v] - data_imp[n][v]);
					dmean_imp[v] += tmp;
					d_allvar += tmp;
					d_contin += tmp;
				}
				else
				{
					cate_nan++;
					tmp = ((int)data_orig[n][v]) == ((int)data_imp[n][v]) ? 0 : 1;
					dmean_imp[v] += tmp;
					d_allvar += tmp;
					d_cate += tmp;
				}
				
			}
		}
	}
	assert(cate_nan + contin_nan == total_nan);
	if (method == 1)
		std::cout << "VARIABLE\t" << "MISSING_COUNTS\t" << "NRMSE\t\t" <<"TYPE" << std::endl;
	else
		std::cout << "VARIABLE\t" << "MISSING_COUNTS\t" << "RMSE\t\t" << "TYPE" << std::endl;

	int coutinuios = 0, categorical = 0;
	float all_coutin_error = 0.f, all_cate_error = 0.f;
	for (int v = 0; v < var_num; v++)
	{
		if (cnt_mv[v] == 0) // feature v does not have missing values
		{
			std::cout << v << "\t\t" << "has no missing values" << std::endl;
			continue;
		}	

		dmean_imp[v] /= cnt_mv[v];

		if (false == is_categorical[v])
		{
			coutinuios++;
			if (method == 1) // NRMSE
			{
				tmp = sqrtf(dmean_imp[v] / (var_true[v]+FLT_MIN));
				std::cout << v << "\t\t" << cnt_mv[v] << "\t\t" << tmp << std::endl;
				all_coutin_error += tmp;
			}
			else
			{
				all_coutin_error += sqrtf(dmean_imp[v]);
				std::cout << v << "\t\t" << cnt_mv[v] << "\t\t" << sqrtf(dmean_imp[v]) << std::endl;
			}
		}
		else
		{
			categorical++;
			std::cout << v << "\t\t" << cnt_mv[v] << "\t\t" << dmean_imp[v]*100 << "%\t Categorical" << std::endl;
			all_cate_error += dmean_imp[v];
		}
	}

	std::cout << "RMSE over all missing values: \t" << sqrtf(d_contin / total_nan) << std::endl;
	
	if (method == 1)
		std::cout << "(1) NRMSE over all variables: \t " << all_coutin_error / (coutinuios == 0 ? 1: coutinuios) <<"\t (2) categorical: "<< all_cate_error / (categorical == 0 ? 1 : categorical)*100 <<"%"<< std::endl;
	else
		std::cout << "(1) RMSE over all variables: \t " << all_coutin_error / (coutinuios == 0 ? 1 : coutinuios) << "\t (2) categorical: " << all_cate_error / (categorical == 0 ? 1 : categorical)*100 <<"%"<< std::endl;
	//std::cout << "======================================================" << std::endl;

	delete[]dmean_imp;
}

float** MissingValuesImputaion(float** data, bool* is_categorical, const float** data_orig, int* label, RandomCForests_info RFinfo, ProximityType prox_type, int max_iteration, bool verbose, int random_state, int jobs)
{
	int rv = chekcDataSetLabel(label, RFinfo.datainfo.samples_num, RFinfo.datainfo.classes_num);
	if (rv < 0)
	{
		std::cout << ">>>[MissingValuesImputaion] labels are not properly set." << std::endl;
		return NULL;
	}

	const int samples_num = RFinfo.datainfo.samples_num;
	const int var_num = RFinfo.datainfo.variables_num;
	const int class_num = RFinfo.datainfo.classes_num;


	bool todel = false;
	if (is_categorical == NULL)
	{
		todel = true;
		is_categorical = new bool[RFinfo.datainfo.variables_num];
		memset(is_categorical, 0, sizeof(bool) * RFinfo.datainfo.variables_num);
	}

	unsigned char* rowmask = new unsigned char[samples_num];
	memset(rowmask, 0, sizeof(unsigned char) * samples_num);

	// generate mask of the sample row and matrix
	unsigned char** nanmask = new unsigned char* [samples_num];
	for (int n = 0; n < samples_num; n++)
	{
		nanmask[n] = new unsigned char[var_num];
		memset(nanmask[n], 0, sizeof(unsigned char) * var_num);
		for (int j = 0; j < var_num; j++)
		{
			if (std::isnan(data[n][j]))
			{
				nanmask[n][j] = 1;
				rowmask[n] = 1;
			}
		}
	}

	float** data_imp = clone_data(data, samples_num, var_num);

	// (1) median-fill method is used to imputate at first
	bool** bzero = MedianFill(data, is_categorical, label, RFinfo, data_imp/*OUT*/);
#if 0
	std::ofstream out("data_roughlyfill.txt");
	for (int n = 0; n < samples_num; n++)
	{
		for (int v = 0; v < var_num; v++)
		{
			out << data_imp[n][v];
			if (v != var_num - 1)
				out << " ";
		}
		out << std::endl;
	}
	out.close();
#endif

	float** data_tmp = clone_data(data_imp, samples_num, var_num);

	int method_nrmse = 0;  // 0: RMSE; 1: NRMSE
	float *var_true = NULL; // variance of the ture(original) values along a variable(feature)
	float* mean_true = NULL;
	int* count_var = NULL; // missing values of a variable(feature)
	float* nrmse = NULL; //  the normalized root mean squared error (NRMSE)
	if (data_orig)
	{
		var_true = new float[var_num * 3];
		mean_true = var_true + var_num;
		nrmse = var_true + var_num * 2;
		memset(var_true, 0, sizeof(float) * var_num * 3);

		count_var = new int[var_num];
		memset(count_var, 0, sizeof(int) * var_num);

		PreparePrecisionCalculation(samples_num, var_num, data_orig, nanmask, count_var, mean_true, var_true);
		std::cout << "imputation results after median filling:" << std::endl;
		PrintPrecision(samples_num, var_num, data_imp, data_orig, nanmask, is_categorical, count_var, var_true, method_nrmse);
	}

	if (jobs <= 0)
		jobs = 1;

	jobs = jobs > omp_get_max_threads() ? omp_get_max_threads() : jobs;
	omp_set_num_threads(jobs);
	
	// (2) random forest proximities is used to impuate
	for (int it = 0; it < max_iteration; it++)
	{
		std::cout << "======================================================" << std::endl;
		std::cout << ">>iteration " << it << " starts." << std::endl;
		LoquatCForest* forest = NULL;
		
		TrainRandomForestClassifier(data_tmp, label, RFinfo, forest, random_state, verbose ? 50 : 0, jobs);
		std::cout << ">>random forest is trained. " << std::endl;
		////////////////
		//std::cout << "start post-training" << std::endl;
		//GrowGreaterDepth(forest, data_tmp, label);
		//PostTrain(forest, data_tmp, label);
		//std::cout << "end of post-training" << std::endl;
		////////////////
		LoquatCTreeNode*** leafMatrix = NULL;
		if (prox_type == ProximityType::PROX_ORIGINAL)
		{
			leafMatrix = createLeafNodeMatrix(forest, data_tmp);  // leafMatrix is calculated in advance for computation efficiency
		}

		//timeIt(1);
#pragma omp parallel for
		for (int n = 0; n < samples_num; n++)
		{
			if (!rowmask[n])
				continue;  // data[n] has no missing value

			float* proximity = NULL;
			int rv = 1;
			switch (prox_type)
			{
			case ProximityType::PROX_ORIGINAL:
				rv = _ClassificationForestOrigProximity(forest, data_tmp[n], leafMatrix, proximity);
				break;
			case ProximityType::PROX_GEO_ACC:
			default:
				rv = ClassificationForestProximity(forest, data_tmp, n, ProximityType::PROX_GEO_ACC, proximity);
			}
			
			if (rv < 0)
			{
				delete[] proximity;
				continue;
			}

			const int lb = label[n];
			for (int v = 0; v < var_num; v++)
			{
				if (!nanmask[n][v])
					continue;

				bool use_label = !bzero[v][lb];
				if (!is_categorical[v])
				{
					float s = 0.f, estimated = 0.f;

					for (int i = 0; i < samples_num; i++)
					{
						if (!nanmask[i][v] && (!use_label || label[i] == lb))
						{
							s += proximity[i];
							estimated += data_tmp[i][v] * proximity[i];
						}
					}

					if (s > 1e-10f)
					{
						estimated = estimated / s;
						data_imp[n][v] = estimated;
					}

				}
				else
				{
					int k;
					std::unordered_map<int, float> categroy_prox;
					for (int i = 0; i < samples_num; i++)
					{
						if (!nanmask[i][v] && (!use_label || label[i] == lb))
						{
							k = int(data_tmp[i][v]);
							if (categroy_prox.end() == categroy_prox.find(k))
								categroy_prox.emplace(k, proximity[i]);
							else
								categroy_prox[k] += proximity[i];
						}
					}

					int k_max_prox = -1;
					float max_prox = -FLT_MAX;
					for (auto it = categroy_prox.begin(); it != categroy_prox.end(); ++it)
					{
						if (it->second > max_prox)
						{
							max_prox = it->second;
							k_max_prox = it->first;
						}
					}
					data_imp[n][v] = k_max_prox;
				}

			}

			delete[] proximity;

			// if (n > 0 && n % 5000 == 0)
			//	 std::cout << "samples " << n << " has been imputated." << std::endl;
		}

		//std::cout << "proximities: " << timeIt(0) << std::endl;
		std::cout << ">>imputation is done. " << std::endl;

		ReleaseClassificationForest(&forest);

		if (prox_type == ProximityType::PROX_ORIGINAL)
		{
			for (int n = 0; n < samples_num; n++)
				delete[] leafMatrix[n];
			delete[] leafMatrix;
		}

		if (data_orig)
		{
			std::cout << "imputation results after iteration " << it << ":" << std::endl;
			PrintPrecision(samples_num, var_num, data_imp, data_orig, nanmask, is_categorical, count_var, var_true, method_nrmse);
		}


		for (int n = 0; n < samples_num; n++)
			memcpy(data_tmp[n], data_imp[n], sizeof(float) * var_num);
	}


	for (int n = 0; n < samples_num; n++)
	{
		delete[] nanmask[n];
		delete[] data_tmp[n];
	}

	delete[] nanmask;
	delete[] data_tmp;
	delete[] rowmask;

	delete[] var_true;
	delete[] count_var;

	for (int v = 0; v < var_num; v++)
		delete[] bzero[v];
	delete[] bzero;

	if (todel)
		delete[] is_categorical;

	return data_imp;
}
 

void MedianFill(float** data, bool* is_categorical, RandomRForests_info RFinfo, float** data_imp/*OUT*/)
{
	const int samples_num = RFinfo.datainfo.samples_num;
	const int var_num = RFinfo.datainfo.variables_num_x;

	// roughly fill missing value with median of the every class

	for (int v = 0; v < var_num; v++)
	{
		//bzero[v] = new bool[class_num];
		if (!is_categorical[v])
		{
			std::vector<float> vvar;

			for (int n = 0; n < samples_num; n++)
			{
				if (!std::isnan(data[n][v]))
					vvar.push_back(data[n][v]);
			}

			std::sort(vvar.begin(), vvar.end());
			float median = vvar[vvar.size() / 2];

			// filling
			for (int n = 0; n < samples_num; n++)
			{
				if (std::isnan(data[n][v]))
					data_imp[n][v] = median;
			}
		}
		else
		{
			std::unordered_map<int, int> freq;
			for (int n = 0; n < samples_num; n++)
			{
				if (!std::isnan(data[n][v]))
				{
					int k = int(data[n][v]);

					if (freq.end() == freq.find(k))
						freq.emplace(k, 1);
					else
						freq[k]++;
				}
			}

			int mostfreq = -1;
			int max_count = -1;
			for (auto it = freq.begin(); it != freq.end(); ++it)
			{
				if (it->second > max_count)
				{
					max_count = it->second;
					mostfreq = it->first;
				}
			}

			// filling
			for (int n = 0; n < samples_num; n++)
			{
				if (std::isnan(data[n][v]))
				{
					data_imp[n][v] = mostfreq;
				}
			}
		}

	}

}


float** MissingValuesImputaion(float** data, bool* is_categorical, const float** data_orig, float* target, RandomRForests_info RFinfo, ProximityType prox_type, int max_iteration, bool verbose, int random_state, int jobs)
{
	const int samples_num = RFinfo.datainfo.samples_num;
	const int var_num = RFinfo.datainfo.variables_num_x;


	bool todel = false;
	if (is_categorical == NULL)
	{
		todel = true;
		is_categorical = new bool[var_num];
		memset(is_categorical, 0, sizeof(bool) * var_num);
	}

	unsigned char* rowmask = new unsigned char[samples_num];
	memset(rowmask, 0, sizeof(unsigned char) * samples_num);

	// generate mask of the sample row and matrix
	unsigned char** nanmask = new unsigned char* [samples_num];
	for (int n = 0; n < samples_num; n++)
	{
		nanmask[n] = new unsigned char[var_num];
		memset(nanmask[n], 0, sizeof(unsigned char) * var_num);
		for (int j = 0; j < var_num; j++)
		{
			if (std::isnan(data[n][j]))
			{
				nanmask[n][j] = 1;
				rowmask[n] = 1;
			}
		}
	}

	float** data_imp = clone_data(data, samples_num, var_num);

	MedianFill(data, is_categorical, RFinfo, data_imp/*OUT*/);
#if 0
	std::ofstream out("data_roughlyfill.txt");
	for (int n = 0; n < samples_num; n++)
	{
		for (int v = 0; v < var_num; v++)
		{
			out << data_imp[n][v];
			if (v != var_num - 1)
				out << " ";
		}
		out << std::endl;
	}
	out.close();
#endif

	float** data_tmp = clone_data(data_imp, samples_num, var_num);

	int method_nrmse = 0;  // 0: RMSE; 1: NRMSE
	float* var_true = NULL;
	float* mean_true = NULL;
	int* count_var = NULL;
	float* dmean_imp = NULL;
	float* nrmse = NULL; //  the normalized root mean squared error (NRMSE)
	if (data_orig)
	{
		var_true = new float[var_num * 4];
		mean_true = var_true + var_num;
		dmean_imp = var_true + var_num * 2;
		nrmse = var_true + var_num * 3;
		memset(var_true, 0, sizeof(float) * var_num * 4);

		count_var = new int[var_num];
		memset(count_var, 0, sizeof(int) * var_num);

		PreparePrecisionCalculation(samples_num, var_num, data_orig, nanmask, count_var, mean_true, var_true);
		std::cout << "imputation results after median filling:" << std::endl;
		PrintPrecision(samples_num, var_num, data_imp, data_orig, nanmask, is_categorical, count_var, var_true, method_nrmse);
	}


	if (jobs <= 0)
		jobs = 1;

	jobs = jobs > omp_get_max_threads() ? omp_get_max_threads() : jobs;
	omp_set_num_threads(jobs);

	for (int it = 0; it < max_iteration; it++)
	{
		std::cout << "======================================================" << std::endl;
		std::cout << ">>iteration " << it << " starts." << std::endl;

		LoquatRForest* forest = NULL;
		TrainRandomForestRegressor(data_tmp, target, RFinfo, 
									forest, 
									RFinfo.datainfo.variables_num_y>1? true:false, 
									random_state, 
									verbose ? 50 : 0, jobs);
		std::cout << ">>random forest is trained" << std::endl;

		LoquatRTreeNode*** leafMatrix = NULL;
		if (prox_type == ProximityType::PROX_ORIGINAL)
		{
			leafMatrix = createLeafNodeMatrix(forest, data_tmp);
		}

#pragma omp parallel for num_threads(jobs) 
		for (int n = 0; n < samples_num; n++)
		{
			if (!rowmask[n])
				continue;

			float* proximity = NULL;
			int rv = 1;
			switch (prox_type)
			{
			case ProximityType::PROX_ORIGINAL:
				rv = RegressionForestOrigProximity2(forest, data_tmp[n], leafMatrix, proximity);
				break;
			case ProximityType::PROX_GEO_ACC:
			default:
				rv = RegressionForestGAPProximity(forest, data_tmp, n, proximity);
			}

			if (rv < 0)
			{
				delete[] proximity;
				continue;
			}

			for (int v = 0; v < var_num; v++)
			{
				if (!nanmask[n][v])
					continue;

				if (!is_categorical[v])
				{
					float s = 0.f, estimated = 0.f;

					for (int i = 0; i < samples_num; i++)
					{
						if (!nanmask[i][v])
						{
							s += proximity[i];
							estimated += data_tmp[i][v] * proximity[i];
						}
					}

					if (s > 1e-10f)
					{
						estimated = estimated / s;
						data_imp[n][v] = estimated;
					}

				}
				else
				{
					int k;
					std::unordered_map<int, float> categroy_prox;
					for (int i = 0; i < samples_num; i++)
					{
						if (!nanmask[i][v] )
						{
							k = int(data_tmp[i][v]);
							if (categroy_prox.end() == categroy_prox.find(k))
								categroy_prox.emplace(k, proximity[i]);
							else
								categroy_prox[k] += proximity[i];
						}
					}

					int k_max_prox = -1;
					float max_prox = -FLT_MAX;
					for (auto it = categroy_prox.begin(); it != categroy_prox.end(); ++it)
					{
						if (it->second > max_prox)
						{
							max_prox = it->second;
							k_max_prox = it->first;
						}
					}
					data_imp[n][v] = k_max_prox;
				}

			}

			delete[] proximity;

		}

		std::cout << ">>imputation is done." << std::endl;

		ReleaseRegressionForest(&forest);

		if (prox_type == ProximityType::PROX_ORIGINAL)
		{
			for (int n = 0; n < samples_num; n++)
				delete[] leafMatrix[n];
			delete[] leafMatrix;
		}

		if (data_orig)
		{
			std::cout << "imputation results after iteration "<<it<<":" << std::endl;
			PrintPrecision(samples_num, var_num, data_imp, data_orig, nanmask, is_categorical, count_var, var_true, method_nrmse);
		}


		for (int n = 0; n < samples_num; n++)
			memcpy(data_tmp[n], data_imp[n], sizeof(float) * var_num);
	}


	for (int n = 0; n < samples_num; n++)
	{
		delete[] nanmask[n];
		delete[] data_tmp[n];
	}

	delete[] nanmask;
	delete[] data_tmp;
	delete[] rowmask;

	delete[] var_true;
	delete[] count_var;

	if (todel)
		delete[] is_categorical;

	return data_imp;
}