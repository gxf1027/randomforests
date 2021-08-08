
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <float.h>
using namespace std;

#include "RandomRLoquatForests.h"
#include "SharedRoutines.h"

//#define STOP_CRITERION_NUM_RATIO_R		0.00005 // 0.008 /*default*/
#define INTERVAL_STEPS_NUM				50
#define VERY_SMALL_VALUE				1e-10f
#define FLOAT_MAX						3.0e38f
#define FLOAT_MIN						1.0e-38f
#define DEFAULT_MIN_SAMPLES				5
#define DEFAULT_MAX_TREE_DEPTH_R		40

//#define new new(_CLIENT_BLOCK, __FILE__, __LINE__)

void UseDefaultSettingsForRFs(RandomRForests_info &RF_info)
{
	RF_info.ntrees = 200;
	RF_info.mvariables = RF_info.datainfo.variables_num_x/3.0;
	if (RF_info.mvariables <= 0)
		RF_info.mvariables = RF_info.datainfo.variables_num_x;
	RF_info.maxdepth = DEFAULT_MAX_TREE_DEPTH_R;
	RF_info.minsamplessplit = DEFAULT_MIN_SAMPLES;
	RF_info.predictionModel = PredictionModel::constant;
	RF_info.randomness = RF_TREE_RANDOMNESS::TREE_RANDOMNESS_WEAK;
}

int CheckRegressionForestParameters(RandomRForests_info &RF_info)
{
	if( RF_info.datainfo.samples_num<=0 || RF_info.datainfo.variables_num_x<= 0 || RF_info.datainfo.variables_num_y<=0 )
	{
		cout<<">>>>>>>>>>>>>>>>>>>>>>>>>Parameters Check Information>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl;
		cout<<"Data Information is not properly assigned to."<<endl;
		if(RF_info.datainfo.variables_num_x <= 0 )
			cout<<"    'variables_num_x' can't be less than 1"<<endl;
		if(RF_info.datainfo.variables_num_y <= 0 )
			cout<<"    'variables_num_y' can't be less than 1"<<endl;
		if( RF_info.datainfo.samples_num <= 0 )
			cout<<"    Samples_num can't be less than 0"<<endl;
		cout<<"You MUST check out your dataset file."<<endl;
		cout<<">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<endl;
		return -1;
	}

	// 以下参数错误那么可以自动修正(使用默认值)
	int rv = 1;
	if( RF_info.maxdepth <= 1 )
	{
		cout<<">>>>'maxdepth' must be more than 1,"<<endl;
		cout<<">>>>thus, "<<DEFAULT_MAX_TREE_DEPTH_R<<" is assigned to 'maxdepth'."<<endl;
		RF_info.maxdepth = DEFAULT_MAX_TREE_DEPTH_R;
		rv = 0;
	}

	if( RF_info.mvariables <= 0 )
	{
		cout<<">>>>'mvariables' must be no less than 1,"<<endl;
		RF_info.mvariables = RF_info.datainfo.variables_num_x / 3.0;
		if (RF_info.mvariables <= 0)
			RF_info.mvariables = RF_info.datainfo.variables_num_x;
		cout<<">>>>thus, "<< RF_info.mvariables <<" is assigned to 'mvariables'."<<endl;
		rv = 0;
	}
	if( RF_info.mvariables > RF_info.datainfo.variables_num_x )
	{
		cout<<">>>>'mvariables' should be no more than number of the variables of the original data set,"<<endl;
		
		RF_info.mvariables = RF_info.datainfo.variables_num_x/3.0;
		if (RF_info.mvariables <= 0)
			RF_info.mvariables = RF_info.datainfo.variables_num_x;
		cout << ">>>>thus, the recommended value: " << RF_info.mvariables << " is assigned to." << endl;
		rv = 0;
	}
	if( RF_info.ntrees <=0 )
	{
		cout<<">>>>'ntrees' must be no less than 1,"<<endl;
		RF_info.ntrees = 200;
		cout<<">>>>thus, the default value:"<< RF_info.ntrees <<" is assigned to 'ntrees'."<<endl;
		rv = 0;
	}

	if (RF_info.minsamplessplit <= 0)
	{
		cout << ">>>>'minsamplessplit' must be no less than 1," << endl;
		RF_info.minsamplessplit = DEFAULT_MIN_SAMPLES;
		cout << ">>>>thus,  the default value:" << RF_info.minsamplessplit << " is assigned to 'minsamplessplit'." << endl;
		rv = 0;
	}
	
	return rv;
}


typedef struct _ForestMat
{
	int rows;
	int cols;
	double* data;
}ForestMat;

inline ForestMat* createMat(int rows, int cols)
{
	ForestMat* m = new ForestMat;
	m->rows = rows;
	m->cols = cols;
	m->data = new double[rows * cols];
	memset(m->data, 0, sizeof(m->data[0] * rows * cols));
	return m;
}

inline ForestMat* createMat(int rows, int cols, double* data)
{
	ForestMat* m = new ForestMat;
	m->rows = rows;
	m->cols = cols;
	m->data = data;
	return m;
}

inline void deleteMat(ForestMat* mat)
{
	mat->rows = 0;
	mat->cols = 0;
	if (mat->data)
		delete[] mat->data;
}

/*LU*/
void ludcmp(double  **a, int  n,  double *d)
{ 
	int i, j, k, imax;
	double big,   dum,   sum,   temp; 
	double *vv=new double[n]; 
	int *indx = new int[n];

	*d = 1.0; 
	for( i=0; i< n; i++) 
	{ 
		big = 0.0; 
		for (j = 0; j<n; j++) 
			if( temp = fabs(a[i][j]) > big)   
				big   =   temp; 
		if(big == 0.0)   
		{ 
			delete [] vv;
			delete [] indx;
			return;
		}   
		vv[i] = 1.0f/big; 
	}   

	for(j=0; j < n; j++) 
	{ 
		for(i= 0; i < j; i++) 
		{ 
			sum = a[i][j]; 
			for   (k = 0; k < i; k++) 
				sum -= a[i][k]*a[k][j]; 
			a[i][j] = sum; 
		} 

		big = 0.0; 
		for   (i = j;  i < n;   i++) 
		{ 
			sum = a[i][j]; 
			for   (k = 0; k < j; k++) 
				sum -= a[i][k]*a[k][j]; 
			a[i][j] = sum; 

			if ( (dum=vv[i]*fabs(sum)) >= big) 
			{ 
				big = dum; 
				imax = i; 
			}       
		} 

		if(j != imax) 
		{ 
			for(k=0; k < n;  k++) 
			{ 
				dum = a[imax][k]; 
				a[imax][k] = a[j][k]; 
				a[j][k] = dum; 
			} 

			*d =  -(*d); 
			vv[imax] =  vv[j]; 
		} 

		indx[j] = imax; 
		if(a[j][j] == 0.0)   
			a[j][j] = 1.0e-20; 
		if(j != n) 
		{ 
			dum = 1.0f / a[j][j]; 
			for(i=j+1; i < n;   i++) 
				a[i][j] *= dum; 
		} 
	}

	delete [] vv; 
	delete [] indx;
}

double CalculateDeterminant(double **matrix, int dim)
{
	if( dim<=0 )
		return 0;
	else if( dim == 1 )
		return matrix[0][0];
	else if( dim == 2 )
		return (matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]);
	else if( dim == 3 )
		return (matrix[0][0]*matrix[1][1]*matrix[2][2]+matrix[0][1]*matrix[1][2]*matrix[2][0]+matrix[0][2]*matrix[2][1]*matrix[1][0]
				-matrix[0][2]*matrix[1][1]*matrix[2][0]-matrix[0][1]*matrix[1][0]*matrix[2][2]-matrix[0][0]*matrix[2][1]*matrix[1][2]);
	// do something if 'else' happens
	double  d;
	int i;
	double **tmpMat = new double *[dim];
	for( i=0; i<dim; i++ )
	{
		tmpMat[i] = new double[dim];
		memcpy(tmpMat[i], matrix[i], sizeof(double)*dim);
	}

	ludcmp(tmpMat, dim,  &d); 
	for( i=0; i< dim; i++ ) 
		d  *= tmpMat[i][i];

	for( i=0; i<dim; i++ )
		delete [] tmpMat[i];
	delete [] tmpMat;
	return d;
}


double CalculateDeterminant(ForestMat* mat)
{
	assert(mat->rows == mat->cols);
	if (mat->rows != mat->cols)
		return 0.0;

	double** data2D = new double* [mat->rows];
	for (int i = 0; i < mat->rows; i++)
	{
		data2D[i] = new double[mat->rows];
		memcpy(data2D[i], &mat->data[i * mat->cols], sizeof(double) * mat->rows);
	}

	double det = CalculateDeterminant(data2D, mat->rows);

	for (int i = 0; i < mat->rows; i++)
		delete[] data2D[i];
	delete[] data2D;

	return det;
}

/******************************************************************
*函数名称：brinv(double a[],int n)
*函数类型：int
*参数说明：a--双精度实型数组，n--整型变量，方阵A的阶数
*函数功能：用全选主元Gauss-Jordan消去法求n阶实矩阵A的逆矩阵
******************************************************************/
int brinv(double a[], int n);
int brinv(ForestMat *mat)
{
	if (mat->cols != mat->rows)
		return -1;

	return brinv(mat->data, mat->rows);
}

int brinv(double a[], int n)
{
	int* is, * js, i, j, k, l, u, v;
	double d, p;
	is = new int[n];
	js = new int[n];
	for (k = 0; k <= n - 1; k++)
	{
		d = 0.0;
		for (i = k; i <= n - 1; i++)
			for (j = k; j <= n - 1; j++)
			{
				l = i * n + j; p = fabs(a[l]);
				if (p > d)
				{
					d = p; is[k] = i; js[k] = j;
				}
			}
		if (d + 1.0 == 1.0)
		{
			//free(is); free(js); printf("err**not inv\n");
			delete[] is;
			delete[] js;
			return -1;
		}
		if (is[k] != k)
			for (j = 0; j <= n - 1; j++)
			{
				u = k * n + j; v = is[k] * n + j;
				p = a[u]; a[u] = a[v]; a[v] = p;
			}
		if (js[k] != k)
			for (i = 0; i <= n - 1; i++)
			{
				u = i * n + k; v = i * n + js[k];
				p = a[u]; a[u] = a[v]; a[v] = p;
			}
		l = k * n + k;
		a[l] = 1.0 / a[l];
		for (j = 0; j <= n - 1; j++)
			if (j != k)
			{
				u = k * n + j; a[u] = a[u] * a[l];
			}
		for (i = 0; i <= n - 1; i++)
			if (i != k)
				for (j = 0; j <= n - 1; j++)
					if (j != k)
					{
						u = i * n + j;
						a[u] = a[u] - a[i * n + k] * a[k * n + j];
					}
		for (i = 0; i <= n - 1; i++)
			if (i != k)
			{
				u = i * n + k; a[u] = -a[u] * a[l];
			}
	}
	for (k = n - 1; k >= 0; k--)
	{
		if (js[k] != k)
			for (j = 0; j <= n - 1; j++)
			{
				u = k * n + j; v = js[k] * n + j;
				p = a[u]; a[u] = a[v]; a[v] = p;
			}
		if (is[k] != k)
			for (i = 0; i <= n - 1; i++)
			{
				u = i * n + k; v = i * n + is[k];
				p = a[u]; a[u] = a[v]; a[v] = p;
			}
	}
	delete [] is;
	delete [] js;
	return 1;
}

// A*B->Res
int matMul(ForestMat* A, ForestMat* B, ForestMat* Res)
{
	if (A->cols != B->rows || A->rows != Res->rows || B->cols != Res->cols)
		return -1;


	int i, j, k;
	const int r = Res->rows, c = Res->cols;

	memset(Res->data, 0, sizeof(Res->data[0]) * Res->rows * Res->cols);

	for (i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			for (k = 0; k < A->cols; k++)
				Res->data[i * c + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];

	return 1;
}

// A*B.T->Res
int matMul_T(ForestMat* A, ForestMat* B, ForestMat* Res)
{
	if (A->cols != B->cols || A->rows != Res->rows || B->rows != Res->cols)
		return -1;


	int i, j, k;
	const int r = Res->rows, c = Res->cols;

	memset(Res->data, 0, sizeof(Res->data[0]) * Res->rows * Res->cols);

	// A*B_T
	for (i = 0; i < r; i++)
		for (j = 0; j < c; j++)
			for (k = 0; k < A->cols; k++)
				Res->data[i * c + j] += A->data[i * A->cols + k] * B->data[j * B->cols + k]; // A的第i行*B的第j行

	return 1;
}

// A.T*A->Res
int A_TmulA(ForestMat* A, ForestMat* Res)
{
	if (A->cols != Res->rows || Res->rows != Res->cols)
		return  -1;
	const int c = A->cols;
	int i, j, k;

	memset(Res->data, 0, sizeof(Res->data[0]) * c * c);

	for (i = 0; i < c; i++)
		for (j = 0; j < c; j++)
			for (k = 0; k < A->rows; k++)
				Res->data[i * c + j] += A->data[k * c + i] * A->data[k * c + j]; // A的第i列.*A的第j列
	return 1;
}

// T = inv(X.T*X)*X.T*Y
int sovleLinearSquares(ForestMat* X, ForestMat* Y, ForestMat* T)
{
	int rv;
	ForestMat* tmpM1 = NULL, * tmpM2 = NULL;
	tmpM1 = createMat(X->cols, X->cols);
	// tmpM1.rows = X->cols;
	// tmpM1.cols = X->cols;
	// tmpM1.data = new double[tmpM1.rows*tmpM1.cols];

	A_TmulA(X, tmpM1);
	// 增加对行列式的判断，如果|tmpM1|很小，认为是无法获得准确的逆矩阵
	double det = CalculateDeterminant(tmpM1);
	if (det < 1e-5)
	{
		//cout<<det<<" "<<cc_det++<<endl;
		deleteMat(tmpM1);
		return -1;
	}
	rv = brinv(tmpM1->data, tmpM1->rows); // inv(X_T*X) -> tmpM1
	if (rv < 0)
	{
		deleteMat(tmpM1);
		return -1;
	}

	tmpM2 = createMat(X->cols, X->rows);
	// tmpM2.rows = X->cols;
	// tmpM2.cols = X->rows;
	// tmpM2.data = new double [tmpM2.rows * tmpM2.cols];
	// memset(tmpM2.data, 0, sizeof(tmpM2.data[0])*tmpM2.rows * tmpM2.cols);

	rv = matMul_T(tmpM1, X, tmpM2); // inv(X_T*X)*X_T -> tmpM2
	if (rv < 0)
	{
		deleteMat(tmpM1);
		deleteMat(tmpM2);
		return -1;
	}
	rv = matMul(tmpM2, Y, T); // inv(X_T*X)*X_T*Y
	if (rv < 0)
	{
		deleteMat(tmpM1);
		deleteMat(tmpM2);
		return -1;
	}

	deleteMat(tmpM1);
	deleteMat(tmpM2);

	return 1;
}


void CalculateCovMat(const float **target, const int *sample_index, const int arrivedNum,  const int variable_num_y, double **Cov)
{
	// double **Cov = new double *[variable_num_y];
	// for(int j=0; j<variable_num_y; j++ )
	// {
	// 	Cov[j] = new double [variable_num_y]; // covariance
	// 	memset(Cov[j], 0, sizeof(double)*variable_num_y);
	// }	
	double *Mean = new double [variable_num_y];
	memset(Mean, 0 , sizeof(Mean[0])*variable_num_y);
	int id = 0;
	for(int i=0; i<arrivedNum; i++ )
	{
		id = sample_index[i];
		for( int m=0; m<variable_num_y; m++ )
		{
			Mean[m] += target[id][m];
			for( int n=0; n<variable_num_y; n++ )
				Cov[m][n] += 1.0*target[id][m]*target[id][n];
		}
	}

	for( int m=0; m<variable_num_y; m++ )
				for( int n=0; n<variable_num_y; n++ )
					Cov[m][n] = Cov[m][n] - Mean[m]*Mean[n]/arrivedNum;

	delete [] Mean;
}

// 归一化方法1: 使用正态分布
void normalizeZScore(const float **data, const int sample_num, const int dim, float **data_norm)
{
	int i, j;
	float *mean = new float[dim];
	float *sigma = new float[dim];

	memset(mean, 0, sizeof(float)*dim);
	memset(sigma, 0, sizeof(float)*dim);

	for (i=0; i<sample_num; i++)
		memset(data_norm[i], 0, sizeof(float)*dim);

	for (i=0; i<sample_num; i++)
		for (j=0; j<dim; j++)
		{
			mean[j] += data[i][j]; // EX
			sigma[j] += data[i][j]*data[i][j]; // E(X^2)
		}

	for (j=0; j<dim; j++)
	{
		mean[j] = mean[j]/sample_num;
		sigma[j] = sigma[j]/sample_num - mean[j]*mean[j]; // E(X^2) - (E(X))^2
	}	

	for (i=0; i<sample_num; i++)
		for (j=0; j<dim; j++)
			data_norm[i][j] = (data[i][j] - mean[j]) / sqrt(sigma[j]);

	delete [] mean;
	delete [] sigma;
}

// 归一化方法2:数据缩放到[0,1]
void normalizeMinMax(const float **data, const int sample_num, const int dim,  float **data_norm)
{
	float *maxv = new float [dim];
	float *minv = new float[dim];
	memset(maxv, 0 , sizeof(float)*dim);
	memset(minv, 0, sizeof(float)*dim);

	int i, j;
	for (i = 0; i<dim; i++)
	{
		maxv[i] = data[0][i];
		minv[i] = data[0][i];
	}

	for (i=1; i<sample_num; i++)
		for (j=0; j<dim; j++)
		{
			if (data[i][j] > maxv[j])
				maxv[j] = data[i][j];
			if (data[i][j] < minv[j])
				minv[j] = data[i][j];
		}

	for (i=0; i<sample_num; i++)
		for (j=0; j<dim; j++)
		{
			data_norm[i][j] = (data[i][j] - minv[j])/(maxv[j] - minv[j]);
		}

	delete [] maxv;
	delete [] minv;
}

int ExtremeRandomlySplitOnRLoquatNode(float** data, int variables_num_x, int* innode_samples_index, int innode_num, int& split_variable_index, float& split_value)
{
	int j, index;
	float maxv = 0.f, minv = 0.f;
	int var_index, totalTryNum = variables_num_x * 2;
	if (totalTryNum < 10)
		totalTryNum = 20;
	// randomly select the variables(attribute) candidate choosing to split on the node
	srand(g_random_seed++);
	while (--totalTryNum)
	{
		var_index = rand_freebsd() % variables_num_x;
		index = innode_samples_index[0];
		maxv = data[index][var_index];
		minv = data[index][var_index];
		for (j = 1; j < innode_num; j++)
		{
			index = innode_samples_index[j];
			if (data[index][var_index] > maxv)
				maxv = data[index][var_index];
			else if (data[index][var_index] < minv)
				minv = data[index][var_index];
		}
		if (maxv - minv > VERY_SMALL_VALUE * 1e6)
		{
			split_variable_index = var_index;
			int s = rand_freebsd() % 100;
			split_value = minv + s / 100.f * (maxv - minv);// (maxv + minv) /2.f;
			break;
		}
	}
	//cout<<"total: "<<2*variables_num-totalTryNum<<" max:"<<maxv<<" min:"<<minv<<endl;
	if (totalTryNum == 0)
		return -1;
	else
		return 1;
}

typedef struct var_target {
	float var;
	float target;
}var_target;

typedef struct var_id {
	float var;
	int index;
}var_id;

int _cmp_r(const void* a, const void* b)
{
	return ((var_target*)a)->var > ((var_target*)b)->var ? 1 : -1;
}

// 采用类积分直方图的方式，在1维target(output)情况下加速
int _SplitOnRLoquatNodeCompletelySearchBySort1D(float** data, float** target, int variables_num_x, int variables_num_y,
										int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{
	assert(variables_num_y == 1);

	int i, j, index, rv = 1;
	int* selSplitIndex = new int[Mvariable];

	float splitv = 0;
	double gini_like, mingini = 1e38;
	int var_index;

	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for (i = 0; i < variables_num_x; i++)
		arrayindx.push_back(i);
	for (i = 0; i < Mvariable; i++)
	{
		int iid = rand_freebsd() % (variables_num_x - i);
		selSplitIndex[i] = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);
	}

	int lcount = 0, rcount = 0, lcount_best, rcount_best;
	double lCov = 0., rCov = 0.;
	double lMean = 0., rMean = 0.;

	double* targetCum = new double[innode_num];
	double* targetSqrCum = new double[innode_num];
	memset(targetCum, 0, sizeof(double) * innode_num);
	memset(targetSqrCum, 0, sizeof(double) * innode_num);
	var_target* vts = new var_target[innode_num];

	bool bfindSplitV = false;
	split_variable_index = -1;
	for (j = 0; j < Mvariable; j++)  // 对于每个被选中的属性
	{
		var_index = selSplitIndex[j];

		// 提取该维度上的值-target对
		for (int k = 0; k < innode_num; k++)
		{
			index = innode_samples_index[k];
			vts[k].var = data[index][var_index];
			vts[k].target = target[index][0];
		}

		// 排序
		qsort(vts, innode_num, sizeof(var_target), _cmp_r);

		// 计算累计直方图(需排序后)
		targetCum[0] = vts[0].target;
		targetSqrCum[0] = 1.0*vts[0].target * vts[0].target;
		for (int k = 1; k < innode_num; k++)
		{
			const double t = vts[k].target;
			targetCum[k] = targetCum[k - 1] + t;
			targetSqrCum[k] = targetSqrCum[k - 1] + t * t;
		}

		// 找最佳的split
		for (int k = 1; k < innode_num; k++)
		{
			if (abs(vts[k - 1].var - vts[k].var) < FLT_EPSILON)
			{
				continue;
			}

			splitv = 0.5f * (vts[k - 1].var + vts[k].var);
			lcount = k;
			rcount = innode_num - k;

			lMean = targetCum[k - 1] / lcount;
			rMean = (targetCum[innode_num - 1] - targetCum[k - 1]) / rcount;
			lCov = targetSqrCum[k - 1] / lcount - lMean * lMean;
			rCov = (targetSqrCum[innode_num - 1] - targetSqrCum[k - 1]) / rcount - rMean * rMean;
			/*
			if (lCov <= 0)
				lCov = 1e-38;
			if (rCov <= 0)
				rCov = 1e-38;

			assert(lCov >= 0 && rCov >= 0);
			gini_like = float(lcount / (double)innode_num * log(lCov) + rcount / (double)innode_num * log(rCov));
			*/
			gini_like = (lcount * lCov + rcount * rCov) / innode_num;

			if (gini_like < mingini)
			{
				bfindSplitV = true;
				mingini = gini_like;
				split_variable_index = var_index;
				split_value = splitv;
				// lcount_best = lcount;
				// rcount_best = rcount;
			}
		}

	}

	// cout << "infunction" << lcount_best << " " << rcount_best << endl;

	if (false == bfindSplitV)
	{
		/*int r = rand() % Mvariable;
		split_variable_index = selSplitIndex[r];
		int rand_index = (int)((rand_freebsd() * 1.0 / RAND_MAX_RF) * innode_num);
		rand_index = rand_index >= innode_num ? innode_num - 1 : rand_index;
		split_value = data[rand_index][split_variable_index];*/
		if (-1 == ExtremeRandomlySplitOnRLoquatNode(data, variables_num_x, innode_samples_index, innode_num, split_variable_index, split_value))
		{
			split_variable_index = -1;
			split_value = 0;
			rv = -1;

			/*if (innode_num > 20)
			{
				ofstream out("bb.txt");
				for (int n = 0; n < innode_num; n++)
				{
					const int index = innode_samples_index[n];
					out << index << " " << target[index][0] << "\t\t";
					for (int k = 0; k < variables_num_x; k++)
						out << data[index][k] << " ";
					out << endl;

				}
				out.close();
				int y = 1;
			}*/
			
		}
		else
			rv = 0;
	}

	delete[] selSplitIndex;
	delete[] vts;
	delete[] targetCum;
	delete[] targetSqrCum;

	return rv;
}

int _SplitOnRLoquatNodeCompletelySearchBySort2D(float** data, float** target, int variables_num_x, int variables_num_y,
	int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{
	assert(variables_num_y == 2);

	int i, j, index, rv = 1;
	int* selSplitIndex = new int[Mvariable];

	float splitv = 0;
	double gini_like, mingini = 1e38;
	int var_index;

	double a, b, c, d;

	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for (i = 0; i < variables_num_x; i++)
		arrayindx.push_back(i);
	for (i = 0; i < Mvariable; i++)
	{
		int iid = rand_freebsd() % (variables_num_x - i);
		selSplitIndex[i] = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);
	}

	int lcount = 0, rcount = 0, lcount_best, rcount_best;
	double lCov = 0., rCov = 0.;
	double lMean = 0., rMean = 0.;

	// 用于计算均值的积分"直方图"
	double* targetCum1 = new double[innode_num];
	double* targetCum2 = new double[innode_num];
	memset(targetCum1, 0, sizeof(double) * innode_num);
	memset(targetCum2, 0, sizeof(double) * innode_num);
	// 用于计算y*y_T的积分"直方图"
	double* covMatCum00 = new double[innode_num];
	double* covMatCum01 = new double[innode_num];
	double* covMatCum11 = new double[innode_num];
	memset(covMatCum00, 0, sizeof(double) * innode_num);
	memset(covMatCum01, 0, sizeof(double) * innode_num);
	memset(covMatCum11, 0, sizeof(double) * innode_num);

	var_id* vts = new var_id[innode_num];

	bool bfindSplitV = false;
	split_variable_index = -1;
	for (j = 0; j < Mvariable; j++)  // 对于每个被选中的属性
	{
		var_index = selSplitIndex[j];

		// 提取该维度上的值-target对
		for (int k = 0; k < innode_num; k++)
		{
			index = innode_samples_index[k];
			vts[k].var = data[index][var_index];
			vts[k].index = index;
		}

		// 排序
		qsort(vts, innode_num, sizeof(var_id), _cmp_r);

		// 计算累计直方图(需排序后)
		targetCum1[0] = target[vts[0].index][0];
		targetCum2[0] = target[vts[0].index][1];
		covMatCum00[0] = 1.0*target[vts[0].index][0] * target[vts[0].index][0];
		covMatCum01[0] = 1.0*target[vts[0].index][0] * target[vts[0].index][1];
		covMatCum11[0] = 1.0*target[vts[0].index][1] * target[vts[0].index][1];

		for (int k = 1; k < innode_num; k++)
		{
			const double t1 = target[vts[k].index][0];
			const double t2 = target[vts[k].index][1];
			targetCum1[k] = targetCum1[k - 1] + t1;
			targetCum2[k] = targetCum2[k - 1] + t2;
			covMatCum00[k] = covMatCum00[k - 1] + t1 * t1;
			covMatCum01[k] = covMatCum01[k - 1] + t1 * t2;
			covMatCum11[k] = covMatCum11[k - 1] + t2 * t2;
		}

		// 找最佳的split
		for (int k = 1; k < innode_num; k++)
		{
			if (abs(vts[k - 1].var - vts[k].var) < FLT_EPSILON)
			{
				continue;
			}

			splitv = 0.5f * (vts[k - 1].var + vts[k].var);
			lcount = k;
			rcount = innode_num - k;

			const int start = k - 1, end = innode_num - 1;
			// 计算2X2协方差矩阵的行列式
			a = covMatCum00[start] - targetCum1[start] * targetCum1[start] / lcount;
			b = covMatCum01[start] - targetCum1[start] * targetCum2[start] / lcount;
			c = b;
			d = covMatCum11[start] - targetCum2[start] * targetCum2[start] / lcount;
			lCov = (a * d - b * c) / lcount;

			// 计算2X2协方差矩阵的行列式
			a = (covMatCum00[end] - covMatCum00[start]) - (targetCum1[end] - targetCum1[start]) * (targetCum1[end] - targetCum1[start]) / rcount;
			b = (covMatCum01[end] - covMatCum01[start]) - (targetCum1[end] - targetCum1[start]) * (targetCum2[end] - targetCum2[start]) / rcount;
			c = b;
			d = (covMatCum11[end] - covMatCum11[start]) - (targetCum2[end] - targetCum2[start]) * (targetCum2[end] - targetCum2[start]) / rcount;
			rCov = (a * d - b * c) / rcount;

			gini_like = (lcount * lCov + rcount * rCov) / innode_num;

			if (gini_like < mingini)
			{
				bfindSplitV = true;
				mingini = gini_like;
				split_variable_index = var_index;
				split_value = splitv;
				lcount_best = lcount;
				rcount_best = rcount;
			}
		}

	}

	if (false == bfindSplitV)
	{
		/*int r = rand() % Mvariable;
		split_variable_index = selSplitIndex[r];
		int rand_index = (int)((rand_freebsd() * 1.0 / RAND_MAX_RF) * innode_num);
		rand_index = rand_index >= innode_num ? innode_num - 1 : rand_index;
		split_value = data[rand_index][split_variable_index];*/
		if (-1 == ExtremeRandomlySplitOnRLoquatNode(data, variables_num_x, innode_samples_index, innode_num, split_variable_index, split_value))
		{
			split_variable_index = -1;
			split_value = 0;
			rv = -1;

			/*ofstream out("bb.txt");
			for (int n = 0; n < innode_num; n++)
			{
				const int index = innode_samples_index[n];
				out<<index<<" " << target[index][0] << " " << target[index][1] << " ";
				for (int k = 0; k < variables_num_x; k++)
					cout << data[index][k] << " ";
				cout << endl;

			}
			out.close();
			int y = 1;*/
		}
		else
			rv = 0;
	}

	//rv = (bfindSplitV == false ? -1 : 0);

	delete[] selSplitIndex;
	delete[] vts;
	delete[] targetCum1;
	delete[] targetCum2;
	delete[] covMatCum00;
	delete[] covMatCum01;
	delete[] covMatCum11;

	return rv;
}

int _SplitOnRLoquatNodeCompletelySearch(float** data, float** target, int variables_num_x, int variables_num_y,
										int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{

	int i, j, m, n, index, rv = 1;
	int* selSplitIndex = new int[Mvariable];

	float splitv = 0;
	double gini_like, mingini = 1e38;
	int var_index;

	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for (i = 0; i < variables_num_x; i++)
		arrayindx.push_back(i);
	for (i = 0; i < Mvariable; i++)
	{
		int iid = rand_freebsd() % (variables_num_x - i);
		selSplitIndex[i] = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);
	}

	int lcount = 0, rcount = 0;
	double detl, detr;
	double** lCov = new double* [variables_num_y];
	double** rCov = new double* [variables_num_y];
	for (j = 0; j < variables_num_y; j++)
	{
		lCov[j] = new double[variables_num_y]; // covariance
		rCov[j] = new double[variables_num_y];
	}
	double* lMean = new double[variables_num_y];
	double* rMean = new double[variables_num_y];

	bool bfindSplitV = false;
	split_variable_index = -1;
	for (j = 0; j < Mvariable; j++)  // 对于每个被选中的属性
	{
		var_index = selSplitIndex[j];

		for (i = 0; i < innode_num; i++)
		{
			splitv = data[innode_samples_index[i]][var_index];

			for (int t = 0; t < variables_num_y; t++)
			{
				memset(lCov[t], 0, sizeof(double) * variables_num_y);
				memset(rCov[t], 0, sizeof(double) * variables_num_y);
			}
			memset(lMean, 0, sizeof(double) * variables_num_y);
			memset(rMean, 0, sizeof(double) * variables_num_y);
			lcount = 0;	rcount = 0;

			for (int k = 0; k < innode_num; k++)
			{
				index = innode_samples_index[k];
				if (data[index][var_index] <= splitv)
				{
					lcount++;
					for (m = 0; m < variables_num_y; m++)
					{
						lMean[m] += target[index][m];
						for (n = 0; n < variables_num_y; n++)
							lCov[m][n] += 1.0*target[index][m] * target[index][n];
					}
				}
				else
				{
					rcount++;
					for (m = 0; m < variables_num_y; m++)
					{
						rMean[m] += target[index][m];
						for (n = 0; n < variables_num_y; n++)
							rCov[m][n] += 1.0*target[index][m] * target[index][n];
					}
				}
			}

			int ld = (lcount == 0) ? 1 : lcount;
			int rd = (rcount == 0) ? 1 : rcount;
			for (m = 0; m < variables_num_y; m++)
				for (n = 0; n < variables_num_y; n++)
				{
					lCov[m][n] = (lCov[m][n] - lMean[m] * lMean[n] / ld)/ld; // 0513 增加 /ld
					rCov[m][n] = (rCov[m][n] - rMean[m] * rMean[n] / rd)/rd; // 0513 增加 /rd
				}

			detl = variables_num_y == 1 ? lCov[0][0] : CalculateDeterminant(lCov, variables_num_y);
			detr = variables_num_y == 1 ? rCov[0][0] : CalculateDeterminant(rCov, variables_num_y);

			/*
			if (detl <= 0) // 协方差矩阵半正定
				detl = 1e-20;
			if (detr <= 0)
				detr = 1e-20;

			assert(lcount + rcount == innode_num);
			gini_like = float(lcount / (double)innode_num * log(detl) + rcount / (double)innode_num * log(detr));// lcount+rcount == innode_num
			*/

			gini_like = (lcount  * detl + rcount * detr)/ innode_num; // lcount+rcount == innode_num

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
		/*int r = rand() % Mvariable;
		split_variable_index = selSplitIndex[r];
		int rand_index = (int)((rand_freebsd() * 1.0 / RAND_MAX_RF) * innode_num);
		rand_index = rand_index >= innode_num ? innode_num - 1 : rand_index;
		split_value = data[rand_index][split_variable_index];*/
		// 0608
		if (-1 == ExtremeRandomlySplitOnRLoquatNode(data, variables_num_x, innode_samples_index, innode_num, split_variable_index, split_value))
		{
			split_variable_index = -1;
			split_value = 0;
			rv = -1;
		}
		else
			rv = 0;
	}

	//rv = (bfindSplitV == false ? -1 : 0);

	delete[] selSplitIndex;
	for (j = 0; j < variables_num_y; j++)
	{
		delete[] lCov[j];
		delete[] rCov[j];
	}
	delete[] lCov;
	delete[] rCov;
	delete[] lMean;
	delete[] rMean;

	return rv;
}


int SplitOnRLoquatNodeCompletelySearch(float** data, float** target, int variables_num_x, int variables_num_y,
										int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{

	int rv = -1;
	switch (variables_num_y)
	{
	case 1:
		rv = _SplitOnRLoquatNodeCompletelySearchBySort1D(data, target, variables_num_x, variables_num_y, innode_samples_index, innode_num, Mvariable, split_variable_index, split_value);
		break;
	case 2:
		rv = _SplitOnRLoquatNodeCompletelySearchBySort2D(data, target, variables_num_x, variables_num_y, innode_samples_index, innode_num, Mvariable, split_variable_index, split_value);
		break;

	default:
		rv = _SplitOnRLoquatNodeCompletelySearch(data, target, variables_num_x, variables_num_y, innode_samples_index, innode_num, Mvariable, split_variable_index, split_value);
		break;
	}

	return rv;
}

int _SplitOnRLoquatNode(float **data, float **target, int variables_num_x, int variables_num_y, 
					   int *innode_samples_index, int innode_num, int Mvariable, int &split_variable_index, float &split_value)
{
	int i, j, m, n, index, rv = 1;
	int *selSplitIndex = new int[Mvariable];
	float *maxv=NULL, *minv=NULL, *step=NULL, *itoa=NULL;
	float splitv = 0;
	double gini_like, mingini = 1e38;
	maxv = new float[Mvariable];
	minv = new float[Mvariable];
	step = new float[Mvariable];
	itoa = new float[Mvariable];
	int var_index;

	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for( i=0; i<variables_num_x; i++ )
		arrayindx.push_back(i);
	for( i=0; i<Mvariable; i++ )
	{
		int iid = rand_freebsd()%(variables_num_x-i);
		selSplitIndex[i] = arrayindx[iid];
		arrayindx.erase(arrayindx.begin()+iid);
	}

	for( i=0; i<Mvariable; i++ )
	{
		var_index = selSplitIndex[i];
		index = innode_samples_index[0];
		maxv[i] = data[index][var_index];
		minv[i] = data[index][var_index];
		for( j=1; j<innode_num; j++ )
		{
			index = innode_samples_index[j];
			if ( data[index][var_index] > maxv[i] )
				maxv[i] = data[index][var_index];
			else if( data[index][var_index] < minv[i] )
				minv[i] = data[index][var_index];
		}
		step[i] = (maxv[i]-minv[i])/INTERVAL_STEPS_NUM;
		itoa[i] = step[i]/100.0f;
	}

	int lcount=0, rcount=0;
	double detl, detr;
	double **lCov = new double *[variables_num_y];
	double **rCov = new double *[variables_num_y];
	for( j=0; j<variables_num_y; j++ )
	{
		lCov[j] = new double [variables_num_y]; // covariance
		rCov[j] = new double [variables_num_y];
	}
	double *lMean = new double [variables_num_y];
	double *rMean = new double [variables_num_y];

	bool bfindSplitV = false;
	split_variable_index = -1;
	for( j=0; j<Mvariable; j++ )  // 对于每个被选中的属性
	{
		var_index = selSplitIndex[j];

		if( step[j] < FLT_EPSILON )	
			continue; // step[j] == 0

		//bfindSplitV = true;

		//for( splitv=minv[j]-itoa[j]; splitv<maxv[j]+itoa[j]; splitv += step[j] ) 
		int counter = 0;
		//splitv = minv[j] - itoa[j];
		splitv = minv[j];
		while (counter <= INTERVAL_STEPS_NUM + 1)
		{
			if (counter == INTERVAL_STEPS_NUM + 1)
			{
				// 当minv 与 maxv差距很小时 splitv+= step可能会因为float精度问题不变化，
				// 此问题在Superconductivty_train数据集出现
				// 因此使用一次均值来避免这个问题
				splitv = (minv[j] + maxv[j]) * 0.5f;
			}

			for (int t=0; t<variables_num_y; t++)
			{
				memset(lCov[t], 0, sizeof(double)*variables_num_y);
				memset(rCov[t], 0, sizeof(double)*variables_num_y);
			}
			memset(lMean, 0, sizeof(double)*variables_num_y);
			memset(rMean, 0, sizeof(double)*variables_num_y);
			lcount = 0;	rcount =0;
			//
// 			ofstream lrecord("left.txt");
// 			ofstream rrecord("right.txt");

			for( i=0; i<innode_num; i++ )
			{
				index = innode_samples_index[i];
				if( data[index][var_index] <= splitv )
				{
					//
// 					for( int g=0; g<variables_num_y; g++ )
// 						lrecord<<target[index][g]<<" ";
// 					lrecord<<endl;

					lcount++;
					for( m=0; m<variables_num_y; m++ )
					{
						lMean[m] += target[index][m];
						for( n=0; n<variables_num_y; n++ )
							lCov[m][n] += 1.0*target[index][m]*target[index][n];
					}
				}
				else
				{
					//
// 					for( int g=0; g<variables_num_y; g++ )
// 						rrecord<<target[index][g]<<" ";
// 					rrecord<<endl;

					rcount++;
					for( m=0; m<variables_num_y; m++ )
					{
						rMean[m] += target[index][m];
						for( n=0; n<variables_num_y; n++ )
							rCov[m][n] += 1.0*target[index][m]*target[index][n];
					}
				}

			}

			// COV = ∑XX_T - n*X_hat*X_hat_T
			// now lCov,rCov = ∑XX_T, lMean,rMean = n*X_hat;
// 			lcount = (lcount == 0) ? 1 : lcount;
// 			rcount = (rcount == 0) ? 1 : rcount; // 计数为0,则d*d维cov中每个元素为0,令count=1,能保证以下代码正常运行获得正确值
			int ld = (lcount == 0) ? 1 : lcount;
			int rd = (rcount == 0) ? 1 : rcount;
			for( m=0; m<variables_num_y; m++ )
				for( n=0; n<variables_num_y; n++ )
				{
					lCov[m][n] = (lCov[m][n] - lMean[m]*lMean[n]/ld)/ld;  // 0513增加 /ld
					rCov[m][n] = (rCov[m][n] - rMean[m]*rMean[n]/rd)/rd;  // 0513增加 /rd
				}
			
			/*detl = CalculateDeterminant(lCov, variables_num_y);
			detr = CalculateDeterminant(rCov, variables_num_y);*/
			detl = variables_num_y == 1 ? lCov[0][0] : CalculateDeterminant(lCov, variables_num_y);
			detr = variables_num_y == 1 ? rCov[0][0] : CalculateDeterminant(rCov, variables_num_y);
			
			/*
			if( detl <= 0 ) // 协方差矩阵半正定
				detl = 1e-20;
			if( detr <= 0 )
				detr = 1e-20;

			gini_like = float(lcount/(double)innode_num*log(detl)+rcount/(double)innode_num*log(detr));// lcount+rcount == innode_num
			*/

			gini_like = (lcount  * detl + rcount  * detr) / innode_num;

			if( gini_like < mingini )
			{
				bfindSplitV = true;
				mingini = gini_like;
				split_variable_index = var_index;
				split_value = splitv;
			}

			++counter;
			splitv += step[j];
		}
	}

	if( bfindSplitV == false ) // 如果所有被选择分量的maxv==minv
	{
		if ( -1 == ExtremeRandomlySplitOnRLoquatNode(data,  variables_num_x, innode_samples_index, innode_num, split_variable_index, split_value) )
        {   
            split_variable_index = -1;
            split_value = 0;
            rv = -1;
        }else
            rv = 0;
	}

	delete [] selSplitIndex;
	delete [] maxv;
	delete [] minv;
	delete [] step;
	delete [] itoa;

	for( j=0; j<variables_num_y; j++ )
	{
		delete [] lCov[j];
		delete [] rCov[j];
	}
	delete [] lCov;
	delete [] rCov;
	delete [] lMean;
	delete [] rMean;

	return rv;
}

int _SplitOnRLoquatNode1D(float** data, float** target, int variables_num_x, int variables_num_y,
	int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{
	int i, j, index, order, rv = 1;
	int lc_best = 0, rc_best = 0;
	int* selSplitIndex = new int[Mvariable];
	float* maxv = NULL, * minv = NULL, * step = NULL, * itoa = NULL;
	float splitv = 0;
	double gini_like, mingini = 1e38;
	maxv = new float[Mvariable];
	minv = new float[Mvariable];
	step = new float[Mvariable];
	itoa = new float[Mvariable];
	int var_index;

	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for (i = 0; i < variables_num_x; i++)
		arrayindx.push_back(i);
	for (i = 0; i < Mvariable; i++)
	{
		int iid = rand_freebsd() % (variables_num_x - i);
		selSplitIndex[i] = arrayindx[iid];
		arrayindx.erase(arrayindx.begin() + iid);
	}

	for (i = 0; i < Mvariable; i++)
	{
		var_index = selSplitIndex[i];
		index = innode_samples_index[0];
		maxv[i] = data[index][var_index];
		minv[i] = data[index][var_index];
		for (j = 1; j < innode_num; j++)
		{
			index = innode_samples_index[j];
			if (data[index][var_index] > maxv[i])
				maxv[i] = data[index][var_index];
			else if (data[index][var_index] < minv[i])
				minv[i] = data[index][var_index];
		}
		step[i] = (maxv[i] - minv[i]) / INTERVAL_STEPS_NUM;
		itoa[i] = step[i] / 100.0f;
	}

	int lcount = 0, rcount = 0;
	double lVar, rVar, lMean, rMean;

	bool bfindSplitV = false;
	split_variable_index = -1;
	for (j = 0; j < Mvariable; j++)  // 对于每个被选中的属性
	{
		var_index = selSplitIndex[j];

		if (step[j] < FLT_EPSILON)	
			continue; // step[j] == 0

		int counter = 0;
		//splitv = minv[j] - itoa[j];
		splitv = minv[j];
		while (counter <= INTERVAL_STEPS_NUM + 1)
		{
			if (counter == INTERVAL_STEPS_NUM + 1)
			{
				// 当minv 与 maxv差距很小时 splitv+= step可能会因为float精度问题不变化，
				// 此问题在Superconductivty_train数据集出现
				// 因此使用一次均值来避免这个问题
				splitv = (minv[j] + maxv[j]) * 0.5f;
			}

			lVar = rVar = lMean = rMean = 0.0;

			lcount = 0;	rcount = 0;
			//
// 			ofstream lrecord("left.txt");
// 			ofstream rrecord("right.txt");

			for (i = 0; i < innode_num; i++)
			{
				index = innode_samples_index[i];
				if (data[index][var_index] <= splitv)
				{
					//
// 					for( int g=0; g<variables_num_y; g++ )
// 						lrecord<<target[index][g]<<" ";
// 					lrecord<<endl;

					lcount++;
					lMean += target[index][0];
					lVar += 1.0*target[index][0] * target[index][0];
				}
				else
				{
					//
// 					for( int g=0; g<variables_num_y; g++ )
// 						rrecord<<target[index][g]<<" ";
// 					rrecord<<endl;

					rcount++;
					// for( m=0; m<variables_num_y; m++ )
					// {
					// 	rMean[m] += target[index][m];
					// 	for( n=0; n<variables_num_y; n++ )
					// 		rCov[m][n] += target[index][m]*target[index][n];
					// }
					rMean += target[index][0];
					rVar += 1.0*target[index][0] * target[index][0];
				}

			}

			// COV = ∑XX_T - n*X_hat*X_hat_T
			// now lCov,rCov = ∑XX_T, lMean,rMean = n*X_hat;
// 			lcount = (lcount == 0) ? 1 : lcount;
// 			rcount = (rcount == 0) ? 1 : rcount; // 计数为0,则d*d维cov中每个元素为0,令count=1,能保证以下代码正常运行获得正确值
			int ld = (lcount == 0) ? 1 : lcount;
			int rd = (rcount == 0) ? 1 : rcount;
			
			lVar = (lVar - lMean * lMean / ld) / ld; // 0513 增加/ld
			rVar = (rVar - rMean * rMean / rd) / rd; // 0513 增加/rd

			gini_like = (lcount  * lVar + rcount * rVar) / innode_num; // lcount+rcount == innode_num

			if (gini_like < mingini)
			{
				bfindSplitV = true;
				mingini = gini_like;
				split_variable_index = var_index;
				split_value = splitv;
				order = j;
				lc_best = lcount;
				rc_best = rcount;
			}

			++counter;
			splitv += step[j];
		}
	}

	if (bfindSplitV == false) // 如果所有被选择分量的maxv==minv
	{
		/*int r = rand() % Mvariable;
		split_variable_index = selSplitIndex[r];
		split_value = (maxv[r] + minv[r]) / 2.f;
		rv = -1;*/
		if (-1 == ExtremeRandomlySplitOnRLoquatNode(data, variables_num_x, innode_samples_index, innode_num, split_variable_index, split_value))
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
			split_value = (maxv[order] + minv[order]) * 0.5f;
		}
	} 

	delete[] selSplitIndex;
	delete[] maxv;
	delete[] minv;
	delete[] step;
	delete[] itoa;

	return rv;
}

int SplitOnRLoquatNode(float** data, float** target, int variables_num_x, int variables_num_y,
			int* innode_samples_index, int innode_num, int Mvariable, int& split_variable_index, float& split_value)
{
	int rv;
	switch (variables_num_y)
	{
	case 1:
		rv = _SplitOnRLoquatNode1D(data, target, variables_num_x, variables_num_y, innode_samples_index, innode_num, Mvariable, split_variable_index, split_value);
		break;
	default:
		rv = _SplitOnRLoquatNode(data, target, variables_num_x, variables_num_y, innode_samples_index, innode_num, Mvariable, split_variable_index, split_value);
		break;
	}
	return rv;
}

int SplitOnRLoquateNodeExtremeRandomly(float **data, float **target, int variables_num_x, int variables_num_y, 
									   int *innode_samples_index, int innode_num ,int Mvariable, int &split_variable_index, float &split_value)
{
	int i, j, m, n, index, rv = 1;
	int *selSplitIndex = new int[Mvariable];
	float *maxv=NULL, *minv=NULL;
	float *splitv_cand = new float[Mvariable];
	double gini_like, mingini=1e38;
	maxv = new float[Mvariable];
	minv = new float[Mvariable];
	int var_index;

	srand(g_random_seed++);
	// randomly select the variables(attribute) candidate choosing to split on the node
	vector<int> arrayindx;
	for( i=0; i<variables_num_x; i++ )
		arrayindx.push_back(i);
	for( i=0; i<Mvariable; i++ )
	{
		int iid = rand_freebsd()%(variables_num_x-i);
		selSplitIndex[i] = arrayindx[iid];
		arrayindx.erase(arrayindx.begin()+iid);
	}

	for( i=0; i<Mvariable; i++ )
	{
		var_index = selSplitIndex[i];
		index = innode_samples_index[0];
		maxv[i] = data[index][var_index];
		minv[i] = data[index][var_index];
		for( j=1; j<innode_num; j++ )
		{
			index = innode_samples_index[j];
			if ( data[index][var_index] > maxv[i] )
				maxv[i] = data[index][var_index];
			else if( data[index][var_index] < minv[i] )
				minv[i] = data[index][var_index];
		}
		splitv_cand[i] = ((float)rand_freebsd())/RAND_MAX_RF*(maxv[i]-minv[i])+minv[i]; //(maxv[i]-minv[i])/INTERVAL_STEPS_NUM;
	}

	int lcount=0, rcount=0;
	double detl, detr;
	double **lCov = new double *[variables_num_y];
	double **rCov = new double *[variables_num_y];
	for( j=0; j<variables_num_y; j++ )
	{
		lCov[j] = new double [variables_num_y]; // covariance
		rCov[j] = new double [variables_num_y];
	}
	double *lMean = new double [variables_num_y];
	double *rMean = new double [variables_num_y];

	bool bfindSplitV = false;
	for( j=0; j<Mvariable; j++ )  // 对于每个被选中的属性
	{
		if (maxv[j] - minv[j] < FLT_EPSILON)
			continue;

		var_index = selSplitIndex[j];

		//bfindSplitV = true;

		for (int t=0; t<variables_num_y; t++)
		{
			memset(lCov[t], 0, sizeof(double)*variables_num_y); // 对二维数组的初始化是否合适?
			memset(rCov[t], 0, sizeof(double)*variables_num_y);
		}
		memset(lMean, 0, sizeof(double)*variables_num_y);
		memset(rMean, 0, sizeof(double)*variables_num_y);

		lcount = 0;	
		rcount =0;

		for( i=0; i<innode_num; i++ )
		{
			index = innode_samples_index[i];
			
			if( data[index][var_index] <= splitv_cand[j] )
			{
				lcount++;
				for( m=0; m<variables_num_y; m++ )
				{
					lMean[m] += target[index][m];
					for( n=0; n<variables_num_y; n++ )
						lCov[m][n] += 1.0*target[index][m]*target[index][n];
				}
			}
			else
			{
				rcount++;
				for( m=0; m<variables_num_y; m++ )
				{
					rMean[m] += target[index][m];
					for( n=0; n<variables_num_y; n++ )
						rCov[m][n] += 1.0*target[index][m]*target[index][n];
				}
			}

		}

		// COV = ∑XX_T - n*X_hat*X_hat_T
		// now lCov,rCov = ∑XX_T, lMean,rMean = n*X_hat;
// 		lcount = (lcount==0) ? 1 : lcount;
// 		rcount = (rcount==0) ? 1 : rcount;
		int ld = (lcount==0) ? 1 : lcount;
		int rd = (rcount==0) ? 1 : rcount;
		for( m=0; m<variables_num_y; m++ )
			for( n=0; n<variables_num_y; n++ )
			{
				lCov[m][n] = (lCov[m][n] - lMean[m]*lMean[n]/ld)/ld; // 0513 增加 /ld
				rCov[m][n] = (rCov[m][n] - rMean[m]*rMean[n]/rd)/rd; // 0513 增加 /rd
			}

			detl = variables_num_y == 1 ? lCov[0][0] : CalculateDeterminant(lCov, variables_num_y) + 1e-38; // 以防止det为0使log(0)发生
			detr = variables_num_y == 1 ? rCov[0][0] : CalculateDeterminant(rCov, variables_num_y) + 1e-38;

			gini_like = (lcount*detl+rcount*detr)/innode_num;

			if( gini_like < mingini )
			{
				bfindSplitV = true; // 0607
				mingini = gini_like;
				split_variable_index = var_index;
				split_value = splitv_cand[j];
			}
	
	}

	if (false == bfindSplitV) //0608
	{
		if (-1 == ExtremeRandomlySplitOnRLoquatNode(data, variables_num_x, innode_samples_index, innode_num, split_variable_index, split_value))
		{
			split_variable_index = -1;
			split_value = 0;
			rv = -1;
		}
		else
			rv = 0;
	}

	delete [] selSplitIndex;
	delete [] maxv;
	delete [] minv;
	delete [] splitv_cand;

	for( j=0; j<variables_num_y; j++ )
	{
		delete [] lCov[j];
		delete [] rCov[j];
	}
	delete [] lCov;
	delete [] rCov;
	delete [] lMean;
	delete [] rMean;

	return rv;
}


void CalculateLeafNodeInformation(float **data, float **target, int *sample_index, int arrivedNum, int variable_num_x, int variable_num_y, PredictionModel predictionModel, LeafNodeInfo *&pLeafNodeInfo)
{
	assert(arrivedNum > 0);

	if( pLeafNodeInfo != NULL )
	{
		if( pLeafNodeInfo->CovMatOfArrived != NULL )
		{
			for( int k=0; k<variable_num_y; k++ )
				delete [] pLeafNodeInfo->CovMatOfArrived[k];
			delete [] pLeafNodeInfo->CovMatOfArrived;
		}
		delete [] pLeafNodeInfo->MeanOfArrived;
		delete pLeafNodeInfo;
	}

	int i, j, k, index;
	pLeafNodeInfo = new LeafNodeInfo;
	pLeafNodeInfo->linearPredictor = NULL;
	pLeafNodeInfo->CovMatOfArrived = new double *[variable_num_y];
	for( i=0; i<variable_num_y; i++ )
	{
		pLeafNodeInfo->CovMatOfArrived[i] = new double [variable_num_y];
		memset(pLeafNodeInfo->CovMatOfArrived[i], 0, sizeof(double)*variable_num_y);
	}
	pLeafNodeInfo->MeanOfArrived = new float [variable_num_y];
	memset(pLeafNodeInfo->MeanOfArrived, 0, sizeof(float)*variable_num_y);

	for( i=0; i<arrivedNum; i++ )
	{
		index = sample_index[i];
		for( j=0; j<variable_num_y; j++ )
		{
			pLeafNodeInfo->MeanOfArrived[j] += target[index][j];
			for( k=0; k<variable_num_y; k++ )
				pLeafNodeInfo->CovMatOfArrived[j][k] += 1.0*target[index][j]*target[index][k];
		}
	}


	for( j=0; j<variable_num_y; j++ )
		pLeafNodeInfo->MeanOfArrived[j] /= (float)arrivedNum;
	for( j=0; j<variable_num_y; j++ )
		for( k=0; k<variable_num_y; k++ )
			pLeafNodeInfo->CovMatOfArrived[j][k] -= 1.0*arrivedNum * pLeafNodeInfo->MeanOfArrived[j] * pLeafNodeInfo->MeanOfArrived[k];	


	// 2021-04-20
	if (PredictionModel::linear == predictionModel)  // 如果使用最小二乘拟合节点样本进行预测
	{
		if (arrivedNum >= 2 * (variable_num_x + 1))
		{
			ForestMat* X = createMat(arrivedNum, variable_num_x + 1);
			ForestMat* Y = createMat(arrivedNum, variable_num_y);
			ForestMat* Y_predict = createMat(arrivedNum, variable_num_y);
			ForestMat* T = createMat(variable_num_x + 1, variable_num_y);
			for (int n = 0; n < arrivedNum; n++)
			{
				for (int m = 0; m < variable_num_x; m++)
					X->data[n * (variable_num_x + 1) + m] = data[sample_index[n]][m];
				X->data[n * (variable_num_x + 1) + variable_num_x] = 1.0;

				for (int m = 0; m < variable_num_y; m++)
					Y->data[n * variable_num_y + m] = target[sample_index[n]][m];
			}
			int rv = sovleLinearSquares(X, Y, T);
			if (rv > 0)
			{
				pLeafNodeInfo->linearPredictor = new double[T->rows * T->cols];
				memcpy(pLeafNodeInfo->linearPredictor, T->data, sizeof(double) * T->rows * T->cols);

				// //////////////////////
				matMul(X, T, Y_predict);
				double mse_linear = 0.0;
				double mse_constant = 0.0;
				for (int n = 0; n < arrivedNum; n++)
					for (int m = 0; m < variable_num_y; m++)
					{
						const int index = n * variable_num_y + m;
						mse_linear += (Y_predict->data[index] - Y->data[index]) * (Y_predict->data[index] - Y->data[index]);
						mse_constant += (Y->data[index] - pLeafNodeInfo->MeanOfArrived[m]) * (Y->data[index] - pLeafNodeInfo->MeanOfArrived[m]);
					}

				if (mse_linear > mse_constant) {
					
					delete[] pLeafNodeInfo->linearPredictor;
					pLeafNodeInfo->linearPredictor = NULL;

					//test
					// ofstream os("mse.txt");
					// for (int kk=0; kk<arrivedNum; kk++)
					// {
					// 	os<<target[sample_index[kk]][0]<<",";
					// 	for (int mm=0; mm<variable_num_x; mm++)
					// 	{
					// 		os<<data[sample_index[kk]][mm]<<",";
					// 	}
					// 	os<<endl;
					// }
					// os.close();
				}
				/////////////////////
			}
			else
			{
				delete[] pLeafNodeInfo->linearPredictor;
				pLeafNodeInfo->linearPredictor = NULL;
				/*ofstream os("test.txt");
				for (int kk = 0; kk < arrivedNum; kk++)
				{
					os << sample_index[kk] << ",";
					os << target[sample_index[kk]][0] << ",";
					for (int mm = 0; mm < variable_num_x; mm++)
					{
						os << data[sample_index[kk]][mm] << ",";
					}
					os << endl;
				}
				os.close();*/
			}



			deleteMat(X);
			deleteMat(Y);
			deleteMat(T);
			deleteMat(Y_predict);
		}
		else
		{
			pLeafNodeInfo->linearPredictor = NULL;
		}
	}
}

int ClearAllocatedMemoryDuringRTraining(struct LoquatRTreeNode* treeNode)
{
	if (NULL == treeNode)
	{
		return 0;
	}
	// clear left subnode
	ClearAllocatedMemoryDuringRTraining(treeNode->pSubNode[0]);
	// clear right subnode
	ClearAllocatedMemoryDuringRTraining(treeNode->pSubNode[1]);
	// clear  this node
	delete[] treeNode->samples_index;
	treeNode->samples_index = NULL;
	treeNode->arrival_samples_num = 0;
	return 1;
}

int ClearAllocatedMemoryDuringRTraining(struct LoquatRTreeStruct* loquatTree)
{
	if (NULL == loquatTree)
	{
		return 0;
	}
	return ClearAllocatedMemoryDuringRTraining(loquatTree->rootNode);
}


// 2021-04-07

struct _GrowNodeInput
{
	int total_samples_num;
	int samples_num_of_tree; // 这棵树的inbag样本数（bootstrap不放回采样的数量）
	int total_variables_num_x;
	int total_variables_num_y;
	int mvariables;

	int leafMinSamples;
	int parent_depth;
	int maxDepth;
	PredictionModel predictionModel;
	int randomness;
};
typedef struct _GrowNodeInput GrowNodeInput;

// target是1维的情况下，计算target的方差
double calculateVariance(float** target, int* sample_index, int sample_num)
{
	if (0 == sample_num)
		return 0.0;

	int i;
	double mean = 0.0;
	double sigma = 0.0;
	double t;

	for (i = 0; i < sample_num; i++)
	{
		t = target[sample_index[i]][0];
		mean += t; // EX
		sigma += t*t; // E(X^2)
	}

	mean = mean / sample_num;
	sigma = sigma / sample_num - mean * mean; // E(X^2) - (E(X))^2

	return sigma;
}
bool isLowVariance(float** target, int variables_num, int* sample_index, int sample_num)
{
	double var = 0.0;
	if (1 == variables_num)
		var = calculateVariance(target, sample_index, sample_num);
	else
		var = 1e10; // TODO: 多维情况下怎么判断?

	return var < 1e-7 ? true : false;
}

inline bool isConstant(const int* sample_index, int sample_num)
{
	if (sample_num == 0)
		return true;
	int index = sample_index[0];
	for (int i = 1; i < sample_num; i++)
	{
		if (index != sample_index[i])
			return false;
	}
	return true;
}

struct LoquatRTreeNode* GrowLoquatRTreeNodeRecursively(float** data, float** target, int*& sample_arrival_index, int arrival_num, const GrowNodeInput* pInputParam, struct LoquatRTreeStruct* loquatTree)
{

	int total_samples_num = pInputParam->total_samples_num;
	int total_variables_num_x = pInputParam->total_variables_num_x;
	int total_variables_num_y = pInputParam->total_variables_num_y;
	int index;
	struct LoquatRTreeNode* treeNode = new struct LoquatRTreeNode;
	assert(NULL != treeNode);
	treeNode->pLeafNodeInfo = NULL;
	treeNode->pParentNode = NULL;
	treeNode->pSubNode = NULL;
	treeNode->samples_index = NULL;

	//treeNode->depth = pInputParam->thisnode_depth; //!!
	treeNode->depth = pInputParam->parent_depth + 1;

	if (treeNode->depth == 0)
		treeNode->nodetype = enRootNode;
	else
		treeNode->nodetype = enLinkNode;//!!暂时在创建时是linknode

	treeNode->subnodes_num = 2;
	treeNode->pParentNode = NULL;
	treeNode->pSubNode = new struct LoquatRTreeNode* [2];
	treeNode->pSubNode[0] = NULL;
	treeNode->pSubNode[1] = NULL;
	treeNode->split_value = 0.0f;
	treeNode->split_variable_index = -1;
	treeNode->train_impurity = 0;
	treeNode->pLeafNodeInfo = NULL;
	treeNode->arrival_samples_num = arrival_num;
	treeNode->samples_index = new int[arrival_num];
	assert(NULL != treeNode->samples_index);
	//memcpy_s(treeNode->samples_index, arrival_num*sizeof(int), sample_arrival_index, arrival_num*sizeof(int));
	memcpy(treeNode->samples_index, sample_arrival_index, arrival_num * sizeof(int));
	if (treeNode->depth != 0) // treeNode->depth==0 为rootnode，它对应的是inbag_sample空间不应该被清除
	{
		delete[] sample_arrival_index; // 在这里清除可以避免递归使分配的空间
		sample_arrival_index = NULL;
	}

	// 以上用到达样本生成一个新节点，以下开始判断这个节点是否可以再分裂
	bool term = false;
	bool isFewSamples = (treeNode->arrival_samples_num <= pInputParam->leafMinSamples);
	bool isMaxDepth = (treeNode->depth == pInputParam->maxDepth - 1);
	bool isLowVar = isLowVariance(target, total_variables_num_y, treeNode->samples_index, treeNode->arrival_samples_num);
	term = isFewSamples || isMaxDepth || isLowVar;
	
	bool isConstantIndex = false;
	if (!term && treeNode->arrival_samples_num <= 2*pInputParam->leafMinSamples)
	{
		isConstantIndex = isConstant(treeNode->samples_index, treeNode->arrival_samples_num);
		term = term || isConstantIndex;
	}
	
	// TODO: 还需要根据covmat的值确定是否还需分裂
	if (term)
	{
		treeNode->nodetype = enLeafNode;
		treeNode->pSubNode[0] = NULL;
		treeNode->pSubNode[1] = NULL;
		loquatTree->leaf_node_num++;
		CalculateLeafNodeInformation(data, target, treeNode->samples_index, treeNode->arrival_samples_num, 
										total_variables_num_x, total_variables_num_y, 
										isLowVar ? PredictionModel::constant : pInputParam->predictionModel, 
										treeNode->pLeafNodeInfo /*OUT*/);
		treeNode->pLeafNodeInfo->arrivedRatio = treeNode->arrival_samples_num / (float)pInputParam->samples_num_of_tree;
		treeNode->pLeafNodeInfo->dimension = total_variables_num_y;
	}
	else
	{
		bool bSplit = true;
		switch (pInputParam->randomness)
		{
		case TREE_RANDOMNESS_WEAK:
			
			SplitOnRLoquatNodeCompletelySearch(data, target, total_variables_num_x, total_variables_num_y, treeNode->samples_index, treeNode->arrival_samples_num,
				pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
			break;
		case TREE_RANDOMNESS_MODERATE:
			SplitOnRLoquatNode(data, target, total_variables_num_x, total_variables_num_y, treeNode->samples_index, treeNode->arrival_samples_num,
				pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
			break;
		case TREE_RANDOMNESS_STRONG:
			SplitOnRLoquateNodeExtremeRandomly(data, target, total_variables_num_x, total_variables_num_y, treeNode->samples_index, treeNode->arrival_samples_num,
				pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
			break;
		default:
			SplitOnRLoquatNode(data, target, total_variables_num_x, total_variables_num_y, treeNode->samples_index, treeNode->arrival_samples_num,
				pInputParam->mvariables, treeNode->split_variable_index, treeNode->split_value);
		}

		float splitv = treeNode->split_value;
		int split_index = treeNode->split_variable_index;
		int leftsubnode_samples_num = 0, rightsubnode_samples_num = 0;
		int* subnode_samples_queue = new int[treeNode->arrival_samples_num];
		assert(subnode_samples_queue);

		if (treeNode->split_variable_index >= 0)
		{
			for (int kk = 0; kk < treeNode->arrival_samples_num; kk++)
			{
				index = treeNode->samples_index[kk];
				if (data[index][treeNode->split_variable_index] <= treeNode->split_value) // 从上个节点划分样本到左右子枝
					subnode_samples_queue[leftsubnode_samples_num++] = index;
				else
					subnode_samples_queue[treeNode->arrival_samples_num - 1 - rightsubnode_samples_num++] = index;
			}
			assert(leftsubnode_samples_num + rightsubnode_samples_num == treeNode->arrival_samples_num);
		}

		//assert(leftsubnode_samples_num + rightsubnode_samples_num == treeNode->arrival_samples_num);
		//cout << "left: " << leftsubnode_samples_num << " right: " << rightsubnode_samples_num << endl;
		// 在非常少的情况下分裂一枝的样本个数为0,
		// 很有可能是因为到达这个节点的样本的属性都一样
		// 连ExtremelRandomSplit都没成功
		if (0 == leftsubnode_samples_num || 0 == rightsubnode_samples_num)
		{
			treeNode->nodetype = enLeafNode;
			loquatTree->leaf_node_num++;
			treeNode->split_value = 0;
			treeNode->split_variable_index = -1;
			treeNode->pSubNode[0] = NULL;
			treeNode->pSubNode[1] = NULL;

			CalculateLeafNodeInformation(data, target, treeNode->samples_index, treeNode->arrival_samples_num,total_variables_num_x, total_variables_num_y, pInputParam->predictionModel, treeNode->pLeafNodeInfo);
			treeNode->pLeafNodeInfo->arrivedRatio = treeNode->arrival_samples_num / (float)pInputParam->samples_num_of_tree;
			treeNode->pLeafNodeInfo->dimension = total_variables_num_y;

			delete[] subnode_samples_queue;//!!!!!!!!!!!!!!!!!!!
			return treeNode;
		}


		int* leftsubnode_index = NULL, * rightsubnode_index = NULL;
		leftsubnode_index = new int[leftsubnode_samples_num];
		rightsubnode_index = new int[rightsubnode_samples_num];
		//memcpy_s(leftsubnode_index, leftsubnode_samples_num*sizeof(int), subnode_samples_queue, leftsubnode_samples_num*sizeof(int));
		memcpy(leftsubnode_index, subnode_samples_queue, leftsubnode_samples_num * sizeof(int));
		//memcpy_s(rightsubnode_index, rightsubnode_samples_num*sizeof(int), subnode_samples_queue+leftsubnode_samples_num, rightsubnode_samples_num*sizeof(int));
		memcpy(rightsubnode_index, subnode_samples_queue + leftsubnode_samples_num, rightsubnode_samples_num * sizeof(int));
		delete[] subnode_samples_queue;

		GrowNodeInput input = *pInputParam;
		input.parent_depth = treeNode->depth;// error happened here 调试了很久!
		if (treeNode->depth + 1 > loquatTree->depth) //到了这一步后，肯定会往下生长两个节点，所以要判断树的总深度是否增加
			loquatTree->depth = treeNode->depth + 1; // 更新整棵树的当前最大深度

		// recursively call the function 
		treeNode->pSubNode[0] = GrowLoquatRTreeNodeRecursively(data, target,
														leftsubnode_index, // 这个变量指向的空间将会在此函数中被释放
														leftsubnode_samples_num,
														&input,
														loquatTree);

		treeNode->pSubNode[1] = GrowLoquatRTreeNodeRecursively(data, target,
														rightsubnode_index,
														rightsubnode_samples_num,
														&input,
														loquatTree);

		//if( rightsubnode_index!=NULL ) delete [] rightsubnode_index;// 在递归调用GrowXXXNodeXXXsively函数中空间被释放
		//if( leftsubnode_index!=NULL ) delete  [] leftsubnode_index; // 假设没有被释放则在这里释放
		treeNode->pSubNode[0]->pParentNode = treeNode;
		treeNode->pSubNode[1]->pParentNode = treeNode;

	}

	return treeNode;
}

// 2021-04-07
int GrowRandomizedRLoquatTreeRecursively(float** data, float** target, const RandomRForests_info RFinfo, struct LoquatRTreeStruct*& loquatTree)
{
	if (loquatTree != NULL)
		return -2;

	int k, j, depth = 0;
	int index = 0;
	unsigned int maxNodeNumThisDepth = 0;
	int maxTreeDepth = RFinfo.maxdepth;

	float ratio = 1.0f;
	int selnum = (int)(RFinfo.datainfo.samples_num * ratio + 0.5);
	int total_samples_num = RFinfo.datainfo.samples_num;
	int variables_num_x = RFinfo.datainfo.variables_num_x;
	int variables_num_y = RFinfo.datainfo.variables_num_y;

	loquatTree = new struct LoquatRTreeStruct;
	assert(loquatTree != NULL);
	loquatTree->inbag_samples_index = NULL;
	loquatTree->outofbag_samples_index = NULL;
	loquatTree->rootNode = NULL;

	loquatTree->depth = 0;
	loquatTree->leaf_node_num = 0;
	loquatTree->inbag_samples_num = selnum;
	loquatTree->inbag_samples_index = new int[selnum]; // 有重复的！
	assert(NULL != loquatTree->inbag_samples_index);

	// int  leafMinSamples = (int)(loquatTree->inbag_samples_num * STOP_CRITERION_NUM_RATIO_R + 0.5);
	// if (leafMinSamples < 5)
	// 	leafMinSamples = 5;

	// (1) Resampling training samples (bootstrap training samples)
	bool* inbagmask = new bool[total_samples_num];
	assert(NULL != inbagmask);
	int inbagcount = 0;   // inbagcount 不包括重叠的计数
	memset(inbagmask, false, total_samples_num * sizeof(bool));

	srand_freebsd(g_random_seed++);

	for (j = 0; j < selnum; j++)
	{
		//index = rand()%total_samples_num;
		index = (int)((rand_freebsd() * 1.0 / RAND_MAX_RF) * total_samples_num + 0.5);
		if (index < 0)
			index = 0;
		else if (index >= total_samples_num)
			index = total_samples_num - 1;
		loquatTree->inbag_samples_index[j] = index;  // resampling from original data with replacement
		if (inbagmask[index] == false)
		{
			inbagmask[index] = true;
			inbagcount++;
		}
	}

	loquatTree->outofbag_samples_num = total_samples_num - inbagcount;
	loquatTree->outofbag_samples_index = new int[loquatTree->outofbag_samples_num];
	assert(NULL != loquatTree->outofbag_samples_index);

	for (k = 0, j = 0; k < total_samples_num; k++)
	{
		if (inbagmask[k] == false)
			loquatTree->outofbag_samples_index[j++] = k;
	}
	delete[]inbagmask;

	// (2) Build the entire tree from the root node.
	GrowNodeInput inputParam = { RFinfo.datainfo.samples_num,
							  selnum,  /*samples_num_of_tree*/
							  RFinfo.datainfo.variables_num_x,
							  RFinfo.datainfo.variables_num_y,
							  RFinfo.mvariables,
							  RFinfo.minsamplessplit,
							  -1/*parent depth of the root node*/,
							  RFinfo.maxdepth,
							  RFinfo.predictionModel,
							  RFinfo.randomness};
	loquatTree->rootNode = GrowLoquatRTreeNodeRecursively(data, target,
		loquatTree->inbag_samples_index,
		loquatTree->inbag_samples_num,
		&inputParam, loquatTree);
	loquatTree->rootNode->pParentNode = NULL;

	ClearAllocatedMemoryDuringRTraining(loquatTree);

	return 1;
}

//int GrowRandomizedRLoquatTree(float **data, float **target, RandomRForests_info RFinfo, struct LoquatRTreeStruct *&loquatTree, bool bExtremeTree)
//{
//	if( loquatTree != NULL )
//		return -2;
//
//	unsigned int i;
//	int k, j, depth=0;
//	int index =0;
//	unsigned int maxNodeNumThisDepth=0;
//	int maxTreeDepth = RFinfo.maxdepth;
//
//	float ratio = RFinfo.selratio;
//	int selnum = (int)(RFinfo.datainfo.samples_num * ratio +0.5);
//	int total_samples_num   = RFinfo.datainfo.samples_num;
//	int variables_num_x = RFinfo.datainfo.variables_num_x;
//	int variables_num_y  = RFinfo.datainfo.variables_num_y;
//
//	// add 0603
//	int leftsubnode_samples_num=0, rightsubnode_samples_num=0;
//	struct LoquatRTreeNode **pPreNode = NULL;
//	struct LoquatRTreeNode **pCurNode = NULL;
//
//	loquatTree = new struct LoquatRTreeStruct; 
//	assert( loquatTree != NULL );
//	loquatTree->inbag_samples_index = NULL;
//	loquatTree->outofbag_samples_index = NULL;
//	loquatTree->rootNode = NULL;
//
//	loquatTree->depth = 0;
//	loquatTree->leaf_node_num = 0;
//	loquatTree->inbag_samples_num = selnum;
//	loquatTree->inbag_samples_index = new int [selnum]; // 有重复的！
//	assert( NULL != loquatTree->inbag_samples_index );
//
//	int  leafMinSamples = (int)( loquatTree->inbag_samples_num * STOP_CRITERION_NUM_RATIO_R + 0.5 );
//	if( leafMinSamples < 5 )
//		leafMinSamples = 5;
//
//	// (1) Resampling training samples (bootstrap training samples)
//	bool *inbagmask = new bool[total_samples_num];
//	assert( NULL != inbagmask );
//	int inbagcount = 0;   // inbagcount 不包括重叠的计数
//	memset(inbagmask, false, total_samples_num*sizeof(bool));
//
//	srand(g_random_seed++);
//
//	for( j=0; j<selnum; j++ )
//	{
//		index = rand()%total_samples_num;
//		if( index<0 )
//			index = 0;
//		else if( index >= total_samples_num )
//			index = total_samples_num-1;
//		loquatTree->inbag_samples_index[j] = index;  // resampling from original data with replacement
//		if( inbagmask[index] == false )
//		{
//			inbagmask[index] = true;
//			inbagcount++;
//		}
//	}
//
//	loquatTree->outofbag_samples_num = total_samples_num - inbagcount;
//	loquatTree->outofbag_samples_index = new int[loquatTree->outofbag_samples_num];
//	assert( NULL != loquatTree->outofbag_samples_index );
//
//	for( k=0,j=0; k<total_samples_num; k++ )
//	{
//		if( inbagmask[k] == false )
//			loquatTree->outofbag_samples_index[j++] = k;
//	}
//	delete []inbagmask;
//
//	// (2) Build the root node
//	struct LoquatRTreeNode *rootNode = new struct LoquatRTreeNode;
//	assert( NULL != rootNode );
//
//	rootNode->depth = 0;
//	rootNode->nodetype = enRootNode;
//	rootNode->subnodes_num = 2;
//	rootNode->pParentNode = NULL;
//	rootNode->pSubNode = new struct LoquatRTreeNode *[2];
//	rootNode->pSubNode[0] = NULL;	rootNode->pSubNode[1] = NULL;
//	rootNode->split_value = 0.0f;
//	rootNode->split_variable_index = -1;
//	rootNode->train_impurity = 0;
//	rootNode->arrival_samples_num = selnum;
//	rootNode->samples_index = new int [selnum];
//	assert( NULL != rootNode->samples_index );
//	//memcpy_s(rootNode->samples_index, selnum*sizeof(int), loquatTree->inbag_samples_index, selnum*sizeof(int));
//	memcpy(rootNode->samples_index, loquatTree->inbag_samples_index, selnum*sizeof(int));
//	rootNode->pLeafNodeInfo = NULL;
//
//	loquatTree->rootNode = rootNode; // rootNode has been grown
//
//	// if rootnode satisfies the stopping criterion then jump to end, else to split it.
//	if( rootNode->arrival_samples_num <=  leafMinSamples )
//	{
//		cout<<"root node = leaf node"<<endl;
//		rootNode->nodetype = enLeafNode;
//		rootNode->pSubNode[0] = NULL;
//		rootNode->pSubNode[1] = NULL;
//		loquatTree->leaf_node_num++;
//		CalculateLeafNodeInformation(data, target, rootNode->samples_index, rootNode->arrival_samples_num, variables_num_x, variables_num_y, RFinfo.predictionModel, rootNode->pLeafNodeInfo);
//		rootNode->pLeafNodeInfo->arrivedRatio = 1.f;
//		rootNode->pLeafNodeInfo->dimension = variables_num_y;
//		goto CLEAN_WORK;
//	}
//	else
//	{
//		if( false == bExtremeTree )
//		{
//			SplitOnRLoquatNode(data, target, variables_num_x, variables_num_y, rootNode->samples_index, rootNode->arrival_samples_num,
//				               RFinfo.mvariables, rootNode->split_variable_index, rootNode->split_value);
//		}
//		else
//		{
//			SplitOnRLoquateNodeExtremeRandomly(data, target, variables_num_x, variables_num_y, rootNode->samples_index, rootNode->arrival_samples_num,
//				               RFinfo.mvariables, rootNode->split_variable_index, rootNode->split_value);
//		}
//	}
//
//	/////////////////////////////////////////
//
//	//int leftsubnode_samples_num=0, rightsubnode_samples_num=0;
//	leftsubnode_samples_num=0;
//	rightsubnode_samples_num=0;
//	depth = 0;
//	maxNodeNumThisDepth = (int)powf(2.f, (float)depth);
//	//struct LoquatRTreeNode **pPreNode = new struct LoquatRTreeNode *[maxNodeNumThisDepth]; // 这只是指针数组, 注意要释放！！！
//	pPreNode = new struct LoquatRTreeNode *[maxNodeNumThisDepth];
//	pPreNode[0] = rootNode;
//	//struct LoquatRTreeNode **pCurNode = NULL;
//	pCurNode = NULL;
//
//	// (3) Build the left tree.
//	while (1)
//	{
//		depth++;
//		maxNodeNumThisDepth = (int)powf(2.f, (float)depth);
//		pCurNode = new struct LoquatRTreeNode *[maxNodeNumThisDepth];
//		bool bSplit = false;
//
//		for ( i=0; i<maxNodeNumThisDepth/2; i++ ) // maxNodeNumThisDepth/2:上一层的个数
//		{
//			if( pPreNode[i] == NULL ) // 这里的if-else顺序不能错！！！
//			{
//				pCurNode[i*2] = NULL;
//				pCurNode[i*2+1] = NULL;
//				continue;
//			}
//			else if( pPreNode[i]->nodetype == enLeafNode )
//			{
//				pCurNode[i*2] = NULL;
//				pCurNode[i*2+1] = NULL;
//
//				pPreNode[i]->pSubNode[0] = NULL;
//				pPreNode[i]->pSubNode[1] = NULL;
//
//				continue;
//			}
//
//
//			float splitv = pPreNode[i]->split_value;
//			int split_index = pPreNode[i]->split_variable_index;
//			int pre_samples_num = pPreNode[i]->arrival_samples_num;
//			int *subnode_samples_queue = new int[pre_samples_num]; // subnode_samples_queue 前一段保存了leftnodeindex, 后一段rightnodeindex
//			assert( NULL != subnode_samples_queue );
//			int leftsubnode_samples_num=0, rightsubnode_samples_num=0;
//
//			for ( int kk=0; kk < pre_samples_num; kk++ )
//			{
//				index = pPreNode[i]->samples_index[kk];
//				if( data[index][split_index] <= splitv ) // 从上个节点划分样本到左右子枝
//					subnode_samples_queue[leftsubnode_samples_num++] = index;
//				else
//				{
//					subnode_samples_queue[pre_samples_num-1-rightsubnode_samples_num ] = index;
//					rightsubnode_samples_num++;
//				}
//			}
//
//			// 在非常少的情况下分裂一枝的样本个数为0,
//			// 很有可能是因为到达这个节点的样本的属性都一样
//			// 连ExtremelRandomSplit都没成功
//			if( 0 == leftsubnode_samples_num || 0 == rightsubnode_samples_num )
//			{
//				//cout<<"No fortune"<<endl;
//				pPreNode[i]->nodetype = enLeafNode;
//				loquatTree->leaf_node_num++;
//				pPreNode[i]->split_value = 0;
//				pPreNode[i]->split_variable_index = -1;
//				pPreNode[i]->pSubNode[0] = NULL;
//				pPreNode[i]->pSubNode[1] = NULL;
//				pCurNode[i*2] = NULL;
//				pCurNode[i*2+1 ] = NULL;
//				CalculateLeafNodeInformation(data, target, pPreNode[i]->samples_index, pPreNode[i]->arrival_samples_num, variables_num_x, variables_num_y, RFinfo.predictionModel, pPreNode[i]->pLeafNodeInfo);
//				pPreNode[i]->pLeafNodeInfo->arrivedRatio = pPreNode[i]->arrival_samples_num/(float)rootNode->arrival_samples_num;
//				pPreNode[i]->pLeafNodeInfo->dimension = variables_num_y;
//				continue;
//			}
//
//			struct LoquatRTreeNode *leftLinkNode = new struct LoquatRTreeNode;
//			assert( NULL != leftLinkNode );
//			struct LoquatRTreeNode *rightLinkNode = new struct LoquatRTreeNode;
//			assert( NULL != rightLinkNode );
//
//			leftLinkNode->depth = depth;				rightLinkNode->depth = depth;
//			leftLinkNode->nodetype = enLinkNode;		rightLinkNode->nodetype = enLinkNode;
//			leftLinkNode->train_impurity = 1;			rightLinkNode->train_impurity = 1;
//			leftLinkNode->arrival_samples_num = leftsubnode_samples_num;
//			leftLinkNode->samples_index = new int[leftsubnode_samples_num];
//			assert( NULL != leftLinkNode->samples_index );
//			rightLinkNode->arrival_samples_num = rightsubnode_samples_num;
//			rightLinkNode->samples_index = new int[rightsubnode_samples_num];
//			assert( NULL != rightLinkNode->samples_index );
//			leftLinkNode->pParentNode  = pPreNode[i];
//			rightLinkNode->pParentNode = pPreNode[i];
//			leftLinkNode->subnodes_num = 2;			rightLinkNode->subnodes_num = 2;
//			leftLinkNode->pSubNode = new struct LoquatRTreeNode *[2];
//			leftLinkNode->pSubNode[0] = NULL; 
//			leftLinkNode->pSubNode[1] = NULL;
//			rightLinkNode->pSubNode = new struct LoquatRTreeNode *[2];
//			rightLinkNode->pSubNode[0] = NULL; 
//			rightLinkNode->pSubNode[1] = NULL;
//			leftLinkNode->split_value = 0.0f;		rightLinkNode->split_value = 0.0f;
//			leftLinkNode->split_variable_index = -1;	rightLinkNode->split_variable_index = -1;
//			leftLinkNode->pLeafNodeInfo = NULL;		rightLinkNode->pLeafNodeInfo = NULL;
//			//memcpy_s(leftLinkNode->samples_index, leftsubnode_samples_num*sizeof(int), subnode_samples_queue, leftsubnode_samples_num*sizeof(int));
//			memcpy(leftLinkNode->samples_index, subnode_samples_queue, leftsubnode_samples_num*sizeof(int));
//			//memcpy_s(rightLinkNode->samples_index, rightsubnode_samples_num*sizeof(int), subnode_samples_queue+leftsubnode_samples_num, rightsubnode_samples_num*sizeof(int));
//			memcpy(rightLinkNode->samples_index, subnode_samples_queue+leftsubnode_samples_num, rightsubnode_samples_num*sizeof(int));
//			delete []subnode_samples_queue;
//
//			bool rule1 = (leftLinkNode->arrival_samples_num <=  leafMinSamples);
//			bool rule2 = (depth == maxTreeDepth - 1);
//
//			if( rule1 || rule2 )
//			{
//				leftLinkNode->nodetype = enLeafNode;
//				leftLinkNode->pSubNode[0] = NULL;
//				leftLinkNode->pSubNode[1] = NULL;
//				loquatTree->leaf_node_num++;
//				CalculateLeafNodeInformation(data, target, leftLinkNode->samples_index, leftLinkNode->arrival_samples_num, variables_num_x, variables_num_y, RFinfo.predictionModel, leftLinkNode->pLeafNodeInfo);
//				leftLinkNode->pLeafNodeInfo->arrivedRatio = leftLinkNode->arrival_samples_num/(float)rootNode->arrival_samples_num;
//				leftLinkNode->pLeafNodeInfo->dimension = variables_num_y;
//				// test code
//// 				double det = CalculateDeterminant(leftLinkNode->pLeafNodeInfo->CovMatOfArrived, variables_num_y);
//// 				if( det<0 )
//// 					int xxx=0;
//			}
//			else
//			{
//				bSplit = true;
//				if( false == bExtremeTree )
//				{
//					SplitOnRLoquatNode(data, target, variables_num_x, variables_num_y, leftLinkNode->samples_index, leftLinkNode->arrival_samples_num,
//						RFinfo.mvariables, leftLinkNode->split_variable_index, leftLinkNode->split_value);
//				}
//				else
//				{
//					SplitOnRLoquateNodeExtremeRandomly(data, target, variables_num_x, variables_num_y, leftLinkNode->samples_index, leftLinkNode->arrival_samples_num,
//						RFinfo.mvariables, leftLinkNode->split_variable_index, leftLinkNode->split_value);
//				}
//			}
//
//			rule1 = (rightLinkNode->arrival_samples_num <=  leafMinSamples);
//			rule2 = (depth == maxTreeDepth -1);
//			
//			if( rule1 || rule2  )
//			{
//				rightLinkNode->nodetype = enLeafNode;
//				rightLinkNode->pSubNode[0] = NULL;
//				rightLinkNode->pSubNode[1] = NULL;
//				loquatTree->leaf_node_num++;
//				CalculateLeafNodeInformation(data, target, rightLinkNode->samples_index, rightLinkNode->arrival_samples_num, variables_num_x, variables_num_y, RFinfo.predictionModel, rightLinkNode->pLeafNodeInfo);
//				rightLinkNode->pLeafNodeInfo->arrivedRatio = rightLinkNode->arrival_samples_num/(float)rootNode->arrival_samples_num;
//				rightLinkNode->pLeafNodeInfo->dimension = variables_num_y;
//			}
//			else
//			{
//				bSplit = true;
//				if( false == bExtremeTree )
//				{
//					SplitOnRLoquatNode(data, target, variables_num_x, variables_num_y, rightLinkNode->samples_index, rightLinkNode->arrival_samples_num,
//						RFinfo.mvariables, rightLinkNode->split_variable_index, rightLinkNode->split_value);
//				}
//				else
//				{
//					SplitOnRLoquateNodeExtremeRandomly(data, target, variables_num_x, variables_num_y, rightLinkNode->samples_index, rightLinkNode->arrival_samples_num,
//						RFinfo.mvariables, rightLinkNode->split_variable_index, rightLinkNode->split_value);
//				}
//			}
//
//			pPreNode[i]->pSubNode[0] = leftLinkNode;
//			pPreNode[i]->pSubNode[1] = rightLinkNode;
//			pCurNode[i*2] = leftLinkNode;
//			pCurNode[i*2+1] = rightLinkNode;
//		}
//
//		delete [] pPreNode;
//		pPreNode = new struct LoquatRTreeNode *[maxNodeNumThisDepth];
//		for ( unsigned int kk=0; kk<maxNodeNumThisDepth; kk++ )
//			pPreNode[kk] = pCurNode[kk];
//		delete [] pCurNode;
//
//		if (  depth >= maxTreeDepth - 1 )
//		{
//			loquatTree->depth++;
//			delete [] pPreNode;
//			break;
//		}
//		if( bSplit == false )
//		{
//			loquatTree->depth++;
//			delete [] pPreNode;
//			break;
//		}
//		else if( bSplit == true )
//			loquatTree->depth++;
//	}
//
//	
//CLEAN_WORK:
//	ClearAllocatedMemoryDuringRTraining(loquatTree);
//
//	return 1;
//}

const struct LoquatRTreeNode *GetArrivedLeafNode(LoquatRForest *RF, int tree_index, float *data)
{
	int total_tree_num = RF->RFinfo.ntrees;
	int variables_num_x = RF->RFinfo.datainfo.variables_num_x;
	if( tree_index < 0 && tree_index >= total_tree_num )
		return NULL;

	int max_depth_index = RF->loquatTrees[tree_index]->depth, cc=0;
	struct LoquatRTreeNode *pNode = RF->loquatTrees[tree_index]->rootNode;
	int test_variables_index;
	float test_splitv;

	while(1)
	{
		if( pNode == NULL )
			return NULL;

		if( pNode->nodetype == enLeafNode )
			return pNode;

		test_variables_index = pNode->split_variable_index;
		test_splitv = pNode->split_value;

		if( test_variables_index >= variables_num_x )
			return NULL;

		if( data[test_variables_index] <= test_splitv )
			pNode = pNode->pSubNode[0]; // 左枝
		else
			pNode = pNode->pSubNode[1]; // 右枝

		if( (++cc) > max_depth_index )
			break;
	}

	return NULL;
}

int TrainRandomForestRegressor(float **data, float **target, RandomRForests_info RFinfo, LoquatRForest *&loquatForest, bool bTargetNormalize /* = true*/, int trace)
{
	GenerateSeedFromSysTime();

	if( loquatForest != NULL )
		return -2;

	int rv = CheckRegressionForestParameters(RFinfo);
	switch (rv)
	{
	case -1:
		cout<<">>>>The data_info structure is assigned with incorrect values."<<endl;
		return -3;
	case 0:
		cout<<">>>>Some incorrectly assigned values are found, and default or recommended values are assigned to them."<<endl;
		break;
	case 1:
		break;
	}

	int i, Ntrees;
	loquatForest = new LoquatRForest;
	assert( loquatForest != NULL );
	loquatForest->loquatTrees = NULL;
	loquatForest->bTargetNormalize = false;
	loquatForest->offset = NULL;
	loquatForest->scale = NULL;
	loquatForest->RFinfo = RFinfo;
	Ntrees = loquatForest->RFinfo.ntrees;
	loquatForest->loquatTrees = new struct LoquatRTreeStruct *[Ntrees];
	for( i=0; i<Ntrees; i++ )
	{
		loquatForest->loquatTrees[i] = NULL;
	}

	float **target_inner = NULL;
	if( bTargetNormalize == true && RFinfo.datainfo.variables_num_y > 1 ) // 如果响应是1维的，那么即使用户要求归一化也不归一化
	{
		int i,j;
		loquatForest->bTargetNormalize = true;
		int samples_num = RFinfo.datainfo.samples_num;
		int variables_num_y = RFinfo.datainfo.variables_num_y;
		
		loquatForest->offset = new float[variables_num_y];
		loquatForest->scale = new float[variables_num_y];

		float *maxv = new float[variables_num_y]; // pre
		float *minv = new float[variables_num_y];
		memcpy(maxv, target[0], sizeof(float)*variables_num_y);
		memcpy(minv, target[0], sizeof(float)*variables_num_y);

		for( i=0; i<samples_num; i++ )
			for( j=0; j<variables_num_y; j++ )
			{
				if( target[i][j] > maxv[j] )
					maxv[j] = target[i][j];
				if( target[i][j] < minv[j] )
					minv[j] = target[i][j];
			}

		
		for( j=0; j<variables_num_y; j++ )
		{
			if (maxv[j] - minv[j] < FLT_EPSILON)
			{
				loquatForest->offset[j] = 0.0f;
				loquatForest->scale[j] = 1.0f;
			}
			else
			{
				loquatForest->offset[j] = -minv[j] / (float)(maxv[j] - minv[j]);
				loquatForest->scale[j] = 1.f / (maxv[j] - minv[j]);
			}
			
		}

		target_inner = new float *[samples_num];
		for( i=0; i<samples_num; i++ )
			target_inner[i] = new float [variables_num_y];
		for( i=0; i<samples_num; i++ )
			for( j=0; j<variables_num_y; j++ )
				target_inner[i][j] = target[i][j]*loquatForest->scale[j] + loquatForest->offset[j]; // normalize
		
		delete [] minv;
		delete [] maxv;
	}

	rv = 1;
	const int ntrees = loquatForest->RFinfo.ntrees;
	for ( i=0; i<Ntrees; i++ )
	{

		float** tgt = loquatForest->bTargetNormalize == false ? target : target_inner;
		rv = GrowRandomizedRLoquatTreeRecursively(data, tgt, RFinfo, loquatForest->loquatTrees[i]);

		if (trace > 0 && (i + 1) % trace == 0)
		{
			float ooberror = 0;
			loquatForest->RFinfo.ntrees = i + 1;
			float* mse = NULL;
			MSEOnOutOfBagSamples(data, tgt, loquatForest, mse);
			cout << "Tree: " << i + 1 << "\tOOB mse: ";
			for (int m = 0; m < loquatForest->RFinfo.datainfo.variables_num_y; m++)
				cout << mse[m] << " ";
			cout << endl;
			loquatForest->RFinfo.ntrees = ntrees;
			delete[] mse;
		}

		if (1 != rv)
		{
			rv = -1;
			break;
		}
			
	}

	if( loquatForest->bTargetNormalize == true )
	{
		for( i=0; i<RFinfo.datainfo.samples_num; i++ )
			delete []target_inner[i];
		delete [] target_inner;
	}

	return rv;
}

int HarvestOneLeafNode(struct LoquatRTreeNode **treeNode)
{
	if( (*treeNode) == NULL )
		return 1;

	if( (*treeNode)->samples_index != NULL )
	{
		delete [] (*treeNode)->samples_index;
		(*treeNode)->samples_index = NULL;
		(*treeNode)->arrival_samples_num = 0;
	}

	if( (*treeNode)->pSubNode != NULL )
	{
		delete [] (*treeNode)->pSubNode;
		(*treeNode)->pSubNode = NULL;
	}

	if( (*treeNode)->nodetype == enLeafNode && (*treeNode)->pLeafNodeInfo )
	{
		if( (*treeNode)->pLeafNodeInfo->CovMatOfArrived )
		{
			int dim = (*treeNode)->pLeafNodeInfo->dimension;
			for( int i=0; i<dim; i++ )
				delete [] (*treeNode)->pLeafNodeInfo->CovMatOfArrived[i];
			delete [] (*treeNode)->pLeafNodeInfo->CovMatOfArrived;
		}
		if( (*treeNode)->pLeafNodeInfo->MeanOfArrived )
			delete [] (*treeNode)->pLeafNodeInfo->MeanOfArrived;
		if ((*treeNode)->pLeafNodeInfo->linearPredictor)
			delete [] (*treeNode)->pLeafNodeInfo->linearPredictor;
		delete (*treeNode)->pLeafNodeInfo; //0519!!!
	}

	delete *treeNode;
	*treeNode = NULL;

	return 1;
}

//void VisitAndHarvestNodes_PostOrder(struct LoquatRTreeNode **pNode)
//{
//	if( (*pNode) == NULL )
//		return;
//
//	if( (*pNode)->pSubNode[0] != NULL || (*pNode)->pSubNode[1] != NULL )
//	{
//		VisitAndHarvestNodes_PostOrder(&((*pNode)->pSubNode[0]));
//		VisitAndHarvestNodes_PostOrder(&((*pNode)->pSubNode[1]));
//		HarvestOneLeafNode(pNode);
//	}
//	else
//	{
//		HarvestOneLeafNode(pNode);
//	}
//}

// 2021-04-09
void VisitAndHarvestNodes_PostOrder(struct LoquatRTreeNode** pNode)
{
	if ((*pNode) == NULL)
		return;

	if ((*pNode)->pSubNode == NULL)
		return;

	VisitAndHarvestNodes_PostOrder(&((*pNode)->pSubNode[0]));
	VisitAndHarvestNodes_PostOrder(&((*pNode)->pSubNode[1]));

	HarvestOneLeafNode(pNode);
}

int HarvestOneRLoquatTreeRecursively(struct LoquatRTreeStruct **loquatTree)
{
	if( (*loquatTree) == NULL )
		return 1;

	struct LoquatRTreeNode *pNode = (*loquatTree)->rootNode;

	VisitAndHarvestNodes_PostOrder(&pNode);

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

int EvaluateOneSample(float *data, LoquatRForest *loquatForest, float *&target_predicted, int nMethod/*=0*/)
{
	int ntrees = loquatForest->RFinfo.ntrees;
	int variables_num_y = loquatForest->RFinfo.datainfo.variables_num_y;
	const struct LoquatRTreeNode *pLeafNode = NULL;
	if( target_predicted != NULL )
		delete [] target_predicted;
	target_predicted = new float [variables_num_y];
	memset(target_predicted, 0, sizeof(float)*variables_num_y);

	int t, effect_trees;
	float w_sum = 0.f;
	int k, rv = 1;
	for( effect_trees=0, t=0; t<ntrees; t++ )
	{
		pLeafNode = GetArrivedLeafNode(loquatForest, t, data);
		if( pLeafNode == NULL )
		{
			rv = 0;
			continue;
		}
		effect_trees++;
		if( nMethod == 0 )
		{
			for( k=0; k<variables_num_y; k++ )
				target_predicted[k] += pLeafNode->pLeafNodeInfo->MeanOfArrived[k];
		}
		else
		{
			float w = 0.f;
			double det = CalculateDeterminant(pLeafNode->pLeafNodeInfo->CovMatOfArrived, variables_num_y);
			if( det <= 0 )	
				det = 1e-20;
			if( nMethod == 1 )
				 w = float(100.0/det);
			else if( nMethod == 2 )
			{
				w = float(det/pLeafNode->pLeafNodeInfo->arrivedRatio+1e-15);
				w = float(100.0/w);
			}
			w_sum += w;
			for( k=0; k<variables_num_y; k++ )
				target_predicted[k] += pLeafNode->pLeafNodeInfo->MeanOfArrived[k] * w;
		}
	}
	
	if( effect_trees == 0 )
		rv = -1;
	else
	{
		if( nMethod == 0 )
			for( k=0; k<variables_num_y; k++ )
				target_predicted[k] = target_predicted[k]/effect_trees;
		else if( nMethod == 1 || nMethod == 2 )
			for( k=0; k<variables_num_y; k++ )
				target_predicted[k] = target_predicted[k]/w_sum;
		if( loquatForest->bTargetNormalize == true ) // 反归一化
		{
			for( k=0; k<variables_num_y; k++ )
				target_predicted[k] = (target_predicted[k]-loquatForest->offset[k])/loquatForest->scale[k];
		}
	}

	return rv;
}

int MSEOnTestSamples(float **data_test, float **target_test, int nTestSamplesNum, LoquatRForest *loquatForest, float *&mean_squared_error, int nMethod/*=0*/, char *RecordName /*=NULL*/)
{
	int rv =1;
	int i, j, effect;
	int variables_num_y = loquatForest->RFinfo.datainfo.variables_num_y;
	float *target_predicted=NULL;

	if( mean_squared_error != NULL )
		delete [] mean_squared_error;
	mean_squared_error = new float [variables_num_y];
	memset(mean_squared_error, 0, sizeof(float)*variables_num_y);

	ofstream record;
	bool bOpened = false;
	if( RecordName != NULL ) 
	{
		record.open(RecordName);
		if( true == record.good() )
			bOpened = true;
	}

	for( effect=0, i=0; i<nTestSamplesNum; i++ )
	{
		if( 1 != EvaluateOneSample(data_test[i], loquatForest, target_predicted, nMethod) )
		{
			rv = 0;
			continue;
		}
		effect++;
		for( j=0; j<variables_num_y; j++ )
			mean_squared_error[j] += (target_predicted[j]-target_test[i][j])*(target_predicted[j]-target_test[i][j]);

		if( true == bOpened )
		{
			for( j=0; j<variables_num_y; j++ )
				record<<target_predicted[j]<<" ";
			record<<" ";
			for( j=0; j<variables_num_y; j++ )
				record<<target_test[i][j]<<" ";
			record<<endl;
		}

		delete [] target_predicted;
		target_predicted = NULL;
	}

	if( effect == 0 )
		rv = -1;
	else{
		for( j=0; j<variables_num_y; j++ )
			mean_squared_error[j] = mean_squared_error[j]/effect;
	}

	delete [] target_predicted;

	return rv;
}

void predictByLinearModel(double* linearModel, float* data, int variable_num_x, int variable_num_y, double* output)
{
	// ForestMat *linearModel, ForestMat *data, ForestMat *output
	ForestMat linearModelM = { variable_num_x + 1, variable_num_y, linearModel };
	ForestMat dataM = { 1, variable_num_x + 1, new double[variable_num_x + 1] };
	ForestMat outputM = { 1, variable_num_y, output };

	for (int i = 0; i < variable_num_x; i++)
		dataM.data[i] = data[i];
	dataM.data[variable_num_x] = 1.0;

	matMul(&dataM, &linearModelM, &outputM);

	delete[] dataM.data;

}

int MSEOnOutOfBagSamples(float **data, float **target, LoquatRForest *loquatForest, float *&mean_squared_error)
{
	int t, i, j, effect, index, rv = 1;
	int nTrees = loquatForest->RFinfo.ntrees;
	int total_samples_num = loquatForest->RFinfo.datainfo.samples_num;
	int variables_num_x = loquatForest->RFinfo.datainfo.variables_num_x;
	int variables_num_y = loquatForest->RFinfo.datainfo.variables_num_y;
	struct LoquatRTreeStruct* pLoquatRTree = NULL;
	const struct LoquatRTreeNode* pLeafNode = NULL;
	float** target_predicted = new float* [total_samples_num];
	for (i = 0; i < total_samples_num; i++)
	{
		target_predicted[i] = new float[variables_num_y];
		memset(target_predicted[i], 0, sizeof(float) * variables_num_y);
	}

	if (mean_squared_error != NULL)
		delete[] mean_squared_error;
	mean_squared_error = new float[variables_num_y];
	memset(mean_squared_error, 0, sizeof(float) * variables_num_y);

	int* oobTreesNum = new int[total_samples_num];
	memset(oobTreesNum, 0, sizeof(int) * total_samples_num);
	float* weight_sum = NULL;
	int nMethod = 0; // 2021-04-19 取消mMethod参数
	if (nMethod == 1 || nMethod == 2)
	{
		weight_sum = new float[total_samples_num];
		memset(weight_sum, 0, sizeof(float) * total_samples_num);
	}

	double* output = new double[variables_num_y];
	memset(output, 0, sizeof(double) * variables_num_y);

	bool bNormalized = loquatForest->bTargetNormalize;

	int cc = 0, cc_tt = 0;
	for (effect = 0, t = 0; t < nTrees; t++)
	{
		pLoquatRTree = loquatForest->loquatTrees[t];
		if (pLoquatRTree->outofbag_samples_index == NULL)
		{
			rv = 0;
			continue;
		}

		effect++;
		int oobnum = pLoquatRTree->outofbag_samples_num;

		for (i = 0; i < oobnum; i++)
		{
			index = pLoquatRTree->outofbag_samples_index[i];
			oobTreesNum[index]++;
			pLeafNode = GetArrivedLeafNode(loquatForest, t, data[index]);
			cc_tt++;
			if (nMethod == 0)
			{
				if (PredictionModel::constant == loquatForest->RFinfo.predictionModel)
				{
					for (j = 0; j < variables_num_y; j++)
						target_predicted[index][j] += pLeafNode->pLeafNodeInfo->MeanOfArrived[j];
				}
				else if (PredictionModel::linear == loquatForest->RFinfo.predictionModel && NULL != pLeafNode->pLeafNodeInfo->linearPredictor)
				{
					cc++;
					predictByLinearModel(pLeafNode->pLeafNodeInfo->linearPredictor, data[index], variables_num_x, variables_num_y, output);
					for (j = 0; j < variables_num_y; j++)
						target_predicted[index][j] += output[j];
				}
				else
				{
					for (j = 0; j < variables_num_y; j++)
						target_predicted[index][j] += pLeafNode->pLeafNodeInfo->MeanOfArrived[j];
				}

			}
			else if (nMethod == 1 || nMethod == 2)
			{
				float w = 0.f;
				double det = CalculateDeterminant(pLeafNode->pLeafNodeInfo->CovMatOfArrived, variables_num_y);
				if (det <= 0)
					det = 1e-20;
				if (nMethod == 1)
					w = float(100.0 / det);
				else if (nMethod == 2)
				{
					w = float(det / pLeafNode->pLeafNodeInfo->arrivedRatio + 1e-15);
					w = float(100.0 / w);
				}
				weight_sum[index] += w;
				for (j = 0; j < variables_num_y; j++)
					target_predicted[index][j] += pLeafNode->pLeafNodeInfo->MeanOfArrived[j] * w;
			}
		}
	}

	//cout << cc << " " << cc_tt << endl;
	if (0 == effect) //  没有一颗树保存了oob信息
		rv = -1;
	else
	{
		for (effect = 0, i = 0; i < total_samples_num; i++)
		{
			if (oobTreesNum[i] == 0)
			{
				rv = 0;
				continue;
			}

			effect++;

			if (nMethod == 0)
				for (j = 0; j < variables_num_y; j++)
				{
					if (bNormalized == false)
						target_predicted[i][j] = target_predicted[i][j] / oobTreesNum[i];
					else
						target_predicted[i][j] = ((target_predicted[i][j] / oobTreesNum[i]) - loquatForest->offset[j]) / loquatForest->scale[j];
					mean_squared_error[j] += (target_predicted[i][j] - target[i][j]) * (target_predicted[i][j] - target[i][j]);
					//mean_squared_error[j] += fabs(target_predicted[i][j]-target[i][j]);
				}
			else if (nMethod == 1 || nMethod == 2)
				for (j = 0; j < variables_num_y; j++)
				{
					if (bNormalized == false)
						target_predicted[i][j] = target_predicted[i][j] / weight_sum[i];
					else
						target_predicted[i][j] = ((target_predicted[i][j] / weight_sum[i]) - loquatForest->offset[j]) / loquatForest->scale[j];
					mean_squared_error[j] += (target_predicted[i][j] - target[i][j]) * (target_predicted[i][j] - target[i][j]);
				}
		}

		if (effect == 0)
			rv = -1;
		else
		{
			for (j = 0; j < variables_num_y; j++)
				mean_squared_error[j] = mean_squared_error[j] / effect; // mean_squared_error有effect个samples参与计算就要除effect
		}
	}

	delete[] oobTreesNum;
	for (i = 0; i < total_samples_num; i++)
		delete[] target_predicted[i];
	delete[] target_predicted;

	if (nMethod == 1 || nMethod == 2)
		delete[] weight_sum;

	delete[] output;

	return rv;
}

//int HarvestOneRLoquatTree(struct LoquatRTreeStruct **loquatTree)
//{
//	if( (*loquatTree) == NULL )
//		return 1;
//
//	if( (*loquatTree)->inbag_samples_index != NULL )
//	{
//		delete [] (*loquatTree)->inbag_samples_index;
//		(*loquatTree)->inbag_samples_index = NULL;
//		(*loquatTree)->inbag_samples_num = 0;
//	}
//
//	if( (*loquatTree)->outofbag_samples_index != NULL )
//	{
//		delete [] (*loquatTree)->outofbag_samples_index;
//		(*loquatTree)->outofbag_samples_index = NULL;
//		(*loquatTree)->outofbag_samples_num = 0;
//	}
//
//	int depth = (*loquatTree)->depth;
//	int i;
//	unsigned int j, maxNodeNumThisDepth=0;
//	struct LoquatRTreeNode **pPreNode = NULL, **pCurNode = NULL;
//	pPreNode = new struct LoquatRTreeNode *[1];
//	pPreNode[0] = (*loquatTree)->rootNode;
//	if( pPreNode[0] == NULL )
//	{
//		delete [] pPreNode;
//		return 1;
//	}
//	else if( pPreNode[0]->nodetype == enLeafNode )
//	{
//		HarvestOneLeafNode(&pPreNode[0]);
//		//delete pPreNode[0]; // 释放指针指向的空间
//		delete [] pPreNode; // 释放指针 
//		return 1;
//	}
//
//	delete [] pPreNode;
//	pPreNode = NULL;
//
//	while( 1 )
//	{
//		pPreNode = new struct LoquatRTreeNode *[1];
//		pPreNode[0] = (*loquatTree)->rootNode;
//
//		for( i=1; i <= depth; i++ )
//		{
//			maxNodeNumThisDepth = (int)powf(2.f, (float)i);
//			if( pCurNode !=NULL )
//				delete []pCurNode;
//			pCurNode = new struct LoquatRTreeNode *[maxNodeNumThisDepth];
//			for ( j=0; j<maxNodeNumThisDepth/2; j++ ) 
//			{
//				if( pPreNode[j] == NULL )
//				{
//					pCurNode[j*2] = NULL;
//					pCurNode[j*2+1] = NULL;
//				}
//				else if( pPreNode[j]->nodetype == enLeafNode )
//				{
//					pCurNode[j*2] = NULL;
//					pCurNode[j*2+1] = NULL;
//				}
//				else{
//					pCurNode[j*2] = pPreNode[j]->pSubNode[0];
//					pCurNode[j*2+1] = pPreNode[j]->pSubNode[1];
//				}
//			}
//
//			delete []pPreNode;
//			pPreNode = new struct LoquatRTreeNode *[maxNodeNumThisDepth];
//			for( j=0; j<maxNodeNumThisDepth; j++ )
//			{
//				pPreNode[j] = pCurNode[j];
//			}
//			//delete []pCurNode;
//		}
//
//		for( j=0; j<maxNodeNumThisDepth; j++ )
//		{
//			if( pCurNode[j] != NULL )
//			{
//				HarvestOneLeafNode(&pCurNode[j]);
//				//delete pCurNode[j];
//			}
//		}
//
//		delete [] pPreNode;	
//		pPreNode = NULL;
//		delete [] pCurNode;
//		pCurNode =NULL;
//
//		depth--;
//
//		if( depth<=0  )
//			break;
//	}
//
//	HarvestOneLeafNode(&((*loquatTree)->rootNode));
//
//	delete *loquatTree;
//	*loquatTree = NULL;
//
//	return 1;
//}

int HarvestOneRLoquatTree2(struct LoquatRTreeStruct **loquatTree)
{
	if( (*loquatTree) == NULL )
		return 1;

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

	// 层序遍历
	deque<LoquatRTreeNode *> dq;
	dq.push_back((*loquatTree)->rootNode);
	LoquatRTreeNode *tmpNode = NULL;
	while(!dq.empty())
	{
		tmpNode = dq.front();
		if( tmpNode->pSubNode[0] )
			dq.push_back(tmpNode->pSubNode[0]);
		if( tmpNode->pSubNode[1] )
			dq.push_back(tmpNode->pSubNode[1]);
		dq.pop_front();
		HarvestOneLeafNode(&tmpNode);
	}

	*loquatTree = NULL;

	return 1;
}

int ReleaseRegressionForest(LoquatRForest **loquatForest)
{
	if( (*loquatForest) == NULL )
		return 1;

	int Ntrees = (*loquatForest)->RFinfo.ntrees;
	for ( int i=0; i<Ntrees; i++ )
	{
		if( (*loquatForest)->loquatTrees[i] == NULL )
			continue;

		HarvestOneRLoquatTreeRecursively(&((*loquatForest)->loquatTrees[i]));
	}

	delete [] (*loquatForest)->loquatTrees; // 二级指针

	if( (*loquatForest)->scale != NULL )
		delete [] (*loquatForest)->scale;
	if( (*loquatForest)->offset !=NULL )
		delete [] (*loquatForest)->offset;

	delete (*loquatForest);
	(*loquatForest) = NULL;
	
	return 1;
}
