#ifndef _SHARED_ROUTINES_H_
#define _SHARED_ROUTINES_H_

#define RF_MAX(a,b)  ((a)>(b) ? (a) : (b))

extern unsigned int g_random_seed;

enum class TreeNodeType: int { ROOT_NODE=0, LINK_NODE=1, LEAF_NODE=2 };

//enum RFError {
//	RF_SUCCESS = 0,
//	RF_WRONG_INIT_FOREST_POINTER,
//	RF_NULL_POINTER_EXCEPTION,
//	RF_WRONG_PARAMETERS,
//	RF_EVALUATE_ERROR,
//	RF_OTHER_ERROR
//};

enum class RandomnessLevel: int
{
	WEAK = 1,
	MODERATE = 2,
	STRONG = 3
};

enum class ProximityType: int
{
	PROX_ORIGINAL = 1,
	PROX_GEO_ACC = 2
};

/*
Description: Returns the time (ms) elapsed between two calls to this function
*/
float timeIt(int reset);

/*
Description: Generate seed for random number generator, 
			 if succeeded, the seed is the second of the current system time,
			 if failed, 0 is assigned to the seed.
*/
unsigned int GenerateSeedFromSysTime();

#define RAND_MAX_RF 0x7ffffffd
int rand_freebsd(void);
void srand_freebsd(unsigned seed);

void permute(const int n, int* order_permuted);

float** clone_data(float** data, int num, int vars);

#ifdef _DEBUG
#define KCF_ASSERT(condition) if(!(condition)) \
{			\
	printf("[TRACK ASSERT] ERROR %s in FILE %s, LINE %u\n", #condition, __FILE__, __LINE__); \
	exit(-1);	\
	}
#else
#define KCF_ASSERT(condition)
#endif

/*
// ÄÚ´æ¹ÜÀí
typedef struct tagKCFMem
{
	void* mem;
	unsigned int tot_size;
	unsigned int used;
}KCFMemPool;

void* kcfAlloc(KCFMemPool* pool, unsigned int size);

void kcfFree(KCFMemPool* pool, void** ptr, unsigned int size);

*/

#endif