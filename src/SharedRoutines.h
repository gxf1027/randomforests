#ifndef _SHARED_ROUTINES_H_
#define _SHARED_ROUTINES_H_

#define RF_MAX(a,b)  ((a)>(b) ? (a) : (b))

extern unsigned int g_random_seed;

enum TreeNodeTpye { enRootNode=0, enLinkNode=1, enLeafNode=2 };

//enum RFError {
//	RF_SUCCESS = 0,
//	RF_WRONG_INIT_FOREST_POINTER,
//	RF_NULL_POINTER_EXCEPTION,
//	RF_WRONG_PARAMETERS,
//	RF_EVALUATE_ERROR,
//	RF_OTHER_ERROR
//};

enum RF_TREE_RANDOMNESS 
{
	TREE_RANDOMNESS_WEAK = 1,
	TREE_RANDOMNESS_MODERATE = 2,
	TREE_RANDOMNESS_STRONG = 3
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
void GenerateSeedFromSysTime();

#define RAND_MAX_RF 0x7ffffffd
int rand_freebsd(void);
void srand_freebsd(unsigned seed);

void permute(const int n, int* order_permuted);


#ifdef _DEBUG
#define KCF_ASSERT(condition) if(!(condition)) \
{			\
	printf("[TRACK ASSERT] ERROR %s in FILE %s, LINE %u\n", #condition, __FILE__, __LINE__); \
	exit(-1);	\
	}
#else
#define KCF_ASSERT(condition)
#endif

// ÄÚ´æ¹ÜÀí
typedef struct tagKCFMem
{
	void* mem;
	unsigned int tot_size;
	unsigned int used;
}KCFMemPool;

void* kcfAlloc(KCFMemPool* pool, unsigned int size);

void kcfFree(KCFMemPool* pool, void** ptr, unsigned int size);

#endif