#include <iostream>
#include <ctime>
#include <cstring>
#include <omp.h>
using namespace std;

#include "SharedRoutines.h"

float timeIt(int reset) 
{
	static time_t startTime, endTime;
	static int timerWorking = 0;

	if (reset)
	{
		startTime = clock();
		timerWorking = 1;
		return -1;
	}
	else
	{
		if (timerWorking) 
		{
			endTime = clock();
			timerWorking = 0;
			return (float) (endTime - startTime);
		} else
		{
			startTime = clock();
			timerWorking = 1;
			return -1;
		}
	}
}

unsigned int GenerateSeedFromSysTime()
{
	time_t ltime;
	//struct tm newtime;
	struct tm *newtime = 0;
	time(&ltime);
	//errno_t err = gmtime_s(&newtime, &ltime);
	newtime = gmtime(&ltime);

	if( 0 == newtime )
	{
		cout<<"-----------------  ERROR:'GenerateSeedFromSysTime' -------------------"<<endl;
		cout<<"Cannot get system time, random seed is initialized with value '0'"<<endl;
		cout<<"----------------------------------------------------------------------"<<endl;
		return 0;
	}
	else
	{
		//g_random_seed = newtime->tm_sec;
		return time(0);
		//cout<<"Inital seed: "<<g_random_seed<<endl;
	}

	return 0;
}

/*

https://svnweb.freebsd.org/base/head/lib/libc/stdlib/rand.c?revision=326025&view=co
*/

#ifdef TEST
#include <stdio.h>
#endif /* TEST */

static int
do_rand(unsigned long* ctx)
{
	/*
	 * Compute x = (7^5 * x) mod (2^31 - 1)
	 * without overflowing 31 bits:
	 *      (2^31 - 1) = 127773 * (7^5) + 2836
	 * From "Random number generators: good ones are hard to find",
	 * Park and Miller, Communications of the ACM, vol. 31, no. 10,
	 * October 1988, p. 1195.
	 */
	long hi, lo, x;

	/* Transform to [1, 0x7ffffffe] range. */
	x = (*ctx % 0x7ffffffe) + 1;
	hi = x / 127773;
	lo = x % 127773;
	x = 16807 * lo - 2836 * hi;
	if (x < 0)
		x += 0x7fffffff;
	/* Transform to [0, 0x7ffffffd] range. */
	x--;
	*ctx = x;
	return (x);
}


int
rand_r(unsigned* ctx)
{
	unsigned long val;
	int r;

	val = *ctx;
	r = do_rand(&val);
	*ctx = (unsigned)val;
	return (r);
}


static unsigned long next_rf = 1;

int
rand_freebsd(void)
{
	return (do_rand(&next_rf));
}

void
srand_freebsd(unsigned seed)
{
	next_rf = seed;
}


/*
 * sranddev:
 *
 * Many programs choose the seed value in a totally predictable manner.
 * This often causes problems.  We seed the generator using pseudo-random
 * data from the kernel.
 */
 // void
 // sranddev(void)
 // {
 // 	int mib[2];
 // 	size_t len;

 // 	len = sizeof(next);

 // 	mib[0] = CTL_KERN;
 // 	mib[1] = KERN_ARND;
 // 	sysctl(mib, 2, (void *)&next, &len, NULL, 0);
 // }


#ifdef TEST

main()
{
	int i;
	unsigned myseed;

	printf("seeding rand with 0x19610910: \n");
	srand(0x19610910);

	printf("generating three pseudo-random numbers:\n");
	for (i = 0; i < 3; i++)
	{
		printf("next random number = %d\n", rand());
	}

	printf("generating the same sequence with rand_r:\n");
	myseed = 0x19610910;
	for (i = 0; i < 3; i++)
	{
		printf("next random number = %d\n", rand_r(&myseed));
	}

	return 0;
}

#endif /* TEST */


void permute(const int n, int* order_permuted)
{
	int i, index;
	int* order = new int[n];
	for (i = 0; i < n; i++)
		order[i] = i;

	for (i = 0; i < n; i++)
	{
		index = rand_freebsd() % (n - i);
		order_permuted[i] = order[index];

		memcpy(order + index, order + index + 1, sizeof(int) * (n - index-i-1));
	}
	delete[] order;
}


/*
void* kcfAlloc(KCFMemPool* pool, unsigned int size)
{
	void* ptr = (void*)((unsigned int)(pool->mem) + pool->used);
	pool->used += size;

	if (pool->used > pool->tot_size)
	{
		pool->used -= size;
		return NULL;
	}

	//printf("[RATIO]: %f\n", pool->used/(float)pool->tot_size); // Õ¼ÓÃÂÊ

	return ptr;
}

void kcfFree(KCFMemPool* pool, void** ptr, unsigned int size)
{
	KCF_ASSERT((unsigned int)(*ptr) >= (unsigned int)(pool->mem)
		&& (unsigned int)(*ptr) + size - 1 < (unsigned int)(pool->mem) + pool->tot_size);
	(*ptr) = NULL;
	KCF_ASSERT((long)pool->used - (long)size >= 0);
	pool->used -= size;
}
*/


float** clone_data(float** data, int num, int vars)
{
	float** data_clone = new float* [num];
	for (int n = 0; n < num; n++)
	{
		data_clone[n] = new float[vars];
		memcpy(data_clone[n], data[n], sizeof(float) * vars);
	}

	return data_clone;
}

void omp_set_threads(int jobs)
{
	const int max_threads = omp_get_max_threads();
	jobs = RF_MAX(1, jobs);
	jobs = jobs > max_threads ? max_threads : jobs;
	omp_set_num_threads(jobs);
}
