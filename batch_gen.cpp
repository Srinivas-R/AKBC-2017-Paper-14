#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <iostream>

using namespace std;

struct Triple {
	int h, r, t;
};

int *lefHead, *rigHead;
int *lefTail, *rigTail;

Triple *trainHead,*trainTail,*trainList;
int relationTotal = 1345, entityTotal = 14951, tripleTotal=483142;
float *left_mean, *right_mean;
int *freqRel, *freqEnt;

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

extern "C"
void init() {

	FILE *fin;
	int tmp;
	
	fin = fopen("data/train.txt", "r");	
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	freqRel = (int *)calloc(relationTotal, sizeof(int));
	freqEnt = (int *)calloc(entityTotal, sizeof(int));


	cout << "Allocated triples data, reading file\n";
	tripleTotal = 0;
	while (fscanf(fin, "%d", &trainList[tripleTotal].h) == 1) {
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].r);
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].t);
		freqEnt[trainList[tripleTotal].t]++;
		freqEnt[trainList[tripleTotal].h]++;
		freqRel[trainList[tripleTotal].r]++;
		trainHead[tripleTotal].h = trainList[tripleTotal].h;
		trainHead[tripleTotal].t = trainList[tripleTotal].t;
		trainHead[tripleTotal].r = trainList[tripleTotal].r;
		trainTail[tripleTotal].h = trainList[tripleTotal].h;
		trainTail[tripleTotal].t = trainList[tripleTotal].t;
		trainTail[tripleTotal].r = trainList[tripleTotal].r;
		tripleTotal++;
	}
	fclose(fin);
	cout << "Staring sort\n";

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());

	cout << "Sorting finished\n";
	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(rigHead));
	memset(rigTail, -1, sizeof(rigTail));
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	cout << "Calculating mean\n";
	left_mean = (float *)calloc(relationTotal,sizeof(float));
	right_mean = (float *)calloc(relationTotal,sizeof(float));
	for (int i = 0; i < entityTotal; i++) {
		for (int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (int i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
	cout << "Init done!\n";

}

// unsigned long long *next_random;
unsigned long long next_random = 3;

unsigned long long randd(int id) {
	next_random = next_random * (unsigned long long)25214903917 + 11;
	return next_random;
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

extern "C"
void getBatch(int *ph, int *pt, int *pr, int *nh, int *nt, int batchSize, int id) {
	for (int batch = 0; batch < batchSize; batch++) {
		int i = rand_max(id, tripleTotal);
		float prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
		if (randd(id) % 1000 < prob) {
			ph[batch] = trainList[i].h;
			pt[batch] = trainList[i].t;
			pr[batch] = trainList[i].r;
			nh[batch] = trainList[i].h;
			nt[batch] = rand_max(id, entityTotal);
		} else {
			ph[batch] = trainList[i].h;
			pt[batch] = trainList[i].t;
			pr[batch] = trainList[i].r;
			nh[batch] = rand_max(id, entityTotal);
			nt[batch] = trainList[i].t;
		}
	}
}
