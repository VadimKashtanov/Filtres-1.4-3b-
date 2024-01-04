#include "main.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

static void kerd_p1e5(float * p, uint i, float _1E5) {
	p[i] += _1E5;
};

static void p1e5(Mdl_t * mdl, uint c, uint i, float _1E5) {
	kerd_p1e5<<<1,1>>>(mdl->p__d[c], i, _1E5);
	ATTENDRE_CUDA();
};

void verif_mdl_1e5() {
	ASSERT(C == 5);
	titre("Comparer MODEL 1e-5");
	//
	uint Y[C] = {
		512,
		128,
		64,
		64,
		P
	};
	uint insts[C] = {
		FILTRES_PRIXS,
		LSTM1D,
		DOT1D,
		LSTM1D,
		DOT1D
	};
	uint lignes[BLOQUES] = {
		0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		3, 3, 3, 3, 3, 3, 3, 3,
		4, 4, 4, 4, 4,
		5, 5, 5, 5, 5, 5,
		6, 6, 6, 6, 6,
		7, 7, 7, 7,
		8, 8, 8, 8, 8, 8, 8,
		9, 9, 9, 9, 9,
		10, 10, 10, 10, 10
	};
	Mdl_t * mdl = cree_mdl(Y, insts, lignes);
	//
	uint plus_T = 16;
	//
	mdl_aller_retour(mdl, DEPART, DEPART+plus_T, 3);
	mdl_gpu_vers_cpu(mdl);
	//
	//	1e-5
	//
	mdl_zero_gpu(mdl);
	float _f = mdl_score(mdl, DEPART, DEPART+plus_T, 3);
	float _1E5 = 1e-3;
	FOR(0, c, C) {
		printf("###############################################################\n");
		printf("#######################   C = %2.i   ##########################\n", c);
		printf("#######################vvvvvvvvvvvvvv##########################\n");
		//
		FOR(0, i, mdl->POIDS[c]) {
			p1e5(mdl, c, i, +_1E5);
			mdl_zero_gpu(mdl);
			float grad_1e5 = (mdl_score(mdl, DEPART, DEPART+plus_T, 3) - _f)/_1E5;
			p1e5(mdl, c, i, -_1E5);
			//
			float a=grad_1e5, b=mdl->dp[c][i];
			EXACTE((fabs(a-b) < log10(-fabs(a+b)/2 -2)))
			printf("%i| %f === %f\033[0m\n", i, a, b);
		}
	};
	printf("  1e5 === df(x)  \n");

	//
	liberer_mdl(mdl);
};