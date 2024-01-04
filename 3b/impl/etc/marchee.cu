#include "mdl.cuh"

//	Sources
float   prixs[PRIXS] = {};
float   macds[PRIXS] = {};
float volumes[PRIXS] = {};
//
float            ema[EMA_INTS * PRIXS *    1  ] = {};
float     normalisee[EMA_INTS * PRIXS * N_FLTR] = {};
float dif_normalisee[EMA_INTS * PRIXS * N_FLTR] = {};

void charger_les_prixs() {
	uint __PRIXS;
	FILE * fp;
	//
	fp = fopen("prixs/prixs.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(prixs, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/volumes.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(volumes, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/macds.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(macds, sizeof(float), PRIXS, fp);
	fclose(fp);
};

void calculer_ema_norm_diff() {
	//	extern float        ema[EMA_INTS][    PRIXS    ];
	float k[EMA_INTS];
	float _k[EMA_INTS];
	for (uint i=0; i < EMA_INTS; i++) {
		k[i] = 1.0/(1.0 + (float)ema_ints[i].ema);
		_k[i] = 1.0 - k[i];
		ema[i*PRIXS+0] = (ema_ints[i].source)[0];

		assert(ema_ints[i].interv <= MAX_INTERVALLE);
	}
	//
	for (uint i=1; i < PRIXS; i++) {
		for (uint j=0; j < EMA_INTS; j++) {
			ema[j*PRIXS+i] = (ema_ints[j].source)[i]*k[j] + ema[j*PRIXS + i-1]*_k[j];
		};
	};


	//	extern float normalisee[EMA_INTS][PRIXS][N_FLTR];
	float _max, _min;
	FOR(DEPART, t, FIN) {
		FOR(0, e, EMA_INTS) {
			_max = ema[e*PRIXS + t-0*ema_ints[e].interv];
			_min = ema[e*PRIXS + t-0*ema_ints[e].interv];
			FOR(1, i, N_FLTR) {
				if (_max < ema[e*PRIXS + t-i*ema_ints[e].interv])
					_max = ema[e*PRIXS + t-i*ema_ints[e].interv];
				if (_min > ema[e*PRIXS + t-i*ema_ints[e].interv])
					_min = ema[e*PRIXS + t-i*ema_ints[e].interv];
			}
			FOR(0, i, N_FLTR) {
				normalisee[e*PRIXS*N_FLTR+t*N_FLTR+i] = (ema[e*PRIXS+t-i*ema_ints[e].interv]-_min)/(_max-_min);
			}
		};
	};

	FOR(DEPART, t, FIN) {
		FOR(0, e, EMA_INTS) {
			FOR(1, i, N_FLTR)
				dif_normalisee[e*PRIXS*N_FLTR+t*N_FLTR+i] = normalisee[e*PRIXS*N_FLTR+t*N_FLTR+i]-normalisee[e*PRIXS*N_FLTR+t*N_FLTR+i-1];
			dif_normalisee[e*PRIXS*N_FLTR+t*N_FLTR+N_FLTR+0] = 0.f;
		}
	}
};

float *          prixs__d = 0x0;
float *          macds__d = 0x0;
float *        volumes__d = 0x0;
float *            ema__d = 0x0;
float *     normalisee__d = 0x0;
float * dif_normalisee__d = 0x0;

void charger_vram_nvidia() {
	CONTROLE_CUDA(cudaMalloc((void**)&  prixs__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&  macds__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&volumes__d, sizeof(float) * PRIXS));
	//
	CONTROLE_CUDA(cudaMalloc((void**)&           ema__d, sizeof(float) * EMA_INTS * PRIXS *    1  ));
	CONTROLE_CUDA(cudaMalloc((void**)&    normalisee__d, sizeof(float) * EMA_INTS * PRIXS * N_FLTR));
	CONTROLE_CUDA(cudaMalloc((void**)&dif_normalisee__d, sizeof(float) * EMA_INTS * PRIXS * N_FLTR));
	//
	CONTROLE_CUDA(cudaMemcpy(  prixs__d,   prixs, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(  macds__d,   macds, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(volumes__d, volumes, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	//
	CONTROLE_CUDA(cudaMemcpy(           ema__d,            ema, sizeof(float) * EMA_INTS * PRIXS *    1  , cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(    normalisee__d,     normalisee, sizeof(float) * EMA_INTS * PRIXS * N_FLTR, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(dif_normalisee__d, dif_normalisee, sizeof(float) * EMA_INTS * PRIXS * N_FLTR, cudaMemcpyHostToDevice));
};

void     liberer_cudamalloc() {
	CONTROLE_CUDA(cudaFree(  prixs__d));
	CONTROLE_CUDA(cudaFree(  macds__d));
	CONTROLE_CUDA(cudaFree(volumes__d));
	//
	CONTROLE_CUDA(cudaFree(           ema__d));
	CONTROLE_CUDA(cudaFree(    normalisee__d));
	CONTROLE_CUDA(cudaFree(dif_normalisee__d));
};

void charger_tout() {
	printf("charger_les_prixs : ");      MESURER(charger_les_prixs());
	printf("calculer_ema_norm_diff : "); MESURER(calculer_ema_norm_diff());
	printf("charger_les_prixs : ");      MESURER(charger_vram_nvidia());
	printf("Méga-octés = %f Mo\n",
		(float)(PRIXS*(sizeof(float)*3 + EMA_INTS + EMA_INTS*N_FLTR + EMA_INTS*N_FLTR)) / 1e6f
	);
};

void liberer_tout() {
	titre("Liberer tout");
	liberer_cudamalloc();
};