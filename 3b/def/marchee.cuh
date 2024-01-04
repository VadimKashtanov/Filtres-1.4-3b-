#pragma once

#define DEBBUG false

#include "etc.cuh"

#define PRIXS 54901	//53170 * NB_DIFF_EMA * 32-bits = 17014400 bits = 17014.4 Ko = 17 Mo
//#define L 1  		//u += u*f*levier*(p[i+L]/p[i]-1)
#define P 1 //2 ou 3

#define N_FLTR  8
#define N N_FLTR

#define MAX_INTERVALLE 1000

#define DEPART (N_FLTR*MAX_INTERVALLE)
#if (DEBBUG == false)
	#define FIN (PRIXS-P-1)
#else
	#define FIN (DEPART+1)
#endif

typedef struct {
	uint     ligne;
	uint       ema;
	uint    interv;
	float * source;
} ema_int;

#define EMA_INTS (39)

/*	Note : dans `normalisee` et `dif_normalisee`
les intervalles sont deja calculee. Donc tout
ce qui est avant DEPART n'est pas initialisee.
*/

//	Sources
extern float   prixs[PRIXS];	//  prixs.bin
extern float   macds[PRIXS];	//   macd.bin
extern float volumes[PRIXS];	// volume.bin

//	ema des sources
extern float            ema[EMA_INTS * PRIXS];
extern float     normalisee[EMA_INTS * PRIXS * N_FLTR];
extern float dif_normalisee[EMA_INTS * PRIXS * N_FLTR];

//	======================================

//	Sources en GPU
extern float *   prixs__d;	//	nVidia
extern float *   macds__d;	//	nVidia
extern float * volumes__d;	//	nVidia

//	gpu ema des sources
extern float *            ema__d;	//	nVidia
extern float *     normalisee__d;	//	nVidia
extern float * dif_normalisee__d;	//	nVidia

void      charger_les_prixs();
void calculer_ema_norm_diff();
void    charger_vram_nvidia();

void     liberer_cudamalloc();

static ema_int ema_ints[EMA_INTS] = {
//	 id   ema  interv  source
	{ 0,    1,    1,   prixs},
	{ 1,    2,    1,   prixs},
	{ 2,    5,    1,   prixs},
// -------  de macd --------
    { 3,    1,    1,   macds},
    { 4,    2,    1,   macds},
    { 5,    5,    1,   macds},
    { 6,    5,    5,   macds},
// --------- volume --------
    { 7,    1,    1,   volumes},
    { 8,    2,    1,   volumes},
    { 9,    5,    1,   volumes},
    // plus grand interv que ema
    {10,    5,   15,   volumes}
};

void charger_tout();
void liberer_tout();