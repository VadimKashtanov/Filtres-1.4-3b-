#pragma once

#include "cuda.cuh"
#include "macro.cuh"

//  Outils
float   rnd();
float signe(float x);

void titre(char * str);
/*
//	Disque
template <typename T>   T*   lire(char * fichier,        uint N);
template <typename T> void ecrire(char * fichier, T * l, uint N);

//	Listes
template <typename T> T * alloc(uint N);
template <typename T> T * copier(T * l, uint N);
template <typename T> T * zero(uint N);

float * lst_rnd(uint N, float a, float b);

void comparer_lst(float * l0, float * l1, uint N, float profondeure);
void comparer_lst_2d(
	float * l0, float * l1,
	uint X,
	uint Y, char * ynom,
	float profondeure);
uint egales_lst(float * l0, float * l1, uint N, float profondeure);

//	CPU <-> GPU
template <typename T> T *     cudalloc(                uint A);
template <typename T> T * cpu_vers_gpu(T * lst,        uint A);
template <typename T> T * gpu_vers_cpu(T * lst__d,     uint A);
					  void   cudaplume(float * lst__d, uint A);
template <typename T> void    cudazero(T * lst__d,     uint A);
*/

//	Inclure tous les patrons

//#include "impl_tmpl/tmpl_etc.cu"