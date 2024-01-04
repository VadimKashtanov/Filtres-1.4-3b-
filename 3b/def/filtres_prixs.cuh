#pragma once

#include "mdl.cuh"

void cree_filtre_prixs(Mdl_t * mdl, uint inst);

//	=====================================

void intel_filtres_prixs___naive(				//	mode == 0
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd);

void nvidia_filtres_prixs___naive(			//	mode == 1
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd);

void f_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);

//	----------------------------

void d_intel_filtres_prixs___naive(				//	mode == 0
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df);

void d_nvidia_filtres_prixs___naive(		//	mode == 1
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df);

void df_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);