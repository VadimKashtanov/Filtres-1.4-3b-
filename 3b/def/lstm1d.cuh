#pragma once

#include "mdl.cuh"

void cree_lstm1d(Mdl_t * mdl, uint inst);

//	============================================

void intel_lstm1d(					//mode = 0
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_lstm1d_naive(			//mode = 1
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_lstm1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);

//	============================================

void d_intel_lstm1d(					//mode = 0
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_lstm1d_naive(			//mode = 1
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void df_lstm1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);