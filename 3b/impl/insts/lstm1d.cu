#include "lstm1d.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_dot1d(Mdl_t * mdl, uint c)
{
	mdl->inst_POIDS        [c] = (mdl->Y[c-1]+1)*mdl->Y[c];
	mdl->inst_VARS         [c] = mdl->Y[c]*6;	//Ft, It, Ot, Tt, Ct, Ht
	mdl->inst_LOCDS        [c] = 0;
	mdl->inst_SORTIES      [c] = mdl->Y[c];		//Ht
	mdl->inst_DEPART_SORTIE[c] = mdl->inst_VARS[c] - mdl->inst_SORTIES[c];
};

void intel_lstm1d(
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	float * _x = x + t*X + DEPART_x;
	//
	float * Ft = y + 0*Y;
	float * It = y + 1*Y;
	float * Ot = y + 2*Y;
	float * Tt = y + 3*Y;
	float * Ct = y + 4*Y;
	float * Ht = y + 5*Y;
	//
	p+=0;     float *Wf=p+0*X*Y, *Wi=p+1*X*Y, *Wo=p+2*X*Y;
	p+=3*X*Y; float *Uf=p+0*Y*Y, *Ui=p+1*Y*Y, *Uo=p+2*Y*Y;
	p+=3*Y*Y; float *Bf=p+0*Y,   *Bi=p+1*Y,   *Bo=p+2*Y;
	p+=3*Y  ; float *Wt=p+0;
	p+=1*X*Y; float *Bt=p+0;
	//
	//(0) Ft[t] = logistic(x@Wf + Ct[t-1]@Uf + Bf)
	//(1) It[t] = logistic(x@Wi + Ct[t-1]@Ui + Bi)
	//(2) Ot[t] = logistic(x@Wo + Ct[t-1]@Uo + Bo)
	//(3) Tt[t] = tanh    (x@Wt + Bt)
	//(4) Ct[t] = Ft[t]*Ct[-1] + It[t]*Tt[t]
	//(5) Ht[t] = Ot[t]*Ct[t]
//#pragma omp parallel
//#pragma omp for
	//	(0), (1), (2)
	FOR(0, _y, Y) {
		float sF=Bf[_y], sI=Bi[_y], sO=Bo[_y], sT=Bt[_y];
		//	--- x@W ---
		FOR(0, k, X) {
			float __x = _x[k];
			sF += __x * Wf[_y*X+k];
			sI += __x * Wi[_y*X+k];
			sO += __x * Wo[_y*X+k];
			sT += __x * Wt[_y*X+k];
		}
		//	--- Ct[t-1]@U ---
		FOR(0, k, Y) {
			float __c = Ct[k - 1*(6*Y)];	//t-1
			sF += __c * Uf[_y*Y+k];
			sI += __c * Ui[_y*Y+k];
			sO += __c * Uo[_y*Y+k];
		}
		//	--- logistic && tanh ---
		Ft[_y] = 1 / (1 + expf(-sF));
		It[_y] = 1 / (1 + expf(-sI));
		Ot[_y] = 1 / (1 + expf(-sO));
		Tt[_y] = tanh(sT);
	}
	//(3) Ct[t] = Ft[t]*Ct[-1] + It[t]*Tt[t]
	FOR(0, _y, Y) {
		Ct[_y] = Ft[_y]*Ct[_y - 1*(6*Y)] + It[_y]*Tt[_y];
		Ht[_y] = Ct[_y] * Ot[_y];
	};
}

void d_intel_lstm1d(
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	float * _x = x + t*X + DEPART_x;
	float * _dx = dx + t*X + DEPART_x;
	//
	float * Ft = y + 0*Y;
	float * It = y + 1*Y;
	float * Ot = y + 2*Y;
	float * Tt = y + 3*Y;
	float * Ct = y + 4*Y;
	float * Ht = y + 5*Y;
	//
	float * dFt = dy + 0*Y;
	float * dIt = dy + 1*Y;
	float * dOt = dy + 2*Y;
	float * dTt = dy + 3*Y;
	float * dCt = dy + 4*Y;
	float * dHt = dy + 5*Y;
	//
	p+=0;     float *Wf=p+0*X*Y, *Wi=p+1*X*Y, *Wo=p+2*X*Y;
	p+=3*X*Y; float *Uf=p+0*Y*Y, *Ui=p+1*Y*Y, *Uo=p+2*Y*Y;
	p+=3*Y*Y; float *Bf=p+0*Y,   *Bi=p+1*Y,   *Bo=p+2*Y;
	p+=3*Y  ; float *Wt=p+0;
	p+=1*X*Y; float *Bt=p+0;
	//
	dp+=0;     float *dWf=dp+0*X*Y, *dWi=dp+1*X*Y, *dWo=dp+2*X*Y;
	dp+=3*X*Y; float *dUf=dp+0*Y*Y, *dUi=dp+1*Y*Y, *dUo=dp+2*Y*Y;
	dp+=3*Y*Y; float *dBf=dp+0*Y,   *dBi=dp+1*Y,   *dBo=dp+2*Y;
	dp+=3*Y  ; float *dWt=dp+0;
	dp+=1*X*Y; float *dBt=dp+0;

	FOR(0, _y, Y) {
		//Ht[_y] = Ct[_y] * Ot[_y];
		dCt[_y] += dHt[_y] * Ot[_y];
		dOt[_y] += dHt[_y] * Ct[_y];

		//Ct[_y] = Ft[_y]*Ct[_y - 1*(6*Y)] + It[_y]*Tt[_y];
		dFt[_y] += dCt[_y] * Ct[_y - 1*(6*Y)];
		dCt[_y - 1*(6*Y)] += dCt[_y] * Ft[_y];
		dIt[_y] += dCt[_y] * Tt[_y];
		dTt[_y] += dCt[_y] * It[_y];
	};

	FOR(0, _y, Y) {
		//	--- logistic && tanh ---
	//	Ft[_y] = 1 / (1 + expf(-sF));
	//	It[_y] = 1 / (1 + expf(-sI));
	//	Ot[_y] = 1 / (1 + expf(-sO));
	//	Tt[_y] = tanh(sT);
		/*float sF = -log(1/Ft[_y]-1);
		float sI = -log(1/It[_y]-1);
		float sO = -log(1/Ot[_y]-1);
		float sT =  atanh(Tt[_y]);*/
		//
		float dsF = dFt[_y] * (Ft[_y] * (1 - Ft[_y]));
		float dsI = dIt[_y] * (It[_y] * (1 - It[_y]));
		float dsO = dOt[_y] * (Ot[_y] * (1 - Ot[_y]));
		float dsT = dTt[_y] * (1 - Tt[_y]*Tt[_y]);

		//	--- Ct[t-1]@U ---
		FOR(0, k, Y) {
			float d__c = 0;
			float __c = Ct[k - 1*(6*Y)];	//t-1
	//		sF += __c * Uf[_y*Y+k];
			d__c += dsF * Uf[_y*Y+k];
			dUf[_y*Y+k] += dsF * __c;
	//		sI += __c * Ui[_y*Y+k];
			d__c += dsI * Ui[_y*Y+k];
			dUi[_y*Y+k] += dsI * __c;
	//		sO += __c * Uo[_y*Y+k];
			d__c += dsO * Uo[_y*Y+k];
			dUo[_y*Y+k] += dsO * __c;
			//
			dCt[k - 1*(6*Y)] += d__c;
		}
		//	--- x@W ---
		FOR(0, k, X) {
	//		float __x = _x[k];
	//		sF += __x * Wf[_y*X+k];
	//		sI += __x * Wi[_y*X+k];
	//		sO += __x * Wo[_y*X+k];
	//		sT += __x * Wt[_y*X+k];
			float d__x = 0;
			float __x = _x[k];	//t-1
	//		sF += __x * Wf[_y*X+k];
			d__x += dsF * Wf[_y*Y+k];
			dWf[_y*Y+k] += dsF * __x;
	//		sI += __x * Wi[_y*X+k];
			d__x += dsI * Wi[_y*Y+k];
			dWi[_y*Y+k] += dsI * __x;
	//		sO += __x * Wo[_y*X+k];
			d__x += dsO * Wo[_y*Y+k];
			dWo[_y*Y+k] += dsO * __x;
	//		sT += __x * Wt[_y*X+k];
			d__x += dsT * Wt[_y*Y+k];
			dWt[_y*Y+k] += dsT * __x;
			//
			_dx[k] += d__x;
		}
		//
	//	float sF=Bf[_y], sI=Bi[_y], sO=Bo[_y], sT=Bt[_y];
		Bf[_y] += dsF;
		Bi[_y] += dsI;
		Bo[_y] += dsO;
		Bt[_y] += dsT;
	}
}

//	=========================================================
__global__
static void kerd_cuda_memset_t(float * p, uint t, uint vars) {
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thx < vars) p[t*vars + thx] = 0;
};
void cuda_memset_t(float * p, uint t, uint vars) {
	kerd_cuda_memset_t<<<dim3(KERD(vars,32)), dim3(32)>>>(p, t, vars);
	ATTENDRE_CUDA();
}

void f_lstm1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint depart = t0;
	uint T = (t1-t0);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	//
	if (mode == 0) {
		memset(mdl->p[inst]+(t0-1)*mdl->inst_VARS[inst], 0, sizeof(float)*mdl->inst_VARS[inst]);
		FOR(t0, t, t1) {
			intel_lstm1d(
				X, Y,
				t,
				DEPART_x,
				mdl->y[inst-1], mdl->y[inst],
				mdl->p[inst],
				mdl->l[inst]);
		}
	} else if (mode == 1) {
		cuda_memset_t(mdl->p[inst], t0-1, mdl->inst_VARS[inst]);
		FOR(t0, t, t1) {
			nvidia_lstm1d_naive(
				X, Y,
				t,
				DEPART_x,
				mdl->y__d[inst-1], mdl->y__d[inst],
				mdl->p__d[inst],
				mdl->l__d[inst]);
		}
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}

//	----------------------------

void df_lstm1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint depart = t0;
	uint T = (t1-t0);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	//
	if (mode == 0) {
		memset(mdl->p[inst]+(t0-1)*mdl->inst_VARS[inst], 0, sizeof(float)*mdl->inst_VARS[inst]);
		RETRO_FOR(t0, t, t1) {
			intel_lstm1d(
				X, Y,
				t,
				DEPART_x,
				mdl->y[inst-1], mdl->y[inst],
				mdl->p[inst],
				mdl->l[inst]);
		}
	} else if (mode == 1) {
		cuda_memset_t(mdl->p[inst], t0-1, mdl->inst_VARS[inst]);
		RETRO_FOR(t0, t, t1) {
			nvidia_lstm1d_naive(
				X, Y,
				t,
				DEPART_x,
				mdl->y__d[inst-1], mdl->y__d[inst],
				mdl->p__d[inst],
				mdl->l__d[inst]);
		}
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}