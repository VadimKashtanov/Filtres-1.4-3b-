#include "lstm1d.cuh"

//(0) Ft[t] = logistic(x@Wf + Ct[t-1]@Uf + Bf)
//(1) It[t] = logistic(x@Wi + Ct[t-1]@Ui + Bi)
//(2) Ot[t] = logistic(x@Wo + Ct[t-1]@Uo + Bo)
//(3) Tt[t] = tanh    (x@Wt + Bt)
//(4) Ct[t] = Ft[t]*Ct[-1] + It[t]*Tt[t]
//(5) Ht[t] = Ot[t]*Ct[t]

#define BLOQUE_Y 32

static __global__ void kerd_lstm1d_naive__0123(
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

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

	if (_y < Y) {
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
};

static __global__ void kerd_lstm1d_naive__45(
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

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

	if (_y < Y) {
		Ct[_y] = Ft[_y]*Ct[_y - 1*(6*Y)] + It[_y]*Tt[_y];
		Ht[_y] = Ct[_y] * Ot[_y];
	}
};

void nvidia_lstm1d_naive(
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	kerd_lstm1d_naive__0123<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
	kerd_lstm1d_naive__45<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
};

//	===========================================================================
//	===========================================================================
//	===========================================================================

static __global__ void deriv_kerd_lstm1d_naive__45(
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
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

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

	if (_y < Y) {
		//Ht[_y] = Ct[_y] * Ot[_y];
		atomicAdd(&dCt[_y], dHt[_y] * Ot[_y]);
		atomicAdd(&dOt[_y], dHt[_y] * Ct[_y]);

		//Ct[_y] = Ft[_y]*Ct[_y - 1*(6*Y)] + It[_y]*Tt[_y];
		atomicAdd(&dFt[_y], dCt[_y] * Ct[_y - 1*(6*Y)]);
		atomicAdd(&dCt[_y - 1*(6*Y)], dCt[_y] * Ft[_y]);
		atomicAdd(&dIt[_y], dCt[_y] * Tt[_y]);
		atomicAdd(&dTt[_y], dCt[_y] * It[_y]);
	}
};

static __global__ void deriv_kerd_lstm1d_naive__0123(
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
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

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

	if (_y < Y) {
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
			atomicAdd(&d__c, dsF * Uf[_y*Y+k]);
			atomicAdd(&dUf[_y*Y+k], dsF * __c);
	//		sI += __c * Ui[_y*Y+k];
			atomicAdd(&d__c, dsI * Ui[_y*Y+k]);
			atomicAdd(&dUi[_y*Y+k], dsI * __c);
	//		sO += __c * Uo[_y*Y+k];
			atomicAdd(&d__c, dsO * Uo[_y*Y+k]);
			atomicAdd(&dUo[_y*Y+k], dsO * __c);
			//
			atomicAdd(&dCt[k - 1*(6*Y)], d__c);
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
			atomicAdd(&d__x, dsF * Wf[_y*Y+k]);
			atomicAdd(&dWf[_y*Y+k], dsF * __x);
	//		sI += __x * Wi[_y*X+k];
			atomicAdd(&d__x, dsI * Wi[_y*Y+k]);
			atomicAdd(&dWi[_y*Y+k], dsI * __x);
	//		sO += __x * Wo[_y*X+k];
			atomicAdd(&d__x, dsO * Wo[_y*Y+k]);
			atomicAdd(&dWo[_y*Y+k], dsO * __x);
	//		sT += __x * Wt[_y*X+k];
			atomicAdd(&d__x, dsT * Wt[_y*Y+k]);
			atomicAdd(&dWt[_y*Y+k], dsT * __x);
			//
			atomicAdd(&_dx[k], d__x);
		}
		//
	//	float sF=Bf[_y], sI=Bi[_y], sO=Bo[_y], sT=Bt[_y];
		atomicAdd(&Bf[_y], dsF);
		atomicAdd(&Bi[_y], dsI);
		atomicAdd(&Bo[_y], dsO);
		atomicAdd(&Bt[_y], dsT);
	}
};


void d_nvidia_lstm1d_naive(
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
	deriv_kerd_lstm1d_naive__45<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();
	deriv_kerd_lstm1d_naive__0123<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();
};