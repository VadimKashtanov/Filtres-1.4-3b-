#include "filtres_prixs.cuh"

#define BLOQUE_T  4//8
#define BLOQUE_B  4//8
#define BLOQUE_FB 8//16//32//8

#include "../../../impl_tmpl/tmpl_etc.cu"

static __device__ float filtre_device(float * x, float * dif_x, float * f, float * locd) {
	float s = 0, d = 0;
	float f_nouveau = f[0];
	s += sqrtf(1 + fabs(x[N-1] - f_nouveau));
	float f_avant   = f_nouveau;
	FOR(1, i, N) {
		f_nouveau = f[i];
		s += sqrtf(1 + fabs(  x[i]   -   f_nouveau  ));
		d += powf((1 + fabs(dif_x[i] - (f_nouveau-f_avant))), 2);
		f_avant   = f_nouveau;
	};

	s = s/8-1;
	d = d/7-1;

	float y = expf(-s*s -d*d);

	locd[0] = -2*2*s*y;
	locd[1] = -2*2*d*y;

	return 2*y-1;
};

static __global__ void kerd_filtre_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _b = threadIdx.y + blockIdx.y * blockDim.y;
	uint _f = threadIdx.z + blockIdx.z * blockDim.z;

	if (_t < T && _b < bloques && _f < f_par_bloque) {
		y[(depart+_t)*(bloques*f_par_bloque) + _b*f_par_bloque + _f] = filtre_device(
			x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			dif_x + ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR,
			f     + _b*f_par_bloque*N     + _f*N,
			locd  + (depart+_t)*bloques*f_par_bloque*2 + _b*f_par_bloque*2 + _f*2
		);
	}
};

void nvidia_filtres_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd)
{
	kerd_filtre_naive<<<dim3(KERD(T, BLOQUE_T), KERD(bloques, BLOQUE_B), KERD(f_par_bloque, BLOQUE_FB)), dim3(BLOQUE_T, BLOQUE_B, BLOQUE_FB)>>>(
		depart, T,
		bloques, f_par_bloque, ligne,
		x, dif_x,
		f,
		y,
		locd);
	ATTENDRE_CUDA();
}

static __global__ void d_kerd_filtre_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df)
{
	uint thz = threadIdx.z;

	uint thy = threadIdx.y;	// les 8 qui vont atomicAdd les df en chaque point qui lui est associé

	__shared__ float __x[N];
	__shared__ float __diff_x[N-1];

	FOR(0, ___t, BLOQUE_T) {
		FOR(0, _b, bloques) {
			uint _t = ___t + blockIdx.x * blockDim.x;
			uint _f = threadIdx.z + blockIdx.z * blockDim.z;

			if (_t < T && _b < bloques && _f < f_par_bloque) {
				//
				if (thy == 0) {
					__x[thz] = x[ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR + thz];
					if (thz != 0)
						__diff_x[thz] = dif_x[ligne[_b]*PRIXS*N_FLTR + (depart+_t)*N_FLTR + thz];
				}
				__syncthreads();
				//
				float * __df = df + _b*f_par_bloque*N + _f*N;
				float * __f = f + _b*f_par_bloque*N + _f*N;
				//
				float _dy0 = dy[(depart+_t)*bloques*f_par_bloque + _b*f_par_bloque + _f];
				float ds = locd[(depart+_t)*bloques*(f_par_bloque*2) + _b*(f_par_bloque*2) + _f*2+0] * _dy0 / 8;
				float dd = locd[(depart+_t)*bloques*(f_par_bloque*2) + _b*(f_par_bloque*2) + _f*2+1] * _dy0 / 7;
				//
				atomicAdd(&__df[thy], ds * 1 / (2*sqrtf(1 + fabs(__x[thy] - __f[thy]))) * (-1) * cuda_signe(x[thy] - __f[thy]));
				if (thy != 0) {
					atomicAdd(&__df[ thy ], dd * 2 * (1 + fabs(__diff_x[thy] - (__f[thy]-__f[thy-1]))) * cuda_signe(__diff_x[thy] - (__f[thy]-__f[thy-1])) * (-1));
					atomicAdd(&__df[thy-1], dd * 2 * (1 + fabs(__diff_x[thy] - (__f[thy]-__f[thy-1]))) * cuda_signe(__diff_x[thy] - (__f[thy]-__f[thy-1])) * (+1));
				}
				__syncthreads();
			}
		}
	}
};

void d_nvidia_filtres_naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df)
{
	d_kerd_filtre_naive<<<dim3(DIV(T, BLOQUE_T), 1, KERD(f_par_bloque, BLOQUE_FB)), dim3(1, N, BLOQUE_FB)>>>(
		depart, T,
		bloques, f_par_bloque, ligne,
		x, dif_x,
		f,
		y,
		locd,
		dy,
		df);
	ATTENDRE_CUDA();
}