#include "filtres_prixs.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_filtre_prixs(Mdl_t * mdl, uint inst)
{
	mdl->inst_POIDS        [inst] = BLOQUES*F_PAR_BLOQUES*N;
	mdl->inst_VARS         [inst] = mdl->Y[inst];
	mdl->inst_LOCDS        [inst] = 0;
	mdl->inst_SORTIES      [inst] = mdl->Y[inst];
	mdl->inst_DEPART_SORTIE[inst] = mdl->Y[inst] - mdl->Y[inst];
};

static float filtre(float * x, float * dif_x, float * f, float * locd) {
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

void intel_filtres_prixs___naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd)
{
	FOR(0, t, T) {
		FOR(0, b, bloques) {
			FOR(0, _f, f_par_bloque) {
				y[(depart+t)*bloques*f_par_bloque + b*f_par_bloque + _f] = filtre(
						x + ligne[b]*PRIXS*N_FLTR + (depart+t)*N_FLTR,
					dif_x + ligne[b]*PRIXS*N_FLTR + (depart+t)*N_FLTR,
					f     + b*f_par_bloque*N     + _f*N,
					locd  + (depart+t)*(bloques*f_par_bloque*2) + b*(f_par_bloque*2) + _f*2
				);
			}
		}
	}
}

static void d_filtre(float * x, float * dif_x, float * f, float * locd, float * dy, float * df) {
	float ds = locd[0] * dy[0] / 8;
	float dd = locd[1] * dy[0] / 7;

	FOR(1, i, N)
	{
		//s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		df[i] += ds * 1 / (2*sqrtf(1 + fabs(x[i] - f[i]))) * (-1) * signe(x[i] - f[i]);
		//d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
		df[ i ] += dd * 2 * (1 + fabs(dif_x[i] - (f[i]-f[i-1]))) * signe(dif_x[i] - (f[i]-f[i-1])) * (-1);
		df[i-1] += dd * 2 * (1 + fabs(dif_x[i] - (f[i]-f[i-1]))) * signe(dif_x[i] - (f[i]-f[i-1])) * (+1);
	}
	df[0] += ds * 1 / (2*sqrtf(1 + fabs(x[0] - f[0]))) * (-1) * signe(x[0] - f[0]);
};

void  d_intel_filtres_prixs___naive(
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * ligne,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df)
{
	FOR(0, t, T) {
		FOR(0, b, bloques) {
			FOR(0, _f, f_par_bloque) {
				d_filtre(
						x + ligne[b]*PRIXS*N_FLTR + (depart+t)*N_FLTR,
					dif_x + ligne[b]*PRIXS*N_FLTR + (depart+t)*N_FLTR,
					f     + b*f_par_bloque*N     + _f*N,
					locd  + (depart+t)*(bloques*f_par_bloque*2) + b*(f_par_bloque*2) + _f*2,
					dy + (depart+t)*bloques*f_par_bloque + b*f_par_bloque + _f,
					df     + b*f_par_bloque*N     + _f*N
				);
			}
		}
	}
}

void f_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint depart = t0;
	uint T = (t1-t0);
	if (mode == 0) {
		intel_filtres_prixs___naive(
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->lignes,
			normalisee, dif_normalisee,
			mdl->p[inst],
			mdl->y[inst],
			mdl->l[inst]);
	} else if (mode == 1 || mode == 2 || mode == 3) {
		nvidia_filtres_prixs___naive(
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->lignes__d,
			normalisee, dif_normalisee,
			mdl->p[inst],
			mdl->y[inst],
			mdl->l[inst]);
	} else {
		ERR("Pas de mode %i pour mes filtres", mode);
	}
};

void df_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint depart = t0;
	uint T = (t1-t0);
	if (mode == 0) {
		d_intel_filtres_prixs___naive(
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->lignes,
			normalisee, dif_normalisee,
			mdl->p[inst],
			mdl->y[inst],
			mdl->l[inst],
			mdl->dy[inst],
			mdl->dp[inst]);
	if (mode == 1 || mode == 2 || mode == 3) {
		d_nvidia_filtres_prixs___naive(
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->lignes__d,
			normalisee, dif_normalisee,
			mdl->p[inst],
			mdl->y[inst],
			mdl->l[inst],
			mdl->dy__d[inst],
			mdl->dp__d[inst]);
	} else {
		ERR("Pas de mode %i pour mes filtres", mode);
	}
};