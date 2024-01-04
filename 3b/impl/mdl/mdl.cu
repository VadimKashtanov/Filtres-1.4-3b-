#include "mdl.cuh"

#include "filtres_prixs.cuh"
#include "dot1d.cuh"
#include "lstm1d.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

mdl_cree_f cree_inst[INSTS] = {
	cree_filtre_prixs,
	cree_dot1d,
	cree_lstm1d
};

mdl_f_f inst_f [INSTS] = {
	f_filtres_prixs,
	f_dot1d,
	f_lstm1d
};

mdl_f_f inst_df[INSTS] = {
	df_filtres_prixs,
	df_dot1d,
	df_lstm1d
};

char * nom_inst[INSTS] = {
	"filtres_prixs",
	"dot1d",
	"lstm1d"
}

PAS_OPTIMISER()
Mdl_t * cree_mdl(uint Y[C], uint insts[C], uint lignes[C]) {
	ASSERT(Y[C-1] == P);
	ASSERT(Y[ 0 ] == 0);
	ASSERT(Y[ 0 ] == BLOQUES * F_PAR_BLOQUES);
	
	Mdl_t * mdl = alloc<Mdl_t>(1);

	//	Architecture
	memcpy(mdl->insts,   insts, sizeof(uint) * C);
	memcpy(mdl->Y,           Y, sizeof(uint) * C);
	memcpy(mdl->lignes, lignes, sizeof(uint) * BLOQUES);
	mdl->lignes__d = cpu_vers_gpu<uint>(lignes, BLOQUES);

	//	Allocation
	FOR(0, c, C) {
		ASSERT(insts[c] != 0);
		ASSERT(Y[c] <= MAX_Y);
		//
		cree_inst[insts[c]](mdl, c);
		//
		mdl->p [c] = alloc<float>(mdl->inst_POIDS[c]);
		mdl->y [c] = alloc<float>(mdl->inst_VARS [c]);
		mdl->l [c] = alloc<float>(mdl->inst_LOCDS[c]);
		mdl->dy[c] = alloc<float>(mdl->inst_VARS [c]);
		mdl->dp[c] = alloc<float>(mdl->inst_POIDS[c]);
		//
		FOR(0, i, mdl->inst_POIDS[c]) mdl->p[c][i] = (2*rnd()-1) * 2.0;
		//
		mdl->p__d [c] = cpu_vers_gpu<float>(mdl->p[c], mdl->inst_POIDS[c]);
		mdl->y__d [c] = cudalloc<float>(mdl->inst_VARS [c]);
		mdl->l__d [c] = cudalloc<float>(mdl->inst_LOCDS[c]);
		mdl->dy__d[c] = cudalloc<float>(mdl->inst_VARS [c]);
		mdl->dp__d[c] = cudalloc<float>(mdl->inst_POIDS[c]);
	}
	ASSERT(mdl->inst_DEPART_SORTIE[C-1] == 0);
	//
	mdl_norme_filtres(mdl);
	//
	return mdl;
};

void mdl_norme_filtres(Mdl_t * mdl) {
	FOR(0, f, BLOQUES*F_PAR_BLOQUES) {
		float max=mdl->p[0][f*N+0], min=mdl->p[0][f*N+0];
		FOR(1, i, N) {
			if (max < mdl->p[0][f*N+i]) max = mdl->p[0][f*N+i];
			if (min > mdl->p[0][f*N+i]) min = mdl->p[0][f*N+i];
		}
		FOR(0, i, N) mdl->p[0][f*N+i] = (mdl->p[0][f*N+i]-min)/(max-min);
	};
	CONTROLE_CUDA(cudaMemcpy(mdl->p__d[0], mdl->p[0], sizeof(float)*BLOQUES*F_PAR_BLOQUES*N, cudaMemcpyHostToDevice))
};

PAS_OPTIMISER()
void mdl_verif(Mdl_t * mdl) {
	FOR(1, c, C) {
		float * r = gpu_vers_cpu<float>(mdl->p__d[c], mdl->inst_POIDS[c]);
		FOR(0, i, mdl->inst_POIDS[c]) ASSERT(fabs(r[i]-mdl->p[c][i]) < 0.01);
		free(r);
	}
};

PAS_OPTIMISER()
void mdl_gpu_vers_cpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p[c],  mdl->p__d[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->y[c],  mdl->y__d[c],  sizeof(float)*mdl->inst_VARS[c]*PRIXS,  cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->l[c],  mdl->l__d[c],  sizeof(float)*mdl->inst_LOCDS[c]*PRIXS, cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->dy[c], mdl->dy__d[c], sizeof(float)*mdl->inst_VARS[c]*PRIXS,  cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->dp[c], mdl->dp__d[c], sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyDeviceToHost));
	}
}

PAS_OPTIMISER()
void mdl_cpu_vers_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p__d[c],  mdl->p[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->y__d[c],  mdl->y[c],  sizeof(float)*mdl->inst_VARS[c]*PRIXS,  cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->l__d[c],  mdl->l[c],  sizeof(float)*mdl->inst_LOCDS[c]*PRIXS, cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->dy__d[c], mdl->dy[c], sizeof(float)*mdl->inst_VARS[c]*PRIXS,  cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->dp__d[c], mdl->dp[c], sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
	}
};

PAS_OPTIMISER()
void liberer_mdl(Mdl_t * mdl) {
	CONTROLE_CUDA(cudaFree(mdl->lignes__d));
	FOR(0, c, C) {
		free(mdl->p [c]);
		free(mdl->y [c]);
		free(mdl->l [c]);
		free(mdl->dy[c]);
		free(mdl->dp[c]);
		//
		CONTROLE_CUDA(cudaFree(mdl->p__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->y__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->l__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->dy__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->dp__d[c]));
	}
};

PAS_OPTIMISER()
void mdl_zero_cpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		memset(mdl->y [c], 0, sizeof(float) * mdl->inst_VARS [c]);
		memset(mdl->dy[c], 0, sizeof(float) * mdl->inst_VARS [c]);
		memset(mdl->dp[c], 0, sizeof(float) * mdl->inst_POIDS[c]);
	}
};

PAS_OPTIMISER()
void mdl_zero_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemset(mdl->y__d [c], 0, sizeof(float) * mdl->inst_VARS [c]));
		CONTROLE_CUDA(cudaMemset(mdl->dy__d[c], 0, sizeof(float) * mdl->inst_VARS [c]));
		CONTROLE_CUDA(cudaMemset(mdl->dp__d[c], 0, sizeof(float) * mdl->inst_POIDS[c]));
	}
};