#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static void plume_pred(Mdl_t * mdl, uint t0, uint t1) {
	float * ancien = mdl_pred(mdl, t0, t1, 3);
	printf("PRED GENERALE = ");
	FOR(0, p, P) printf(" %f%% ", 100*ancien[p]);
	printf("\n");
	free(ancien);
};

float pourcent_masque_nulle[C] = {0};

float pourcent_masque[C] = {0.10};
	/*0.00,
	0.40,
	0.10,0.10,
	0.10,0.10,0.10,
	0.10,0.10,0.10,0.10,
	0.20,0.20,0.20,
	0.20,0.20,0.20,0.20,
	0.20,0.30,0.30,0.20,
	0.00
};*/

float * alpha = de_a(5e-4, 5e-4, C);

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");   charger_tout();

	//	-- Verification --
	titre("Verifier MDL");     verif_mdl_1e5();

	//===============
	titre("  Programme Generale  ");

	//	Verifier si tout est bon (verification verif)
	//	reecrire l'integralitée des prixs/macd/volumes

	/*uint ST[C] = {
		512,
		256,
		128,128,
		64,64,
		32,32,32,
		16,16,16,
		8,8,8,8,8,8,8,
		4,4,4,
		P
	};
	uint activations[C] = {TANH};
	uint bloques      = 64;
	uint f_par_bloque =  8;
	uint lignes[bloques] = {
		0,0,0,0,0,0,0,0,
		1,1,1,2,2,2,3,3,3,4,4,4,
		5,5,6,6,7,7,
		8,8,9,9,10,10,
		11,11,12,12,13,13,14,15,16,17,
		18,18,
		19,19,
		20,20,
		21,21,
		22,22,
		23,23,
		24,25,26,27,
		28,29,31,32,33,38
	};
	Mdl_t * mdl = cree_mdl(ST, activations, bloques, f_par_bloque, lignes);*/

	/*Mdl_t * mdl = ouvrire_mdl("mdl.bin");

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = ROND_MODULO(FIN, 16);
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%32=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%16);
	//
	plume_pred(mdl, t0, t1);
	//
	INIT_CHRONO(chrono)
	//
	DEPART_CHRONO(chrono)
	mdl_aller_retour(mdl, t0, t1, 3);
	float sec_opti = VALEUR_CHRONO(chrono);
	uint OPTIMISATIONS = 150*1500;
	printf("\033[3;92;m1 opti = %+f s, donc %i*%+f = %+f s = %+f mins\033[0m\n",
		sec_opti,
		OPTIMISATIONS, sec_opti,
		OPTIMISATIONS * sec_opti,
		OPTIMISATIONS * sec_opti / 60.0);
	//
	uint REP = 150;
	FOR(0, rep, REP) {
		FOR(0, i, 5) {
			optimisation_mini_packet(
				mdl,
				t0, t1, 16*300,
				alpha, 1.0,
				RMSPROP, 1500,
				pourcent_masque);
			plume_pred(mdl, t0, t1);
			mdl_gpu_vers_cpu(mdl);
			ecrire_mdl(mdl, "mdl.bin");
		}
		//
		optimiser(
			mdl,
			t0, t1,
			alpha, 1.0,
			RMSPROP, 100,
			pourcent_masque_nulle);
		//
		mdl_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, "mdl.bin");
		plume_pred(mdl, t0, t1);
		printf("===================================================\n");
		printf("==================TERMINE %i/%i=======================\n", rep+1, REP);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);*/

	//	-- Fin --
	liberer_tout();
};