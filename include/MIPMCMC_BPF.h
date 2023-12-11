/*
 * MIPMCMC_BPF.h
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */
#include<gsl/gsl_rng.h>

#ifndef MIPMCMC_BPF_H_
#define MIPMCMC_BPF_H_

void MIPMCMC_TREE_BPF(double *data, int len, int data_dim, long size, double *parm, int parm_dim, int state_dim, int *Ms,
		      gsl_rng *rng, double *marginal_likelihood, double *history, double *ave_ess, int rank, int world_size,
		      double *occupation, double* times, int naive, double *ml_seq);
void MIPMCMC_BPF(double *data, int len, int data_dim, long size, double *parm, int parm_dim, int state_dim, int *Ms,
		 gsl_rng *rng, double *marginal_likelihood, double *history, double *ave_ess, int rank, int world_size);
void BPFresample_old(long size, double *w, long *ind, gsl_rng *r, double normaliser);
void MIPMCMC_Naive_BPF(double *data, int len, int data_dim, long size, double *parm, int parm_dim, int state_dim,
		       gsl_rng *rng, double *marginal_likelihood, double *history, double *ave_ess,
		       int rank, int world_size, double *ml_seq);
#endif /* MIPMCMC_BPF_H_ */

