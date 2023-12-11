/*
 * APS_BPF.h
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */
#include<gsl/gsl_rng.h>

#ifndef APS_BPF_H_
#define APS_BPF_H_

void APS_CreateInitialSample(double *X, long size, int state_dim, gsl_rng *rng);
void APS_DebugLikelihood(double *X, double *W, long size);
void debug_bpf(double *data, double *history, int n_data, FILE *out, int iter);

#endif /* APS_BPF_H_ */
