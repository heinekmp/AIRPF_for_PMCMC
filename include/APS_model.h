/*
 * APS_model.h
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */
#include<gsl/gsl_rng.h>

#ifndef APS_MODEL_H_
#define APS_MODEL_H_

void APS_Mutate(double *x, long size, double *parm, int parm_dim, gsl_rng *rng, double *iw);
void APS_Mutate2D(double *x, long size, double *parm, int parm_dim, gsl_rng *rng, double *iw);
void APS_Likelihood(unsigned int *observation, int data_dim, double *x, long size, double *w, double *iw,
		double *normaliser, double *ess);
void APS_Int2Double(int *data, double *ddata, int len);
void APS_MutationDebug(double *x, double *jumps, long size, char *filename);

#endif /* APS_MODEL_H_ */
