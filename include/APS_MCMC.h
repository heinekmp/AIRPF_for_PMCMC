/*
 * APS_MCMC.h
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */
#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>

#ifndef APS_MCMC_H_
#define APS_MCMC_H_

void APS_GenerateProposal(double *current, double *proposal, int dim, gsl_rng *rng);
void APS_GenerateProposal2D(double *current, double *proposal, int dim, gsl_rng *rng);
double APS_ProposalDensity(double *proposal, double *current, int dim);
double APS_ProposalDensity2D(double *proposal, double *current, int dim);
double APS_ParametrePrior(double *parm, int dim);
double APS_ParametrePrior2D(double *parm, int dim);
void APS_ProposalDebug(double *current, double *proposal, int dim);

#endif /* APS_MCMC_H_ */
