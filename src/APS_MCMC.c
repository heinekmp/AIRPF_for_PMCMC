/*
 * APS_MCMC.c
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */
#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_cdf.h>

const double DIFFUSION_PROPOSAL_SD = 0.1/2.5;
const double JUMP_INTENSITY_PROPOSAL_SD = 20/2;
const double JUMP_SIZE_PROPOSAL_SD = 0.2;
const double JUMP_CORR_SD = 0.15;
const double JUMP_SIZE_CORR_SD = 0.15;
const double DIFF_CORR_SD = 0.15;

void APS_GenerateProposal(double *current,
			  double *proposal,
			  int dim,
			  gsl_rng *rng) {
  proposal[0] = gsl_ran_beta(rng, (double) 365 * current[0], (double) 365);
}


void APS_GenerateProposal2D(double *current,
			    double *proposal,
			    int dim,
			    gsl_rng *rng) {
  if(current[0] <= 0)
    printf("Proposal error!\n");
  proposal[0] = gsl_ran_beta(rng, (double) 4, (double) 365);
}


double APS_ProposalDensity(double *proposal,
			   double *current,
			   int dim) {
  
  double density = 1;
  
  for (int i = 0; i < dim; i++)
    density *= gsl_ran_beta_pdf(proposal[i], (double) 356 * current[0], (double) 365);
  
  return density;
}

double APS_ProposalDensity2D(double *proposal,
			     double *current,
			     int dim) {
  
  double sd[3] = { DIFFUSION_PROPOSAL_SD, JUMP_INTENSITY_PROPOSAL_SD, JUMP_SIZE_PROPOSAL_SD };
  double density = 1;
  
  for (int i = 0; i < 6; i++)
    density *= gsl_ran_gaussian_pdf(proposal[i] - current[i], sd[i % 3])
      / ((double) 1.0 - gsl_cdf_gaussian_P(-current[i], sd[i % 3]));
  
  density *= gsl_ran_gaussian_pdf(proposal[6] - current[6], JUMP_CORR_SD)
    / ((double) 1.0 - gsl_cdf_gaussian_P(-current[6] - 1, JUMP_CORR_SD)
       - gsl_cdf_gaussian_Q(1 - current[6], JUMP_CORR_SD));
  
  density *= gsl_ran_gaussian_pdf(proposal[7] - current[7], JUMP_SIZE_CORR_SD)
    / ((double) 1.0 - gsl_cdf_gaussian_P(-current[7] - 1, JUMP_SIZE_CORR_SD)
       - gsl_cdf_gaussian_Q(1 - current[7], JUMP_SIZE_CORR_SD));
  
  density *= gsl_ran_gaussian_pdf(proposal[8] - current[8], DIFF_CORR_SD)
    / ((double) 1.0 - gsl_cdf_gaussian_P(-current[8] - 1, DIFF_CORR_SD)
       - gsl_cdf_gaussian_Q(1 - current[8], DIFF_CORR_SD));
  
  return density;
}


double APS_ParametrePrior(double *parm,
			  int dim) {

  return gsl_ran_beta_pdf(parm[0], (double) 4, (double) 365);

}

double APS_ParametrePrior2D(double *parm,
			    int dim) {
  
  double d1 = 1, d2 = 1, d3 = 1, d4 = 1, d5 = 1, d6 = 1, d7 = 1, d8 = 1, d9 = 1;
  
  // Parameters are shape and scale
  d1 = gsl_ran_gamma_pdf(parm[0], 1, 0.05);
  d2 = gsl_ran_gamma_pdf(parm[1], 2.5, 4);
  d3 = gsl_ran_gamma_pdf(parm[2], 2, 0.5);
  d4 = gsl_ran_gamma_pdf(parm[3], 1, 0.05);
  d5 = gsl_ran_gamma_pdf(parm[4], 2.5, 4);
  d6 = gsl_ran_gamma_pdf(parm[5], 2, 0.5);
  
  // We take uniform priors for the correlations
  
  return d1 * d2 * d3 * d4 * d5 * d6 * d7 * d8 * d9;
  
}


void APS_ProposalDebug(double *current,
		       double *proposal,
		       int dim) {

	FILE *file = fopen("proposal_debug.txt", "w");
	for (int i = 0; i < dim; i++) {
		fprintf(file, "%10e %10e\n", current[i], proposal[i]);
	}
	fclose(file);
	printf("Paused by Proposal debug\n");
	getchar();
}

