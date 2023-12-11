/*
 * APS_model.c
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */
#include<float.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_cdf.h>
#include<math.h>
#include "APS_model.h"
/*
 * double *x		- Particles (resampled) that will be mutated. size-by-2 array
 * long size		- Number of particles
 * double *parm		- Filter parametre array
 * int parm_dim		- Number of parametres
 * gsl_rng *rng		- Random generator
 * double *iw       - Importance contribution to weights, if not using the signal kernel
 */
void APS_Mutate(double *x,
		long size,
		double *parm,
		int parm_dim,
		gsl_rng *rng,
		double *iw) {
  
  double jump_prob = parm[0];
  int articles_per_day = 75;
  double u;
  
  for (int i = 0; i < size; i++) {

    u = gsl_ran_flat(rng, 0, 1);
   
    if(u < jump_prob) {
      x[i] =  gsl_ran_beta(rng, 0.05 * (double)(articles_per_day), (double)articles_per_day);
    }
  }
}

void APS_Mutate2D(double *x,
		  long size,
		  double *parm,
		  int parm_dim,
		  gsl_rng *rng,
		  double *iw) {
  
  double diffusion_sd0 = parm[0];
  double jump_rate0 = parm[1];
  double jump_size_inv0 = (double) parm[2];
  double diffusion_sd1 = parm[3];
  double jump_rate1 = parm[4];
  double jump_size_inv1 = (double) parm[5];
  double rho = parm[6];
  double diff_corr = parm[7];
  double jump_size_corr = parm[8];
  
  double p00, p01, p10, p11, p0, p1;
  p0 = gsl_cdf_exponential_P(1.0, jump_rate0);
  p1 = gsl_cdf_exponential_P(1.0, jump_rate1);
  p11 = rho * sqrt(p0 * (1.0 - p0) * p1 * (1.0 - p1)) + p0 * p1;
  p01 = p0 - p11;
  p10 = p1 - p11;
  p00 = 1.0 - p01 - p10 - p11;
  double u, jump0[1], jump1[1];
  double diffusion_term0[1], diffusion_term1[1];
  
  for (int i = 0; i < size; i++) {
    
    // Jump mutation
    // -------------
    u = gsl_ran_flat(rng, 0, 1);
    if (u < p00) { // No jumps
      jump0[0] = 0;
      jump1[0] = 0;
    } else if (u < p00 + p01) { // first signal jumps bu second does not
      jump0[0] = gsl_ran_gaussian(rng, jump_size_inv0);
      jump1[0] = 0;
    } else if (u < p00 + p01 + p10) { // second jumps but first does not
      jump0[0] = 0;
      jump1[0] = gsl_ran_gaussian(rng, jump_size_inv1);
		} else { // both jump
      gsl_ran_bivariate_gaussian(rng, jump_size_inv0, jump_size_inv1, jump_size_corr, jump0,
				 jump1);
    }
    
    x[size + i] += jump0[0];
    x[3 * size + i] += jump1[0];
    iw[i] = 1;
    
    // Diffusion mutation
    // ------------------
    gsl_ran_bivariate_gaussian(rng, diffusion_sd0, diffusion_sd1, diff_corr, diffusion_term0,
			       diffusion_term1);
    
    x[i] += x[size + i] + diffusion_term0[0];
    x[2 * size + i] += x[3 * size + i] + diffusion_term1[0];
  }
  
}


/*
 * int obervation 	- Integer observation count
 * double *x		- Array of particles *assumed to be size-by-2 where size is the number of particles
 *                    and first column is the diffusion part and the second column is the jump process
 * long size		- Number of particles
 * double *w		- Weights (output)
 * double *normaliser - Normaliser (output)
 */
void APS_Likelihood(unsigned int *observation,
		    int data_dim,
		    double *x,
		    long size,
		    double *w,
		    double *iw,
		    double *normaliser,
		    double *ess) {
  
  normaliser[0] = 0;
  double tmp = 0;
    
  for (int i = 0; i < size; i++) {
      
  
    if(x[i] <= 0)
      printf("Likelihood error! x[i] = %10e\n",x[i]);
    w[i] = gsl_ran_binomial_pdf(observation[0], x[i], observation[1]);
        
    normaliser[0] += w[i];
  }
    
  tmp = 0;
  for(int i = 0; i < size; i++) {
    tmp += (w[i] / normaliser[0]) * (w[i] / normaliser[0]);
  }

  ess[0] = (double) 1.0 / tmp;

}

void APS_Int2Double(int *data,
		    double *ddata,
		    int len) {
  
  for (int i = 0; i < len; i++)
    ddata[i] = (double) data[i];
  
}

void APS_MutationDebug(double *x,
		       double *jumps,
		       long size,
		       char *filename) {

	FILE *file = fopen(filename, "w");

	for (long i = 0; i < size; i++)
		fprintf(file, "%10e %10e %10e\n", x[i], x[size + i], jumps[i]);
	fclose(file);
}

