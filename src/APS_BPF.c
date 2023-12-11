/*
 * APS_BPF.c
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */

#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_cdf.h>
#include<math.h>
#include<stdio.h>

void APS_CreateInitialSample(double *X,
			     long size,
			     int state_dim,
			     gsl_rng *rng) {

  double prior_prob = 0.03;
  double articles_per_day = 75;
  for(int i = 0; i < size; i++) {
    X[i] = gsl_ran_beta(rng, prior_prob * articles_per_day, articles_per_day);
  }
  
}

void APS_DebugLikelihood(double *X,
			 double *W,
			 long size) {
  
  FILE *file = fopen("weight_debug.txt","w");
  
  for(long i = 0; i < size; i++) {
    fprintf(file,"%10e %10e %10e\n", X[i],X[i + size],W[i]);
  }
  
  fclose(file);
  printf("Paused by Likelihood debugging");
  getchar();
}

void debug_bpf(double *data,
	       double *history,
	       int n_data,
	       FILE *out,
	       int iter) {

	for(int i = 0; i < n_data; i++) {
		fprintf(out,"%i %i %f %10e\n",i, iter, data[i],exp(history[i]));
	}

}
