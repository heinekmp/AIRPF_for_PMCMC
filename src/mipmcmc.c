/*
 * mipmcmc.c
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */

#include<gsl/gsl_rng.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include<mpi.h>

#include "myMPICollectives.h"

#include "APS_fileio.h"
#include "APS_MCMC.h"
#include "APS_model.h"
#include "APS_BPF.h"
#include "MIPMCMC_BPF.h"

const int MAX_DATA_ENTRIES = 10000;

int sign(double x) {
	return (x > 0) - (x < 0);
}

void run_ml_variance_test(double *data, int len, int data_dim, long size, double* parm,
			  int parm_dim, int state_dim, int *Ms, gsl_rng *rng,
			  double *marginal_likelihood, double* history, double *ave_ess,
			  int rank, int world_size, double *occupation, double *times, int Niter);
void ml_variance(FILE* ml_var_out, double *MLs, int iter, int NMCMC, int rank, int world_size);
void recalculate_pess(double *ml_seq, int len, int iter, int world_size);
void read_final_state(char *final_state_file_name, double *trace, int trace_dim);
int parse_command_line(int argc, char *argv[], char *data_file_name, int *N,
			int *NMCMC, char *out_file_name, int *method, int *data_dim,
		       char *start_state_file, int rank);
int *create_stagewise_samplesizes(int world_size, short *number_of_stages, int NSMC);
int get_parm_dimension(int data_dim);
int get_state_dimension(int data_dim);
void initialise_derived_variable_statistics(int data_dim,  double *current_track,
					    int sgnl_len, double *ave_aux, double *curr_aux,
					    double **drift, double **p_jump, double **n_jump,
					    double ** curr_n_jump, double **curr_p_jump);
int create_output_files(FILE** out, FILE** time_out, FILE** track_out, FILE** final_state_out,
			FILE** latent_trace, FILE** enf_out, char *out_file_name, int NSMC, int NMCMC,
			int world_size, int method, int rank);
void timing_output(int j, FILE *time_out, time_t *stopwatch, time_t *secondstopwatch);
void generate_proposal(int data_dim, double *current, int parm_dim, double *proposal,
		       gsl_rng *rng);
void run_SMC(int method, double *data, int n_data, int data_dim, int NSMC, double *parm,
	     int parm_dim, int state_dim, int *Ms, gsl_rng *rng, double *ml, double *history,
	     double *ESS, int rank, int world_size, double *occupation, double *times, double* enf);
double acceptance_probability(double *current, double *proposal, int parm_dim, double current_ml,
			      double *ml, int data_dim);
void run_variance_test(int method, double *data, int n_data, int data_dim, int NSMC, double *parm,
	     int parm_dim, int state_dim, int *Ms, gsl_rng *rng, double *ml, double *history,
		       double *ESS, int rank, int world_size, double *occupation, double *times);
int main(int argc,
	 char *argv[]) {
  
  clock_t begin = clock();
  
  // Initialize the MPI environment
  // ------------------------------
  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Parse command line
  // ------------------
  char data_file_name[100];
  int NSMC = 0;
  int NMCMC = 0;
  int method;
  int data_dim;
  double *parm;
  double *current;
  char out_file_name[100];
  char start_state_file[100];
    
  if(parse_command_line(argc, argv, data_file_name, &NSMC, &NMCMC, out_file_name, &method,
			&data_dim, start_state_file, rank)==0) {
    printf("Failing\n");
    fflush(stdout);
    MPI_Finalize();
    return 0;
  }

  if(rank == 0) 
    printf("Workers found: \t%i\n", world_size);

  short number_of_stages;
  int *Ms = create_stagewise_samplesizes(world_size, &number_of_stages, NSMC);

  int parm_dim = 1;
  if( parm_dim == 0 ) {
    MPI_Finalize();
    return 0;
  }
  int state_dim = 1;
  parm = (double*) malloc(parm_dim * sizeof(double));

  // Allocate memory for the trace
  // -----------------------------
  int total_dim = parm_dim + 4; // The additional 4 dimensions are hard coded intentionally
  double *trace = (double*) malloc(NMCMC * total_dim * sizeof(double));
  // Pointers to different parts of the trace
  double *curr_state, *prev_state;

  read_final_state(start_state_file, trace, total_dim);
  
  // Read data
  // ---------
  int *rawdata = (int*) malloc(MAX_DATA_ENTRIES * sizeof(int));
  int n_data = APS_ReadDataFile(data_file_name, rawdata);
  if(rank == 0){
    printf("%i data elements read from %s\n", n_data, data_file_name);
  }

  // Convert the integer data to double precision and discard the integer data
  // (this is done to increase general purposeness of the algorithms).
  double *data = (double*) malloc(n_data * sizeof(double));
  APS_Int2Double(rawdata, data, n_data);
  free(rawdata);

  // Allocate memory for summarised statistic outputs
  // ------------------------------------------------
  int aux_stats = 0; // Number of derived statistics
  int sgnl_len = n_data / data_dim; // actual signal length
  double *track = (double*) malloc(2 * (state_dim + aux_stats) * sgnl_len * sizeof(double));
  double *ave_aux = track + state_dim * sgnl_len;
  double *current_track = track + (state_dim + aux_stats) * sgnl_len;

  // Indices to various key quantities in the trace
  int ml_index = parm_dim;
  int alpha_index = parm_dim + 1;
  int prop_ml_index = parm_dim + 3;

  // Set up randomness
  // -----------------
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(rng, (unsigned long int) rank + 42 * (unsigned long int) time(NULL));

  // Few additional variables
  double marginal_likelihood[1] = {0};
  double *history = (double*) malloc(n_data * state_dim * sizeof(double));
  double ESS[1];
  double occupation[2] = {0.0,0.0};
  double times[4] = {0,0,0,0};
  double *ml_seq = (double *)malloc(sgnl_len * (world_size + 2) * sizeof(double)); 
  int track_len = state_dim * sgnl_len;
  int accept_count = 0;
  double alpha, current_ml;
  double *MLs = (double *)malloc(world_size * sizeof(double));
  double *ESSs = (double *)malloc(world_size * sizeof(double));
  double *enf = (double *)malloc(sgnl_len * sizeof(double));
  double *ave_enf = (double *)malloc(sgnl_len * sizeof(double));
  
  // ----------------------
  // Output file management
  // ----------------------
  FILE *track_out;
  FILE *out;
  FILE *time_out;
  FILE *final_state_out;
  FILE *latent_trace_out;
  FILE *enf_out;
  
  create_output_files(&out, &time_out, &track_out, &final_state_out, &latent_trace_out, &enf_out,
		      out_file_name, NSMC, NMCMC, world_size, method, rank);

  time_t stopwatch = clock();
  time_t secondstopwatch = time(NULL);

  // Particle MCMC main loop
  // -----------------------
  if(rank==0)
    printf("\n");
  
  for (int j = 1; j < NMCMC; j++) {
    
    if(rank == 0) {
    
      printf("\b\b\b\b\b\b\b%7i",j);
      fflush(stdout);

      // Write code timing info into a file
      timing_output(j, time_out, &stopwatch, &secondstopwatch);

      // Update the pointers to the current previoud etc. states...
      current = trace + (j - 1) * total_dim;
      // ...and get the current marginal likelihood estimate value
      current_ml = trace[(j - 1) * total_dim + ml_index];

      // Generate proposal. double *parm is an array for the proposal 
      generate_proposal(data_dim, current,  parm_dim, parm, rng);

    }

    // Master broadcasts the proposed parameter setting to all processes
    MPI_Bcast(parm, parm_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Run the BPF
    // -----------
    run_SMC(method, data, n_data, data_dim, NSMC, parm, parm_dim, state_dim, Ms, rng,
	    marginal_likelihood, history, ESS, rank, world_size, occupation, times, enf);

    // Master collects the marginal log likelihoods from all procs
    MPI_Gather(marginal_likelihood, 1, MPI_DOUBLE, MLs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Masater collects the ESS diagnostics from all procs (not essential)
    MPI_Gather(ESS, 1, MPI_DOUBLE, ESSs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {

      // Acceptance probability
      if(j == 1){
	alpha = 1;
      } else {
	alpha = acceptance_probability(current, parm, parm_dim, current_ml, marginal_likelihood,
				       data_dim);
      }

      if(j == 1) {
 	for(int i = 0; i < sgnl_len; i++)
	  ave_enf[i] = enf[i];
      } else {
	for(int i = 0; i < sgnl_len; i++)
	  ave_enf[i] = (ave_enf[i] * (double) (j - 1) + enf[i] ) / (double) j;
      }
          
      // Update trace pointers
      prev_state = current;
      curr_state = trace + total_dim * j;
      
      // Acceptance test
      if (gsl_ran_flat(rng, 0, 1) < alpha) { // ACCEPT
	
	// Parametres
	for (int p = 0; p < parm_dim; p++)
	  curr_state[p] = parm[p];
	
	// Marginal likelihood
	curr_state[ml_index] = marginal_likelihood[0];
	curr_state[alpha_index] = 1;
	
	// Update the track ( and average )
	for (int i = 0; i < track_len; i++) {
	  track[i] = (track[i] * (double) j + history[i]) / (double) (j + 1);
	  current_track[i] = history[i]; // store the current trajectory
	}
	
	accept_count++;
	
      } else { // REJECT

	for (int i = 0; i < total_dim; i++)
	  curr_state[i] = prev_state[i];

	curr_state[alpha_index] = 0;

	// Update the track  average
	for (int i = 0; i < track_len; i++)
	  track[i] = (track[i] * (double) j + current_track[i]) / (double) (j + 1);

      }

      // Regardless of acceptance, make a record of proposed marginal likelihood for
      // diagnostic and analytic purposes
      curr_state[prop_ml_index] = marginal_likelihood[0];
      
      // Write out the trace in a file
      for (int i = 0; i < total_dim; i++)
	fprintf(out, "%10e ", curr_state[i]);
      fprintf(out, "\n");

      // Write out the latent process trace in a file
      for (int i = 0; i < sgnl_len; i++)
	fprintf(latent_trace_out, "%5.4f ", current_track[i]);
      fprintf(latent_trace_out, "\n");

      // flush the stream to make sure file is updated as it is calculated
      fflush(out);

    }
   
  }


  if(rank==0){

    // Print the final state into a file
    // ---------------------------------

    printf("Printing out the final state\n");
    fflush(stdout);
    // Trace
    for (int i = 0; i < total_dim; i++)
      fprintf(final_state_out, "%10e ", trace[total_dim * (NMCMC-1) + i]);

    printf("Trace done\n");
    fflush(stdout);
    // Track sample
    for (int i = 0; i < track_len; i++) {
      fprintf(final_state_out, "%10e ", current_track[i]); // store the current trajectory
    }
    
    fprintf(final_state_out, "\n");

    printf("Done\n");
    fflush(stdout);
    // Close the files 
    fclose(out);
    fclose(time_out);
    fclose(final_state_out);

    // Print smoothed time series
    if (data_dim == 1) {
      
      for (int i = 0; i < sgnl_len; i++) {
	fprintf(track_out, "%i %10e %10e ", i, track[i], track[track_len + i]);
	for (int s = 0; s < aux_stats; s++)
	  fprintf(track_out, "%10e ", ave_aux[s * sgnl_len + i]);
	fprintf(track_out, "\n");
      }
      
    } else if (data_dim == 2) {
      
      for (int i = 0; i < sgnl_len; i++) {
	fprintf(track_out, "%i %10e %10e ", i, track[i],track[track_len + i]);
	for (int s = 0; s < aux_stats; s++)
	  fprintf(track_out, "%10e ", ave_aux[s * sgnl_len + i]);
	fprintf(track_out, "\n");
	    
      }
      
    } else {
      
      printf("Unknown dimension\n");
      
    }
    fclose(track_out);

    for(int i = 0; i < sgnl_len; i++) {
      fprintf(enf_out,"%10e\n",ave_enf[i]);
    }
    fclose(enf_out);
  }
  
  clock_t end = clock();
  double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
  if(rank == 0){
    printf("\nAcceptance ratio %f\tElapsed %f\n\n", (double) accept_count / (double) NMCMC, time_spent);
    fflush(stdout);
  }
  free(data);
  free(history);
  free(track);
  free(trace);
  free(Ms);
  free(parm);
  free(MLs);
  free(ESSs);
  free(ml_seq);

  MPI_Finalize();
  return 0;
}

void run_ml_variance_test(double *data,
			  int len,
			  int data_dim,
			  long size,
			  double* parm,
			  int parm_dim,
			  int state_dim,
			  int *Ms,
			  gsl_rng *rng,
			  double *marginal_likelihood,
			  double* history,
			  double *ave_ess,
			  int rank,
			  int world_size,
			  double *occupation,
			  double *times,
			  int Niter) {
  
  FILE* ml_test_out;
  double *ml_seq = (double *)malloc(3 * len / data_dim * world_size * sizeof(double));
  double *ml_seq_MIPS = ml_seq;
  double *ml_seq_MIPS_naive = ml_seq + len / data_dim * world_size;
  double *ml_seq_NAIVE = ml_seq + 2 * len / data_dim * world_size;
  double tmp, tmp2;
  char filename[500];
  
  if(rank == 0) {
    sprintf(filename, "/home/ba-kheine/MIPMCMC/data/ml_test%i_%ld.txt", world_size, size);
    ml_test_out  = fopen(filename,"w");
  }

  printf("      ");
  fflush(stdout);
  for(int i = 0; i < Niter; i++) {
    
    MIPMCMC_TREE_BPF(data, len, data_dim, size, parm, parm_dim, state_dim, Ms, rng, marginal_likelihood,
		     history, ave_ess, rank, world_size, occupation, times, 0, ml_seq_MIPS);
    
    MIPMCMC_TREE_BPF(data, len, data_dim, size, parm, parm_dim, state_dim, Ms, rng, marginal_likelihood,
    		     history, ave_ess, rank, world_size, occupation, times, 1, ml_seq_MIPS_naive);
        
    MIPMCMC_Naive_BPF(data, len, data_dim, size, parm, parm_dim, state_dim, rng, marginal_likelihood,
		      history, ave_ess, rank, world_size, ml_seq_NAIVE);
    
    if(rank == 0) {

      for(int j = 0; j < len / data_dim; j++) {
	
	fprintf(ml_test_out,"%i ", i);

	// MIPS
	tmp = 0;
	tmp2 = 0;
	for(int p = 0; p < world_size; p++) {
	  tmp += exp(ml_seq_MIPS[j * world_size + p]);
	}
	for(int p = 0; p < world_size; p++) {
	  tmp2 += (exp(ml_seq_MIPS[j * world_size + p]) / tmp) * (exp(ml_seq_MIPS[j * world_size + p]) / tmp);
	}
	fprintf(ml_test_out,"%10.6e %10.6e ",(double) 1.0 / tmp2, tmp / (double) world_size);

	// Naive MIPS
	tmp = 0;
	tmp2 = 0;
	for(int p = 0; p < world_size; p++) {
	  tmp += exp(ml_seq_MIPS_naive[j * world_size + p]);
	}
	for(int p = 0; p < world_size; p++) {
	  tmp2 += (exp(ml_seq_MIPS_naive[j * world_size + p]) / tmp) * (exp(ml_seq_MIPS_naive[j * world_size + p]) / tmp);
	}
	fprintf(ml_test_out,"%10.6e %10.6e ",(double) 1.0 / tmp2, tmp / (double) world_size);
	
	// Naive
	tmp = 0;
	tmp2 = 0;
	for(int p = 0; p < world_size; p++) {
	  tmp += exp(ml_seq_NAIVE[j * world_size + p]);
	}
	for(int p = 0; p < world_size; p++) {
	  tmp2 += (exp(ml_seq_NAIVE[j * world_size + p]) / tmp) * (exp(ml_seq_NAIVE[j * world_size + p]) / tmp);
	}
	fprintf(ml_test_out,"%10.6e %10.6e ",(double) 1.0 / tmp2, tmp / (double) world_size);

	fprintf(ml_test_out,"\n");
      }

      printf("\b\b\b\b\b\b%6i",i);
      fflush(stdout);

    }
       
  }
  if(rank == 0) {
    fclose(ml_test_out);
    printf("\n");
  }
  free(ml_seq);
}

void ml_variance(FILE* ml_var_out,
		 double *MLs,
		 int iter,
		 int NMCMC,
		 int rank,
		 int world_size) {
 
  double sample_mu = 0, sample_var = 0;
  
  if(rank == 0) { // compute the variance of MLs
      
    // Sample average
    for(int r = 0; r < world_size; r++)
      sample_mu += MLs[r]/ (double) world_size;
    
    // Sample variance
    for(int r = 0; r < world_size; r++)
      sample_var += (MLs[r] - sample_mu) * (MLs[r] - sample_mu) / (double) (world_size-1);
    
    fprintf(ml_var_out, "%i %5.6e\n", iter, sample_var);
  }  
}

void recalculate_pess(double *ml_seq,
		      int len,
		      int iter,
		      int world_size) {

  double normaliser = 0;
  double normal_weight;
  double tmp;
  
  for(int i = 0; i < len; i++) {

    normaliser = 0;
    
    for(int p = 0; p < world_size; p++) 
      normaliser += exp(ml_seq[i * world_size + p]);

    tmp = 0;
    for(int p = 0; p < world_size; p++) {

      normal_weight = exp(ml_seq[i * world_size + p]) / normaliser;
      tmp += normal_weight * normal_weight;
      
    }
    
    if(iter > 1) {
      ml_seq[len * world_size + i] = (ml_seq[len * world_size + i] * (double) (iter - 1) + 
				      (double) 1.0 / tmp ) / (double) (iter);
      ml_seq[len * (world_size + 1) + i] = (ml_seq[len * (world_size + 1) + i] * (double) (iter - 1) +
					    (double) 1.0 / (tmp * tmp)) / (double) iter;
    } else {
      ml_seq[len * world_size + i] = (double) 1.0 / tmp;
      ml_seq[len * (world_size + 1) + i] = (double) 1.0 / (tmp * tmp);
    }
  }
}
  
void read_final_state(char *final_state_file_name,
		      double *trace,
		      int trace_dim) {
    
  FILE *input_file = fopen(final_state_file_name, "r");

  if(input_file) {
    
  
  size_t linelength = 10000;
  char *line = (char*) malloc(linelength * sizeof(char));
  int element_count = 0;
  const char s[2] = " ";
  char *token;
  
  while (fgets(line, linelength, input_file)) {
    
    /* get the first token */
    token = strtok(line, s);
    trace[element_count++] = atof(token);
    
    /* walk through other tokens */
    while (token != NULL && element_count < trace_dim) {
      
      token = strtok(NULL, s);
      trace[element_count++] = atof(token);
      
    }
    
  }

  fclose(input_file);
  free(line);
  } else {
    printf("Starting state file (%s) not found.\n", final_state_file_name);
    fflush(stdout);

    trace[0] = 0.02;
    //trace[1] = 0.05;
    for(int i = 1; i < trace_dim; i++)
      trace[i] = 0.0;
  }
}

int parse_command_line(int argc,
		       char *argv[],
		       char *data_file_name,
		       int *N,
		       int *NMCMC,
		       char *out_file_name,
		       int *method,
		       int *data_dim,
		       char *start_state_file,
		       int rank) {

  const int required_argument_count = 8;

  // Terminate for illegal command line
  if (argc < required_argument_count) {
    
    if(rank == 0) {
      
      printf("Usage: mpimcmc data_file.txt NSMC NMCMC method data_dim outfile start_state");
      printf("starting_state_file\n");
      printf("NB! outfile must have extension .txtout\n");
      
    }
    
    return 0;
  }

  strcpy(data_file_name,argv[1]);
  N[0] = atoi(argv[2]);
  NMCMC[0] = atoi(argv[3]);
  method[0] = atoi(argv[4]);
  data_dim[0] = atoi(argv[5]);
  strcpy(out_file_name, argv[6]);
  strcpy(start_state_file, argv[7]);
  
  if (rank == 0) {
    printf("Input file.....:%s\n", data_file_name);
    printf("N..............:%i\n", N[0]);
    printf("NMCMC..........:%i\n", NMCMC[0]);
    printf("Output.........:%s\n", out_file_name);
    printf("Method.........:%i\n", method[0]);
    printf("Data dimension.:%i\n", data_dim[0]);
    printf("Start state....:%s\n", start_state_file);
  }

  return 1;
}

int *create_stagewise_samplesizes(int world_size,
				  short *number_of_stages,
				  int NSMC) {

  /* Within processor resampling is the zeroth stage, so the number_of_stages is
     log2(world_size) + 1 */
  number_of_stages[0] = (short) log2(world_size) + 1;
  
  /* Determine the stage specific sample sizes. These numbers are the maximum number of
     particles communicated between processes */
  int* Ms = (int*) malloc((number_of_stages[0] + 1) * sizeof(int));
  for(int i = 0; i < number_of_stages[0] + 1; i++)
    Ms[i] = (int)NSMC; 
  return Ms;
}

int get_parm_dimension(int data_dim) {
  
  int parm_dim;
  
  if(data_dim == 1) {
    
    parm_dim = 3;
  
  } else if (data_dim == 2) {
    
    parm_dim = 9;
  
  } else {
    printf("Unknown data dimension! Can be only 1 or 2.\n");
    parm_dim = 0;
  }
  
  return parm_dim;
}

int get_state_dimension(int data_dim) {
  if (data_dim == 1)
    return 2;
  if (data_dim == 2)
    return 4;
  return 0;
}

void initialise_derived_variable_statistics(int data_dim,
					    double *current_track,
					    int sgnl_len,
					    double *ave_aux,
					    double *curr_aux,
					    double **drift,
					    double **p_jump,
					    double **n_jump,
					    double ** curr_n_jump,
					    double **curr_p_jump) {

  double diff;

  for (int s = 0; s < data_dim; s++) {
    
    drift[s] = current_track + (2 * s + 1) * sgnl_len;
    n_jump[s] = ave_aux + 2 * s * sgnl_len;
    p_jump[s] = ave_aux + (2 * s + 1) * sgnl_len;
    curr_n_jump[s] = curr_aux + 2 * s * sgnl_len;
    curr_p_jump[s] = curr_aux + (2 * s + 1) * sgnl_len;
    
    for (int i = 0; i < (sgnl_len - 1); i++) {
      
      diff = drift[s][i + 1] - drift[s][i];
      
      n_jump[s][i] = sign(diff) == -1 ? 1.0 : 0.0;
      p_jump[s][i] = sign(diff) == 1 ? 1.0 : 0.0;
      
      curr_n_jump[s][i] = n_jump[s][i];
      curr_p_jump[s][i] = p_jump[s][i];
      
    }
    
    n_jump[s][sgnl_len - 1] = 0;
    p_jump[s][sgnl_len - 1] = 0;
    
    curr_n_jump[s][sgnl_len - 1] = 0;
    curr_p_jump[s][sgnl_len - 1] = 0;
    
  }

}

int create_output_files(FILE** out,
			FILE** time_out,
			FILE** track_out,
			FILE** final_state_out,
			FILE** latent_trace,
			FILE** enf_out,
			char *out_file_name,
			int NSMC,
			int NMCMC,
			int world_size,
			int method,
			int rank) {
  if(rank != 0)
    return -1;

  char track_file_name[150];
  char timing_file_name[150];
  char final_state_out_name[150];
  char latent_trace_out_name[150];
  char enf_out_name[150];

  // The main trce output
  out[0] = fopen(out_file_name, "w");

  // Timing code timing info for analysis and optimisation
  // TODO: Remove the hard coding
  sprintf(timing_file_name,"/home/ba-kheine/MIPMCMC3/data/timing_%i_%i_%i_%i.txt",
	  NSMC, method, NMCMC, world_size);
  time_out[0] = fopen(timing_file_name,"w");
  
  printf("Timing data output file: %s\n",timing_file_name);
  fflush(stdout);
  
    // Store final state of the chain
  sprintf(final_state_out_name,"%s_final",out_file_name);
  final_state_out[0] = fopen(final_state_out_name,"w");
  printf("Final state output file: %s\n",final_state_out_name);
  fflush(stdout);
  
  // Track output filename manipulations
  // -----------------------
  // Taking out 7 elements means removing the extension .txtout, which has 7 character
  strcpy(track_file_name, out_file_name);
  track_file_name[strlen(out_file_name) - 7] = '\0';
  strcat(track_file_name, "_track.txt");
  
  printf("Smoothed time series output file: %s\n",track_file_name);
  track_out[0] = fopen(track_file_name, "w");

  // Trace of the latent process
  strcpy(latent_trace_out_name, out_file_name);
  latent_trace_out_name[strlen(out_file_name) - 7] = '\0';
  strcat(latent_trace_out_name, "_track_trace.txt");
  
  printf("Latent process trace output file: %s\n",latent_trace_out_name);
  latent_trace[0] = fopen(latent_trace_out_name, "w");

  // Effective number of filters
  strcpy(enf_out_name, out_file_name);
  enf_out_name[strlen(out_file_name) - 7] = '\0';
  strcat(enf_out_name, "_enf.txt");
  
  printf("Average ENF output file: %s\n",enf_out_name);
  enf_out[0] = fopen(enf_out_name, "w");

  return 0;

}

void timing_output(int j,
		   FILE *time_out,
		   time_t *stopwatch,
		   time_t *secondstopwatch) {
  
  // Print elapsed time into a file on each 1000 iterations
  // ------------------------------------------------------
  if(j % 1000 == 0) {
    
    fprintf(time_out,"%5.4f %i\n",(double) (clock() - stopwatch[0]) / CLOCKS_PER_SEC,
	    (int)(time(NULL)-secondstopwatch[0]));
    fflush(time_out);
    stopwatch[0] = clock();
    secondstopwatch[0] = time(NULL);
  }
  
}

void generate_proposal(int data_dim,
		       double *current,
		       int parm_dim,
		       double *proposal,
		       gsl_rng *rng) {
  
  // Use different proposal generation function depending on the dimension of the problem
  if(data_dim == 1)    
    APS_GenerateProposal(current, proposal, parm_dim, rng);
  else
    APS_GenerateProposal2D(current, proposal, parm_dim, rng);
  
}

void run_SMC(int method,
	     double *data,
	     int n_data,
	     int data_dim,
	     int NSMC,
	     double *parm,
	     int parm_dim,
	     int state_dim,
	     int *Ms,
	     gsl_rng *rng,
	     double *ml,
	     double *history,
	     double *ESS,
	     int rank,
	     int world_size,
	     double *occupation,
	     double *times,
	     double* enf) {
  
  if(method == 1) {
    
    MIPMCMC_TREE_BPF(data, n_data, data_dim, NSMC, parm, parm_dim, state_dim, Ms, rng,
		     ml, history, ESS, rank, world_size, occupation,
		     times,0, enf);
    
  } else if(method == 0)  {
    
    MIPMCMC_Naive_BPF(data, n_data, data_dim, NSMC, parm, parm_dim, state_dim, rng,
		      ml, history, ESS, rank, world_size, enf);
    
  }
  
}

void run_variance_test(int method,
		       double *data,
		       int n_data,
		       int data_dim,
		       int NSMC,
		       double *parm,
		       int parm_dim,
		       int state_dim,
		       int *Ms,
		       gsl_rng *rng,
		       double *ml,
		       double *history,
		       double *ESS,
		       int rank,
		       int world_size,
		       double *occupation,
		       double *times) {

  int NMC = 100000;
  int sgnl_len = n_data/data_dim;
  double * ml_seq = (double *) malloc(sgnl_len * sizeof(double));
  double * ml_ave_0 = (double *) malloc(sgnl_len * sizeof(double));
  double * ml_ave_1 = (double *) malloc(sgnl_len * sizeof(double));
  double * ml_ave_01 = (double *) malloc(sgnl_len * sizeof(double));
  double * ml_ave_11 = (double *) malloc(sgnl_len * sizeof(double));
  char filename[100];

  FILE *variance_out;
  if(rank == 0) {
    sprintf(filename,"/home/ba-kheine/MIPMCMC3/data/var_N%i_m%i_%ld.txt",NSMC,world_size,time(NULL));
    variance_out = fopen(filename,"w");
  }

  for(int n = 0; n < sgnl_len; n++) {
    ml_ave_1[n] = 0;
    ml_ave_0[n] = 0;
    ml_ave_11[n] = 0;
    ml_ave_01[n] = 0;
    
  }
  
  for(int i = 0; i < NMC; i++) {
    
    if(rank == 0) {
      printf("%i\n",i);
      fflush(stdout);
    }
    
    MIPMCMC_TREE_BPF(data, n_data, data_dim, NSMC, parm, parm_dim, state_dim, Ms, rng,
		     ml, history, ESS, rank, world_size, occupation,
		     times,0, ml_seq);
    
    for(int n = 0; n < sgnl_len; n++) {
      ml_ave_1[n] += ml_seq[n] * ml_seq[n];
      ml_ave_11[n] += ml_seq[n];
    }
    
    MIPMCMC_Naive_BPF(data, n_data, data_dim, NSMC, parm, parm_dim, state_dim, rng,
		      ml, history, ESS, rank, world_size, ml_seq);

    for(int n = 0; n < sgnl_len; n++) {
      ml_ave_0[n] += ml_seq[n] * ml_seq[n];
      ml_ave_01[n] += ml_seq[n];
    }

    if(i % 1000 == 0 && rank == 0) {

      variance_out = fopen(filename,"w");
      
      for(int n = 0; n < sgnl_len; n++)
	fprintf(variance_out, "%i %i %10.12e %10.12e\n",0,1,ml_ave_1[n] / (double)NMC, ml_ave_11[n] / (double)NMC);
      
      for(int n = 0; n < sgnl_len; n++) 
	fprintf(variance_out, "%i %i %10.12e %10.12e\n",0,0,ml_ave_0[n] / (double)NMC, ml_ave_01[n] / (double)NMC);  
      
      fclose(variance_out);      
    }
    
  }
    
  free(ml_seq);
  free(ml_ave_0);
  free(ml_ave_1);
  free(ml_ave_01);
  free(ml_ave_11);
}

double acceptance_probability(double *current,
			      double *proposal,
			      int parm_dim,
			      double current_ml,
			      double *ml,
			      int data_dim) {

  double proposal_ratio;
  double prior_ratio;
  double marginal_likelihood_ratio;
  
  // Proposal and prior density ratios
    
    proposal_ratio = APS_ProposalDensity(current, proposal, parm_dim)
      / APS_ProposalDensity(proposal, current, parm_dim);
    
    prior_ratio = APS_ParametrePrior(proposal, parm_dim)
      / APS_ParametrePrior(current, parm_dim);
    
  // Marginal (log) likelihood ratio
  marginal_likelihood_ratio = exp(ml[0] - current_ml);
  
  double alpha = proposal_ratio * marginal_likelihood_ratio * prior_ratio;
  alpha = alpha > 1 ? 1 : alpha;

  return alpha;
}

