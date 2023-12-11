/*
 * myMPICollectives.c
 *
 *  Created on: 3 Jun 2021
 *      Author: heine
 */

#include"myMPICollectives.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>


void Island_communication(
			  double *sample,
			  double *buffer,
			  double *send_buffer,
			  double *rec_buffer,
			  int dim,
			  int M,
			  int n_results,
			  MPI_Comm comm,
			  gsl_rng *rng)
{

  int world_size = 0;
  int offset;
  int rank;
  double normaliser = 0;
  double *weights;
  
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Create book keeping for processes that have received already */
  int* have_i_received = (int*) malloc(world_size * sizeof(int));;
  for(int i = 0; i < world_size; i++)
    have_i_received[i] = 0;
  
  /* Only root wil do the resampling */
  int* ind = (int*) malloc(world_size * sizeof(int));
  if(rank == 0) {
    
    /* Collect the weights */
    weights = (double*)malloc(world_size*sizeof(double));
    offset = 2 * dim + 1;
    for(int i = 0; i < world_size; i++) {
      weights[i] = buffer[offset + i * n_results];
      normaliser += weights[i];
    }
    
    BPFresample_MIPS(world_size, weights, ind, rng, normaliser, rank);

    free(weights);
  }
  
  /* Let all processes know their ancestors */
  
  MPI_Bcast(ind, world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Prepare the send_buffer */
  move2sendbuffer(sample, dim, send_buffer, M);

  for(int i = 0; i < world_size;i++){
    
    if(have_i_received[i] == 0){
      
      move2sendbuffer(send_buffer,dim,rec_buffer,M);

      MPI_Bcast(rec_buffer, M * dim, MPI_DOUBLE, ind[i], MPI_COMM_WORLD);	
     
      if(ind[rank]==ind[i]) {// if my ancestor is the one broadcasating, read the buffer
	for(int i = 0; i < dim*M;i++) 
	  sample[i] = rec_buffer[i];
      }
      MPI_Barrier(MPI_COMM_WORLD);

      /* Update the reception status */
      for(int j = i; j < world_size; j++) 
	have_i_received[j] = ind[j] == ind[i] ? 1 : 0; 
      
    }
  }

  /* Finally we need to reset the weights */
  for(int i = 0; i < M; i++)
    sample[dim * M + i] = (double)1.0 / (double)M;

  free(have_i_received);
  free(ind);
}

void naive_interaction(double* serial_forest,
		       double *rec_buffer,
		       int* msg_sizes,
		       double *weights,
		       gsl_rng *rng,
		       int world_size,
		       int rank) {

  double my_weight[1] = {weights[0]};
  double *all_weights = (double *) malloc(world_size * sizeof(double));
  double total_w = 0;
  int *parents = (int *) malloc(world_size * sizeof(int));
  
  // Get weights from the processes
  MPI_Allgather(my_weight, 1, MPI_DOUBLE, all_weights, 1, MPI_DOUBLE, MPI_COMM_WORLD);

  // Each proc chooses its parent
  // ----------------------------

  // Sum of weights
  for(int p = 0; p < world_size; p++)
    total_w += all_weights[p];

  double u = gsl_rng_uniform(rng) * total_w;
  int parent = 0;
  double cum_weights = all_weights[0];

  while(u > cum_weights) {
     cum_weights += all_weights[++parent];
  }

  // All procs know who they get the data from
  // -----------------------------------------
  MPI_Allgather(&parent, 1, MPI_INT, parents, 1, MPI_INT, MPI_COMM_WORLD);
  
  // This is a very slow interaction but used only for debugging, so its use is justified
  for(int p = 0; p < world_size; p++) {
    
    if(p != parents[p]) { // Don't send to yourself
      
      if(rank == parents[p]) {
	MPI_Send(serial_forest, msg_sizes[0] + 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
      } else if (rank == p) {
	MPI_Recv(rec_buffer, msg_sizes[0] + 1, MPI_DOUBLE, parents[p], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  // Move received stuff to the workspace
  if(rank != parents[rank])
    for (int i = 0; i < msg_sizes[0] + 1; i++) 
      serial_forest[i] = rec_buffer[i];
  
  free(all_weights);
  free(parents);
}
  
void Tree_AIRPF_Recursive_doubling(double* serial_forest,
				   double *rec_buffer,
				   int* msg_sizes,
				   int state_dim,
				   double *weights,
				   MPI_Comm comm,
				   gsl_rng *rng) {
  
  int world_size, rank, n_stages, pair;
  int sample_size = (int)(serial_forest[0]);
  double paired_weight;
  
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  n_stages = (int) (round(log2(world_size))+0.1);

  for (int s = 0; s < n_stages; s++) {
    
    serial_forest[msg_sizes[0]] = weights[0]; 
    
    /* The actual communication */
    // NB: the '+1' in the size is due to passing also the weight
    if (left_sender(s, &pair))
      MPI_Send(serial_forest, msg_sizes[0] + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD);
    else
      MPI_Recv(rec_buffer, msg_sizes[0] + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (!left_sender(s, &pair))
      MPI_Send(serial_forest, msg_sizes[0] + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD);
    else
      MPI_Recv(rec_buffer, msg_sizes[0] + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Update the weights
    paired_weight = rec_buffer[msg_sizes[0]] * sample_size;
    Tree_AIRPF_recursive_doubling(serial_forest, rec_buffer, sample_size, msg_sizes[0],
				  weights[0] * sample_size, paired_weight, weights, rng);

    MPI_Barrier(MPI_COMM_WORLD);
  }
  for(long i = 1; i < sample_size;i++)
    weights[i] = weights[0];
}
 
void AIRPF_Recursive_doubling(double *sample,
			      double *send_buffer,
			      double *rec_buffer,
			      int dim,
			      int *sizes,
			      double *weights,
			      MPI_Datatype dtype,
			      MPI_Comm comm,
			      gsl_rng *rng) {
  
  int world_size, rank, n_stages, pair;
  
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  n_stages = (int) round(log2(world_size));
  
  for (int s = 0; s < n_stages; s++) {
    
    /* At the beginning of a stage each rank copies communicateable particles to the
     * send_buffer.
     */
    move2sendbuffer(sample, dim, send_buffer, sizes[s + 1]);
    send_buffer[sizes[s + 1] * dim] = weights[0];
    
    /* The actual communication */
    // NB: the '+1' in the size is due to passing also the weight
    if (left_sender(s, &pair))
      MPI_Send(send_buffer, sizes[s + 1] * dim + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD);
    else
      MPI_Recv(rec_buffer, sizes[s + 1] * dim + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (!left_sender(s, &pair))
      MPI_Send(send_buffer, sizes[s + 1] * dim + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD);
    else
      MPI_Recv(rec_buffer, sizes[s + 1] * dim + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    
    // This moves the sample from the buffer to working memory and updates the weights
     AIRPF_recursive_doubling(sample, rec_buffer, dim, sizes[s + 1], weights[0] * sizes[s],
			     rec_buffer[sizes[s + 1] * dim] * sizes[s], weights, rng);
     MPI_Barrier(MPI_COMM_WORLD);
  }
  for(long i = 1; i < sizes[0];i++)
    weights[i] = weights[0];
  
}

void Recursive_doubling(double *sample,
			double *send_buffer,
			double *rec_buffer,
			int dim,
			int *sizes,
			double *weights,
			MPI_Datatype dtype,
			MPI_Comm comm,
			gsl_rng *rng) {
  
  int world_size, rank, n_stages, pair;
  
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  n_stages = (int) round(log2(world_size));
  
  for (int s = 0; s < n_stages; s++) {
    
    /* At the beginning of a stage each rank copies communicateable particles to the
     * send_buffer.
     */
    sample2sendbuffer(sample, send_buffer, dim, sizes[s + 1], sizes[s], rng);
    send_buffer[sizes[s + 1] * dim] = weights[0];
    
    /* The actual communication */
    // NB: the '+1' in the size is due to passing also the weight
    if (left_sender(s, &pair))
      MPI_Send(send_buffer, sizes[s + 1] * dim + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD);
    else
      MPI_Recv(rec_buffer, sizes[s + 1] * dim + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (!left_sender(s, &pair))
      MPI_Send(send_buffer, sizes[s + 1] * dim + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD);
    else
      MPI_Recv(rec_buffer, sizes[s + 1] * dim + 1, MPI_DOUBLE, pair, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
    recursive_doubling(sample, rec_buffer, dim, sizes[s + 1], weights[0] * sizes[s],rec_buffer[sizes[s + 1] * dim] * sizes[s], weights, rng);
    
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
}

void Tree_AIRPF_recursive_doubling(double *serial_forest,
				   double *rec_buffer,
				   int sample_size,
				   int msg_size,
				   double my_weight,
				   double paired_weight,
				   double *weights,
				   gsl_rng *rng) {

  weights[0] = (my_weight + paired_weight) / (double) 2.0 / (double) sample_size;

  if (gsl_rng_uniform(rng) < paired_weight / (my_weight + paired_weight)) {
    for (int i = 0; i < msg_size; i++) 
      serial_forest[i] = rec_buffer[i];
  }
 
}

void AIRPF_recursive_doubling(double *sample,
			      double *rec_buffer,
			      int dim,
			      int size,
			      double my_weight,
			      double paired_weight,
			      double *weights,
			      gsl_rng *rng) {
  
  double prob = paired_weight / (my_weight + paired_weight);
  double new_weight = (my_weight + paired_weight) / (double) 2.0 / (double) size;
  
  if (gsl_rng_uniform(rng) < prob) 
    for (int i = 0; i < size * dim; i++) 
      sample[i] = rec_buffer[i];
  weights[0] = new_weight;
}

void recursive_doubling(double *sample,
			double *rec_buffer,
			int dim,
			int size,
			double my_weight,
			double paired_weight,
			double *weights,
			gsl_rng *rng) {
  
  double prob = paired_weight / (my_weight + paired_weight);
  double new_weight = (my_weight + paired_weight) / (double) 2.0 / (double) size;

  int offset;
  for (int i = 0; i < size; i++) {
    if (gsl_rng_uniform(rng) < prob) {
      offset = i * dim;
      for (int d = 0; d < dim; d++)
	sample[offset + d] = rec_buffer[offset + d];
    }
    weights[i] = new_weight;
  }
}

void sample2sendbuffer(double *sample,
		       double *buffer,
		       int dim,
		       int size,
		       int outof,
		       gsl_rng *rng) {

  int ind, offset;
  for (int i = 0; i < size; i++) {
    ind = gsl_rng_uniform_int(rng, outof);
    offset = i * dim;
    for (int d = 0; d < dim; d++)
      buffer[offset + d] = sample[ind * dim + d];
  }
}

void move2sendbuffer(double *sample,
		     int dim,
		     double *buffer,
		     int size) {

  for(int i = 0; i < size * dim; i++)
    buffer[i] = sample[i];
  
}

unsigned short left_sender(int stage,
			   int *pair) {
  
  int rank;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int p1;
  int tmp1, tmp2;
  double p2;
  
  p1 = (int) round(pow(2, stage));
  p2 = pow(2, stage + 1);
  
  tmp1 = rank % p1 + p2 * (int) round(floor(rank / p2));
  tmp2 = tmp1 + p1;
  
  if (tmp1 < tmp2) {
    if (rank == tmp1) {
      pair[0] = tmp2;
      return 1;
    } else {
      pair[0] = tmp1;
      return 0;
    }
  } else {
    if (rank == tmp2) {
      pair[0] = tmp1;
      return 1;
    } else {
      pair[0] = tmp2;
      return 0;
    }
  }
  
}


void BPFresample_MIPS(int size,
		      double *w,
		      int *ind,
		      gsl_rng *r,
		      double normaliser,
		      int rank) {

  /* Generate the exponentials */
  double *e = (double*) malloc((size + 1) * sizeof(double));
  double g = 0;
  for (long i = 0; i <= size; i++) {
    e[i] = gsl_ran_exponential(r, 1.0);
    g += e[i];
  }
  
  /* Generate the uniform order statistics */
  double *u = (double*) malloc((size + 1) * sizeof(double));
  u[0] = 0;
  for (long i = 1; i <= size; i++)
    u[i] = u[i - 1] + e[i - 1] / g;
  
  /* Do the actual sampling with inverse cdf */
  double cdf = w[0] / normaliser;
  long j = 0;
  for (long i = 0; i < size; i++) {
    while (cdf < u[i + 1]) {
      j++;
      cdf += w[j] / normaliser;
    }
    ind[i] = j;
  }
  
  free(e);
  free(u);
  
}

void BPFresample(long size,
		 double *w,
		 long *ind,
		 gsl_rng *r,
		 double normaliser){
  
  /* Generate the exponentials */
  double *e = (double*) malloc((size + 1) * sizeof(double));
  double g = 0;
  for (long i = 0; i <= size; i++) {
    e[i] = gsl_ran_exponential(r, 1.0);
    g += e[i];
  }
  /* Generate the uniform order statistics */
  double *u = (double*) malloc((size + 1) * sizeof(double));
  u[0] = 0;
  for (long i = 1; i <= size; i++)
    u[i] = u[i - 1] + e[i - 1] / g;
  
  /* Do the actual sampling with inverse cdf */
  double cdf = w[0] / normaliser;
  long j = 0;
  for (long i = 0; i < size; i++) {
    while (cdf < u[i + 1]) {
      j++;
      cdf += w[j] / normaliser;
    }
    ind[i] = j;
  }
  
  free(e);
  free(u);
  
}
