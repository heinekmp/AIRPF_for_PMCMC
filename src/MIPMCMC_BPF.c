/*
 * MIPMCMC_BPF.c
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */

#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>
#include<math.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#include "myMPICollectives.h"
#include "APS_BPF.h"
#include "APS_model.h"
#include "MIPMCMC_BPF.h"

const int MAX_TREE_CAPACITY = 8*100000;
const int TREE_COMM_BUFFER_SIZE = 8*250000; // 25 times the single processor sample size. This multiplied by two to include send and recv.
//const double eps = 0.01;
double calculate_pess(double *ml, double *all_mls, int world_size);
double logsum(double *log_val, int len);
  
void kill_node(int i,
	       int* n_child,
	       int *parents,
	       int* dead,
	       int *number_of_dead) {
  
  if( (--n_child[i]) == 0){
    
    dead[number_of_dead[0]++] = i;
    
    if(parents[i] >= 0)
      kill_node(parents[i], n_child, parents, dead, number_of_dead);
  }
}

int find_max(int* vals,
	     int length) {
  int record = 0;
  for(int i = 0; i < length; i++)
    record = vals[i] > record ? vals[i] : record;
  return(record);
}

void print_tree(int *leaves,
		int *parents,
		int *n_child,
		double* values,
		int tree_size,
		int size) {

  printf("Tree (size %i):\n-------------------\n",tree_size);
  printf("Leaves: ");
  for(int i = 0 ; i < size; i++) 
    printf("%8i ",leaves[i]);
  printf("\n\n");

  printf("Index:  ");
  for(int i = 0 ; i < tree_size; i++) 
    printf("%8i ",i);
  printf("\n");
  
  printf("Parents:");
  for(int i = 0 ; i < tree_size; i++) 
    printf("%8i ",parents[i]);
  printf("\n");
  
  printf("n_child:");
  for(int i = 0 ; i < tree_size; i++) 
    printf("%8i ",n_child[i]);
  printf("\n");

  printf("Values :");
  for(int i = 0 ; i < tree_size; i++) 
    printf("%8.1e ",values[2*i]);
  printf("\nValues :");
  for(int i = 0 ; i < tree_size; i++) 
    printf("%8.1e ",values[2*i+1]);
  printf("\n\n");

  fflush(stdout);
  getchar();
}

/*
 This converts the forest into single one dimensional array of doubles. The resulting 
 array will have

 2 + size + 4 * tree_size

 elements. 
 */
void serialise_forest(int* leaves,
		      double* values,
		      int* dead_nodes,
		      int number_of_dead,
		      int tree_size,
		      int size,
		      int state_dim,
		      double *serial_forest) {

  serial_forest[0] = (double) size;

  serial_forest[1] = (double) tree_size;

  serial_forest[2] = (double) number_of_dead;

  double *serial_leaves, *serial_parents, *serial_n_child, *serial_values, *serial_dead_nodes;
  int *parents, *n_child;

  serial_leaves = serial_forest + 3;
  for(int i = 0; i < size; i++) // Leaves
    serial_leaves[i] = (double) leaves[i];

  serial_parents = serial_forest + 3 + size;
  parents = leaves + size;
  serial_n_child = serial_forest + 3 + size + tree_size;
  n_child = leaves + size + MAX_TREE_CAPACITY;
  for(int i = 0; i < tree_size; i++) {// Parents and n_child 
    serial_parents[i] = (double) parents[i];
    serial_n_child[i] = (double) n_child[i];
  }

  serial_values = serial_forest + 3 + size + 2 * tree_size;
  for(int i = 0; i < state_dim * tree_size; i++) // values
    serial_values[i] = values[i];
    
  serial_dead_nodes = serial_forest + 3 + size + (2 + state_dim) * tree_size;
  for(int i = 0; i < number_of_dead; i++)
    serial_dead_nodes[i] = (double)dead_nodes[i];

}

void deserialise_forest(int* leaves,
			double* values,
			int *tree_size,
			long *size,
			int* dead,
			int* number_of_dead,
			int state_dim,
			double *serial_forest) {

  size[0] = (int) (serial_forest[0] + 0.1);
  tree_size[0] = (int) (serial_forest[1] + 0.1);
  number_of_dead[0] = (int) ( serial_forest[2] + 0.1);

  double *serial_leaves = serial_forest + 3;
  for(int i = 0; i < size[0]; i++) // Leaves
    leaves[i] = (int) (serial_leaves[i] + 0.1);

  double *serial_parents = serial_forest + 3 + size[0];
  int *parents = leaves + size[0];
  double *serial_n_child = serial_forest + 3 + size[0] + tree_size[0];
  int *n_child = leaves + size[0] + MAX_TREE_CAPACITY;
  for(int i = 0; i < tree_size[0]; i++) { // n_child
    n_child[i] = (int) (serial_n_child[i] + 0.1);
    parents[i] = (int) (serial_parents[i] + 0.1); // Parent
  }

  double *serial_values = serial_forest + 3 + size[0] + 2 * tree_size[0];
  for(int i = 0; i < state_dim * tree_size[0]; i++) // values
    values[i] = serial_values[i];

  double *serial_dead_nodes  = serial_forest + 3 + size[0] + (2 + state_dim) * tree_size[0];
  for(int i = 0; i < number_of_dead[0]; i++)
    dead[i] = (int)(serial_dead_nodes[i] + 0.1);
  
}

void MIPMCMC_TREE_BPF(double *data,
		      int len,
		      int data_dim,
		      long size,
		      double *parm,
		      int parm_dim,
		      int state_dim,
		      int *Ms,
		      gsl_rng *rng,
		      double *marginal_likelihood,
		      double *history,
		      double *ave_ess,
		      int rank,
		      int world_size,
		      double *occupation,
		      double* times,
		      int naive,
		      double *ml_seq) {
  
  // Create the tree (this could be regarded as a struct with fields leafs, parents,
  // n_child, and values
  // -------------------------------------------------------------------------------
  
  int *leaves = (int*) malloc((size + 2 * MAX_TREE_CAPACITY) * sizeof(int));
  int *parents = leaves + size;
  int *n_child = leaves + size + MAX_TREE_CAPACITY;
  double *values = (double *) malloc(state_dim * MAX_TREE_CAPACITY * sizeof(double));
  int tree_size = 0; // How many entries are used
  int *new_leaves = (int *)malloc(size * sizeof(int));
  double* tree_comm_buffer = (double *) malloc(TREE_COMM_BUFFER_SIZE * sizeof(double));
  double* rec_buffer = tree_comm_buffer + TREE_COMM_BUFFER_SIZE / 2;
  int *dead = (int*)malloc(MAX_TREE_CAPACITY * sizeof(int));
  int number_of_dead = 0;
  int dead_node;
  int resampled_node;
  int nodes_to_overwrite;
  int nodes_to_add;
  double *node_state;
  
  // Allocate memory for the SMC sample
  double *X = (double*) malloc(size * state_dim * sizeof(double));
  APS_CreateInitialSample(X, size, state_dim, rng);
  
  double *W = (double*) malloc(size * sizeof(double));
  double *iw = (double*) malloc(size * sizeof(double));
  for(long i = 0; i < size;i++)
    iw[i] = 1;

  int sgnl_len = len / data_dim;

  // This is just a one step genealogy, i.e. the indices of the parents
  // in a single run of resampling.
  long *genealogy = (long*) malloc(size * sizeof(long));
  
  // Add the first set of leaves to the tree
  for(int i = 0; i < size; i++){
    
    // We think values as state_dim - by - N array, while
    // X is N - by - state_dim array (obviously it is a one dimensional array,
    // but the interpretaiton applies to the indexing)
    node_state = values + state_dim * i;
    for(int s = 0; s < state_dim; s++)
      node_state[s] = X[s * size + i];

    parents[i] = -1; // Root nodes do not have parents (actually this is a forest)
    n_child[i] = 0; // No resampling -> no children yet
    leaves[i] = i; // All nodes are leaves
  }
  tree_size = size; // Number of nodes stored in the tree
  
  // Various variable introductions
  // ------------------------------
  double normaliser[1];
  double ess[1];
  ave_ess[0] = 0;
  marginal_likelihood[0] = 0;
  double *W0 = (double*) malloc(size * sizeof(double));
  int message_sizes[1];
  int msg_size;
  unsigned int *obss = (unsigned int*) malloc(data_dim * sizeof(unsigned int));

  clock_t like_st, comm_st, mut_st, rs_st;
  double like_elap = 0, comm_elap = 0, mut_elap = 0, rs_elap = 0;

  // Effective number of processes and threshold for communication
  double pess, pess_threshold = 0.3 * world_size;
  double *all_MLs = (double *) malloc( world_size * sizeof(double));
  double total_logml = 0;
  double tmp;
  int dead_tmp;
    
  // ----------------
  // Filter main loop
  // ----------------
  for (int n = 0; n < sgnl_len; n++) {
      
    like_st = clock();
    
    if (data_dim == 1) {
      obss[0] = (unsigned int) data[n];
    } else if (data_dim == 2) {
      obss[0] = (unsigned int) data[2 * n];
      obss[1] = (unsigned int) data[1 + 2 * n];
    }

    // Weights
    APS_Likelihood(obss, data_dim, X, size, W, iw, normaliser, ess);
    ave_ess[0] += ess[0] / (double) sgnl_len;

    marginal_likelihood[0] += log(normaliser[0] / (double) size);

    MPI_Barrier(MPI_COMM_WORLD);      
    // Master process collects the accumulated log weight per process
    MPI_Allgather(marginal_likelihood, 1, MPI_DOUBLE, all_MLs, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    like_elap += (double) (clock() - like_st) / CLOCKS_PER_SEC;
    rs_st = clock();

    // Resample
    // --------
    // NB: The way we index genealogy here means that genealogy should be
    // interpreted as size-by-length array
    BPFresample(size, W, genealogy, rng, normaliser[0]);
    
    // Process level ESS
    pess = 0; 
    
    total_logml = logsum(all_MLs, world_size);
    
    for(int p = 0; p < world_size; p++) {
      tmp = exp(all_MLs[p] - total_logml);
      pess += tmp * tmp;
    }
    pess = (double) 1.0 / pess;
    ml_seq[n] = pess;
    
    // Set the weight for the process (uniform weight over all particles)
    tmp = exp(all_MLs[rank] - total_logml);
    for(long i = 0; i < size; i++) 
      W0[i] = tmp; // Proportion of the weight for the current process

    // Update the child counts
    // -----------------------
    for(int i = 0; i < size; i++)
      n_child[leaves[genealogy[i]]]++;
    
    // Kill childless branches
    // -----------------------
    for(int i = 0; i < size; i++) {
      if(n_child[leaves[i]] == 0) {
	dead[number_of_dead++] = leaves[i];
	if(parents[leaves[i]] >= 0) // not a root node
	  kill_node(parents[leaves[i]], n_child, parents, dead, &number_of_dead);
      }
    }

    dead_tmp = number_of_dead;
    
    // Insert resampled particles to the tree
    // --------------------------------------
    if(number_of_dead >= size) {
      nodes_to_overwrite = size;
      nodes_to_add = 0;
    } else {
      nodes_to_overwrite = number_of_dead;
      nodes_to_add = size - number_of_dead;
    }

    // overwrite dead nodes
    for(int j = 0; j < nodes_to_overwrite; j++){ // iterate over resampled nodes

      // take the last dead node for overwriting
      // and reduce the number of dead by one
      dead_node = dead[--number_of_dead];
      
      resampled_node = genealogy[j];

      // Insert parent info
      parents[dead_node] = leaves[resampled_node];
      n_child[dead_node] = 0;
      
      // Insert state
      node_state = values + dead_node * state_dim;
      for(int s = 0; s < state_dim; s++)
	node_state[s] = X[s * size + resampled_node];
      
      // Mark dead node as a new leaf
      new_leaves[j] = dead_node;
      
    }

    // expand the tree
    for(int j = 0; j < nodes_to_add; j++) { // iterate over resampled nodes
      
      resampled_node = genealogy[nodes_to_overwrite + j];

      // Insert parent info
      parents[tree_size] = leaves[resampled_node];
      n_child[tree_size] = 0;

      // Insert state
      node_state = values + tree_size * state_dim;
      for(int s = 0; s < state_dim; s++)
	node_state[s] = X[s * size + resampled_node];
      
      // Mark dead node as a new leaf
      new_leaves[nodes_to_overwrite + j] = tree_size++;

      occupation[0] = (double) tree_size / (double) MAX_TREE_CAPACITY;
      if(occupation[0] > 0.5) {
	printf("High occupation[0]! %f\n",occupation[0]);
	fflush(stdout);
      }
    }

    // Finally set the new leaves
    for(int i = 0; i < size; i++)
      leaves[i] = new_leaves[i];

    // Diagnostics
    occupation[1] = (double)(3 + size + (2 + state_dim) * tree_size + number_of_dead) /
      (double) TREE_COMM_BUFFER_SIZE * 2.0;

    if(occupation[1] > 0.9) {
      printf("High occupation! %f\n",occupation[1]);
      fflush(stdout);
    }

    rs_elap += (double) (clock() - rs_st) / CLOCKS_PER_SEC;
    comm_st = clock();
    
    // Check is communication is needed
    // --------------------------------
    if(pess < pess_threshold) {

      // Inter processor communication
      // -----------------------------
      serialise_forest(leaves, values, dead, number_of_dead, tree_size, size,
		       state_dim, tree_comm_buffer);
      
      msg_size = 3 + size + (2 + state_dim ) * tree_size + number_of_dead + 1;
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allreduce(&msg_size, message_sizes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      // Add rank info
      tree_comm_buffer[message_sizes[0]-1] = (double) rank;

      Tree_AIRPF_Recursive_doubling(tree_comm_buffer, rec_buffer, message_sizes,
				    state_dim, W0, MPI_COMM_WORLD, rng);
      
      deserialise_forest(leaves, values, &tree_size, &size, dead, &number_of_dead,
			 state_dim, tree_comm_buffer);
           
      marginal_likelihood[0] = total_logml - log( (double) world_size);
      MPI_Barrier(MPI_COMM_WORLD);

    }

    // ----------------------------
    comm_elap += (double) (clock() - comm_st) / CLOCKS_PER_SEC;
    mut_st = clock();
    
    // Mutate if there is one more observation left
    if (n < (sgnl_len - 1)) {
      
      for(int s = 0; s < state_dim; s++)
	for(long i = 0; i < size; i++) 
	  X[s * size + i] = values[leaves[i] * state_dim + s];
      
      if (data_dim == 1)
	APS_Mutate(X, size, parm, parm_dim, rng, iw);
      else if (data_dim == 2)
	APS_Mutate(X, size, parm, parm_dim, rng, iw);
      else
	printf("Unknown dimension\n");

    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    mut_elap += (double) (clock() - mut_st) / CLOCKS_PER_SEC;

  }
  
  // Create the smoothed particle
  int next = leaves[0];
  for (int t = 0; t < sgnl_len; t++) {
    
    node_state = values + state_dim * next;
    for(int s = 0; s < state_dim; s++) 
      history[s * sgnl_len + sgnl_len - 1 - t] = node_state[s];
    
    next = parents[next];
    
  }
  
  if(rank==0){
    times[0] += like_elap;
    times[1] += rs_elap;
    times[2] += comm_elap;
    times[3] += mut_elap;
  }
  
  if(rank == 0) {
    marginal_likelihood[0] = logsum(all_MLs,world_size) - log((double) world_size);
  }

  free(values);
  free(leaves);
  free(X);
  free(W);
  free(W0);
  free(iw);
  free(genealogy);
  free(new_leaves);
  free(tree_comm_buffer);
  free(obss);
  free(all_MLs);
  free(dead);
  
}

void MIPMCMC_BPF(double *data,
		 int len,
		 int data_dim,
		 long size,
		 double *parm,
		 int parm_dim,
		 int state_dim,
		 int *Ms,
		 gsl_rng *rng,
		 double *marginal_likelihood,
		 double *history,
		 double *ave_ess,
		 int rank,
		 int world_size) {
  
  // Allocate memory for the SMC sample
  double *X = (double*) malloc(size * state_dim * sizeof(double));
  APS_CreateInitialSample(X, size, state_dim, rng);
  
  // Allocate memory for the particle weights
  double *W = (double*) malloc(size * sizeof(double));
  double *iw = (double*) malloc(size * sizeof(double));
  for(long i = 0; i < size;i++)
    iw[i] = 1;
  
  // This is just a one step genealogy, i.e. the indices of the parents
  // in a single run of resampling.
  long *genealogy = (long*) malloc(size * sizeof(long));

  // We allocate meomry for the butterfly communication. We need to communicate
  // the whole particle history, which means that the total dimension of the
  // particles is state_dim * len
  int total_dim = state_dim * len / data_dim;
  double *sample = (double*) malloc((size * total_dim  + size) * sizeof(double));
  double *trajectories_rs = (double*) malloc((size * total_dim  + size) * sizeof(double));
  
  /*
   * We have one buffer for communication, but we split it in two (send/receive)
   */
  int buffer_size = 2 * (size * total_dim + 1);
  double *buffer = (double*) malloc(buffer_size * sizeof(double));
  double *send_buffer = buffer, *rec_buffer = buffer + size * total_dim + 1;
  
  double *W0 = (double*) malloc(size * sizeof(double));
  
  marginal_likelihood[0] = 1;
  double ml_increment[0];
  double normaliser[1];
  double ess[1];
  ave_ess[0] = 0;

  unsigned int *obss = (unsigned int*) malloc(data_dim * sizeof(unsigned int));

  // Filter main loop
  for (int n = 0; n < len; n++) {

    if (data_dim == 1) {
      obss[0] = (unsigned int) data[n];
    } else if (data_dim == 2) {
      obss[0] = (unsigned int) data[n];
      obss[1] = (unsigned int) data[len / data_dim + n];
    }
    
    APS_Likelihood(obss, data_dim, X, size, W, iw, normaliser, ess);
    ave_ess[0] += ess[0] / (double) len;
   
    // Update the marginal likelihood
    MPI_Reduce(normaliser,ml_increment,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    marginal_likelihood[0] *= (ml_increment[0] / (double) (size * world_size));
    
    // Resample
    // --------
    // NB: The way we index genealogy here means that genealogy should be interpreted as size-by-length array
    BPFresample(size, W, genealogy, rng, normaliser[0]);
    for(long i = 0; i < size; i++) 
      W0[i] = normaliser[0] / (double)size;
    
    // Apply the resampled indices
    for (long i = 0; i < size; i++) {

      // History
      for(int t = 0; t < n; t++) 
	for(int s = 0; s < state_dim; s++) 
	  trajectories_rs[t * (state_dim * size) + s * size + i] = sample[t * (state_dim * size) + s * size + genealogy[i]];
      
      // Final state
      for(int s = 0; s < state_dim; s++) 
	trajectories_rs[n * (state_dim * size) + s * size + i] = X[s * size + genealogy[i]];	
    }
    
    //   Actual MPI interaction 
    AIRPF_Recursive_doubling(trajectories_rs, send_buffer, rec_buffer, total_dim, Ms, W0,
    			     MPI_DOUBLE, MPI_COMM_WORLD, rng);

    // Mutate only if there is one more observation left
    if (n < (len - 1)) {

      for(long i = 0; i < size * state_dim; i++)
	X[i] = trajectories_rs[n * (state_dim * size) + i];
    
      APS_Mutate(X, size, parm, parm_dim, rng, iw);

      // Apply the resampled indices
      for( int j = 0; j < (n+1) * state_dim * size; j++)
	sample[j] = trajectories_rs[j];
      
      // Final state      
      for (long i = 0; i < size; i++) 
	for(int s = 0; s < state_dim ; s++)
	  sample[(n + 1) * (state_dim * size) + s*size + i] = X[s * size + i];	

    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  // Create the smoothed particle
  for (int t = 0; t < len; t++) 
    for(int s = 0; s < state_dim; s++)
      history[s * len + t] = sample[t * state_dim * size + s*size];

  free(X);
  free(W);
  free(iw);
  free(genealogy);
  free(sample);
  free(buffer);
  free(trajectories_rs);
  free(W0);
  free(obss);
}

void BPFresample_old(long size,
		     double *w,
		     long *ind,
		     gsl_rng *r,
		     double normaliser) {
  
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

void MIPMCMC_Naive_BPF(double *data,
		       int len,
		       int data_dim,
		       long size,
		       double *parm,
		       int parm_dim,
		       int state_dim,
		       gsl_rng *rng,
		       double *marginal_likelihood,
		       double *history,
		       double *ave_ess,
		       int rank,
		       int world_size,
		       double *ml_seq) {
  
  // Allocate memory for the SMC sample
  double *X = (double*) malloc(size * state_dim * sizeof(double));
  APS_CreateInitialSample(X, size, state_dim, rng);
  
  // Allocate memory for the particle weights
  double *W = (double*) malloc(size * sizeof(double));
  double *iw = (double*) malloc(size * sizeof(double));
  for (long i = 0; i < size; i++)
    iw[i] = 1;

  int sgnl_len = len / data_dim;
  
  // Allocate memory for the genealogy
  long *genealogy = (long*) malloc(sgnl_len * size * sizeof(long));
  
  // Allocate memory for the sample history
  double *sample_history = (double*) malloc(sgnl_len * size * state_dim * sizeof(double));
  
  for (long i = 0; i < state_dim * size; i++)
    sample_history[i] = X[i];
  
  marginal_likelihood[0] = 0;
  double normaliser[1];
  double ess[1], pess, total_logml, tmp;
  long ind;
  ave_ess[0] = 0;

  double *all_MLs = (double *) malloc(world_size * sizeof(double));
   unsigned int *obss = (unsigned int*) malloc(data_dim * sizeof(unsigned int));

  // ----------------
  // Filter main loop
  // ----------------
  for (int n = 0; n < sgnl_len; n++) {

    if (data_dim == 1) {
      obss[0] = (unsigned int) data[n];
    } else if (data_dim == 2) {
      obss[0] = (unsigned int) data[2 * n];
      obss[1] = (unsigned int) data[1 + 2 * n];
    }

    // Weights
    APS_Likelihood(obss, data_dim, X, size, W, iw, normaliser, ess);
    ave_ess[0] += ess[0] / (double) sgnl_len;

    // Update the marginal likelihood
    marginal_likelihood[0] += log((normaliser[0] / (double) size));
      
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(marginal_likelihood, 1, MPI_DOUBLE, all_MLs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Resample
    // --------
    // NB: The way we index genealogy here means that genealogy should be
    // interpreted as size-by-length array
    BPFresample(size, W, genealogy + n * size, rng, normaliser[0]);

    pess = 0; 
    total_logml = logsum(all_MLs, world_size);
    
    for(int p = 0; p < world_size; p++) {
      tmp = exp(all_MLs[p] - total_logml);
      pess += tmp * tmp;
    }
    pess = (double) 1.0 / pess;
    ml_seq[n] = pess;
    
    // Apply the resampled indices
    for (long i = 0; i < size; i++) {

      ind = genealogy[n * size + i];
      
      for(int s = 0; s < state_dim; s++)
	X[s * size + i] = sample_history[n * state_dim * size + s * size + ind];
      
    }
    
    // Mutate only if there is one more observation left
    if (n < (sgnl_len - 1)) {
            
      if (data_dim == 1)
	APS_Mutate(X, size, parm, parm_dim, rng, iw);
      else if (data_dim == 2)
	APS_Mutate(X, size, parm, parm_dim, rng, iw);
      else
	printf("Unknown dimension\n");
            
      for (long i = 0; i < size; i++) 
	for(int s = 0; s < state_dim; s++) 
	  sample_history[(n + 1) * state_dim * size + s * size + i] = X[s * size + i];
      
    }
  }
  
  // Create the smoothed particle
  long new_index = 0;
  for (int i = 0; i < sgnl_len; i++) {

    new_index = genealogy[(sgnl_len - 1 - i) * size + new_index];
    
    for(int s = 0; s < state_dim; s++)
      history[s * sgnl_len + sgnl_len - 1 - i] =
	sample_history[(sgnl_len - 1 - i) * state_dim * size + s * size +  new_index];
    
  }

  if(rank == 0) {    
    marginal_likelihood[0] = logsum(all_MLs,world_size) - log((double) world_size);
  }

  free(X);
  free(W);
  free(iw);
  free(genealogy);
  free(sample_history);
  free(obss);
  free(all_MLs);
}

double calculate_pess(double *ml,
		      double *all_mls,
		      int world_size) {

  double normaliser = 0;
  double pess = 0, tmp;
  
  for(int p = 0; p < world_size; p++) {
    normaliser += exp(all_mls[p]);
  }

  for(int p = 0; p < world_size; p++) {
    tmp = exp(all_mls[p]) / normaliser;
    pess += tmp * tmp; 
  }

  return (double) 1.0 / pess;
}

double logsum(double *log_val,
	      int len) {

  double tmp = 1.0;
  
  for(int i = 1; i < len; i++) 
    tmp += exp(log_val[i] - log_val[0]);

  return log_val[0] + log(tmp);
}
