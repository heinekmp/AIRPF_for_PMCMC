/*
 * myMPICollectives.h
 *
 *  Created on: 3 Jun 2021
 *      Author: heine
 */

#include <mpi.h>
#include <gsl/gsl_rng.h>

#ifndef MYMPICOLLECTIVES_H_
#define MYMPICOLLECTIVES_H_

void Tree_AIRPF_Recursive_doubling(double* serial_forest, double *rec_buffer, int* msg_sizes,
				   int state_dim,
				   double *weights, MPI_Comm comm,  gsl_rng *rng);
void AIRPF_Recursive_doubling(double *sample, double *send_buffer, double *rec_buffer, int dim,
			      int *sizes, double *weights, MPI_Datatype dtype, MPI_Comm comm, gsl_rng *rng);
void Recursive_doubling(double *sample, double *send_buffer, double *rec_buffer, int dim,
			int *sizes, double *weights, MPI_Datatype dtype, MPI_Comm comm, gsl_rng *rng);
unsigned short left_sender(int stage, int* pair);
void sample2sendbuffer(double *sample, double *buffer, int dim, int size, int outof, gsl_rng *rng);
void move2sendbuffer(double *sample, int dim, double *buffer, int size);
void recursive_doubling(double *sample, double *rec_buffer, int dim, int size, double my_weight,
			double paired_weight, double *weights, gsl_rng *rng);
void Tree_AIRPF_recursive_doubling(double *serial_forest, double *rec_buffer, int sample_size, int msg_size,
				   double my_weight, double paired_weight, double *weights, gsl_rng *rng);
void AIRPF_recursive_doubling(double *sample, double *rec_buffer, int dim, int size,
			      double my_weight, double paired_weight, double *weights, gsl_rng *rng);
void Island_communication( double *sample, double *buffer, double *send_buffer,
			   double *rec_buffer, int dim, int M, int n_results, MPI_Comm comm,
			   gsl_rng *rng);
void BPFresample_MIPS(int size, double *w, int *ind, gsl_rng *r, double normaliser, int rank);
void BPFresample(long size, double *w, long *ind, gsl_rng *r, double normaliser);
void naive_interaction(double* serial_forest, double *rec_buffer, int* msg_sizes, double *weights,
		       gsl_rng *rng, int world_size, int rank);
#endif /* MYMPICOLLECTIVES_H_ */
