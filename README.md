# AIRPF_for_PMCMC

This repository contains C codes for a Particle MCMC [[1]](#1) algorithm deploying the Augmented Island Resampling Particle Filter (AIRPF) [[2]](#2). To run, the implementation requires MPI. As the implementation is made for simple algorithmic experimenting, makefiles or compiling instrauctions are not included. As an experimental product, the code also contains some hard coded variables, that one should change in order to be able to run the algorithm. The main file is `mipmcmcm.c`

### References
<a id="1">[1]</a> 
Andrieu, C., Doucet, A. and Holenstein, R. (2010). 
Particle Markov chain Monte Carlo methods. 
*Journal of the Royal Statistical Society: Series B (Statistical Methodology)* **72** 269-342.

<a id="2">[2]</a> 
Heine, K., Whiteley, N. and Cemgil A. T. (2020).
Parallelizing particle filters with butterfly interactions. 
*Scandinavian Journal of Statistics* **47** 361-396.
