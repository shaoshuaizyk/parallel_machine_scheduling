## Mathematical Model for Parallel Machine Scheduling with Multiple Depots

### Sets

1. Jobs: J={0,1,2,...,N+1} 
    - 0 represents the dummy start job
    - N+1 represents the dummy end job
    - 1,2,..., N are real jobs
2. D = {0,1,...,D-1}, set of depots
3. M = {0,1,...,M-1}, set of machines

### Parameters
1. L: length of the region
2. W: width of the region
3. $N_J$: number of jobs
4. D: number of depots
5. $N_M$: number of machines
6. $R_i$: release time for job $i$
7. $A_k$: availability time for machine $k,k\in M$
8. $L_i(x^J_i, y^J_i)$: location of jobs
9. $D_k(x^D_i, y^D_i)$: location of depots
10. $P_k^{d_1,d_2}$: processing time for job i when departing from $d_1$ and going to $d_2$