GPUFEM Notes
============

## Lists
### TODO
* Add microbench to interface
* Add differences in memory initialization (dynamic/static) regarding host and kernel executions.
Ideas!:
* No transfer by zeroing only at the GPU?
* Change the implementation of the kernel when setting OpenCL strategy using the
interface
* Check clearing matrix really necessary?
* Init GlblKaux as zero? in GPU?
* Amdahl's law and Gustafson -> potential limits to paralellism / development of
  new parallel algorithms
* Refactor AxyGPU to decide on local OCL dims Polymophically

### BUGS
* Fix batch metis(?) memory bug

### IDEEZ
* FEM class initialization: femdata in init?
* More clarity in assembly in OpenMP
* 2.4 finite element history

### References to add
* Chris Jang from GATLAS:
ATI is often criticized very heavily for immaturity in the software stack
supporting their GPGPU product line. However, my (limited) experience thus
far is that NVIDIA has many issues too. Using the GPU for general purpose
computation is like using GCC with -O3 and lots of dangerous compiler
options. The higher performance is not free. This partially explains why
GATLAS has so many kernel generation options. It is not possible to know
the best kernel variants ahead of time. They must all be tested for both
performance and correctness.

* AMD Accellerated Parallel Processing Guide:
"Global memory reads generate a reference to the off-chip memory and
experience a latency of 300 to 600 cycles. The wavefront that generates the
global memory access is made idle until the memory request completes. During
this time, the compute unit can process other independent wavefronts, if they are
available."

* "A large power of two stride
results in a channel conflict; a larger power of two stride results in a bank conflict.
The size of the power of two stride that causes a specific type of conflict depends
Compute Unit <> Memory Channel XbarAMD ACCELERATED  PARALLEL  PROCESSING
6.1 Global Memory Optimization 6-3
Copyright © 2012 Advanced Micro Devices, Inc. All rights reserved.
on the chip. A stride that results in a channel conflict on a machine with eight
channels might result in a bank conflict on a machine with four."

## Naive SpMV Kernel
Each global thread is responsible for a row of non-zeros. The output for each vector row multiplication (dot product) is calculated by iterating over all of each threads non-zeros and accumulating the sum to a private variable. Indexing is calculated by (i * matdim + globalid) where i is the iterator varying from 0 to the number of non-zeros for the row. The number of nonzeros for each row is passed to the GPU and is accessible reading from global memory. Given the Ellpack storage characteristics, this indexing causes strided accesses (where the stride is matdim) which degrades performance. It is noteworthy that while local memory is not explicitly used, varying local memory size does impact performance. This happens because the form in which we divide groupsizes affects caching and group scheduling. More groups means less memory per workgroup but also more efficient scheduling. Scheduling also affects how wavefronts are executed which can entail underutilized resources.

## Staggered Shared SpMV
The SpMVStag function provides staggered access to the matrix entries like the naive version but adds a shared buffer utilizing local memory. Also, contrary to the naive implementation, each row has multiple threads which belong to a single workgroup. The calculation of a row vector dot product is, therefore, performed in a group based manner and not in a thread based manner. The higher bandwidth and ability to share information and implement barriers internal to a workgroup is the rationale for utilizing local memory. This added functionality also permits us to employ parallel reduction, which is only possible with synchronization primitives, which speeds up reduction processes considerably. The downside is that only one workgroup per row permits little utilization of local resources, and since global memory is accessed the same number of times as the naive implementation, copying from global memory to local memory does not yield considerable benefit.

## Blocked Shared SpMV
The blocked implementation is similar to the shared implementation with the difference that workgroups span multiple rows. The added benefit of loading more data into local memory and possible implementation utilizing image buffers which take advantage of memory locality is the reasoning to implement a block-based strategy. Although the bottelneck is global memory access, some gain can also be made here utilizing parallel reductons. In newer graphics cards where the available local memory becomes larger, this added benefit tends to increase.

## Loop Unrolled Versions
Loop unrolling allows the compiler to more easily identify vectorizable code. This allows the compiler to extract instruction level parallelism out of code which is seemingly parallel.

## CG Outline
The conjugate gradient method is one of the most widely employed iterative methods for solving large systems of linear equations. The matrix must be positive definite. Iterative methods are especially suited for solving linear systems of sparse matrices, since traditional factorization methods would destroy matrix sparsity (and the whole purpose of using a sparse matrix in a first place). The foremost reference regarding the conjugate gradient method used for this section is Shewchuk's seminal, aptly named paper "An Introduction to the Conjugate Gradient Method Without Agonizing Pain". For a more detailed explanation I would urge the reader to refer to this reference, and most material exposed in this section should pertain to particularities of implementation and also regarding the finite element method.

Precond(asphalt): A rule of thumb is that M must resemble the original matrix, K, to obtain eigenvalues that cluster around 1. Obviously, M = K would be the best choice, but this choice is equivalent to solving the original system! A more common choice is the diagonal of K, known as diagonal scaling or Incomplete Cholesky factorization using a drop tolerance to control the fill-in.

PCG widely used because: easy to implement, PCG iterations are cheap and the storage demands are modest and fixed. Cons: performance depends on the conditioning and/or spectrum of the matrix.

From [A. Van der Sluis and H.A. Van der Vorst, The rate of convergence of conjugate gradients] the smallest eigenvalues correspond to the slow converging components of the solution. The number of rigid body modes of any unconstrained volume equals the number of zero-valued eigenvalues of its corresponding stiffness matrix.

Diagonal scaling may improve the matrix characteristics that are most important to convergence because the scaled matrix is still symmetric and positive definite, and the condition number of A is minimizes [Forsythe and Strauss, 1955]

Preconditioners, in reducing the number of iterations also minimize potential roundoff errors, particularly important when using float precision.

[Reid] Showed that for large sparse matrices that are reasonably well conditioned, the conjugate gradient method is a very powerful iterative scheme that yields good approximate solutions in far fewer than n steps.

Excellent historic perspective on iterative methods [survey preconditioning]

recent hardware trends towards computing hedge against future of non-normal rendering, raytracing, voxel cone whatever, etc.

blue cranes - STILL


