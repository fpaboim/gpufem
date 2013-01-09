GPUFEM Notes
============

## Naive SpMV Kernel
Each global thread is responsible for a row of nonzeros. The output for each vector row multiplication (dot product) is calculated by iterating over all of each threads nonzeros and accumulating the sum to a private variable. Indexing is calculated by (i * matdim + globalid) where i is the iterator varying from 0 to the number of nonzeros for the row. The number of nonzeros for each row is passed to the GPU and is accessible reading from global memory. Given the ellpack storage characteristics, this indexing causes strided accesses (where the stride is matdim) which degrades performance. It is noteworthy that while localmemory is not explicitly used, varying local memory size does impact performance. This happens because the form in which we devide groupsizes affects caching and group scheduling. More groups means less memory per workgroup but also more efficient scheduling. Scheduling also affects how wavefronts are executed which can entail underutilized resources.

## Staggered Shared SpMV
The SpMVStag function provides staggered access to the matrix entries like the naive version but adds a shared buffer utilizing local memory. Also, contrary to the naive implementation, each row has multiple threads which belong to a single workgroup. The calculation of a row vector dot product is, therefore, performed in a group based manner and not in a thread based manner. The higher bandwidth and ability to share information and implement barriers internal to a workgroup is the rationale for utilizing local memory. This added functionality also permits us to employ parallel reduction, which is only possible with synchronization primitives, which speeds up reduction processes considerably. The downside is that only one workgroup per row permits little utilization of local resources, and since global memory is accessed the same number of times as the naive implementation, copying from global memory to local memory does not yield considerable benefit.

## Blocked Shared SpMV
The blocked implementation is similar to the shared implementation with the difference that workgroups span multiple rows. The added benefit of loading more data into local memory and possible implementation utilizing image buffers which take advantage of memory locality is the reasoning to implement a block-based strategy. Although the bottelneck is global memory access, some gain can also be made here utilizing parallel reductons. In newer graphics cards where the available local memory becomes larger, this added benefit tends to increase.


## Loop Unrolled Versions
Loop unrolling allows the compiler to more easily identify vectorizable code. This allows the compiler to extract instruction level parallelism out of code which is seemingly parallel.


## CG Outline
The conjugate gradient method is one of the most widely employed iterative methods for solving large systems of linear equations. The matrix must be positive definite. Iterative methods are especially suited for solving linear systems of sparse matrices, since traditional factorization methods would destroy matrix sparsity (and the whole purpose of using a sparse matrix in a first place). The foremost reference regarding the conjugate gradient method used for this section is Shewchuk's seminal, aptly named paper "An Introduction to the Conjugate Gradient Method Without Agonizing Pain". For a more detailed explanation I would urge the reader to refer to this reference, and most material exposed in this section should pertain to particularities of implementation and also regarding the finite element method.


TODO:
=====
+ Add differences in memory initialization (dynamic/static) regardin host and kernel executions.
Ideas!:
+ No transfer by zeroing only at the gpu?
+ Change the implementation of the kernel when setting opencl strategy using the
interface
+ Check clearing matrix really necessary?
+ Init GlblKaux as zero? in GPU?

BUGS:
=====
+ Fix batch metis(?) memory bug

IDEEZ:
=====
+ FEM class initialization: femdata in init?

## References to add:
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



ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEA0BdpfuWhkmzSSLpM64KpCutAM6olnxsqpv6ZjF2PcYDUbtaoWYDyd998CvbRpy3US2tt8hkmYMABMq9Qydb6ZQvKiF6kUPqEkIv4KLKrBu7LyPVEuEiTnd3n3yvYeEdSo+ghFEgEXv7XuUdbrAj9d24uUy+35iOWJY3pyU8n2lG7qb3H87ssqJeDr8yKMOAYXOj1e1C2ak/7TnkM5/Q+VPl76W4JZn0eutzopWz3bQ077Iz0yNPqB6XNFu0c/Zk03Uv00UIldED/nhuPUPILBdFE5BtdwCrWpK0Q5ZnCUljtn1xCC0ZwwmQ2HtJVf9bWoK0NPi1ANbk/Ky2U8opw3w== fr4n@hotmail.com
