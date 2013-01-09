// Copyright 2012, Francisco Aboim (fpaboim@tecgraf.puc-rio.br)
// All Rights Reserved
//
// This Program is licensed under the MIT License as follows:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

////////////////////////////////////////////////////////////////////////////////
// SprSolver.cc - Basic Linear Algebra Functions
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////

#include "LAops/SprSolver.h"

#include <math.h>
#include <time.h>
#include <windows.h>
#include <assert.h>
#include <omp.h>
#include <cstdio>

#include "OpenCL/OCLwrapper.h"
#include "Utils/util.h"
#include "SPRmatrix/SPRmatrix.h"

/******************************************************************************/
//                           Direct Solvers                                //
/******************************************************************************/
////////////////////////////////////////////////////////////////////////////////
// Modifies contents of U address to contain cholesky decomposed lower matrix L
////////////////////////////////////////////////////////////////////////////////
int cholesky(SPRmatrix* matrixA, fem_float* vectorX, fem_float* VectorY,
             int dim, bool print) {
  int i, j, k;
  fem_float sum;
  fem_float*  DiagL = allocVector(dim, false);

  for (i = 0; i < dim; i++) {
    DiagL[i] = matrixA->GetElem(i, i);
  }

  // Decomposition
  for (i = 0; i < dim; i++) {
    DiagL[i] = sqrt(DiagL[i]);
    matrixA->SetElem(i, i, DiagL[i]);

    // Divide all below diagonal by diag value
    for (j = i+1; j < dim; j++) {
      fem_float newval = matrixA->GetElem(j, i) / DiagL[i];
      matrixA->SetElem(j, i, newval);
    }

    for (j = i+1; j < dim; j++) {
      DiagL[j] -= matrixA->GetElem(j, i) * matrixA->GetElem(j, i);
      for (k = j+1; k < dim; k++) {
        fem_float newval = matrixA->GetElem(k, j) - (matrixA->GetElem(j, i) *
                           matrixA->GetElem(k, i));
        matrixA->SetElem(k, j, newval);
      }
    }
  }

  if (print) {
    printf("Cholesky Decomposed Matrix (ignore above diag):\n");
    matrixA->PrintMatrix();
  }

  // Solves the system - L*x = y
  for (i = 0; i < dim; ++i) {
    sum = VectorY[i];
    for (k = i-1; k >= 0; --k) {
      sum = sum - (matrixA->GetElem(i, k)*vectorX[k]);
    }
    vectorX[i] = sum / DiagL[i];
  }
  // Solves L^t*x = y
  for (int i = (dim-1); i >= 0; --i) {
    sum = vectorX[i];
    for ( k = i+1; k < dim; k++ ) {
      sum = sum - (matrixA->GetElem(k, i)*vectorX[k]);
    }
    vectorX[i] = sum / DiagL[i];
  }

  free(DiagL);

  return 1;
}

/******************************************************************************/
//                           Iterative Solvers                                //
/******************************************************************************/
////////////////////////////////////////////////////////////////////////////////
// Computes CG using CPU device
////////////////////////////////////////////////////////////////////////////////
int CPU_CG(SPRmatrix* matrix_A,
           fem_float* vector_X,
           fem_float* vector_Y,
           int n,
           int n_iterations,
           fem_float epsilon,
           bool print) {
  int i;
  // Direction Vector
  fem_float* vector_d = (fem_float*)malloc(n * sizeof(fem_float));
  fem_float* vector_r = (fem_float*)malloc(n * sizeof(fem_float));
  fem_float* vector_q = (fem_float*)malloc(n * sizeof(fem_float));
  fem_float delta_new = 0;
  fem_float delta_old = 0;
  fem_float err_bound = 0;
  fem_float alpha     = 0;
  fem_float beta      = 0;

  // zero initial value for x
  // r = b - Ax
  // d = r
#pragma omp parallel for private(i)
  for (i = 0; i < n; ++i) {
    vector_X[i] = 0;
    vector_d[i] = vector_Y[i];
    vector_r[i] = vector_Y[i];
    vector_q[i] = 0;
  }

  delta_new = dotProductOMP(n, vector_r, vector_r);
  err_bound = epsilon * epsilon * delta_new;
  for (i = 0; (i < n_iterations) && (delta_new > err_bound); ++i) {
    // q = Ad
    matrix_A->Ax_y(vector_d, vector_q);
    // alpha = rDotrNew / (d dot q)
    alpha = delta_new / dotProductOMP(n, vector_d, vector_q);
    // x = x + alpha * d
    addScaledVectToSelfOMP(vector_X, vector_d, alpha, n);
    // r = r - alpha * q
    addScaledVectToSelfOMP(vector_r, vector_q, (-alpha), n);
    // rDotrOld = rDotrNew
    delta_old = delta_new;
    // rDotrNew = r dot r
    delta_new = dotProductOMP(n, vector_r, vector_r);
    // beta = rDotrNew / rDotrOld
    beta = delta_new / delta_old;
    // d = r + beta * d
    addSelfScaledToVectOMP(vector_d, vector_r, beta, n);
  }

  if (print == 1) {
    if (i == n_iterations) {
      printf("\n\n***********\nReached max num of iterations!\n***********\n");
    } else {
      printf("\n\n***********\n          Solved!             \n***********\n");
      printf("Vector X:\n");
      printVectorf(vector_X, n);
      printf("solver CG iterations:%i\n", i);
    }
  }


  free(vector_d);
  free(vector_r);
  free(vector_q);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////
// GPU Conjugate Gradient Method
////////////////////////////////////////////////////////////////////////////////
void GPU_CG(fem_float* matrix_A,
            fem_float* vector_X,
            fem_float* vector_B,
            int n, int n_iterations,
            fem_float epsilon,
            size_t local_work_size) {
    size_t VEC_buffer_size, MAT_buffer_size;
    cl_mem  MatA_mem, VecX_mem, VecD_mem, VecQ_mem, VecR_mem;  // VecB_mem;

    fem_float* vector_D = allocVector(n, false);
    fem_float* vector_R = allocVector(n, false);
    fem_float* vector_Q = allocVector(n, false);

    // Initializes Vectors
    int i;
    for (i = 0; i < n; ++i) {
      vector_X[i] = 0;
      vector_D[i] = vector_B[i];
      vector_R[i] = vector_D[i];
      vector_Q[i] = 0;
    }

    OCL.loadSource("../src/clKernels/gpuCG.cl");
    OCL.loadKernel("CGdense");

    // Memory Allocation
    {
      VEC_buffer_size = sizeof(float) * n;
      MAT_buffer_size = sizeof(float) * n * n;

      MatA_mem = OCL.createBuffer(MAT_buffer_size, CL_MEM_READ_ONLY);
      VecX_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_WRITE);
      VecD_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_WRITE);
      VecQ_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_WRITE);
      VecR_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_WRITE);

      // Enqueue Buffers For Execution
      OCL.enqueueWriteBuffer(MatA_mem, MAT_buffer_size, matrix_A, false);
      OCL.enqueueWriteBuffer(VecX_mem, VEC_buffer_size, vector_X, false);
      OCL.enqueueWriteBuffer(VecD_mem, VEC_buffer_size, vector_D, false);
      OCL.enqueueWriteBuffer(VecQ_mem, VEC_buffer_size, vector_Q, false);
      OCL.enqueueWriteBuffer(VecR_mem, VEC_buffer_size, vector_R, false);

      // Get all of the stuff written and allocated
      OCL.finish();
    }

    // Kernel Arguments
    {
      i = 0;
      // Now setup the arguments to our kernel
      OCL.setKernelArg(i, sizeof(int),    &MatA_mem);
      OCL.setKernelArg(i, sizeof(cl_mem), &VecX_mem);
      OCL.setKernelArg(i, sizeof(cl_mem), &VecD_mem);
      OCL.setKernelArg(i, sizeof(cl_mem), &VecQ_mem);
      OCL.setKernelArg(i, sizeof(cl_mem), &VecR_mem);
      OCL.setKernelArg(i, local_work_size * sizeof(fem_float), NULL);

      OCL.finish();
    }

    // Execution and Read
    {
      int global_work_size = n;

      OCL.setGlobalWorksize(1, global_work_size);
      OCL.setLocalWorksize(1, local_work_size);
      OCL.enquequeNDRangeKernel(1, true);
      OCL.enqueueReadBuffer(VecX_mem, VEC_buffer_size, vector_X, true);
    }

    // Teardown
    {
      OCL.releaseMem(MatA_mem);
      OCL.releaseMem(VecX_mem);
      OCL.releaseMem(VecD_mem);
      OCL.releaseMem(VecQ_mem);
      OCL.releaseMem(VecR_mem);
    }

    //   printf("Vector X:\n");
    //   printVectorf(vector_X, n);
}
