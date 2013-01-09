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
// Sparsely Sparse Matrix Library - ELLpack Sparse Matrix Implementation
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////

// Headers
#include "SPRmatrix/ELLmatrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>

#include "OpenCL/OCLwrapper.h"
#include "utils/util.h"
#include "LAops/LAops.h"

//-----------------------------------------------------------------------
ELLmatrix::ELLmatrix(int matdim) {
  m_matdim        = matdim;
  m_maxrowlen     = 64;
  m_growstep      = 64;
  m_matdata       = NULL;
  m_colidx        = NULL;
  m_rownnz        = NULL;
  m_iskernelbuilt = false;

  AllocateMatrix(matdim);
}

//-----------------------------------------------------------------------
ELLmatrix::~ELLmatrix() {
  Teardown();
}
//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

// SetElem: sets sparse matrix element to val
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::SetElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  int rownnz = m_rownnz[row];
  // Strided binary search for insertion position
  int pos = BinSearchIntStep(m_colidx, col, rownnz, row, m_matdim);

  // In case element exists sets the element
  if (m_colidx[pos] == col) {
    m_matdata[pos] = val;
  } else {  // Else inserts element into matrix
    // If matrix is full grows matrix size
    #pragma omp critical (memalloc)
    {
      InsertElem(rownnz, pos, val, col, row);
    }
  }
}

// AddElem: adds number to sparse matrix
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::AddElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  int rownnz = m_rownnz[row];
  // Strided binary search for insertion position
  int pos = BinSearchIntStep(m_colidx, col, rownnz, row, m_matdim);

  // In case element exists adds to the element
  #pragma omp critical (memalloc)
  {
    if (m_colidx[pos] == col) {
      m_matdata[pos] += val;
    } else { // Else inserts element into matrix
      InsertElem(rownnz, pos, val, col, row);
    }
  }
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::InsertElem(int rownnz, int pos, const fem_float val,
                           const int col, const int row ) {
  // Moves data forward before insert
  int zeropos = row + (rownnz * m_matdim);
  for (int i = zeropos; i > pos; i = i - m_matdim) {
    m_matdata[i] = m_matdata[i-m_matdim];
    m_colidx[i]  = m_colidx[i-m_matdim];
  }
  m_matdata[pos] = val;
  m_colidx[pos]  = col;
  // increments number of nonzeros and checks if matrix needs to be grown
  m_rownnz[row]++;
  if (m_rownnz[row] == m_maxrowlen) {
    GrowMatrix();
  }
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
fem_float ELLmatrix::GetElem(const int row, const int col) {
  if (!BoundsOK(row, col))
    return 0;

  // Linear Search
  int pos = -1;
  for (int i = row; i < (m_maxrowlen * m_matdim); i = i + m_matdim) {
    if (m_colidx[i] == col) {
      pos = i;
      break;
    }
  }

  if (pos == -1)
    return 0;

  return m_matdata[pos];
}

// GetMatSize: considers size of matrix data and column index arrays
////////////////////////////////////////////////////////////////////////////////
size_t ELLmatrix::GetMatSize() {
  size_t datasize = sizeof(fem_float) * m_matdim * m_maxrowlen;
  size_t idxsize  = sizeof(int) * m_matdim * m_maxrowlen;

  return (idxsize + datasize);
}

// GetNNZ: gets number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix::GetNNZ() {
  int nnz = 0;
  int totalentries = (m_matdim * m_maxrowlen);
  for (int i = 0; i < totalentries; ++i) {
    if (m_matdata[i] != 0) ++nnz;
  }

  return nnz;
}

// SetNNZ: Sets the number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::SetNNZInfo(int nnz, int band) {
  if (nnz < 0 || band < 0)
    return;

  ReallocateForBandsize(band);
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::Ax_y(fem_float* x, fem_float* y) {
  #pragma omp parallel for
  for (int i = 0; i < m_matdim; ++i) {
    y[i] = 0;
    for (int j = 0; j < m_rownnz[i]; ++j) {
      int idx = i + (j * m_matdim);
      y[i] += m_matdata[idx] * x[m_colidx[idx]];
    }
  }
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::AxyGPU(fem_float* x, fem_float* y, size_t local_worksize) {
  bool printkerneltime = false;
  OCL.setDir("C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//src//OpenCL//clKernels//");
  //OCL.setDir(".\\..\\src\\OpenCL\\clKernels\\");

  OCL.loadSource("LAopsEll.cl");
  switch (m_optimizationstrat) {
    case NAIVE:
      OCL.loadKernel("SpMVNaive"); break;
    case NAIVEUR:
      OCL.loadKernel("SpMVNaiveUR"); break;
    case SHARE:
      OCL.loadKernel("SpMVStag"); break;
    case BLOCK:
      OCL.loadKernel("SpMVCoal"); break;
    case BLOCKUR:
      OCL.loadKernel("SpMVCoalUR"); break;
    default:
      OCL.loadKernel("SpMVNaive");
  }

  // Memory Allocation Sizes
  size_t VEC_buffer_size, MAT_buffer_size, IDX_buffer_size;
  MAT_buffer_size = sizeof(fem_float) * m_matdim * m_maxrowlen;
  IDX_buffer_size = sizeof(int) * m_matdim * m_maxrowlen;
  VEC_buffer_size = sizeof(fem_float) * m_matdim;

  cl_mem  MatA_mem, IdxA_mem, NnzA_mem, VecX_mem, VecY_mem;
  MatA_mem = OCL.createBuffer(MAT_buffer_size, CL_MEM_READ_ONLY);
  IdxA_mem = OCL.createBuffer(IDX_buffer_size, CL_MEM_READ_ONLY);
  NnzA_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_ONLY);
  VecX_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_ONLY);
  VecY_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_WRITE);

  // Enqueue Buffers For Execution
  OCL.enqueueWriteBuffer(MatA_mem, MAT_buffer_size, m_matdata, false);
  OCL.enqueueWriteBuffer(IdxA_mem, IDX_buffer_size, m_colidx,  false);
  OCL.enqueueWriteBuffer(NnzA_mem, VEC_buffer_size, m_rownnz,  false);
  OCL.enqueueWriteBuffer(VecX_mem, VEC_buffer_size, x,         false);

  // Get all of the stuff written and allocated
  OCL.finish();

  // Kernel Arguments
  int i = 0;
  OCL.setKernelArg(i, sizeof(cl_mem), &MatA_mem); ++i;
  OCL.setKernelArg(i, sizeof(cl_mem), &IdxA_mem); ++i;
  OCL.setKernelArg(i, sizeof(cl_mem), &NnzA_mem); ++i;
  OCL.setKernelArg(i, sizeof(int), &m_maxrowlen); ++i;
  OCL.setKernelArg(i, sizeof(int), &m_matdim   ); ++i;
  OCL.setKernelArg(i, sizeof(cl_mem), &VecX_mem); ++i;
  OCL.setKernelArg(i, sizeof(cl_mem), &VecY_mem); ++i;
  if (m_optimizationstrat == BLOCK)
    OCL.setKernelArg(i,
      local_worksize * local_worksize * sizeof(fem_float), NULL);
  else
    OCL.setKernelArg(i, m_maxrowlen * sizeof(fem_float), NULL);
  OCL.finish();

  // Block, aka Tiled strategy
  if ((m_optimizationstrat == BLOCK) || (m_optimizationstrat == BLOCKUR)) {
    size_t localszX = 4;
    size_t gpumatdim = m_matdim;
    // make divisible by 32
    if (gpumatdim % 32 != 0) {
      gpumatdim = gpumatdim >> 5;
      gpumatdim++;
      gpumatdim = gpumatdim << 5;
    }
    OCL.setLocalWorksize (0, localszX);
    OCL.setLocalWorksize (1, local_worksize);
    OCL.setGlobalWorksize(0, localszX);
    OCL.setGlobalWorksize(1, gpumatdim);
    OCL.enquequeNDRangeKernel(2, printkerneltime);
  } else {
    OCL.setLocalWorksize (0, local_worksize);
    OCL.setGlobalWorksize(0, m_matdim * local_worksize);
    OCL.enquequeNDRangeKernel(1, printkerneltime);
  }

  OCL.enqueueReadBuffer(VecY_mem, VEC_buffer_size, y, true);

  // Teardown
  OCL.releaseMem(MatA_mem);
  OCL.releaseMem(IdxA_mem);
  OCL.releaseMem(NnzA_mem);
  OCL.releaseMem(VecX_mem);
  OCL.releaseMem(VecY_mem);
}


// GPU_CG: Solve is controlled by host with partial work (mainly LA Ops
// done by the GPU - for CG see Shewchuk
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::SolveCgGpu(fem_float* vector_X,
                           fem_float* vector_B,
                           int n_iterations,
                           fem_float epsilon,
                           size_t local_worksize) {
  size_t localszX = 8;
  size_t VEC_buffer_size, MAT_buffer_size, IDX_buffer_size;
  cl_mem  MatA_mem, IdxA_mem, NnzA_mem, VecD_mem, VecQ_mem;
  fem_float* vector_D = (fem_float*)_aligned_malloc(m_matdim * sizeof(fem_float), 16);
  fem_float* vector_R = (fem_float*)_aligned_malloc(m_matdim * sizeof(fem_float), 16);
  fem_float* vector_Q = (fem_float*)_aligned_malloc(m_matdim * sizeof(fem_float), 16);
  // Initializes Vectors
  int i;
  //#pragma omp parallel for private (i)
  for (i = 0; i < m_matdim; ++i) {
    vector_X[i] = 0;
    vector_D[i] = vector_B[i];
    vector_R[i] = vector_B[i];
  }
  OCL.setDir("C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//src//OpenCL//clKernels//");
  //OCL.setDir(".\\..\\src\\OpenCL\\clKernels\\");
  OCL.loadSource("LAopsEll.cl");
  switch (m_optimizationstrat) {
    case NAIVE:
      OCL.loadKernel("SpMVNaive"); break;
    case NAIVEUR:
      OCL.loadKernel("SpMVNaiveUR"); break;
    case SHARE:
      OCL.loadKernel("SpMVStag"); break;
    case BLOCK:
      OCL.loadKernel("SpMVCoal"); break;
    case BLOCKUR:
      OCL.loadKernel("SpMVCoalUR"); break;
    case TEST:
      OCL.loadKernel("SpMVCoalUR2"); break;
    default:
      OCL.loadKernel("SpMVNaive");
  }
  // Memory Allocation Sizes
  MAT_buffer_size = sizeof(fem_float) * m_matdim * m_maxrowlen;
  IDX_buffer_size = sizeof(int) * m_matdim * m_maxrowlen;
  VEC_buffer_size = sizeof(fem_float) * m_matdim;
  MatA_mem = OCL.createBuffer(MAT_buffer_size, CL_MEM_READ_ONLY);
  IdxA_mem = OCL.createBuffer(IDX_buffer_size, CL_MEM_READ_ONLY);
  NnzA_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_ONLY);
  VecD_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_WRITE);
  VecQ_mem = OCL.createBuffer(VEC_buffer_size, CL_MEM_READ_WRITE);
  // Enqueue Buffers For Execution
  OCL.enqueueWriteBuffer(MatA_mem, MAT_buffer_size, m_matdata, false);
  OCL.enqueueWriteBuffer(IdxA_mem, IDX_buffer_size, m_colidx,  false);
  OCL.enqueueWriteBuffer(NnzA_mem, VEC_buffer_size, m_rownnz,  false);
  OCL.enqueueWriteBuffer(VecD_mem, VEC_buffer_size, vector_D,  false);
  // Get all of the stuff written and allocated
  OCL.finish();
  // Kernel Arguments
  i = 0;
  // Now setup the arguments to our kernel
  OCL.setKernelArg(i, sizeof(cl_mem), &MatA_mem); ++i;
  OCL.setKernelArg(i, sizeof(cl_mem), &IdxA_mem); ++i;
  OCL.setKernelArg(i, sizeof(cl_mem), &NnzA_mem); ++i;
  OCL.setKernelArg(i, sizeof(int), &m_maxrowlen); ++i;
  OCL.setKernelArg(i, sizeof(int), &m_matdim   ); ++i;
  OCL.setKernelArg(i, sizeof(cl_mem), &VecD_mem); ++i;
  OCL.setKernelArg(i, sizeof(cl_mem), &VecQ_mem); ++i;

  // Sets kernel arguments and local worksize limits
  if ((m_optimizationstrat == BLOCK) ||
      (m_optimizationstrat == BLOCKUR) ||
      (m_optimizationstrat == TEST)) {
    while (local_worksize * localszX > 256) {
      local_worksize /= 2;
    }
    OCL.setKernelArg(i,
                     local_worksize * localszX * sizeof(fem_float),
                     NULL);
  } else {
    OCL.setKernelArg(i, local_worksize * sizeof(fem_float), NULL);
  }
  OCL.finish();

  // Fixes dimensions to ensure divisibility by local sizes and
  // sets OpenCL worksizes
  if ((m_optimizationstrat == BLOCK) ||
      (m_optimizationstrat == BLOCKUR) ||
      (m_optimizationstrat == TEST)) {
    // make global matdim divisible by 32
    size_t gpumatdim = m_matdim;
    if (gpumatdim % 32 != 0) {
      gpumatdim = gpumatdim >> 5;
      gpumatdim++;
      gpumatdim = gpumatdim << 5;
    }
    OCL.setLocalWorksize (0, localszX);
    OCL.setLocalWorksize (1, local_worksize);
    OCL.setGlobalWorksize(0, localszX);
    OCL.setGlobalWorksize(1, gpumatdim);
  } else {
    if (m_optimizationstrat == SHARE) {
      if (local_worksize >= 64) {
        local_worksize = 32;
      }
    }
    OCL.setLocalWorksize (0, local_worksize);
    OCL.setGlobalWorksize(0, m_matdim * local_worksize);
  }

  // Direction Vector
  fem_float delta_new = 0;
  fem_float delta_old = 0;
  fem_float err_bound = 0;
  fem_float alpha     = 0;
  fem_float beta      = 0;
  // zero initial value for x
  // r = b - Ax
  // d = r
  delta_new = dotProduct(m_matdim, vector_R, vector_R);
  err_bound = epsilon * epsilon * delta_new;
  for (i = 0; (i < n_iterations) && (delta_new > err_bound); ++i) {
    // q = Ad
    //matrix_A->Ax_y(vector_d, vector_q);
    if ((m_optimizationstrat == BLOCK) ||
        (m_optimizationstrat == BLOCKUR) ||
        (m_optimizationstrat == TEST)) {
      OCL.enquequeNDRangeKernel(2, false);
    } else {
      OCL.enquequeNDRangeKernel(1, false);
    }
    OCL.enqueueReadBuffer(VecQ_mem, VEC_buffer_size, vector_Q, true);
    //AxyGPU(vector_D, vector_Q, local_worksize);

    // alpha = rDotrNew / (d dot q)
    alpha = delta_new / dotProductOMP(m_matdim, vector_D, vector_Q);
    // x = x + alpha * d
    addScaledVectToSelfOMP(vector_X, vector_D, alpha, m_matdim);
    // r = r - alpha * q
    addScaledVectToSelfOMP(vector_R, vector_Q, (-alpha), m_matdim);
    // rDotrOld = rDotrNew
    delta_old = delta_new;
    // rDotrNew = r dot r
    delta_new = dotProductOMP(m_matdim, vector_R, vector_R);
    // beta = rDotrNew / rDotrOld
    beta = delta_new / delta_old;
    // d = r + beta * d
    addSelfScaledToVectOMP(vector_D, vector_R, beta, m_matdim);
    OCL.enqueueWriteBuffer(VecD_mem, VEC_buffer_size, vector_D, true);
  }
  if (false) {
    if (i == n_iterations) {
      printf("\n\n***********\nReached max num of iterations!\n***********\n");
    } else {
      //printf("\n\n***********\n          Solved!             \n***********\n");
      //printf("Vector X:\n");
      //printVectorf(vector_X, m_matdim);
      printf("solver CG iterations:%i\n", i);
    }
  }
  // Teardown
  OCL.releaseMem(MatA_mem);
  OCL.releaseMem(IdxA_mem);
  OCL.releaseMem(NnzA_mem);
  OCL.releaseMem(VecD_mem);
  OCL.releaseMem(VecQ_mem);
  _aligned_free(vector_D);
  _aligned_free(vector_Q);
  _aligned_free(vector_R);
}

// Clear: resets the matrix data but otherwise keeps matrix information
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::Clear() {
  // Frees mallocd value and column info
  Teardown();
  AllocateMatrix(m_matdim);
}

// Teardown: frees memory used by matrix
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix::Teardown() {
  // Frees mallocd value array and column info array
  if (m_colidx) {
    _aligned_free(m_colidx);
    m_colidx = NULL;
  }
  if (m_matdata) {
    _aligned_free(m_matdata);
    m_matdata = NULL;
  }
  if (m_rownnz) {
    _aligned_free(m_rownnz);
    m_rownnz = NULL;
  }
  m_maxrowlen = 32;
}

// GrowMatrix: Reallocates matrix data for added nonzeros
////////////////////////////////////////////////////////////////////////////////
inline void ELLmatrix::GrowMatrix() {
    m_maxrowlen += m_growstep;
    m_matdata =
      (fem_float*)_aligned_realloc(m_matdata,
                                   m_maxrowlen * m_matdim * sizeof(fem_float),
                                   16);
    m_colidx  =
      (int*)_aligned_realloc(m_colidx,
                             m_maxrowlen * m_matdim * sizeof(int),
                             16);
  // Sets new matrix data to zero
  for (int i = (m_maxrowlen - m_growstep) * m_matdim;
    i < (m_maxrowlen * m_matdim);
    i++) {
      m_matdata[i] = 0;
  }
}

// AllocateMatrix: allocated memory for the matrix
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix::AllocateMatrix(const int matdim) {
  m_matdata = (fem_float*)_aligned_malloc(
                            m_maxrowlen * matdim * sizeof(fem_float),
                            16
                          );
  m_colidx  = (int*)_aligned_malloc(m_maxrowlen * matdim * sizeof(int), 16);
  m_rownnz  = (int*)_aligned_malloc(matdim * sizeof(int), 16);

  for (int i = 0; i < matdim; ++i) {
    m_rownnz[i] = 0;
    for (int j = 0; j < m_maxrowlen; j++)
      m_matdata[i+matdim*j] = 0;
  }

  return 1;
}

// ReallocateForBandsize: Uses band information to preallocate whole matrix
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix::ReallocateForBandsize(const int band) {
  #pragma omp critical (memalloc)
  {
    m_maxrowlen = band;

    m_matdata = (fem_float*)_aligned_realloc(m_matdata,
      m_maxrowlen * m_matdim * sizeof(fem_float),
      16);
    m_colidx  = (int*)_aligned_realloc(m_colidx,
      m_maxrowlen * m_matdim * sizeof(int),
      16);
  }

  #pragma omp parallel for
  for (int i = 0; i < m_matdim; ++i) {
    m_rownnz[i] = 0;
    for (int j = 0; j < m_maxrowlen; j++)
      m_matdata[i + (m_matdim * j)] = 0;
  }

  return 1;
}

// BinSearchInt: performs binary search over vector with stepped values, length
// is of stepped search and NOT of whole intvector
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix::BinSearchIntStep(int* intvector,
                                int  val,
                                int  steppedlen,
                                int  startpos,
                                int  step) {
  int min = 0, max = steppedlen;
  const int* rebasedvector = &intvector[startpos];
  while (min < max) {
    int middle = (min + max) >> 1;
    if (val > rebasedvector[middle * step])
      min = middle + 1;
    else
      max = middle;
  }
  return startpos + (min * step);
}
