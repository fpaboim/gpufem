////////////////////////////////////////////////////////////////////////////////
// Sparsely Sparse Matrix Library - ELLpack Sparse Matrix Implementation
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////

// Headers
#include "SPRmatrix/ELLmatrix2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>

#include "OpenCL/OCLwrapper.h"
#include "utils/util.h"
#include "LAops/LAops.h"

//-----------------------------------------------------------------------
ELLmatrix2::ELLmatrix2(int matdim) {
  m_matdim        = matdim;
  m_maxrowlen     = 32;
  m_growthfactor  = 2;
  m_matdata       = NULL;
  m_colidx        = NULL;
  m_rownnz        = NULL;
  m_iskernelbuilt = false;

  AllocateMatrix(matdim);
}

//-----------------------------------------------------------------------
ELLmatrix2::~ELLmatrix2() {
  Teardown();
}
//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

// SetElem: sets sparse matrix element to val
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix2::SetElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  int rownnz = m_rownnz[row];
  // Strided binary search for insertion position
  int pos = LinSearchRow(m_colidx, col, row, rownnz, m_maxrowlen);

  if (pos == -1) { // key not found
    InsertElem(rownnz, (row * m_maxrowlen + rownnz), val, col, row);
    return;
  }
  // In case element exists sets the element
  if (m_colidx[pos] == col) {
    m_matdata[pos] = val;
  } else {  // Else inserts element into matrix
    InsertElem(rownnz, pos, val, col, row);
  }
}

// AddElem: adds number to sparse matrix
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix2::AddElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  int rownnz = m_rownnz[row];
  // Strided binary search for insertion position
  int pos = BinSearchRow(m_colidx, col, row, rownnz, m_maxrowlen);

  if (pos == -1) { // key not found
    InsertElem(rownnz, (row * m_maxrowlen + rownnz), val, col, row);
    return;
  }
  // In case element exists adds to the element
  if (m_colidx[pos] == col) {
    m_matdata[pos] += val;
  } else { // Else inserts element into matrix
    InsertElem(rownnz, pos, val, col, row);
  }
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix2::InsertElem(int rownnz, int pos, const fem_float val,
                           const int col, const int row ) {
  // Moves data forward before insert
  int zeropos = (row * m_maxrowlen) + rownnz;
  for (int i = zeropos; i > pos; i--) {
    m_matdata[i] = m_matdata[i-1];
    m_colidx[i]  = m_colidx[i-1];
  }
  m_matdata[pos] = val;
  m_colidx[pos]  = col;
  // increments number of nonzeros and checks if matrix needs to be grown
  m_rownnz[row]++;
  if (m_rownnz[row] == m_maxrowlen - 3) {
    m_prealloctrigger = true;
    return;
  }
  if (m_rownnz[row] == m_maxrowlen) {
    GrowMatrix();
    m_prealloctrigger = false;
  }
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
fem_float ELLmatrix2::GetElem(const int row, const int col) {
  if (!BoundsOK(row, col))
    return 0;
  int rownnz = m_rownnz[row];
  int pos = BinSearchRow(m_colidx, col, row, rownnz, m_maxrowlen);
  if (pos == -1)
    return 0;

  return m_matdata[pos];
}

// GetMatSize: considers size of matrix data and column index arrays
////////////////////////////////////////////////////////////////////////////////
size_t ELLmatrix2::GetMatSize() {
  size_t datasize = sizeof(fem_float) * m_matdim * m_maxrowlen;
  size_t idxsize  = sizeof(int) * m_matdim * m_maxrowlen;

  return (idxsize + datasize);
}

// GetNNZ: gets number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix2::GetNNZ() {
  int nnz = 0;
  for (int i = 0; i < m_matdim; ++i) {
    nnz += m_rownnz[i];
  }

  return nnz;
}

// SetNNZ: Sets the number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix2::SetNNZInfo(int nnz, int band) {
  if (nnz < 0 || band < 0 || (band < m_maxrowlen))
    return;

  ReallocateForBandsize(band);
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix2::Ax_y(fem_float* x, fem_float* y) {
  #pragma omp parallel for
  for (int i = 0; i < m_matdim; ++i) {
    y[i] = 0;
    for (int j = 0; j < m_rownnz[i]; ++j) {
      int idx = (i * m_maxrowlen) + j;
      y[i] += m_matdata[idx] * x[m_colidx[idx]];
    }
  }
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix2::AxyGPU(fem_float* x, fem_float* y, size_t local_worksize) {
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
      printf("ERROR LOADING KERNEL!!!!\n");
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
  if (m_optimizationstrat == BLOCK) {
    size_t blocksharedsize = local_worksize * local_worksize * sizeof(fem_float);
    bool blksharedmemok = OCL.localSizeIsOK(blocksharedsize);
    assert(blksharedmemok);
    OCL.setKernelArg(i, blocksharedsize, NULL);
  } else {
    size_t sharedmemsize = m_maxrowlen * sizeof(fem_float);
    bool sharedmemok = OCL.localSizeIsOK(sharedmemsize);
    OCL.setKernelArg(i, sharedmemsize, NULL);
    assert(sharedmemok);
  }
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
void ELLmatrix2::SolveCgGpu(fem_float* vector_X,
                           fem_float* vector_B,
                           int n_iterations,
                           fem_float epsilon,
                           size_t local_worksize) {
  size_t localszX = 8;
  size_t VEC_buffer_size, MAT_buffer_size, IDX_buffer_size;
  cl_mem  MatA_mem, IdxA_mem, NnzA_mem, VecD_mem, VecQ_mem;
  fem_float* vector_D = (fem_float*)malloc(m_matdim * sizeof(fem_float));
  fem_float* vector_R = (fem_float*)malloc(m_matdim * sizeof(fem_float));
  fem_float* vector_Q = (fem_float*)malloc(m_matdim * sizeof(fem_float));
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
  free(vector_D);
  free(vector_Q);
  free(vector_R);
}

// Clear: resets the matrix data but otherwise keeps matrix information
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix2::Clear() {
  // Frees mallocd value and column info
  Teardown();
  AllocateMatrix(m_matdim);
}

// Teardown: frees memory used by matrix
////////////////////////////////////////////////////////////////////////////////
void ELLmatrix2::Teardown() {
  // Frees mallocd value array and column info array
  if (m_colidx) {
    free(m_colidx);
    m_colidx = NULL;
  }
  if (m_matdata) {
    free(m_matdata);
    m_matdata = NULL;
  }
  if (m_rownnz) {
    free(m_rownnz);
    m_rownnz = NULL;
  }
  m_maxrowlen = 64;
}

// GrowMatrix: Reallocates matrix data for added nonzeros
////////////////////////////////////////////////////////////////////////////////
inline void ELLmatrix2::GrowMatrix() {
  int lastrowlen = m_maxrowlen;
  m_maxrowlen *= m_growthfactor;
  m_matdata = (fem_float*)realloc(m_matdata,
                                  m_maxrowlen * m_matdim * sizeof(fem_float));
  m_colidx  = (int*) realloc(m_colidx, m_maxrowlen * m_matdim * sizeof(int));
  // Moves Memory: (to, from, blocksize)
  size_t datablksize   = sizeof(fem_float) * lastrowlen;
  size_t colidxblksize = sizeof(int) * lastrowlen;
  for (int i = (m_matdim - 1); i > 0; i--) {
    memmove(&m_matdata[i * m_maxrowlen], &m_matdata[i * lastrowlen],
            datablksize);
    memmove(&m_colidx[i * m_maxrowlen], &m_colidx[i * lastrowlen],
            colidxblksize);
  }
}

// AllocateMatrix: allocated memory for the matrix
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix2::AllocateMatrix(const int matdim) {
  if (m_matdata != NULL)
    return 0;
  m_matdata = (fem_float*) malloc(m_maxrowlen * matdim * sizeof(fem_float));
  if (m_colidx != NULL)
    return 0;
  m_colidx  = (int*) malloc(m_maxrowlen * matdim * sizeof(int));
  if (m_rownnz != NULL)
    return 0;
  m_rownnz  = (int*) calloc(matdim, sizeof(int));

  return 1;
}

// ReallocateForBandsize: Uses band information to preallocate whole matrix
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix2::ReallocateForBandsize(const int band) {
  int lastrowlen = m_maxrowlen;
  #pragma omp critical (memalloc)
  {
    m_maxrowlen = band;
    m_matdata = (fem_float*) realloc(m_matdata,
      m_maxrowlen * m_matdim * sizeof(fem_float));
    m_colidx  = (int*) realloc(m_colidx, m_maxrowlen * m_matdim * sizeof(int));
  }

  return 1;
}

// BinSearchInt: performs binary search to find insertion point
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix2::BinSearchRow(int* intvector,
                             int  val,
                             int  row,
                             int  nrownnz,
                             int  rowlen) {
  int min = 0, max = nrownnz;
  int startpos = row * rowlen;
  const int* rebasedvector = &intvector[startpos];
  while (min < max) {
    int middle = (min + max) >> 1;
    if (val > rebasedvector[middle])
      min = middle + 1;
    else
      max = middle;
  }
  if (val <= rebasedvector[min]) {
    return startpos + min;
  } else {
    return -1; // not found in array (larger then all values)
  }
}

// LinSearchRow: Linearly searches for insertion point
////////////////////////////////////////////////////////////////////////////////
int ELLmatrix2::LinSearchRow(int* intvector,
                             int  val,
                             int  row,
                             int  nrownnz,
                             int  rowlen) {
  int startpos = row * rowlen;
  int maxpos   = startpos + nrownnz;
  for (int i = startpos; i < maxpos; i++) {
    if (intvector[i] >= val) {
      return i;
    }
  }

  return -1; // not found in array (larger then all values)
}
