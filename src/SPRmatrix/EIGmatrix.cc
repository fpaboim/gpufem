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
#include "SPRmatrix/EIGmatrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "LAops/LAops.h"

//-----------------------------------------------------------------------
EIGmatrix::EIGmatrix(int matdim) {
  m_matdim      = matdim;
  m_maxrowlen   = 32;
  m_growstep    = 32;

  m_matrix.resize(m_matdim, m_matdim);
}

//-----------------------------------------------------------------------
EIGmatrix::~EIGmatrix() {
}
//-----------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

// SetElem: sets sparse matrix element to val
////////////////////////////////////////////////////////////////////////////////
void EIGmatrix::SetElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;
  #pragma omp critical
  {
    m_matrix.coeffRef(row, col) = val;
  }
}

// AddElem: adds number to sparse matrix
////////////////////////////////////////////////////////////////////////////////
void EIGmatrix::AddElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;
  #pragma omp critical
  {
    m_matrix.coeffRef(row, col) += val;
  }
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
fem_float EIGmatrix::GetElem(const int row, const int col) {
  if (!BoundsOK(row, col))
    return 0;
  return m_matrix.coeff(row, col);
}

// GetMatSize: considers size of matrix data and column index arrays
////////////////////////////////////////////////////////////////////////////////
size_t EIGmatrix::GetMatSize() {
  return m_matrix.size();
}

// GetNNZ: gets number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
int EIGmatrix::GetNNZ() {
  return m_matrix.nonZeros();
}

// SetNNZ: Sets the number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
void EIGmatrix::SetNNZInfo(int nnz, int band) {
  m_matrix.reserve(nnz);
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void EIGmatrix::Axy(fem_float* x, fem_float* y) {
  spax_yOMP(this, x, y, m_matdim, false);
}

////////////////////////////////////////////////////////////////////////////////
void EIGmatrix::EIG_CG(fem_float* vector_X,
                       fem_float* vector_B,
                       int        n_iterations,
                       fem_float  epsilon) {
  using namespace Eigen;
  VectorXf xf(m_matdim);
  VectorXf bf(m_matdim);
  for (int i = 0; i < m_matdim; i++) {
    bf[i] = vector_B[i];
  }
  ConjugateGradient<SparseMatrix<float> > cg;
  cg.compute(m_matrix);
  cg.setMaxIterations(n_iterations);
  cg.setTolerance(epsilon);
  xf = cg.solve(bf);
  for (int i = 0; i < m_matdim; i++) {
    vector_X[i] = xf[i];
  }
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void EIGmatrix::CG(fem_float* vector_X,
                   fem_float* vector_B,
                   int        n_iterations,
                   fem_float  epsilon) {
  switch (m_devicemode) {
    case DEV_CPU:
      EIG_CG(vector_X, vector_B, n_iterations, epsilon);
      break;
    case DEV_OMP:
      EIG_CG(vector_X, vector_B, n_iterations, epsilon);
      break;
    case DEV_GPU:
      EIG_CG(vector_X, vector_B, n_iterations, epsilon);
      break;
    default:  // default falls back to CPU single threaded
      EIG_CG(vector_X, vector_B, n_iterations, epsilon);
      assert(false);
      break;
  }
}

// Clear: resets the matrix
////////////////////////////////////////////////////////////////////////////////
void EIGmatrix::Clear() {
  return m_matrix.setZero();
}

// Teardown: frees memory used by matrix
////////////////////////////////////////////////////////////////////////////////
void EIGmatrix::Teardown() {
  m_matrix.resize(0,0);
}

// GrowROW: reallocates row memory to make room for new NZ entries
////////////////////////////////////////////////////////////////////////////////
inline void EIGmatrix::GrowMatrix() {
}

// CheckBounds: checks if matrix access is within bounds
////////////////////////////////////////////////////////////////////////////////
int EIGmatrix::CheckBounds(const int row, const int col ) {
  // checks for valid row
  if (row < 0 || row >= m_matdim || col < 0 || col >= m_matdim) {
    if (m_verboseerrors)
      printf("**ERROR** EIGmatrix.cpp - Out of bounds matrix access.\n");
    return 0;
  }

  return 1;
}
