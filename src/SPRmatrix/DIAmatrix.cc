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
// Sparsely Sparse Matrix Library - Diagonal Sparse Matrix Implementation
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////

// Headers
#include "DIAmatrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "LAops/LAops.h"

// Constructor
////////////////////////////////////////////////////////////////////////////////
DIAmatrix::DIAmatrix(int matdim) {
  m_matdim     = matdim;
  m_ndiag      = 0;
  m_maxentries = 16;
  m_growstep   = 16;
  m_DIAdata    = (fem_float*)calloc(m_maxentries*m_matdim, sizeof(fem_float*));
  m_posvect    = (int*)malloc(m_maxentries*sizeof(int));
}

// Destructor
////////////////////////////////////////////////////////////////////////////////
DIAmatrix::~DIAmatrix() {
  Teardown();
}


////////////////////////////////////////////////////////////////////////////////
// Matrix Manipulation Functions
////////////////////////////////////////////////////////////////////////////////

// SetElem: sets sparse matrix element to val
////////////////////////////////////////////////////////////////////////////////
void DIAmatrix::SetElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  const int diag = col - row;
  // finds matrix position
  int pos = BinSearchInt2(m_posvect, diag, m_ndiag);
  // if diagonal of element not found at position allocates memory
  if (m_posvect[pos] != diag) {
    insertDiag(pos, diag);
  }

  m_DIAdata[pos * m_matdim + row] = val;
}

// AddElem: adds number to sparse matrix
////////////////////////////////////////////////////////////////////////////////
void DIAmatrix::AddElem(const int row, const int col, const fem_float val) {
  if (!InputIsOK(val, row, col))
    return;

  const int diag = col - row;
  // finds matrix position
  int pos = BinSearchInt2(m_posvect, diag, m_ndiag);
  // if diagonal of element not found allocates memory
  if (m_posvect[pos] != diag) {
    insertDiag(pos, diag);
  }
  m_DIAdata[pos * m_matdim + row] += val;
}

// GetElem: gets number from sparse matrix
////////////////////////////////////////////////////////////////////////////////
fem_float DIAmatrix::GetElem(const int row, const int col) {
  if (!BoundsOK(row, col))
    return 0;

  int diag = col - row;
  int pos = BinSearchInt2(m_posvect, diag, m_ndiag);
  // if diagonal of element not found returns zero else returns element
  if (m_posvect[pos] != diag) {
    return 0;
  } else {
    return m_DIAdata[pos * m_matdim + row];
  }
}

// GetMatSize: gets matrix size, does not consider size of pointers!
////////////////////////////////////////////////////////////////////////////////
size_t DIAmatrix::GetMatSize() {
  return (m_maxentries*m_matdim*sizeof(fem_float));
}

// GetNNZ: gets number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
int DIAmatrix::GetNNZ() {
  int nnz=0;
  for (int diag=0; diag<m_ndiag; diag++){
    for (int j=0; j<m_matdim; j++){
      if(m_DIAdata[diag * m_matdim + j] != 0)
        nnz++;
    }
  }
  return nnz;
}

// SetNNZ: Sets the number of non-zero entries in matrix
////////////////////////////////////////////////////////////////////////////////
void DIAmatrix::SetNNZInfo(int nnz, int band) {
}

// Ax_y: does matrix vector multiply and stores result in y (which should be
// allocated by calling function)
////////////////////////////////////////////////////////////////////////////////
void DIAmatrix::Ax_y(fem_float* x, fem_float* y) {
  spax_yOMP(this, x, y, m_matdim, false);
}

// GPU_CG: Solves Ax = b for x by conjugate gradient method using GPU
////////////////////////////////////////////////////////////////////////////////
void DIAmatrix::SolveCgGpu(fem_float* vector_X,
                           fem_float* vector_B,
                           int n_iterations,
                           fem_float epsilon,
                           size_t local_work_size){
}

// Clear: frees memory and reallocates memory from zero
////////////////////////////////////////////////////////////////////////////////
void DIAmatrix::Clear() {
  Teardown();
  m_ndiag      = 0;
  m_maxentries = 8;
  m_growstep   = 8;
  m_DIAdata    = (fem_float*)calloc(m_maxentries*m_matdim, sizeof(fem_float*));
  m_posvect    = (int*)malloc(m_maxentries*sizeof(int));
}

// Teardown: frees memory used by matrix
////////////////////////////////////////////////////////////////////////////////
void DIAmatrix::Teardown() {
  if (m_DIAdata) {
    free(m_DIAdata);
    m_DIAdata = NULL;
  }
  // Frees vector with position information
  if (m_posvect) {
    free(m_posvect);
    m_posvect = NULL;
  }
}

// insertDiag: inserts a new diagonal in sparse matrix!
////////////////////////////////////////////////////////////////////////////////
inline void DIAmatrix::insertDiag(const int pos, const int diag) {
  m_ndiag++;
  if (m_ndiag > m_maxentries) {
    m_maxentries += m_growstep;
    m_DIAdata = (fem_float*)realloc(m_DIAdata, (m_maxentries * m_matdim *
                                                sizeof(fem_float)));
    m_posvect = (int*)realloc(m_posvect, (m_maxentries * sizeof(int)) );
  }

  // Moves Memory: (to, from, block_size)
  memmove(&m_posvect[pos + 1], &m_posvect[pos],
          (m_ndiag-(pos+1)) * sizeof(int));
  memmove(&m_DIAdata[(pos + 1) * m_matdim], &m_DIAdata[pos * m_matdim],
          (m_ndiag-(pos+1)) * m_matdim * sizeof(fem_float));

  // Fills new diagonal with zeros
  int startpos = pos * m_matdim;
  for (int i = startpos; i < startpos + m_matdim; i++) {
    m_DIAdata[i] = 0;
  }
  m_posvect[pos] = diag;
}
