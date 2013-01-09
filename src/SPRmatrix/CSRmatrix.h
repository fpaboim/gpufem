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
// Sparsely Sparse Matrix Library - Compressed Row Sparse Matrix Implementation
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////
#ifndef CSRMATRIX_H_
#define CSRMATRIX_H_

#include "SPRmatrix/SPRmatrix.h"
#include "Utils/util.h"

class CSRmatrix : public SPRmatrix
{
public:
  CSRmatrix(int matdim);
  ~CSRmatrix();

  void         SetElem(const int row, const int col, const fem_float val);
  void         AddElem(const int row, const int col, const fem_float val);
  fem_float    GetElem(const int row, const int col);
  size_t       GetMatSize();
  int          GetNNZ();
  void         SetNNZInfo(int nnz, int band);
  void         Ax_y(fem_float* x, fem_float* y);
  void         SolveCgGpu(fem_float* vector_X,
                          fem_float* vector_B,
                          int n_iterations,
                          fem_float epsilon,
                          size_t local_work_size);
  void         Clear();
  void         Teardown();

private:
  inline void  GrowROW(const int row);
  int          CheckBounds(const int row, const int col);
  int          AllocateMatrix(const int matdim);

  //-------------------------------------------
  // member variables
  //-------------------------------------------
  typedef struct ROWdata {
    fem_float* val;
    int* col;
    int  nNZ;
    int  maxSz;
  } ROWdata;

  // CSR Matrix Format Data Structure
  ROWdata* m_ROWdata;  /* matrix data vector */
  int      m_initrowsize;
  int      m_growstep;

};

#endif  // CSRMATRIX_H_
