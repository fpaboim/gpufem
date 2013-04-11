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
#ifndef ELLMATRIX_H_
#define ELLMATRIX_H_

#include "SPRmatrix/SPRmatrix.h"
#include "Utils/util.h"

// Matrix M Data is stored in format aligned to 16 byte boundary (4 floats or
// multiple thereof). The data is stored scanning columns of the compressed
// matrix matdata:
// M = [4 0 5 0   matdata = [4 5 0 0   colidx = [0 2 * *  rownnz = [2
//      0 3 0 1              3 1 0 0             1 3 * *            2
//      4 0 2 6              4 2 6 0             0 2 3 *            3
//      0 0 0 4]             4 0 0 0]            3 * * *]           1]
// -> Matrices are converted to arrays scanned in columnwise fashion: e.g.:
// matdata = [4 3 4 4 5 1 2 0 0 0 6 0 ... 0] -> padded with zeros
// colidx  = [0 1 0 3 2 3 2 * * * 3 * ... *] -> * is garbage

class ELLmatrix : public SPRmatrix {
 public:
  ELLmatrix(int matdim);
  ~ELLmatrix();

  void         SetElem(const int row, const int col, const fem_float val);
  void         AddElem(const int row, const int col, const fem_float val);
  fem_float    GetElem(const int row, const int col);
  size_t       GetMatSize();
  int          GetNNZ();
  void         SetNNZInfo(int nnz, int band);
  void         Ax_y(fem_float* x, fem_float* y);
  void         AxyGPU(fem_float* x, fem_float* y, size_t local_worksize);
  void         SolveCgGpu(fem_float* vector_X,
                          fem_float* vector_B,
                          int n_iterations,
                          fem_float epsilon,
                          size_t local_work_size);
  void         SolveCgGpu2(fem_float* vector_X,
                           fem_float* vector_B,
                           int n_iterations,
                           fem_float epsilon,
                           size_t local_work_size);
  void         Clear();
  void         Teardown();
  static int   BinSearchIntStep(int* intvector,
                                int  val,
                                int  steppedlength,
                                int  startpos,
                                int  step) ;

 private:
  void         InsertElem(int rownnz, int pos, const fem_float val,
                          const int col, const int row);
  void         GrowMatrix();
  int          AllocateMatrix(const int matdim);
  int          ReallocateForBandsize(const int matdim);
  int          CheckBounds(const int row, const int col);

  //-------------------------------------------
  // member variables
  //-------------------------------------------
  // ELLpack Matrix Format Data Structure
  fem_float* m_matdata;    // matrix data vector - empty data is 0!
  int*       m_colidx;     // column index - empty data is GARBAGE!
  int*       m_rownnz;     // row is index, value is nnz
  int*       m_rownnztrigger; // given independent row access num of concurrent 
                              // threads
  int        m_maxrowlen;  // maximum row length (number of compressed columns
                           // stored)
  int        m_growstep;   // step to grow compressed row length
  bool       m_iskernelbuilt;
};

#endif  // ELLMATRIX_H_
