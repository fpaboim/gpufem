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
// SPRmatrix.h - Sparse matrix class interface header
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////
#ifndef SPRMATRIX_H
#define SPRMATRIX_H

#include "Utils/util.h"

class SPRmatrix
{
public:
  // Sparse Matrix Type Enum
  typedef enum SPRformat {
    NIL = 0,
    DEN = 1,
    DIA = 2,
    CSR = 3,
    ELL = 4,
    EL2 = 5,
    EIG = 6,
    BCR = 7,
  } SPRformat;

  // Optimization strategy used in opencl kernels
  typedef enum OclStrategy {
    UNDEF   = 0,
    NAIVE   = 1,
    NAIVEUR = 2,
    SHARE   = 3,
    BLOCK   = 4,
    BLOCKUR = 5,
    TEST    = 6,
  } OclStrategy;

  SPRmatrix();
  virtual ~SPRmatrix();

  // virtual (interface) functions
  virtual void      SetElem(const int row, const int col, const fem_float val)=0;
  virtual void      AddElem(const int row, const int col, const fem_float val)=0;
  virtual fem_float GetElem(const int row, const int col)=0;
  virtual size_t    GetMatSize() = 0;
  virtual void      SetNNZInfo(int nnz, int band) = 0;
  virtual int       GetNNZ() = 0;
  virtual void      Clear() = 0;
  virtual void      Ax_y(fem_float* x, fem_float* y) = 0;
  virtual void      AxyGPU(fem_float* x, fem_float* y, size_t local_worksize){};
  virtual void      SolveCgGpu(fem_float* vector_X,
                               fem_float* vector_B,
                               int n_iterations,
                               fem_float epsilon,
                               size_t local_work_size) = 0;
  virtual void      Teardown() = 0; // Deallocates matrix

  // common sparse matrix public functions
  static SPRmatrix* CreateMatrix(const int dim, SPRformat matformat);
  int               BinSearchInt(const int* intvector, const int val,
                                 const int length) const;
  int               BinSearchInt2(const int* intvector, const int val,
                                  const int length) const;
  int               BinSearchIntCmov(const int* intvector, const int val,
                                     const int len) const;
  int               BinSmart(const int* intvector, const int val,
                             const int len) const;
  int               LinSearchI(const int* intvector, const int val,
                               const int length) const;
  int               LinSearchISSE(const int* intvector, const int val,
                                  const int length) const;
  int               LinSearchISSE2(const int* intvector, const int val,
                                   const int length) const;
  void              PrintMatrix();
  SPRformat         GetFormat() {return m_matformat;};
  void              SetOclStrategy(OclStrategy strat) {
                      m_optimizationstrat = strat;
                    };
  OclStrategy       GetOclStrategy() {return m_optimizationstrat;};
  void              VerboseErrors(bool isverbose) {
                      m_verboseerrors = isverbose;
                    };

protected:
  int               InputIsOK(fem_float val, int row, int col);
  int               BoundsOK(int row, int col);

  // common data
  int         m_matdim;     /* square matrix dimension      */
  SPRformat   m_matformat;  /* sparse matrix storage format */
  OclStrategy m_optimizationstrat;  /* sparse matrix storage format */
  bool        m_verboseerrors;
};

#endif
