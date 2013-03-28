////////////////////////////////////////////////////////////////////////////////
// Sparsely Sparse Matrix Library - Second ELLpack Sparse Matrix Implementation
// Author: Francisco Aboim
// TecGraf / PUC-RIO
////////////////////////////////////////////////////////////////////////////////
#ifndef ELLMATRIX2_H_
#define ELLMATRIX2_H_

#include "SPRmatrix/SPRmatrix.h"
#include "Utils/util.h"

// Matrix M Data is stored in format aligned to 16 byte boundary (4 floats or
// multiple thereof). The data is stored scanning columns of the compressed
// matrix matdata:
// M = [4 0 5 0   matdata = [4 5 0 0   colidx = [0 2 * *  rownnz = [2
//      0 3 0 1              3 1 0 0             1 3 * *            2
//      4 0 2 6              4 2 6 0             0 2 3 *            3
//      0 0 0 4]             4 0 0 0]            3 * * *]           1]
// -> Matrices are converted to arrays scanned in row-wise fashion: e.g.:
// matdata = [4 5 0 0 3 1 0 0 4 2 6 0 ... 0] -> padded with zeros
// colidx  = [0 2 * * 1 3 * * 0 2 3 * ... *] -> * is garbage

class ELLmatrix2 : public SPRmatrix {
 public:
  ELLmatrix2(int matdim);
  ~ELLmatrix2();

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

  static int   BinSearchRow(int* intvector, int  val, int  row,
                            int  nrownnz, int  rowlen);
  static int   LinSearchRow(int* intvector, int  val, int  row,
                            int  nrownnz, int  rowlen);

 private:
  void         InsertElem(int rownnz, int pos, const fem_float val,
                          const int col, const int row);
  void         SyncForAllocation();
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
  int        m_maxrowlen;  // maximum row length (number of compressed columns
                           // stored)
  int        m_growthfactor;   // step to grow compressed row length
  bool       m_iskernelbuilt;
};

#endif  // ELLMATRIX2_H_
