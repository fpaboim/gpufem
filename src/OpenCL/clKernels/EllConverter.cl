#define TILE_SIZEx 4
#define TILE_SIZEy 16

#include "EllUtils.h"

////////////////////////////////////////////////////////////////////////////////
//  SparseMV: Ellpack version has matrix data structure with matData matrix
//  arrays scanned column-wise and colIdx vector with corresponding cols for
//  each data entry. ELLwidth is number of columns saved (usually more than
//  used to mainain 2^n scalability. vector_d is input vector and vector_q
//  is CG output vector, result of MV operation. auxShare is a local vector
//  of the same size as ELLwidth used for parallel optimizations
////////////////////////////////////////////////////////////////////////////////

// SpMVCoal: Coalesced version of SpMV kernel (using blocking)
///////////////////////////////////////////////////////////////////////////////
__kernel void SpMVCoal(__global  float* matData,     // INPUT MATRIX DATA
                       __global  float* global_Kaux,
                       __global  int*   colIdx,
                       __global  int*   rowNnz,
                       __private int    ELLwidth,
                       __private int    matDim,
                       __global  float* vector_x,    // INPUT
                       __global  float* vector_y,    // OUTPUT
                       __local   float* auxShared) { // LOCAL SHARED BUFFER
  //uint grpidy  = get_group_id(1)
}

void AssembleGPUColoring(int elemdofs,
                         int nelem,
                         int modeldim,
                         int nelemnodes,
                         int* elemconnect,
                         SPRmatrix* stiffmat,
                         fem_float* auxstiffmat,
                         __global  float* auxstiffmat,
                         const ivecvec colorelem) {
    // Loops Over Elements to Perform Serial Assembly
    int rowStride  = elemdofs * nelem;
    int elemStride;
    int gblDOFi, gblDOFj;
    size_t nColors = colorelem.size();
    if(modeldim == 2) {
      for (size_t color = 0; color < nColors; ++color) {
        int nColorElems = (int)colorelem[color].size();
        // Loops over current color doing stiffness calculation in parallel
#pragma omp parallel for private(gblDOFi, gblDOFj, elemStride)
        for (int elemPos = 0; elemPos < nColorElems; ++elemPos) {
          int elem = colorelem[color][elemPos];
          elemStride = elem * elemdofs;
          for (int i = 0; i < nelemnodes; ++i) {
            gblDOFi = (elemconnect[nelemnodes*elem+i]-1)*2;
            for (int j = 0; j < nelemnodes; ++j) {
              gblDOFj = (elemconnect[nelemnodes*elem+j]-1)*2;
              if (stiffmat->GetAllocTrigger())
              {
                #pragma omp critical (memalloc)
                {
                AssembleMatrix2D(stiffmat, gblDOFi, gblDOFj, auxstiffmat, elemStride, rowStride, i, j);
                }
              } else {
                AssembleMatrix2D(stiffmat, gblDOFi, gblDOFj, auxstiffmat, elemStride, rowStride, i, j);
              }
            }
          }
        }
      }
    } else {
      for (size_t color = 0; color < nColors; ++color) {
        int nColorElems = (int)colorelem[color].size();
        // Loops over current color doing stiffness calculation in parallel
#pragma omp parallel for private(gblDOFi, gblDOFj, elemStride)
        for (int elemPos = 0; elemPos < nColorElems; ++elemPos) {
          int elem = colorelem[color][elemPos];
          elemStride = elem * elemdofs;
          for (int i = 0; i < nelemnodes; ++i) {
            gblDOFi = (elemconnect[nelemnodes*elem+i]-1)*3;
            for (int j = 0; j < nelemnodes; ++j) {
              gblDOFj = (elemconnect[nelemnodes*elem+j]-1)*3;
              if (stiffmat->GetAllocTrigger())
              {
                #pragma omp critical (memalloc)
                {
                AssembleMatrix3D(stiffmat, gblDOFi, gblDOFj, auxstiffmat, elemStride, rowStride, i, j);
                }
              } else {
                AssembleMatrix3D(stiffmat, gblDOFi, gblDOFj, auxstiffmat, elemStride, rowStride, i, j);
              }

            }
          }
        }
      }
    }
}

