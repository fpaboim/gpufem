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
// StiffAlgoGpuOmp.h - GPU based stiffness calculation and assembly by CPU in
// parallel using OpenMP and Graph Coloring
// Author: Francisco Paulo de Aboim (fpaboim@gmail.com)
////////////////////////////////////////////////////////////////////////////////

#ifndef STIFFNESS_ALGO_GPUOMP_H_
#define STIFFNESS_ALGO_GPUOMP_H_

#include "FEM/fem.h"
#include "FEM/FemData.h"
#include "FEM/StiffAlgo.h"
#include "SPRmatrix/SPRmatrix.h"
#include "Utils/util.h"

class StiffAlgoGpuOmp : public StiffAlgo {
public:
  StiffAlgoGpuOmp();
  ~StiffAlgoGpuOmp() {};

  // GPU FEM Operations
  double CalcGlobalStiffness(FemData* femdata);

protected:
  // Performs serial assembly
  inline void AssembleGPUSerial(int elemdofs,
    int nelem,
    int modeldim,
    int nelemnodes,
    int* elemconnect,
    SPRmatrix* stiffmat,
    fem_float* auxstiffmat);
  // Performs parallel assembly by element coloring
  inline void AssembleGPUColoring(int elemdofs,
    int nelem,
    int modeldim,
    int nelemnodes,
    int* elemconnect,
    SPRmatrix* stiffmat,
    fem_float* auxstiffmat,
    const ivecvec colorelem);

  void AssembleMatrix2D( SPRmatrix* stiffmat, int gblDOFi, int gblDOFj, fem_float* auxstiffmat, int elemStride, int rowStride, int i, int j ) 
  {
    stiffmat->AddElem(gblDOFi  , gblDOFj  , auxstiffmat[elemStride + (rowStride*(2*i  )) + (2*j)  ]);
    stiffmat->AddElem(gblDOFi+1, gblDOFj  , auxstiffmat[elemStride + (rowStride*(2*i+1)) + (2*j)  ]);
    stiffmat->AddElem(gblDOFi  , gblDOFj+1, auxstiffmat[elemStride + (rowStride*(2*i  )) + (2*j)+1]);
    stiffmat->AddElem(gblDOFi+1, gblDOFj+1, auxstiffmat[elemStride + (rowStride*(2*i+1)) + (2*j)+1]);
  }
  void AssembleMatrix3D( SPRmatrix* stiffmat, int gblDOFi, int gblDOFj, fem_float* auxstiffmat, int elemStride, int rowStride, int i, int j ) 
  {
    stiffmat->AddElem(gblDOFi  , gblDOFj  , auxstiffmat[elemStride + (rowStride*(3*i  )) + (3*j)  ]);
    stiffmat->AddElem(gblDOFi+1, gblDOFj  , auxstiffmat[elemStride + (rowStride*(3*i+1)) + (3*j)  ]);
    stiffmat->AddElem(gblDOFi+2, gblDOFj  , auxstiffmat[elemStride + (rowStride*(3*i+2)) + (3*j)  ]);
    stiffmat->AddElem(gblDOFi  , gblDOFj+1, auxstiffmat[elemStride + (rowStride*(3*i  )) + (3*j)+1]);
    stiffmat->AddElem(gblDOFi+1, gblDOFj+1, auxstiffmat[elemStride + (rowStride*(3*i+1)) + (3*j)+1]);
    stiffmat->AddElem(gblDOFi+2, gblDOFj+1, auxstiffmat[elemStride + (rowStride*(3*i+2)) + (3*j)+1]);
    stiffmat->AddElem(gblDOFi  , gblDOFj+2, auxstiffmat[elemStride + (rowStride*(3*i  )) + (3*j)+2]);
    stiffmat->AddElem(gblDOFi+1, gblDOFj+2, auxstiffmat[elemStride + (rowStride*(3*i+1)) + (3*j)+2]);
    stiffmat->AddElem(gblDOFi+2, gblDOFj+2, auxstiffmat[elemStride + (rowStride*(3*i+2)) + (3*j)+2]);
  }
private:
  void loadKernelAndProgram(int modeldim, int numelemnodes);
};

#endif  // STIFFNESS_ALGO_GPUOMP_H_
