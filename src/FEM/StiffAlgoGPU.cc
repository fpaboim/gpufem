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

#include "FEM/StiffAlgoGPU.h"

#include <omp.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include <assert.h>
#include <cstdio>
#include <cstdlib>

#include "FEM/fem.h"
#include "FEM/FemData.h"
#include "LAops/LAops.h"
#include "OpenCL/OCLwrapper.h"
#include "Utils/util.h"
#include "Utils/fileIO.h"
#include "SPRmatrix/SPRmatrix.h"

////////////////////////////////////////////////////////////////////////////////
// Constructor Sets Basic Information to Make Code Less Error Prone
////////////////////////////////////////////////////////////////////////////////
StiffAlgoGPU::StiffAlgoGPU(FemData* femdata) {
  m_femdata = femdata;
}

/******************************************************************************/
//                            CPU FEM Operations                              //
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Calculates the Global Sparse Stiffness Matrix K
////////////////////////////////////////////////////////////////////////////////
double StiffAlgoGPU::CalcGlobalStiffness() {
  double start_time = omp_get_wtime();  // TIMESTAMP

  int modeldim                     = m_femdata->GetModelDim();
  int numdof                       = m_femdata->GetNumDof();
  int nelemdof                     = m_femdata->GetElemDof();
  int numelem                      = m_femdata->GetNumElem();
  int numelemnodes                 = m_femdata->GetNumElemNodes();
  int* elemconnect                 = m_femdata->GetElemConnect();
  fem_float* nodecoords            = m_femdata->GetNodeCoords();
  int ngpts                        = m_femdata->GetNumGaussPts();
  int nloopgpts                    = m_femdata->GetNumGaussLoopPts();
  fem_float* gausspts_vec          = m_femdata->GetGaussPtsVecGPU();
  fem_float* gaussweights_vec      = m_femdata->GetGaussWeightVec();
  fem_float elasticmod             = m_femdata->GetElasticModulus();
  fem_float poissoncoef            = m_femdata->GetPoissonCoef();
  SPRmatrix* stiffmat              = m_femdata->GetStiffnessMatrix();
  SPRmatrix::SPRformat sprseformat = m_femdata->GetSparseFormat();
  const ivecvec colorelem          = m_femdata->GetColorVector();

  double alloc_time = omp_get_wtime();  // TIMESTAMP

  // Checks global stiffness matrix for allocation or clears it
  if (!stiffmat)
    stiffmat = SPRmatrix::CreateMatrix(numdof, sprseformat);
  else
    stiffmat->Clear();

  double opencl_time = omp_get_wtime();  // TIMESTAMP
  // OpenCL used Variables
  size_t NodeCoorBufSz, ElemConBufSz, GlblKDataBufSz;//, GlblKColIdxBufSz, ColorVecBufSz;
  cl_mem GlbK_mem, ElmCon_mem, NodeCor_mem, Xgauss_mem, Wgauss_mem;

  // Program and Kernel Creation
  loadKernelAndProgram(modeldim, numelemnodes);

  double perfstart_time = omp_get_wtime();  // TIMESTAMP

  fem_float* global_Kaux =
    allocVector(nelemdof * nelemdof * numelem, true);

  double memalloc_time = omp_get_wtime();  // TIMESTAMP

  // Allocate memory on the device to hold our data and store the results into
  GlblKDataBufSz = sizeof(fem_float) * nelemdof * nelemdof * numelem;
  ElemConBufSz   = sizeof(int) * numelemnodes * numelem;
  NodeCoorBufSz  = sizeof(fem_float) * numdof;

  printf("    Glbl_Kaux Buffersize: %iMB\n", GlblKDataBufSz/(1024*1024));
  printf("    ElemConn  Buffersize: %iMB\n", ElemConBufSz/(1024*1024));
  printf("    NodeCoord Buffersize: %iMB\n", NodeCoorBufSz/(1024*1024));

  // Input array matrix_A
  Xgauss_mem  = OCL.createBuffer(sizeof(fem_float) * nloopgpts * modeldim,
                                 CL_MEM_READ_ONLY);
  Wgauss_mem  = OCL.createBuffer(sizeof(fem_float) * nloopgpts,
                                 CL_MEM_READ_ONLY);
  ElmCon_mem  = OCL.createBuffer(ElemConBufSz, CL_MEM_READ_ONLY);
  NodeCor_mem = OCL.createBuffer(NodeCoorBufSz, CL_MEM_READ_ONLY);
  GlbK_mem    = OCL.createBuffer(GlblKDataBufSz, CL_MEM_WRITE_ONLY);

  // Enqueue Buffers For Execution
  OCL.enqueueWriteBuffer(Xgauss_mem, sizeof(fem_float) * nloopgpts * modeldim,
                         (void*)gausspts_vec, CL_FALSE);
  OCL.enqueueWriteBuffer(Wgauss_mem, sizeof(fem_float) * nloopgpts,
                         (void*)gaussweights_vec, CL_FALSE);
  OCL.enqueueWriteBuffer(ElmCon_mem, ElemConBufSz,
                         (void*)elemconnect, CL_FALSE);
  OCL.enqueueWriteBuffer(NodeCor_mem, NodeCoorBufSz,
                         (void*)nodecoords, CL_FALSE);

  // Get all of the stuff written and allocated
  OCL.finish();

  double writebuff_time = omp_get_wtime();  // TIMESTAMP

  // Set Kernel Arguments
  int i = 0;
  OCL.setKernelArg(i, sizeof(fem_float),(void*)&elasticmod); i++;
  OCL.setKernelArg(i, sizeof(fem_float),(void*)&poissoncoef); i++;
  OCL.setKernelArg(i, sizeof(int),      (void*)&ngpts); i++;
  OCL.setKernelArg(i, sizeof(cl_mem),   (void*)&Xgauss_mem); i++;
  OCL.setKernelArg(i, sizeof(cl_mem),   (void*)&Wgauss_mem); i++;
  OCL.setKernelArg(i, sizeof(cl_mem),   (void*)&NodeCor_mem); i++;
  OCL.setKernelArg(i, sizeof(cl_mem),   (void*)&ElmCon_mem); i++;
  OCL.setKernelArg(i, sizeof(cl_mem),   (void*)&GlbK_mem); i++;

  double kernel_time = omp_get_wtime();  // TIMESTAMP

  // Sets Dimensions and Executes Kernel
  OCL.setGlobalWorksize(0, numelem*numelemnodes);
  OCL.setLocalWorksize(0, numelemnodes);
  OCL.enquequeNDRangeKernel(1, true);
  // Read results back to globalK array
  OCL.enqueueReadBuffer(GlbK_mem, GlblKDataBufSz, global_Kaux, CL_TRUE);

  double enqueue_time = omp_get_wtime();  // TIMESTAMP
  double end_time = omp_get_wtime();  // TIMESTAMP
  double totaltime = end_time - perfstart_time;
  // Prints OpenCL Performance Timings
  printf("+Total GPU Time:%.3fms (x- marks included times)\n", totaltime);
  printf("    o-Data Read Time:%.3fms\n", alloc_time-start_time);
  printf("    o-Sparse Matrix Creation/Clearing Time:%.3fms\n",
         opencl_time-alloc_time);
  printf("    o-Kernel Creation Time:%.3fms\n", perfstart_time-opencl_time);
  printf("    x-Kaux Allocation Time:%.3fms\n", memalloc_time-perfstart_time);
  printf("    x-WriteBuffer Alloc/Enqueue Time + GaussPts:%.3fms\n",
         writebuff_time-memalloc_time);
  printf("    x-Kernel Set Time:%.3fms\n", kernel_time-writebuff_time);
  printf("    x-Enqueue Time:%.3fms\n", enqueue_time-kernel_time);
  printf("    x-Kernel Execution Time:%.3fms\n", enqueue_time-perfstart_time);
  printf("    x-Assembly Time:%.3fms\n", end_time-enqueue_time);

  free(global_Kaux);

  // Teardown
  OCL.releaseMem(Xgauss_mem);
  OCL.releaseMem(Wgauss_mem);
  OCL.releaseMem(GlbK_mem);
  OCL.releaseMem(ElmCon_mem);
  OCL.releaseMem(NodeCor_mem);

  return (totaltime);
}

////////////////////////////////////////////////////////////////////////////////
// Loads OpenCL kernel and program
////////////////////////////////////////////////////////////////////////////////
void StiffAlgoGPU::loadKernelAndProgram(int modeldim, int numelemnodes) {
  const char* filename;
  const char* kernelname;
  if (modeldim == 2) {
    if (numelemnodes == 4) {
      filename   = "gpuFEM_Q4.cl";
      kernelname = "getStiffnessQ4";
    }
    if (numelemnodes == 8) {
      filename   = "gpuFEM_Q8.cl";
      kernelname = "getStiffnessQ8";
    }
  } else {
    if (numelemnodes == 8) {
      filename   = "gpuFEM_UR8.cl";
      kernelname = "getStiffnessUR8";
    }
    if (numelemnodes == 20) {
      filename   = "gpuFEM_UR20.cl";
      kernelname = "getStiffnessUR20";
    }
  }
  OCL.setDir(".\\..\\src\\OpenCL\\clKernels\\");
  #ifdef GTEST_INCLUDE_GTEST_GTEST_H_
  OCL.setDir(".\\..\\..\\src\\OpenCL\\clKernels\\");
  #endif

  OCL.loadSource(filename);
  OCL.loadKernel(kernelname);
}
