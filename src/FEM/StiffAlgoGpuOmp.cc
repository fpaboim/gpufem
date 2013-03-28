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

#include "FEM/StiffAlgoGpuOmp.h"

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

// Constructor Sets Basic Information to Make Code Less Error Prone
////////////////////////////////////////////////////////////////////////////////
StiffAlgoGpuOmp::StiffAlgoGpuOmp() {
}

// Calculates the Global Sparse Stiffness Matrix K
////////////////////////////////////////////////////////////////////////////////
double StiffAlgoGpuOmp::CalcGlobalStiffness(FemData* femdata) {
  bool   verbose    = false;
  double start_time = omp_get_wtime();  // TIMESTAMP

  int modeldim                     = femdata->GetModelDim();
  int numdof                       = femdata->GetNumDof();
  int nelemdof                     = femdata->GetElemDof();
  int numelem                      = femdata->GetNumElem();
  int numelemnodes                 = femdata->GetNumElemNodes();
  int* elemconnect                 = femdata->GetElemConnect();
  fem_float* nodecoords            = femdata->GetNodeCoords();
  int ngpts                        = femdata->GetNumGaussPts();
  int nloopgpts                    = femdata->GetNumGaussLoopPts();
  fem_float* gausspts_vec          = femdata->GetGaussPtsVecGPU();
  fem_float* gaussweights_vec      = femdata->GetGaussWeightVec();
  fem_float elasticmod             = femdata->GetElasticModulus();
  fem_float poissoncoef            = femdata->GetPoissonCoef();
  SPRmatrix* stiffmat              = femdata->GetStiffnessMatrix();
  SPRmatrix::SPRformat sprseformat = femdata->GetSparseFormat();
  const ivecvec colorelem          = femdata->GetColorVector();

  double alloc_time = omp_get_wtime();  // TIMESTAMP

  // Checks global stiffness matrix for allocation or clears it
  if (!stiffmat)
    stiffmat = SPRmatrix::CreateMatrix(numdof, sprseformat);
  else
    stiffmat->Clear();

  double opencl_time = omp_get_wtime();  // TIMESTAMP
  // OpenCL used Variables
  size_t NodeCor_buffer_size, ElmCon_buffer_size, GlbKaux_buffer_size;
  cl_mem GlbK_mem, ElmCon_mem, NodeCor_mem, Xgauss_mem, Wgauss_mem;

  // Program and Kernel Creation
  loadKernelAndProgram(modeldim, numelemnodes);

  double perfstart_time = omp_get_wtime();  // TIMESTAMP

  fem_float* global_Kaux =
    allocVector(nelemdof * nelemdof * numelem, true);

  double memalloc_time = omp_get_wtime();  // TIMESTAMP

  // Allocate memory on the device to hold our data and store the results into
  GlbKaux_buffer_size = sizeof(fem_float) * nelemdof * nelemdof * numelem;
  ElmCon_buffer_size  = sizeof(int) * numelemnodes * numelem;
  NodeCor_buffer_size = sizeof(fem_float) * numdof;

  if (verbose) {
    printf("   Glbl_Kaux Buffersize: %iMB\n", GlbKaux_buffer_size/(1024*1024));
    printf("   ElemConn  Buffersize: %iMB\n", ElmCon_buffer_size/(1024*1024));
    printf("   NodeCoord Buffersize: %iMB\n", NodeCor_buffer_size/(1024*1024));
  }

  // Input array matrix_A
  Xgauss_mem  = OCL.createBuffer(sizeof(fem_float) * nloopgpts * modeldim,
                                 CL_MEM_READ_ONLY);
  Wgauss_mem  = OCL.createBuffer(sizeof(fem_float) * nloopgpts,
                                 CL_MEM_READ_ONLY);
  ElmCon_mem  = OCL.createBuffer(ElmCon_buffer_size, CL_MEM_READ_ONLY);
  NodeCor_mem = OCL.createBuffer(NodeCor_buffer_size, CL_MEM_READ_ONLY);
  GlbK_mem    = OCL.createBuffer(GlbKaux_buffer_size, CL_MEM_WRITE_ONLY);


  // Enqueue Buffers For Execution
  OCL.enqueueWriteBuffer(Xgauss_mem, sizeof(fem_float) * nloopgpts * modeldim,
                         (void*)gausspts_vec, CL_FALSE);
  OCL.enqueueWriteBuffer(Wgauss_mem, sizeof(fem_float) * nloopgpts,
                         (void*)gaussweights_vec, CL_FALSE);
  OCL.enqueueWriteBuffer(ElmCon_mem, ElmCon_buffer_size,
                         (void*)elemconnect, CL_FALSE);
  OCL.enqueueWriteBuffer(NodeCor_mem, NodeCor_buffer_size,
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
  OCL.setGlobalWorksize(0, numelem * numelemnodes);
  OCL.setLocalWorksize(0, numelemnodes);
  OCL.enquequeNDRangeKernel(1, true);
  // Read results back to globalK array
  OCL.enqueueReadBuffer(GlbK_mem, GlbKaux_buffer_size, global_Kaux, CL_TRUE);

  double enqueue_time = omp_get_wtime();  // TIMESTAMP

  if (m_usecoloring) {
    // Parallel assembly by element coloring
    AssembleGPUColoring(nelemdof,
                        numelem,
                        modeldim,
                        numelemnodes,
                        elemconnect,
                        stiffmat,
                        global_Kaux,
                        colorelem);
  } else {
    // Performs serial assembly
    AssembleGPUSerial(nelemdof,
                      numelem,
                      modeldim,
                      numelemnodes,
                      elemconnect,
                      stiffmat,
                      global_Kaux);
  }
  double end_time = omp_get_wtime();  // TIMESTAMP
  double totaltime = end_time - perfstart_time;

  if (true) {
    // Prints OpenCL Performance Timings
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
    printf("------------------------------------------------\n");
    printf("+ Total GPU Time:%.3fms (x- marks included times)\n", totaltime);
  }

  free(global_Kaux);

  // Teardown
  OCL.releaseMem(Xgauss_mem);
  OCL.releaseMem(Wgauss_mem);
  OCL.releaseMem(GlbK_mem);
  OCL.releaseMem(ElmCon_mem);
  OCL.releaseMem(NodeCor_mem);

  return (totaltime);
}

// Performs serial assembly of stiffness matrix
////////////////////////////////////////////////////////////////////////////////
void StiffAlgoGpuOmp::AssembleGPUSerial(int elemdofs,
                                        int nelem,
                                        int modeldim,
                                        int nelemnodes,
                                        int* elemconnect,
                                        SPRmatrix* stiffmat,
                                        fem_float* auxstiffmat) {
  // Loops Over Elements to Perform Serial Assembly
  int rowStride  = elemdofs * nelem;
  int elemStride;
  int gblDOFi, gblDOFj;
  if(modeldim == 2) {
    for (int elem=0; elem < nelem; ++elem) {
      elemStride = elem * elemdofs;
      for (int i=0; i < nelemnodes; ++i) {
        gblDOFi = (elemconnect[nelemnodes*elem+i]-1)*2;
        for (int j=0; j < nelemnodes; ++j) {
          gblDOFj = (elemconnect[nelemnodes*elem+j]-1)*2;
          stiffmat->AddElem(gblDOFi  , gblDOFj  , auxstiffmat[elemStride + ( rowStride*(2*i  ) ) + (2*j)   ]);
          stiffmat->AddElem(gblDOFi+1, gblDOFj  , auxstiffmat[elemStride + ( rowStride*(2*i+1) ) + (2*j)   ]);
          stiffmat->AddElem(gblDOFi  , gblDOFj+1, auxstiffmat[elemStride + ( rowStride*(2*i  ) ) + (2*j)+1 ]);
          stiffmat->AddElem(gblDOFi+1, gblDOFj+1, auxstiffmat[elemStride + ( rowStride*(2*i+1) ) + (2*j)+1 ]);
        }
      }
    }
  } else {
    for (int elem=0; elem < nelem; ++elem) {
      elemStride = elem*elemdofs;
      for (int i=0; i < nelemnodes; ++i) {
        gblDOFi = (elemconnect[nelemnodes*elem+i]-1)*3;
        for (int j=0; j < nelemnodes; ++j) {
          gblDOFj = (elemconnect[nelemnodes*elem+j]-1)*3;
          stiffmat->AddElem(gblDOFi  , gblDOFj  , auxstiffmat[elemStride + ( rowStride*(3*i  ) ) + (3*j)   ]);
          stiffmat->AddElem(gblDOFi+1, gblDOFj  , auxstiffmat[elemStride + ( rowStride*(3*i+1) ) + (3*j)   ]);
          stiffmat->AddElem(gblDOFi+2, gblDOFj  , auxstiffmat[elemStride + ( rowStride*(3*i+2) ) + (3*j)   ]);
          stiffmat->AddElem(gblDOFi  , gblDOFj+1, auxstiffmat[elemStride + ( rowStride*(3*i  ) ) + (3*j)+1 ]);
          stiffmat->AddElem(gblDOFi+1, gblDOFj+1, auxstiffmat[elemStride + ( rowStride*(3*i+1) ) + (3*j)+1 ]);
          stiffmat->AddElem(gblDOFi+2, gblDOFj+1, auxstiffmat[elemStride + ( rowStride*(3*i+2) ) + (3*j)+1 ]);
          stiffmat->AddElem(gblDOFi  , gblDOFj+2, auxstiffmat[elemStride + ( rowStride*(3*i  ) ) + (3*j)+2 ]);
          stiffmat->AddElem(gblDOFi+1, gblDOFj+2, auxstiffmat[elemStride + ( rowStride*(3*i+1) ) + (3*j)+2 ]);
          stiffmat->AddElem(gblDOFi+2, gblDOFj+2, auxstiffmat[elemStride + ( rowStride*(3*i+2) ) + (3*j)+2 ]);
        }
      }
    }
  }
}


// Performs serial assembly of stiffness matrix
////////////////////////////////////////////////////////////////////////////////
void StiffAlgoGpuOmp::AssembleGPUColoring(int elemdofs,
                                          int nelem,
                                          int modeldim,
                                          int nelemnodes,
                                          int* elemconnect,
                                          SPRmatrix* stiffmat,
                                          fem_float* auxstiffmat,
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

// Loads OpenCL kernel and program
////////////////////////////////////////////////////////////////////////////////
void StiffAlgoGpuOmp::loadKernelAndProgram(int modeldim, int numelemnodes) {
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
