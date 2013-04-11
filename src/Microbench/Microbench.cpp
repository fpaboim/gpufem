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

#include "Microbench/Microbench.h"

#include <cstdio>
#include <cstdlib>
#include "stdlib.h"
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#include <omp.h>


#include "Utils/util.h"
#include "Utils/fileIO.h"
#include "FEM/fem.h"
#include "FEM/FemData.h"
#include "FEM/femColor.h"
#include "LAops/LAops.h"
#include "LAops/SprSolver.h"
#include "SPRmatrix/SPRmatrix.h"
#include "OpenCL/OCLwrapper.h"

// BenchSearch: Simple Microbenchmark for comparing SSE Linear and Binary
// Searches
////////////////////////////////////////////////////////////////////////////////
void Microbench::BenchSearch() {
  int maxtestsize = 4096;
  int ntestloops = 128;

  SPRmatrix* dummymatrix = SPRmatrix::CreateMatrix(1, SPRmatrix::DEN);

  for (int testarraysz = 16; testarraysz <= maxtestsize; testarraysz *= 2) {
    printf("array size: %i\n", testarraysz);
    // Testarray declared, defined and filled
    int* testarray = (int*)_aligned_malloc(testarraysz * sizeof(int), 16);
    for (int i = 0; i < testarraysz; i++) {
      testarray[i] = i;
    }
    // Auxiliary vars for performance measuring and debugging
    double t1, t2;
    int pos1, pos2, pos3;
    // Normal binary search
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      pos1 = dummymatrix->BinSearchInt(testarray, testarraysz/4, testarraysz);
      pos2 = dummymatrix->BinSearchInt(testarray, testarraysz/2, testarraysz);
      pos3 = dummymatrix->BinSearchInt(testarray, (3*testarraysz)/4, testarraysz);
    }
    t2 = omp_get_wtime() - t1;
    printf("Binsearch avg time: %.4fms -- poscheck(%i,%i,%i)\n", (t2*1000),
      pos1, pos2, pos3);
    // Binary search with Cmov
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      pos1 = dummymatrix->BinSearchInt2(testarray, testarraysz/4, testarraysz);
      pos2 = dummymatrix->BinSearchInt2(testarray, testarraysz/2, testarraysz);
      pos3 = dummymatrix->BinSearchInt2(testarray, (3*testarraysz)/4, testarraysz);
    }
    t2 = omp_get_wtime() - t1;
    printf("BinSearchInt2 avg time: %.4fms -- poscheck(%i,%i,%i)\n", (t2*1000),
      pos1, pos2, pos3);
    // Traditional linear search
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      pos1 = dummymatrix->LinSearchI(testarray, testarraysz/4, testarraysz);
      pos2 = dummymatrix->LinSearchI(testarray, testarraysz/2, testarraysz);
      pos3 = dummymatrix->LinSearchI(testarray, (3*testarraysz)/4, testarraysz);
    }
    t2 = omp_get_wtime() - t1;
    printf("LinSearchI avg time: %.4fms -- poscheck(%i,%i,%i)\n", (t2*1000),
      pos1, pos2, pos3);
    // Linear search with SSE
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      pos1 = dummymatrix->LinSearchISSE(testarray, testarraysz/4, testarraysz);
      pos2 = dummymatrix->LinSearchISSE(testarray, testarraysz/2, testarraysz);
      pos3 = dummymatrix->LinSearchISSE(testarray, (3*testarraysz)/4, testarraysz);
    }
    t2 = omp_get_wtime() - t1;
    printf("LinSearchISSE avg time: %.4fms -- poscheck(%i,%i,%i)\n", (t2*1000),
      pos1, pos2, pos3);
    // Linear search with SSE (second implementation)
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      pos1 = dummymatrix->LinSearchISSE2(testarray, testarraysz/4, testarraysz);
      pos2 = dummymatrix->LinSearchISSE2(testarray, testarraysz/2, testarraysz);
      pos3 = dummymatrix->LinSearchISSE2(testarray, (3*testarraysz)/4, testarraysz);
    }
    t2 = omp_get_wtime() - t1;
    printf("LinSearchISSE2 avg time: %.4fms -- poscheck(%i,%i,%i)\n\n", (t2*1000),
      pos1, pos2, pos3);

    _aligned_free(testarray);
  }
}

// BenchCG: Benchmark for testing performance of CG solver
////////////////////////////////////////////////////////////////////////////////
void Microbench::BenchCG() {
  int ntestloops = 2;
  int localsize  = 8;
  int iterations = 10000;
  fem_float precision  = 0.0001f;

  SPRmatrix::SPRformat   matformat = SPRmatrix::ELL;
  SPRmatrix::OclStrategy oclstrat  = SPRmatrix::BLOCK;

  // Builds kernel and loads source so that dynamic loading does not interfere
  // in benchmarking
  OCL.setDir(".\\..\\src\\OpenCL\\clKernels\\");
  OCL.loadSource("LAopsEll.cl");
  OCL.loadKernel("SpMVNaive");
  OCL.loadKernel("SpMVNaiveUR");
  OCL.loadKernel("SpMVStag");
  OCL.loadKernel("SpMVCoal");
  OCL.loadKernel("SpMVCoalUR");

  std::vector<std::string> testfiles;
  // testfiles.push_back(
  //   "C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//test_models//Q4//placa96_Q4.nf");
  // testfiles.push_back(
  //   "C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//test_models//Q8//placa64_Q8.nf");
  testfiles.push_back(
    "C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//test_models//Brk8//rubik8_brk8.nf");
  testfiles.push_back(
    "C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//test_models//Brk8//rubik12_brk8.nf");
  testfiles.push_back(
    "C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//test_models//Brk8//rubik16_brk8.nf");
  testfiles.push_back(
    "C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//test_models//Brk20//rubik12_brk20.nf");

  for (size_t i = 0; i < testfiles.size(); i++) {
    // Finite Element Test Case
    bool usecolor = true;

    // Reads Files and Sets Output File
    FileIO* fileIO = new FileIO();
    int err = fileIO->ReadNF(testfiles[i]);
    if (err == FEM_ERR) {
      return;
    }
    // Creates new FEM analysis object and initializes it
    FEM* FEM_test = new FEM();
    FEM_test->Init(true, FEM::GPUOMP);
    // Sets-up finite element info
    FemData* femdata = FEM_test->GetFemData();
    femdata->Init(matformat,
                  2,
                  10,
                  0.25,
                  fileIO);
    femdata->GetStiffnessMatrix()->SetOclStrategy(oclstrat);
    printf("\n# MATRIX SIZE: %.2fMB\n",
      (float) femdata->GetStiffnessMatrix()->GetMatSize() / (1024 * 1024));

    // Preprocessing node coloring
    if (usecolor) {
      femColor* mshColorObj = new femColor();
      double t1 = omp_get_wtime();
      mshColorObj->makeMetisGraph(femdata, false);
      double t2 = omp_get_wtime();
      mshColorObj->MakeGreedyColoring(femdata);
      delete(mshColorObj);
    }

    // Gets global stiffness matrix using selected device
    FEM_test->SetUseColoring(usecolor);
    double tstiff = FEM_test->CalcStiffnessMat();
    FEM_test->ApplyConstraint(FEM::PEN,
                              fileIO->getNumSupports(),
                              fileIO->getNodeSupports());

    // Solves linear system for displacements
    // CPU CG Bench
    omp_set_num_threads(4);
    double tsolvecpu = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      CPU_CG(femdata->GetStiffnessMatrix(),
             femdata->GetDisplVector(),
             femdata->GetForceVector(),
             femdata->GetNumDof(),
             iterations,
             precision,
             false);
    }
    tsolvecpu = (omp_get_wtime() - tsolvecpu)/ntestloops;
    double tsolvegpu1 = BenchCGGPU(femdata, localsize, ntestloops);
    double tsolvegpu2 = BenchCGGPU(femdata, localsize*2, ntestloops);
    double tsolvegpu3 = BenchCGGPU(femdata, localsize*4, ntestloops);
    double tsolvegpu4 = BenchCGGPU(femdata, localsize*8, ntestloops);

    printf("Solver Time CPU: %3.4f -> %.2f%% of Stiffness Time\n",
           tsolvecpu, (tsolvecpu/tstiff)*100);
    printf("Solver Time GPU(%i): %3.4f -> %.0f%% of Stiffness Time -> %.0f%% of CPU Time\n",
           localsize  , tsolvegpu1, (tsolvegpu1/tstiff)*100, (tsolvegpu1/tsolvecpu)*100);
    printf("Solver Time GPU(%i): %3.4f -> %.0f%% of Stiffness Time -> %.0f%% of CPU Time\n",
           localsize*2, tsolvegpu2, (tsolvegpu2/tstiff)*100, (tsolvegpu2/tsolvecpu)*100);
    printf("Solver Time GPU(%i): %3.4f -> %.0f%% of Stiffness Time -> %.0f%% of CPU Time\n",
           localsize*4, tsolvegpu3, (tsolvegpu3/tstiff)*100, (tsolvegpu3/tsolvecpu)*100);
    printf("Solver Time GPU(%i): %3.4f -> %.0f%% of Stiffness Time -> %.0f%% of CPU Time\n",
           localsize*8, tsolvegpu4, (tsolvegpu4/tstiff)*100, (tsolvegpu4/tsolvecpu)*100);
  }

}
// BenchCG: Benchmark for testing performance of CG solver
////////////////////////////////////////////////////////////////////////////////
void Microbench::BenchCG2() {
  int initestsize = 4 * 1024;
  int maxtestsize = 16 * 1024;
  int ntestloops = 2;
  int localsize = 4;
  SPRmatrix::SPRformat matformat = SPRmatrix::ELL;

  // Builds kernel and loads source so that dynamic loading does not interfere
  // in benchmarking
  OCL.setDir(".\\..\\src\\OpenCL\\clKernels\\");
  OCL.loadSource("LAopsEll.cl");
  OCL.loadKernel("SpMVNaive");
  OCL.loadKernel("SpMVNaiveUR");
  OCL.loadKernel("SpMVStag");
  OCL.loadKernel("SpMVCoal");
  OCL.loadKernel("SpMVCoalUR");

  for (int testsize = initestsize; testsize <= maxtestsize; testsize *= 2) {
    printf("-----------------\nCG TestSize: %i\n", testsize);
    SPRmatrix* dummymatrix = SPRmatrix::CreateMatrix(testsize, matformat);
    // Test array declared, defined and filled
    fem_float* xvec = (fem_float*)malloc(testsize * sizeof(fem_float));
    fem_float* yvec = (fem_float*)malloc(testsize * sizeof(fem_float));
    for (int i = 0; i < testsize; i++) {
      int j = i + 1;
      yvec[i] = (float)j;
      dummymatrix->AddElem(i, i, (float)j);
    }

    // Print Results
    double t1, t2; // aux vars for getting timestamps

    // CPU CG Microbench
    ////////////////////////////////////////////////////////////////////////////
    omp_set_num_threads(1);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      CPU_CG(dummymatrix, xvec, yvec, testsize, 1000, 0.00001f, false);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));
    // CPU CG Microbench
    omp_set_num_threads(2);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      CPU_CG(dummymatrix, xvec, yvec, testsize, 1000, 0.00001f, false);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));
    // CPU CG Microbench
    omp_set_num_threads(4);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      CPU_CG(dummymatrix, xvec, yvec, testsize, 1000, 0.00001f, false);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));

    // GPU CG Microbench
    ////////////////////////////////////////////////////////////////////////////
    omp_set_num_threads(4);
    BenchCGGPU(dummymatrix, xvec, yvec, localsize, ntestloops);
    BenchCGGPU(dummymatrix, xvec, yvec, localsize*2, ntestloops);
    BenchCGGPU(dummymatrix, xvec, yvec, localsize*4, ntestloops);
    BenchCGGPU(dummymatrix, xvec, yvec, localsize*8, ntestloops);

    free(yvec);
    free(xvec);
    delete(dummymatrix);
  }

  // Finite Element Test Case
  std::string testfile =
    "C://Users//fpaboim//Desktop//parallel_projects//GPU_FEM//fpaboim_gpufem//test_models//Brk8//rubik12_brk8.nf";
  bool usecolor = true;

  // Reads Files and Sets Output File
  FileIO* fileIO = new FileIO();
  int err = fileIO->ReadNF(testfile);
  if (err == FEM_ERR) {
    return;
  }
  // Creates new FEM analysis object and initializes it
  FEM* FEM_test = new FEM();
  FEM_test->Init(true, FEM::GPUOMP);
  // Sets-up finite element info
  FemData* femdata = FEM_test->GetFemData();
  femdata->Init(matformat,
                2,
                10,
                0.25,
                fileIO);
  printf("\n# MATRIX SIZE: %.2fMB\n",
    (float) femdata->GetStiffnessMatrix()->GetMatSize() / (1024 * 1024));
  // Preprocessing node coloring
  if (usecolor) {
    femColor* mshColorObj = new femColor();
    double t1 = omp_get_wtime();
    mshColorObj->makeMetisGraph(femdata, false);
    double t2 = omp_get_wtime();
    std::cout << "Connect Graph Build Time: " << (t2-t1) << std::endl;
    mshColorObj->MakeGreedyColoring(femdata);
    delete(mshColorObj);
  }

  // Gets global stiffness matrix using selected device
  FEM_test->SetUseColoring(usecolor);
  double tstiff = FEM_test->CalcStiffnessMat();

  printf("..Applying Constraints:\n");
  FEM_test->ApplyConstraint(FEM::PEN,
                            fileIO->getNumSupports(),
                            fileIO->getNodeSupports());

  // Solves linear system for displacements
  printf("..Solving for displacements:\n");
  double tsolvecpu = omp_get_wtime();
  CPU_CG(femdata->GetStiffnessMatrix(),
         femdata->GetDisplVector(),
         femdata->GetForceVector(),
         femdata->GetNumDof(),
         3000,
         0.001f,
         false);
  double tsolvegpu = BenchCGGPU(femdata, localsize, ntestloops);

  printf("Solver Time CPU: %3.4f -> %.2f%% of Stiffness Time\n",
         tsolvecpu, (tsolvecpu/tstiff)*100);
  printf("Solver Time GPU: %3.4f -> %.2f%% of Stiffness Time\n",
         tsolvegpu, (tsolvegpu/tstiff)*100);
}

// BenchMV: Microbenchmark for testing performance of MV product
////////////////////////////////////////////////////////////////////////////////
void Microbench::BenchMV() {
  int initestsize = 16 * 1024;
  int maxtestsize = 64 * 1024;
  int ntestloops  = 4;
  int bandsize    = 64;
  SPRmatrix::SPRformat matformat = SPRmatrix::ELL;

  // Builds kernel and loads source so that dynamic loading does not interfere
  // in benchmarking
  OCL.setDir(".\\..\\src\\OpenCL\\clKernels\\");
  // Preloads all kernels and source not to interfere in benchmark time
  OCL.loadSource("LAopsEll.cl");
  OCL.loadKernel("SpMVNaive");
  OCL.loadKernel("SpMVNaiveUR");
  OCL.loadKernel("SpMVStag");
  OCL.loadKernel("SpMVCoal");
  OCL.loadKernel("SpMVCoalUR");

  for (int testsize = initestsize; testsize <= maxtestsize; testsize *= 2) {
    int localsize = 16;
    double t1, t2; // aux vars for getting timestamps
    SPRmatrix* dummymatrix = SPRmatrix::CreateMatrix(testsize, matformat);
    dummymatrix->VerboseErrors(false);
    // test arrays setup
    fem_float* xvec = (fem_float*)malloc(testsize * sizeof(fem_float));
    fem_float* yvec = (fem_float*)malloc(testsize * sizeof(fem_float));
    for (int i = 0; i < testsize; i++) {
      int val = i + 1;
      xvec[i] = (float)val;
      for (int j = 0; j < bandsize; j++)
        dummymatrix->AddElem(i, i + j, 1.0f);
    }

    printf("\n==================================================\n");
    printf("MV microbench testsize: %i\n", testsize);

    // CPU MV Microbench
    ////////////////////////////////////////////////////////////////////////////
    printf("==================================================\n");

    omp_set_num_threads(1);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      dummymatrix->Ax_y(xvec, yvec);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));
    // CPU CG Microbench
    omp_set_num_threads(2);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      dummymatrix->Ax_y(xvec, yvec);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU  time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));
    // CPU CG Microbench
    omp_set_num_threads(4);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      dummymatrix->Ax_y(xvec, yvec);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));
    double gflops = (2.0f * testsize * (double)bandsize * 1.0e-6) / (t2 / ntestloops);
    printf("CPU MFLOP(%i): %.4f Mflops \n", omp_get_max_threads(), gflops);

    // GPU CG Microbench
    ////////////////////////////////////////////////////////////////////////////
    printf("--------------------------------------------------\n");
    BenchAxyGPU(dummymatrix, xvec, yvec, localsize, ntestloops);
    localsize *= 2;
    BenchAxyGPU(dummymatrix, xvec, yvec, localsize, ntestloops);
    localsize *= 2;
    double tgpu = BenchAxyGPU(dummymatrix, xvec, yvec, localsize, ntestloops);
    gflops = (2.0f * testsize * (double)bandsize * 1.0e-6) / tgpu;
    printf("GPU MFLOP: %.4f Mflops \n", gflops);

    free(yvec);
    free(xvec);
    delete(dummymatrix);
  }
}

// BenchCG: Benchmark for testing performance of CG solver
////////////////////////////////////////////////////////////////////////////////
double Microbench::BenchAxyGPU(SPRmatrix* dummymatrix,
                               fem_float* xvec,
                               fem_float* yvec,
                               size_t     localsize,
                               int        nloops) {
  // Naive Bench
  double tini, tnaive, tnaiveur, tshare, tblock, tblockur;
  dummymatrix->SetOclStrategy(SPRmatrix::NAIVE);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->AxyGPU(xvec, yvec, localsize);
  }
  tnaive = omp_get_wtime() - tini;
  // NaiveUR Bench
  dummymatrix->SetOclStrategy(SPRmatrix::NAIVEUR);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->AxyGPU(xvec, yvec, localsize);
  }
  tnaiveur = omp_get_wtime() - tini;
  // Shared Bench
  dummymatrix->SetOclStrategy(SPRmatrix::SHARE);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->AxyGPU(xvec, yvec, localsize);
  }
  tshare = omp_get_wtime() - tini;
  // Blocked Bench
  dummymatrix->SetOclStrategy(SPRmatrix::BLOCK);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->AxyGPU(xvec, yvec, localsize);
  }
  tblock = omp_get_wtime() - tini;
  // Blocked Bench
  dummymatrix->SetOclStrategy(SPRmatrix::BLOCKUR);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->AxyGPU(xvec, yvec, localsize);
  }
  tblockur = omp_get_wtime() - tini;

  printf("GPU time(%2i): N:%.0fms NUR:%.0fms S:%.0fms B:%.0fms BUR:%.0fms\n",
          localsize,
          (tnaive*1000),
          (tnaiveur*1000),
          (tshare*1000),
          (tblock*1000),
          (tblockur*1000));

  return tblock/nloops;
}

// BenchCG: Benchmark for testing performance of CG solver
////////////////////////////////////////////////////////////////////////////////
double Microbench::BenchCGGPU(SPRmatrix* dummymatrix,
                              fem_float* xvec,
                              fem_float* yvec,
                              size_t     localsize,
                              int        nloops) {
  // Naive Bench
  int       iterations = 10000;
  fem_float precision = 0.0001f;

  double tini, tnaive, tnaiveur, tshare, tblock, tblockur;
  dummymatrix->SetOclStrategy(SPRmatrix::NAIVE);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->SolveCgGpu(xvec, yvec, iterations, precision, localsize);
  }
  tnaive = omp_get_wtime() - tini;
  // NaiveUR Bench
  dummymatrix->SetOclStrategy(SPRmatrix::NAIVEUR);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->SolveCgGpu(xvec, yvec, iterations, precision, localsize);
  }
  tnaiveur = omp_get_wtime() - tini;
  // Shared Bench
  dummymatrix->SetOclStrategy(SPRmatrix::SHARE);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->SolveCgGpu(xvec, yvec, iterations, precision, localsize);
  }
  tshare = omp_get_wtime() - tini;
  // Blocked Bench
  dummymatrix->SetOclStrategy(SPRmatrix::BLOCK);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->SolveCgGpu(xvec, yvec, iterations, precision, localsize);
  }
  tblock = omp_get_wtime() - tini;
  // Blocked Bench
  dummymatrix->SetOclStrategy(SPRmatrix::BLOCKUR);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    dummymatrix->SolveCgGpu(xvec, yvec, iterations, precision, localsize);
  }
  tblockur = omp_get_wtime() - tini;

  printf("CGGPU time(%2i): N:%.0fms NUR:%.0fms S:%.0fms B:%.0fms BUR:%.0fms\n",
          localsize,
          (tnaive*1000),
          (tnaiveur*1000),
          (tshare*1000),
          (tblock*1000),
          (tblockur*1000));

  return tblock/nloops;
}

// BenchCG: Benchmark for testing performance of CG solver
////////////////////////////////////////////////////////////////////////////////
double Microbench::BenchCGGPU(FemData* femdata,
                              size_t   localsize,
                              int      nloops) {
  SPRmatrix* matrix = femdata->GetStiffnessMatrix();
  fem_float* xvect = femdata->GetDisplVector();
  fem_float* yvect = femdata->GetForceVector();
  // Naive Bench
  double tini, tnaive, tnaiveur, tshare, tblock, tblockur, ttest;
  matrix->SetOclStrategy(SPRmatrix::NAIVE);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    matrix->SolveCgGpu(xvect, yvect, 1000, 0.00001f, localsize);
  }
  tnaive = omp_get_wtime() - tini;
  // NaiveUR Bench
  matrix->SetOclStrategy(SPRmatrix::NAIVEUR);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    matrix->SolveCgGpu(xvect, yvect, 1000, 0.00001f, localsize);
  }
  tnaiveur = omp_get_wtime() - tini;
  // Shared Bench
  matrix->SetOclStrategy(SPRmatrix::SHARE);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    matrix->SolveCgGpu(xvect, yvect, 1000, 0.00001f, localsize);
  }
  tshare = omp_get_wtime() - tini;
  // Blocked Bench
  matrix->SetOclStrategy(SPRmatrix::BLOCK);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    matrix->SolveCgGpu(xvect, yvect, 1000, 0.00001f, localsize);
  }
  tblock = omp_get_wtime() - tini;
  // Blocked Bench
  matrix->SetOclStrategy(SPRmatrix::BLOCKUR);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    matrix->SolveCgGpu(xvect, yvect, 1000, 0.00001f, localsize);
  }
  tblockur = omp_get_wtime() - tini;
  // Blocked Bench
  matrix->SetOclStrategy(SPRmatrix::TEST);
  tini = omp_get_wtime();
  for (int i = 0; i <= nloops; i++) {
    matrix->SolveCgGpu(xvect, yvect, 1000, 0.00001f, localsize);
  }
  ttest = omp_get_wtime() - tini;

  printf("CGGPU time(%2i): N:%.0fms NUR:%.0fms S:%.0fms B:%.0fms BUR:%.0fms TST:%.0fms\n",
          localsize,
          (tnaive*1000),
          (tnaiveur*1000),
          (tshare*1000),
          (tblock*1000),
          (tblockur*1000),
          (ttest*1000));

  return __min(tnaive,__min(tnaiveur, __min(tblock, tblockur)))/nloops;
}
