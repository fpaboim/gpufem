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
  int iterations = 15000;
  fem_float precision  = 0.001f;
  SPRmatrix::SPRformat   matformat = SPRmatrix::ELL;
  SPRmatrix::OclStrategy oclstrat  = SPRmatrix::STRAT_BLOCK;

  // Builds kernel and loads source so that dynamic loading does not interfere
  // in benchmarking
  preloadEllKernels();

  std::vector<std::string> testfiles;
  // testfiles.push_back(".//..//test_models//Q4//placa96_Q4.nf");
  testfiles.push_back(".//..//test_models//Q8//placa48_Q8.nf");
  //  testfiles.push_back(".//..//test_models//Q8//placa64_Q8.nf");
  testfiles.push_back(".//..//test_models//Brk8//rubik8_brk8.nf");
  testfiles.push_back(".//..//test_models//Brk8//rubik12_brk8.nf");
  testfiles.push_back(".//..//test_models//Brk8//rubik16_brk8.nf");
  // testfiles.push_back(".//..//test_models//Brk20//rubik12_brk20.nf");
  testfiles.push_back(".//..//test_models//Brk20//rubik20_brk20.nf");

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
    femdata->GetStiffnessMatrix()->SetDeviceMode(SPRmatrix::DEV_OMP);
    omp_set_num_threads(4);
    double tsolvecpu = omp_get_wtime();
    for (int i = 0; i < ntestloops; i++) {
      femdata->GetStiffnessMatrix()->CG(femdata->GetDisplVector(),
                                        femdata->GetForceVector(),
                                        iterations,
                                        precision);
    }
    tsolvecpu = (omp_get_wtime() - tsolvecpu)/ntestloops;
    stratTime tsolvegpu1 = BenchCGGPU(femdata, localsize, ntestloops);
    stratTime tsolvegpu2 = BenchCGGPU(femdata, localsize*2, ntestloops);
    stratTime tsolvegpu3 = BenchCGGPU(femdata, localsize*4, ntestloops);
    stratTime tsolvegpu4 = BenchCGGPU(femdata, localsize*8, ntestloops);
    stratTime tsolvegpu5 = BenchCGGPU(femdata, localsize*16, ntestloops);

    stratTime fastest = _min(tsolvegpu1, _min(tsolvegpu2,
                          _min(tsolvegpu3, _min(tsolvegpu4, tsolvegpu5))));

    printf("Solver Time CPU: %3.4f -> %.2f%% of Stiffness Time\n",
           tsolvecpu, (tsolvecpu/tstiff) * 100);
    printf("===================== Comparison ======================\n");
    printGPUSolveTime(tsolvegpu1.time, tsolvecpu, tstiff, localsize*1);
    printGPUSolveTime(tsolvegpu2.time, tsolvecpu, tstiff, localsize*2);
    printGPUSolveTime(tsolvegpu3.time, tsolvecpu, tstiff, localsize*4);
    printGPUSolveTime(tsolvegpu4.time, tsolvecpu, tstiff, localsize*8);
    printGPUSolveTime(tsolvegpu5.time, tsolvecpu, tstiff, localsize*16);
    if (tsolvecpu < fastest.time) {
      printf("Fastest Time (CPU):%3.4f\n", tsolvecpu);
    } else {
      std::string fasteststrat = getStratString(fastest.strat);
      std::cout << "Fastest Time (" << fasteststrat << "):" << fastest.time <<
        std::endl;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
std::string Microbench::getStratString(SPRmatrix::OclStrategy strategy) {
  std::string stratstring;
  switch (strategy) {
  case SPRmatrix::STRAT_NAIVE:
    stratstring = "STRAT_NAIVE";
    break;
  case SPRmatrix::STRAT_NAIVEUR:
    stratstring = "STRAT_NAIVEUR";
    break;
  case SPRmatrix::STRAT_SHARE:
    stratstring = "STRAT_SHARE";
    break;
  case SPRmatrix::STRAT_BLOCK:
    stratstring = "STRAT_BLOCK";
    break;
  case SPRmatrix::STRAT_BLOCKUR:
    stratstring = "STRAT_BLOCKUR";
    break;
  case SPRmatrix::STRAT_TEST:
    stratstring = "STRAT_TEST";
    break;
  default:  //default falls back to CPU
    assert(false);
  }
  return stratstring;
}

///////////////////////////////////////////////////////////////////////////////
void Microbench::printGPUSolveTime(double tsolvegpu, double tsolvecpu, double tstiff,
                       int localsize) {
    printf("GPU Solve Time(%i): %3.4f-> %.0f%% of Stiffness-> %.0f%% of CPU\n",
      localsize, tsolvegpu, (tsolvegpu/tstiff)*100, (tsolvegpu/tsolvecpu)*100);
}

// BenchCG: Benchmark for testing performance of CG solver
///////////////////////////////////////////////////////////////////////////////
void Microbench::BenchCG2() {
  int initestsize = 4 * 1024;
  int maxtestsize = 16 * 1024;
  int ntestloops = 2;
  int localsize = 4;
  SPRmatrix::SPRformat matformat = SPRmatrix::ELL;

  // Builds kernel and loads source so that dynamic loading does not interfere
  // in benchmarking
  preloadEllKernels();

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
    dummymatrix->SetDeviceMode(SPRmatrix::DEV_OMP);
    omp_set_num_threads(1);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      dummymatrix->CG(xvec, yvec, 1000, 0.00001f);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));
    // CPU CG Microbench
    omp_set_num_threads(2);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      dummymatrix->CG(xvec, yvec, 1000, 0.00001f);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));
    // CPU CG Microbench
    omp_set_num_threads(4);
    t1 = omp_get_wtime();
    for (int i = 0; i <= ntestloops; i++) {
      dummymatrix->CG(xvec, yvec, 1000, 0.00001f);
    }
    t2 = omp_get_wtime() - t1;
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (t2*1000));

    // GPU CG Microbench
    omp_set_num_threads(4);
    BenchCGGPU(dummymatrix, xvec, yvec, localsize * 1, ntestloops);
    BenchCGGPU(dummymatrix, xvec, yvec, localsize * 2, ntestloops);
    BenchCGGPU(dummymatrix, xvec, yvec, localsize * 4, ntestloops);
    BenchCGGPU(dummymatrix, xvec, yvec, localsize * 8, ntestloops);

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
  femdata->GetStiffnessMatrix()->CG(femdata->GetDisplVector(),
                                    femdata->GetForceVector(),
                                    3000,
                                    0.001f);
  stratTime tsolvegpu = BenchCGGPU(femdata, localsize, ntestloops);

  printf("Solver Time CPU: %3.4f -> %.2f%% of Stiffness Time\n",
         tsolvecpu, (tsolvecpu/tstiff) * 100);
  printf("Solver Time GPU: %3.4f -> %.2f%% of Stiffness Time\n",
         tsolvegpu.time, (tsolvegpu.time/tstiff) * 100);
}

// BenchMV: Microbenchmark for testing performance of MV product
////////////////////////////////////////////////////////////////////////////////
void Microbench::BenchMV() {
  int initestsize = 8 * 1024;
  int maxtestsize = 64 * 1024;
  int ntestloops  = 8;
  int bandsize    = 128;
  SPRmatrix::SPRformat matformat = SPRmatrix::ELL;

  // Builds kernel and loads source so that dynamic loading does not interfere
  // in benchmarking
  preloadEllKernels();

  for (int testsize = initestsize; testsize <= maxtestsize; testsize *= 2) {
    int localsize = 16;
    SPRmatrix* dummymatrix = SPRmatrix::CreateMatrix(testsize, matformat);
    dummymatrix->VerboseErrors(false);
    // test arrays setup
    fem_float* xvec = (fem_float*)malloc(testsize * sizeof(fem_float));
    fem_float* yvec = (fem_float*)malloc(testsize * sizeof(fem_float));
    for (int i = 0; i < testsize; i++) {
      int val = i + 1;
      xvec[i] = (float)val;
      for (int j = 0; j < bandsize; j++) {
        dummymatrix->AddElem(i, i + j, 1.0f);
      }
    }

    printf("\n==================================================\n");
    printf("MV microbench testsize: %i\n", testsize);

    // CPU MV Microbench
    printf("==================================================\n");
    double cputime = 0; 
    omp_set_num_threads(1);
    cputime = getAxyTime(dummymatrix, xvec, yvec, ntestloops);
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (cputime * 1000));
    // CPU 2 threads
    omp_set_num_threads(2);
    cputime = getAxyTime(dummymatrix, xvec, yvec, ntestloops);
    printf("CPU  time(%i): %.4fms \n", omp_get_max_threads(), (cputime * 1000));
    // CPU 4 threads
    omp_set_num_threads(4);
    cputime = getAxyTime(dummymatrix, xvec, yvec, ntestloops);
    printf("CPU time(%i): %.4fms \n", omp_get_max_threads(), (cputime * 1000));
    double gflops = calcMFlops(testsize, bandsize, cputime);
    printf("CPU MFLOP(%i): %.4f Mflops \n", omp_get_max_threads(), gflops);

    // GPU CG Microbench
    printf("--------------------------------------------------\n");
    BenchAxyGPU(dummymatrix, xvec, yvec, localsize, ntestloops);
    localsize *= 2;
    BenchAxyGPU(dummymatrix, xvec, yvec, localsize, ntestloops);
    localsize *= 2;
    stratTime gpu = BenchAxyGPU(dummymatrix, xvec, yvec, localsize, ntestloops);
    gflops = calcMFlops(testsize, bandsize, gpu.time);
    std::cout << "GPU MFLOP(" << getStratString(gpu.strat);
    printf("): %.4f Mflops \n", gflops);

    printf("Best GPU (%.1f%%) faster then CPU \n",
      (100 * ((cputime / gpu.time) - 1)));

    free(yvec);
    free(xvec);
    delete(dummymatrix);
  }
}

////////////////////////////////////////////////////////////////////////////////
double Microbench::calcMFlops(int testsize, int bandsize, double time) {
    return (2.0f * testsize * (double)bandsize * 1.0e-6) / time;
}

////////////////////////////////////////////////////////////////////////////////
void Microbench::preloadEllKernels() {
  OCL.setDir(".\\..\\src\\OpenCL\\clKernels\\");
  // Preloads all kernels and source not to interfere in benchmark time
  OCL.loadSource("LAopsEll.cl");
  OCL.loadKernel("SpMVNaive");
  OCL.loadKernel("SpMVNaiveUR");
  OCL.loadKernel("SpMVStag");
  OCL.loadKernel("SpMVCoal");
  OCL.loadKernel("SpMVCoalUR");
  OCL.loadKernel("SpMVCoalUR2");
}

// BenchCG: Benchmark for testing performance of CG solver
////////////////////////////////////////////////////////////////////////////////
Microbench::stratTime Microbench::BenchAxyGPU(SPRmatrix* dummymatrix,
                               fem_float* xvec,
                               fem_float* yvec,
                               size_t     localsize,
                               int        nloops) {
  dummymatrix->SetDeviceMode(SPRmatrix::DEV_GPU);
  dummymatrix->SetOclLocalSize(localsize);
  stratTime naive, naiveur, share, block, blockur, test;
  // Naive Bench
  naive.strat = SPRmatrix::STRAT_NAIVE;
  dummymatrix->SetOclStrategy(naive.strat);
  naive.time = getAxyTime(dummymatrix, xvec, yvec, nloops);
  // NaiveUR Bench
  naiveur.strat = SPRmatrix::STRAT_NAIVEUR;
  dummymatrix->SetOclStrategy(naiveur.strat);
  naiveur.time = getAxyTime(dummymatrix, xvec, yvec, nloops);
  // Shared Bench
  share.strat = SPRmatrix::STRAT_SHARE;
  dummymatrix->SetOclStrategy(share.strat);
  share.time = getAxyTime(dummymatrix, xvec, yvec, nloops);
  // Blocked Bench
  block.strat = SPRmatrix::STRAT_BLOCK;
  dummymatrix->SetOclStrategy(block.strat);
  block.time = getAxyTime(dummymatrix, xvec, yvec, nloops);
  // Blocked Bench
  blockur.strat = SPRmatrix::STRAT_BLOCKUR;
  dummymatrix->SetOclStrategy(blockur.strat);
  blockur.time = getAxyTime(dummymatrix, xvec, yvec, nloops);
  // Test Bench
  test.strat = SPRmatrix::STRAT_TEST;
  dummymatrix->SetOclStrategy(test.strat);
  test.time = getAxyTime(dummymatrix, xvec, yvec, nloops);

  stratTime mintime = _min(naive, _min(naiveur, _min(share,
    _min(block, _min(blockur, test)))));

  printf("CGGPU time(%2i): N:%.0fms NUR:%.0fms S:%.0fms B:%.0fms BUR:%.0fms TST:%.0fms\n",
         localsize,
         (naive.time * 1000),
         (naiveur.time * 1000),
         (share.time * 1000),
         (block.time * 1000),
         (blockur.time * 1000),
         (test.time * 1000));

  return mintime;
}


////////////////////////////////////////////////////////////////////////////////
double Microbench::getAxyTime(SPRmatrix* sprmat,
                              fem_float* xvec,
                              fem_float* yvec,
                              int nloops) {
  double tini = omp_get_wtime();
  for (int i = 0; i < nloops; i++) {
    sprmat->Axy(xvec, yvec);
  }
  return ((omp_get_wtime() - tini)/(double)nloops);
}

// BenchCG: Benchmark for testing performance of CG solver
////////////////////////////////////////////////////////////////////////////////
Microbench::stratTime Microbench::BenchCGGPU(FemData* femdata,
                              size_t   localsize,
                              int      nloops) {
  SPRmatrix* matrix = femdata->GetStiffnessMatrix();
  fem_float* xvect = femdata->GetDisplVector();
  fem_float* yvect = femdata->GetForceVector();

  return BenchCGGPU(matrix, xvect, yvect, localsize, nloops);
}

// BenchCG: Benchmark for testing performance of CG solver
////////////////////////////////////////////////////////////////////////////////
Microbench::stratTime Microbench::BenchCGGPU(SPRmatrix* matrix,
                              fem_float* xvec,
                              fem_float* yvec,
                              size_t     localsize,
                              int        nloops) {
  int niter = 5000;
  fem_float precision = 0.0001f;

  stratTime naive, naiveur, share, block, blockur, btest;
  // Naive Bench
  matrix->SetDeviceMode(SPRmatrix::DEV_GPU);
  matrix->SetOclLocalSize(localsize);
  naive.strat = SPRmatrix::STRAT_NAIVE;
  matrix->SetOclStrategy(naive.strat);
  naive.time = getCGTime(matrix, xvec, yvec, nloops, niter, precision);
  // NaiveUR Bench
  naiveur.strat = SPRmatrix::STRAT_NAIVEUR;
  matrix->SetOclStrategy(naiveur.strat);
  naiveur.time = getCGTime(matrix, xvec, yvec, nloops, niter, precision);
  // Shared Bench
  share.strat = SPRmatrix::STRAT_SHARE;
  matrix->SetOclStrategy(share.strat);
  share.time = getCGTime(matrix, xvec, yvec, nloops, niter, precision);
  // Blocked Bench
  block.strat = SPRmatrix::STRAT_BLOCK;
  matrix->SetOclStrategy(block.strat);
  block.time = getCGTime(matrix, xvec, yvec, nloops, niter, precision);
  // Blocked Bench
  blockur.strat = SPRmatrix::STRAT_BLOCKUR;
  matrix->SetOclStrategy(blockur.strat);
  blockur.time = getCGTime(matrix, xvec, yvec, nloops, niter, precision);
  // Blocked Bench
  btest.strat = SPRmatrix::STRAT_TEST;
  matrix->SetOclStrategy(btest.strat);
  btest.time = getCGTime(matrix, xvec, yvec, nloops, niter, precision);

  stratTime mintime = _min(naive, _min(naiveur, _min(share,
    _min(block, _min(blockur, btest)))));

  printf("CGGPU time(%2i): N:%.0fms NUR:%.0fms S:%.0fms B:%.0fms BUR:%.0fms TST:%.0fms\n",
         localsize,
         (naive.time * 1000),
         (naiveur.time * 1000),
         (share.time * 1000),
         (block.time * 1000),
         (blockur.time * 1000),
         (btest.time * 1000));

  return mintime;
}

///////////////////////////////////////////////////////////////////////////////
Microbench::stratTime Microbench::_min(stratTime t1, stratTime t2) {
  return (t1.time < t2.time ? t1 : t2);
}

///////////////////////////////////////////////////////////////////////////////
double Microbench::getCGTime(SPRmatrix* sprmat,
                              fem_float* xvec,
                              fem_float* yvec,
                              int nloops,
                              int niterations,
                              fem_float precision) {
  if (nloops <= 0) return 0;
  double tini = omp_get_wtime();
  for (int i = 0; i < nloops; i++) {
    sprmat->CG(xvec, yvec, niterations, precision);
  }
  return (omp_get_wtime() - tini) / nloops;
}

