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

#include "gtest/gtest.h"
#include "checkcrt.h"

#include "SPRmatrix/SPRmatrix.h"
#include "SPRmatrix/CSRmatrix.h"
#include "SPRmatrix/DIAmatrix.h"
#include "SPRmatrix/DENmatrix.h"
#include "SPRmatrix/ELLmatrix.h"
#include "SPRmatrix/EIGmatrix.h"
#include "SPRmatrix/EIGmatrix.h"
#include "OpenCL/OCLwrapper.h"
#include "LAops/SprSolver.h"

#include <iostream>
#include <stdlib.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "helpers.h"


// The fixture for value parameterized testing class AxyGPUTest.
////////////////////////////////////////////////////////////////////////////////
class SolverValidationGPU : public ::testing::TestWithParam<SPRmatrix::OclStrategy> {
};

TEST_P(SolverValidationGPU, Compare_Eigen_CG_With_EllpackGPU_Solver) {
  CheckMemory check;
  bool verbose = false;
  using namespace Eigen;
  using namespace std;
  int localsz = 4;
  int matdim  = 80;
  int maxiter = 100;
  double tol  = 0.001;

  SparseMatrix<float> Mf;
  SparseMatrix<double> Md;
  SPRmatrix* mysparse = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  mysparse->SetOclStrategy(GetParam());
  genSparsePosDefMatrices(Mf, Md, mysparse, matdim);
  VectorXf xf(matdim);
  VectorXf bf(matdim);
  fem_float* vecx = (fem_float*)malloc(matdim * sizeof(float));
  fem_float* vecy = (fem_float*)malloc(matdim * sizeof(float));
  srand((unsigned int)time(NULL));
  for (int i = 0; i < matdim; i++) {
    float randval;
    randval = (float) (rand() % 9);
    bf[i]   = randval;
    vecy[i] = randval;
  }

  // fill A and b
  ConjugateGradient<SparseMatrix<float> > cg;
  cg.compute(Mf);
  xf = cg.solve(bf);
  cg.setMaxIterations(maxiter);
  cg.setTolerance(tol);
  if (verbose) {
    std::cout << "\nCGSolver:" << std::endl;
    std::cout << "max iterations:  " << cg.maxIterations() << std::endl;
    std::cout << "tolerance:       " << cg.tolerance() << std::endl;
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;
    std::cout << "Mf: " << Mf << std::endl;
    std::cout << "xf:\n" << xf << std::endl;
  }

  BiCGSTAB<SparseMatrix<float>> bicsolverf(Mf);
  bicsolverf.setMaxIterations(maxiter);
  bicsolverf.setTolerance(tol);
  xf = bicsolverf.solve(bf);
  if (verbose) {
    std::cout << "\nBiCGSolver(float):" << std::endl;
    std::cout << "max iterations:  " << bicsolverf.maxIterations() << std::endl;
    std::cout << "tolerance:       " << bicsolverf.tolerance() << std::endl;
    std::cout << "#iterations:     " << bicsolverf.iterations() << std::endl;
    std::cout << "estimated error: " << bicsolverf.error()      << std::endl;
    std::cout << "xf:\n" << xf << std::endl << std::endl;
  }

  mysparse->CG(vecx, vecy, maxiter, tol);
  //printVectorf(vecx, matdim);
  for (int i = 0; i < matdim; i++) {
    ASSERT_NEAR(vecx[i], xf[i], .01f);
  }

  // Cleanup
  delete(mysparse);
  free(vecx);
  free(vecy);
}

using ::testing::Values;
INSTANTIATE_TEST_CASE_P(AxyNAIVE,   SolverValidationGPU, Values(SPRmatrix::STRAT_NAIVE));
INSTANTIATE_TEST_CASE_P(AxyNAIVEUR, SolverValidationGPU, Values(SPRmatrix::STRAT_NAIVEUR));
INSTANTIATE_TEST_CASE_P(AxySHARE,   SolverValidationGPU, Values(SPRmatrix::STRAT_SHARE));
INSTANTIATE_TEST_CASE_P(AxyBLOCK,   SolverValidationGPU, Values(SPRmatrix::STRAT_BLOCK));
INSTANTIATE_TEST_CASE_P(AxyBLOCKUR, SolverValidationGPU, Values(SPRmatrix::STRAT_BLOCKUR));
