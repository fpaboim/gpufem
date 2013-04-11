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


// The fixture for value parameterized testing class AxyGPUTest.
////////////////////////////////////////////////////////////////////////////////
class AxyGPUTest : public ::testing::TestWithParam<SPRmatrix::OclStrategy> {
};

// Ax_y GPU matrix vector multiply for diagonal matrix
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, 2D_Q4_analysis_cpugpu_comparison) {
  int matdim  = 8;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yres = (fem_float*)malloc(matdim * sizeof(fem_float));

  // Populate matrix to solve A * x = y (being that x == y), and A is a diagonal
  // matrix where diagonal values are i^2 (i == j)
  for (int i = 0; i < matdim; ++i) {
    int num = i + 1;
    testmatrix->SetElem(i, i, (float)num);
    xvec[i] = (float)num;
    yvec[i] = 0.0f;
    yres[i] = (float)num * num;
  }

  // Test all kernels
  testmatrix->Ax_y(xvec, yvec);
  for (int i = 0; i < matdim; ++i) {
    ASSERT_NEAR(yvec[i], yres[i], 0.001);
  }

  // Teardown
  free(xvec);
  free(yvec);
  free(yres);
  delete(testmatrix);
  OCL.teardown();
}

// Ax_y GPU matrix vector multiply for diagonal matrix
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, Ell_diagonal_Ax_y_GPU) {
  CheckMemory check;
  int matdim  = 2 * 1024;
  int localsz = 8;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yres = (fem_float*)malloc(matdim * sizeof(fem_float));

  // Populate matrix to solve A * x = y (being that x == y), and A is a diagonal
  // matrix where diagonal values are i^2 (i == j)
  for (int i = 0; i < matdim; ++i) {
    fem_float num = (fem_float) i + 1;
    testmatrix->SetElem(i, i, (float)num);
    xvec[i] = num;
    yvec[i] = 0.0f;
    yres[i] = num * num;
  }

  // Test all kernels
  testmatrix->SetOclStrategy(GetParam());
  testmatrix->AxyGPU(xvec, yvec, localsz);
  for (int i = 0; i < matdim; ++i) {
    ASSERT_NEAR(yvec[i], yres[i], 0.001);
  }

  // Teardown
  free(xvec);
  free(yvec);
  free(yres);
  delete(testmatrix);
  OCL.teardown();
}

// Ax = y - vector banded matrix multiply test
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, ELL_band_Axy) {
  CheckMemory check;
  int matdim  = 64;
  int localsz = 16;
  int bandsz  = 8;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yres = (fem_float*)malloc(matdim * sizeof(fem_float));

  for (int i = 0; i < matdim; ++i) {
    int num = i + 1;
    for (int j = 0; j < bandsz; ++j)
      testmatrix->AddElem(i, j, 1);
    xvec[i] = (float)2;
    yres[i] = (float)2 * bandsz;
  }

  // Test with all strategies
  testmatrix->SetOclStrategy(GetParam());
  testmatrix->AxyGPU(xvec, yvec, localsz);
  for (int i = 0; i < matdim; ++i) {
    ASSERT_NEAR(yvec[i], yres[i], 0.001);
  }

  // Teardown
  free(xvec);
  free(yvec);
  free(yres);
  delete(testmatrix);
  OCL.teardown();
}

// Ax = y - vector banded matrix multiply test
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, ELL_diag_band_Axy) {
  CheckMemory check;
  int matdim  = 64;
  int localsz = 16;
  int bandsz  = 4;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yres = (fem_float*)malloc(matdim * sizeof(fem_float));

  for (int i = 0; i < matdim; ++i) {
    int num = i + 1;
    for (int j = 0; j < bandsz; ++j)
      testmatrix->AddElem(i, i+j, 1);
    xvec[i] = (float)2;
    yres[i] = (float)2 * bandsz;
  }

  int startpos = matdim - bandsz;
  for (int i = 1; i < bandsz; ++i) {
    yres[startpos + i] -= 2 * i;
  }

  // Test with all strategies
  testmatrix->SetOclStrategy(GetParam());
  testmatrix->AxyGPU(xvec, yvec, localsz);
  for (int i = 0; i < matdim-1; ++i) {
    ASSERT_NEAR(yvec[i], yres[i], 0.001);
  }

  // Teardown
  free(xvec);
  free(yvec);
  free(yres);
  delete(testmatrix);
  OCL.teardown();
}

// Ax = y - vector banded matrix multiply test
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, ELL_banded_4to64_localsize) {
  CheckMemory check;
  int matdim     = 2 * 1024;
  int minlocalsz = 4;
  int maxlocalsz = 32;
  int bandsz     = 16;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yres = (fem_float*)malloc(matdim * sizeof(fem_float));
  // Fill test matrix
  for (int i = 0; i < matdim; ++i) {
    int num = i + 1;
    for (int j = 0; j < bandsz; ++j)
      testmatrix->AddElem(i, j, 1);
    xvec[i] = (float)1;
    yres[i] = (float)1 * bandsz;
  }
  // Test with all strategies
  testmatrix->SetOclStrategy(GetParam());
  for (int localsz = minlocalsz; localsz <= maxlocalsz; localsz *= 2) {
    testmatrix->AxyGPU(xvec, yvec, localsz);
    for (int i = 0; i < matdim; ++i) {
      ASSERT_NEAR(yvec[i], yres[i], 0.001);
    }
  }
  // Teardown
  free(xvec);
  free(yvec);
  free(yres);
  delete(testmatrix);
  OCL.teardown();
}

// Conjugate Gradient Test
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, Ell_CGGPU) {
  CheckMemory check;
  int matdim = 32;
  int localsz = 4;
  int maxiter = 100;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));

  // Populate matrix to solve I * x = y (being that x == y)
  for (int i = 0; i < matdim; ++i) {
    int num = i + 1;
    testmatrix->AddElem(i, i, (float)num * num);
    yvec[i] = (float)num * num;
    xvec[i] = 0.0f;
  }
  testmatrix->SetElem(0,0,2);
  testmatrix->SetElem(0,1,2);
  testmatrix->SetElem(1,0,2);

  // Solve CG with all three strategies
  testmatrix->SetOclStrategy(GetParam());
  testmatrix->SolveCgGpu(xvec, yvec, maxiter, 0.000001f, localsz);
  ASSERT_NEAR(xvec[0],  -1, 0.001);
  ASSERT_NEAR(xvec[1], 1.5, 0.001);
  for (int i = 2; i < matdim; ++i) {
    int j = i + 1;
    fem_float comp = (fem_float)j;
    ASSERT_NEAR(xvec[i], 1, 0.001);
  }

  // Teardown
  free(xvec);
  free(yvec);
  delete(testmatrix);
  OCL.teardown();
}

// Conjugate Gradient Test
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, Ell_CGGPU_Localsize_8to64) {
  CheckMemory check;
  int matdim = 64;
  int minlocalsz = 16;
  int maxlocalsz = 64;
  int maxiter = 500;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));

  // Populate matrix to solve I * x = y (being that x == y)
  for (int i = 0; i < matdim; ++i) {
    int num = i + 1;
    testmatrix->AddElem(i, i, (float)num * num);
    yvec[i] = (float)num * num;
    xvec[i] = 0.0f;
  }
  testmatrix->SetElem(0,0,2);
  testmatrix->SetElem(0,1,2);
  testmatrix->SetElem(1,0,2);

  // Solve CG with all three strategies
  testmatrix->SetOclStrategy(GetParam());
  for (int localsz = minlocalsz; localsz <= maxlocalsz; localsz *= 2) {
    testmatrix->SolveCgGpu(xvec, yvec, maxiter, 0.000001f, localsz);
    ASSERT_NEAR(xvec[0],  -1, 0.001);
    ASSERT_NEAR(xvec[1], 1.5, 0.001);
    for (int i = 2; i < matdim; ++i) {
      int j = i + 1;
      fem_float comp = (fem_float)j;
      ASSERT_NEAR(xvec[i], 1, 0.001);
    }
  }

  // Teardown
  free(xvec);
  free(yvec);
  delete(testmatrix);
  OCL.teardown();
}

// Conjugate Gradient Test for small positive definite difference matrix
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, Ell_CGGPU_small_difference_mat) {
  CheckMemory check;
  int matdim = 4;
  int maxiter = 200;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* xres = (fem_float*)malloc(matdim * sizeof(fem_float));

  for (int i = 0; i < matdim; ++i) {
    testmatrix->AddElem(i, i    , 2);
    testmatrix->AddElem(i, i + 1, -1);
    testmatrix->AddElem(i, i - 1, -1);
    xvec[i] = 0.0f;
    yvec[i] = 1.0f;
  }
  xres[0] = 2;
  xres[1] = 3;
  xres[2] = 3;
  xres[3] = 2;

  // Solve CG with all three strategies
  testmatrix->SetOclStrategy(GetParam());
  testmatrix->SolveCgGpu(xvec, yvec, maxiter, 0.00001f, 4);

    // check result
  for (int i = 0; i < matdim; i++) {
    ASSERT_NEAR(xvec[i], xres[i], 0.001);
  }

  // Teardown
  free(xres);
  free(xvec);
  free(yvec);
  delete(testmatrix);
  OCL.teardown();
}

// Conjugate Gradient Test for banded matrix, the multiplying vector for band
// with a width of 4 and the resulting vector y all equal to 1 will always be
// a vector with the repeating sequence [4 0 0 0] and the last 3 elements will
// be: [-1 -1 -1] (i.e. the matrix will always be invertible)
////////////////////////////////////////////////////////////////////////////////
TEST_P(AxyGPUTest, Ell_CGGPU_Banded_Localsize_8to64) {
  CheckMemory check;
  int matdim = 32; // assert result must be changed if matrix dimension changes
  int minlocalsz = 8;
  int maxlocalsz = 64;
  int maxiter = 500;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));

  for (int i = 0; i < matdim; ++i) {
    for (int j = 0; j < matdim; ++j) {
      testmatrix->SetElem(i, j, 1);
      yvec[i] = 1;
    }
    testmatrix->SetElem(i, i, 2);
  }

  // tests varying local sizes
  testmatrix->SetOclStrategy(GetParam());
  for (int localsz = minlocalsz; localsz <= maxlocalsz; localsz *= 2) {
    CPU_CG(testmatrix, xvec, yvec, matdim, maxiter, 0.00001f, false);
    for (int i = 0; i < matdim; i++) {
      ASSERT_NEAR(xvec[i], 0.0303f, 0.001);
    }
  }

  // Teardown
  free(xvec);
  free(yvec);
  delete(testmatrix);
  OCL.teardown();
}

using ::testing::Values;
INSTANTIATE_TEST_CASE_P(AxyNAIVE,   AxyGPUTest, Values(SPRmatrix::NAIVE));
INSTANTIATE_TEST_CASE_P(AxyNAIVEUR, AxyGPUTest, Values(SPRmatrix::NAIVEUR));
INSTANTIATE_TEST_CASE_P(AxySHARE,   AxyGPUTest, Values(SPRmatrix::SHARE));
INSTANTIATE_TEST_CASE_P(AxyBLOCK,   AxyGPUTest, Values(SPRmatrix::BLOCK));
INSTANTIATE_TEST_CASE_P(AxyBLOCKUR, AxyGPUTest, Values(SPRmatrix::BLOCKUR));
INSTANTIATE_TEST_CASE_P(AxyTEST,    AxyGPUTest, Values(SPRmatrix::TEST));
