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

#include "LAops/SprSolver.cc"
#include "LAops/SprSolver.h"
#include "LAops/LAops.cc"
#include "LAops/LAops.h"
#include "Utils/util.cc"
#include "Utils/util.h"
#include "OpenCL/OCLwrapper.cc"
#include "OpenCL/OCLwrapper.h"


// The fixture for testing class SPRmatrixTest.
////////////////////////////////////////////////////////////////////////////////
class LAopsTest : public ::testing::Test {
 protected:
  LAopsTest() {
    m_matdim = 20;
    m_vecdim = 20;
    m_testmat = NULL;
    m_testvec = NULL;
  }
  virtual ~LAopsTest() {
  }

  virtual void SetUp() {
    m_testmat = (fem_float**)malloc(m_matdim * sizeof(fem_float*));
    for (int i = 0; i < m_vecdim; ++i)
      m_testmat[i] = (fem_float*)malloc(m_matdim * sizeof(fem_float));
    m_testvec = (fem_float*)malloc(m_vecdim * sizeof(fem_float));
  }
  virtual void TearDown() {
    free(m_testmat);
    free(m_testvec);
  }

  int m_matdim;
  int m_vecdim;
  fem_float** m_testmat;
  fem_float*  m_testvec;
};

// Tests Dot Product Function
////////////////////////////////////////////////////////////////////////////////
TEST_F(LAopsTest, result_dotprod_ok) {
  fem_float* testvec2 = (fem_float*)malloc(m_vecdim * sizeof(fem_float));
  float accumulator = 0;
  for (int i = 0; i < m_vecdim; ++i) {
    m_testvec[i] = 2;
    testvec2[i]  = i+1.0f;
    accumulator += 2 * (i+1);
  }
  fem_float dotprod1 = dotProduct(m_vecdim, m_testvec, testvec2);
  fem_float dotprod2 = dotProductOMP(m_vecdim, m_testvec, testvec2);
  ASSERT_FLOAT_EQ(dotprod1, accumulator);
  ASSERT_FLOAT_EQ(dotprod2, accumulator);
}

// Tests Identity Matrix Function
////////////////////////////////////////////////////////////////////////////////
TEST_F(LAopsTest, create_identity_matrix) {
  GetEye(m_matdim, m_testmat);
  ASSERT_FLOAT_EQ(m_testmat[2][2], 1);
  ASSERT_FLOAT_EQ(m_testmat[0][0], 1);
  ASSERT_FLOAT_EQ(m_testmat[2][3], 0);
  ASSERT_FLOAT_EQ(m_testmat[7][2], 0);
  float accumulator = 0;
  for (int i = 0; i < m_matdim; ++i) {
    accumulator += m_testmat[3][i];
  }
  ASSERT_FLOAT_EQ(accumulator, 1);
}

// Tests Determinant of Identity
////////////////////////////////////////////////////////////////////////////////
TEST_F(LAopsTest, det_of_identity_matrix) {
  GetEye(m_matdim, m_testmat);
  fem_float detval = det(m_matdim, m_testmat);
  ASSERT_FLOAT_EQ(detval, 1);
}

// Conjugate Gradient Test
////////////////////////////////////////////////////////////////////////////////
TEST_F(LAopsTest, CGCPU_solver_nonsingular) {
  int matdim = 8;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);

  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  // Populate matrix to solve I * x = y (being that x == y)
  for (int i = 0; i < matdim; ++i) {
    int num = i + 1;
    testmatrix->AddElem(i, i, (float)num * num);
    yvec[i] = (float)num * num;
  }
  testmatrix->SetElem(0,0,2);
  testmatrix->SetElem(0,1,2);
  testmatrix->SetElem(1,0,2);

  CPU_CG(testmatrix, xvec, yvec, matdim, 3000, 0.00001f, false);
  //cholesky(testmatrix, xvec, yvec, matdim, true);

  ASSERT_NEAR(xvec[0], -1, 0.001);
  ASSERT_NEAR(xvec[1], 1.5, 0.001);
  for (int i = 2; i < matdim; ++i) {
    int j = i + 1;
    fem_float comp = (fem_float)j;
    ASSERT_NEAR(xvec[i], 1, 0.001);
  }

  free(xvec);
  free(yvec);
}

// Ax = y - vector diagonal matrix multiply test
////////////////////////////////////////////////////////////////////////////////
TEST_F(LAopsTest, spmv_diagonal_mat_multiply) {
  CheckMemory check;
  int matdim  = 16;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(matdim, SPRmatrix::ELL);
  fem_float* xvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(matdim * sizeof(fem_float));
  fem_float* yres = (fem_float*)malloc(matdim * sizeof(fem_float));

  // Populate matrix to solve A * x = y (being that x == y), and A is a diagonal
  // matrix where diagonal values are i^2 (i == j)
  for (int i = 0; i < matdim; ++i) {
    int num = i + 1;
    testmatrix->AddElem(i, i, (float)num);
    xvec[i] = (float)num;
    yres[i] = (float)num * num;
  }

  spax_y(testmatrix, xvec, yvec, matdim, false);

  for (int i = 0; i < matdim; ++i) {
    ASSERT_NEAR(yvec[i], yres[i], 0.001);
  }
  free(xvec);
  free(yvec);
  free(yres);
  delete(testmatrix);
}
