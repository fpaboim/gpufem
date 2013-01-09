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
#include "omp.h"

#include "SPRmatrix/SPRmatrix.cc"
#include "SPRmatrix/SPRmatrix.h"
#include "SPRmatrix/CSRmatrix.cc"
#include "SPRmatrix/CSRmatrix.h"
#include "SPRmatrix/DIAmatrix.cc"
#include "SPRmatrix/DIAmatrix.h"
#include "SPRmatrix/DENmatrix.cc"
#include "SPRmatrix/DENmatrix.h"
#include "SPRmatrix/ELLmatrix.cc"
#include "SPRmatrix/ELLmatrix.h"
#include "SPRmatrix/EIGmatrix.cc"
#include "SPRmatrix/EIGmatrix.h"

#include "LAops/SprSolver.h"

using ::testing::TestWithParam;
using ::testing::Values;

////////////////////////////////////////////////////////////////////////////////
// Tests common to all classes (interface)
////////////////////////////////////////////////////////////////////////////////

// The fixture for testing class SPRmatrixTest.
////////////////////////////////////////////////////////////////////////////////
class SPRmatrixTest : public ::testing::TestWithParam<SPRmatrix::SPRformat> {
 protected:
  SPRmatrixTest() {
    m_testmatrix = NULL;
    m_matdim = 256;
  }

  virtual ~SPRmatrixTest() {
  }

  virtual void SetUp() {
    m_testmatrix = SPRmatrix::CreateMatrix(m_matdim, GetParam());
  }

  virtual void TearDown() {
    delete(m_testmatrix);
  }

  int        m_matdim;
  SPRmatrix* m_testmatrix;
};

// CreationDoesNotLeak: Tests matrix creation
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, CreationDoesNotLeak) {
  CheckMemory check;

  for (int i = 1; i < 8; i++) {
    SPRmatrix* dummymatrix = SPRmatrix::CreateMatrix(256, GetParam());
    delete(dummymatrix);
  }
}

// BigMatrixDoesNotLeak: Tests matrix creation
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, BigMatrixDoesNotLeak) {
  CheckMemory check;
  int dummydim = 1*1024;
  SPRmatrix* dummymatrix = SPRmatrix::CreateMatrix(dummydim, GetParam());
  for (int i = 0; i < dummydim; i++) {
    dummymatrix->SetElem(0, i, 1);
    dummymatrix->SetElem(i, 0, 1);
    dummymatrix->SetElem(100, i, 1);
    dummymatrix->SetElem(i, 100, 1);
    dummymatrix->SetElem(1000, i, 1);
    dummymatrix->SetElem(i, 1000, 1);
  }
  for (int i = 0; i < 64; i++) {
    dummymatrix->AddElem(0, 0, 1);
    dummymatrix->AddElem(10, 10, 1);
    dummymatrix->AddElem(1000, 0, 1);
    dummymatrix->AddElem(0, 70, 1);
  }
  delete(dummymatrix);
}

// InitializesToZero: Tests ELLpack initializes elements to zero
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, InitializesToZero) {
  fem_float testfloat;
  testfloat = m_testmatrix->GetElem(0, 0);
  EXPECT_FLOAT_EQ(testfloat, 0);
  testfloat = m_testmatrix->GetElem(3, 3);
  EXPECT_FLOAT_EQ(testfloat, 0);
  testfloat = m_testmatrix->GetElem(2, 1);
  EXPECT_FLOAT_EQ(testfloat, 0);
  testfloat = m_testmatrix->GetElem(78, 87);
  EXPECT_FLOAT_EQ(testfloat, 0);
}

// OutOfBoundsAccess: Test getting and setting out of bounds matrix entries
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, OutOfBoundsAccess) {
  CheckMemory check;
  SPRmatrix* dummymatrix = SPRmatrix::CreateMatrix(256, GetParam());

  dummymatrix->AddElem(256, 256, 1);
  EXPECT_FLOAT_EQ(dummymatrix->GetElem(256,256), 0);
  dummymatrix->SetElem(256, 256, 1);
  EXPECT_FLOAT_EQ(dummymatrix->GetElem(256,256), 0);
  delete(dummymatrix);
}

// Tests Simple Set/Get Methods
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, SetGetSimple) {
  fem_float testfloat;

  m_testmatrix->SetElem(0, 0, 4);
  testfloat = m_testmatrix->GetElem(0, 0);
  EXPECT_FLOAT_EQ(testfloat, 4);

  m_testmatrix->SetElem(2, 9, 8);
  testfloat = m_testmatrix->GetElem(2, 9);
  EXPECT_FLOAT_EQ(testfloat, 8);

  m_testmatrix->SetElem(91, 19, 4.56f);
  testfloat = m_testmatrix->GetElem(91, 19);
  EXPECT_FLOAT_EQ(testfloat, 4.56f);

  m_testmatrix->SetElem(91, 19, -4.56f);
  testfloat = m_testmatrix->GetElem(91, 19);
  EXPECT_FLOAT_EQ(testfloat, -4.56f);
}

// Tests Set/Get on elements of the same row
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, SetSameRow) {
  fem_float testfloat;

  m_testmatrix->SetElem(15, 3, 0);
  testfloat = m_testmatrix->GetElem(15, 3);
  EXPECT_FLOAT_EQ(testfloat, 0);

  m_testmatrix->SetElem(15, 6, 3);
  testfloat = m_testmatrix->GetElem(15, 6);
  EXPECT_FLOAT_EQ(testfloat, 3);

  m_testmatrix->SetElem(15, 4, 4);
  testfloat = m_testmatrix->GetElem(15, 4);
  EXPECT_FLOAT_EQ(testfloat, 4);
}

// Tests Set/Get on elements of the same row
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, SetFullRow) {
  fem_float accumulator = 0;
  for (int i = 0; i < m_matdim; i++) {
    m_testmatrix->SetElem(0, i, 1);
    m_testmatrix->SetElem(10, i, 2);
    accumulator += i;
  }
  fem_float testaccumulator1 = 0;
  fem_float testaccumulator2 = 0;
  for (int i = 0; i < m_matdim; i++) {
    testaccumulator1 += m_testmatrix->GetElem(0, i);
    testaccumulator2 += m_testmatrix->GetElem(10, i);
  }

  EXPECT_FLOAT_EQ(testaccumulator1, (float)m_matdim);
  EXPECT_FLOAT_EQ(testaccumulator2, (float)m_matdim*2);
}

// Tests Set/Get on elements of the same row
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, SetSmallFullMatrix) {
  fem_float accumulator = 0;
  int i,j;
  int dim = 256;
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      m_testmatrix->SetElem(i, j, (fem_float)i+j);
      accumulator += i + j;
    }
  }
  fem_float testaccumulator = 0;
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      testaccumulator += m_testmatrix->GetElem(i, j);
    }
  }

  EXPECT_FLOAT_EQ(testaccumulator, accumulator);
}

// Tests Set/Get on whole diagonal
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, SetFullDiagonal) {
  fem_float testfloat = 0;
  fem_float accumulator = 0;
  for (int i = 0; i < m_matdim; ++i) {
    m_testmatrix->SetElem(i, i, (fem_float)i);
    accumulator += i;
  }
  for (int i = 0; i < m_matdim; ++i) {
    testfloat += m_testmatrix->GetElem(i, i);
  }
  EXPECT_FLOAT_EQ(testfloat, accumulator);
}

// Tests using add method to set unset variables
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, AddAsSet) {
  fem_float testfloat;

  m_testmatrix->AddElem(15, 3, 0);
  testfloat = m_testmatrix->GetElem(15, 3);
  EXPECT_FLOAT_EQ(testfloat, 0);

  m_testmatrix->AddElem(15, 6, 3);
  testfloat = m_testmatrix->GetElem(15, 6);
  EXPECT_FLOAT_EQ(testfloat, 3);

  m_testmatrix->AddElem(15, 4, 4);
  testfloat = m_testmatrix->GetElem(15, 4);
  EXPECT_FLOAT_EQ(testfloat, 4);
}

// Tests Set/Add/Get on elements of the same row
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, AddElemSimple) {
  fem_float testfloat;

  m_testmatrix->SetElem(15, 3, 1);
  m_testmatrix->AddElem(15, 3, 2);
  testfloat = m_testmatrix->GetElem(15, 3);
  EXPECT_FLOAT_EQ(testfloat, 3);

  m_testmatrix->SetElem(9, 0, -5);
  m_testmatrix->AddElem(9, 0, 10);
  testfloat = m_testmatrix->GetElem(9, 0);
  EXPECT_FLOAT_EQ(testfloat, 5);

  m_testmatrix->SetElem(30, 30, 6);
  m_testmatrix->AddElem(30, 30, -10);
  testfloat = m_testmatrix->GetElem(30, 30);
  EXPECT_FLOAT_EQ(testfloat, -4);
}

// Tests Set/Add/Get on elements of the same row
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, AddElemSameRow) {
  fem_float testfloat;

  m_testmatrix->SetElem(15, 3, 1);
  m_testmatrix->AddElem(15, 3, 2);
  testfloat = m_testmatrix->GetElem(15, 3);
  EXPECT_FLOAT_EQ(testfloat, 3);

  m_testmatrix->AddElem(15, 3, 8);
  m_testmatrix->AddElem(15, 3, 32);
  testfloat = m_testmatrix->GetElem(15, 3);
  EXPECT_FLOAT_EQ(testfloat, 43);

  m_testmatrix->SetElem(15, 0, -5);
  m_testmatrix->AddElem(15, 0, 10);
  testfloat = m_testmatrix->GetElem(15, 0);
  EXPECT_FLOAT_EQ(testfloat, 5);

  m_testmatrix->SetElem(15, 30, 6);
  m_testmatrix->AddElem(15, 30, -10);
  testfloat = m_testmatrix->GetElem(15, 30);
  EXPECT_FLOAT_EQ(testfloat, -4);
}

// Tests Add/Get by adding elements to two full rows and counting the sum
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, AddFullRow) {
  int rowlen = m_matdim;
  for (int i = 0; i < rowlen; i++) {
    m_testmatrix->AddElem(0, i, 1);
    m_testmatrix->AddElem(10, i, 2);
  }
  fem_float testaccumulator1 = 0;
  fem_float testaccumulator2 = 0;
  for (int i = 0; i < rowlen; i++) {
    testaccumulator1 += m_testmatrix->GetElem(0, i);
    testaccumulator2 += m_testmatrix->GetElem(10, i);
  }

  EXPECT_FLOAT_EQ(testaccumulator1, (float)rowlen);
  EXPECT_FLOAT_EQ(testaccumulator2, (float)rowlen*2);

  for (int i = rowlen - 1; i >= 0; i--) {
    m_testmatrix->AddElem(0, i, 1);
    m_testmatrix->AddElem(10, i, 2);
  }
  testaccumulator1 = 0;
  testaccumulator2 = 0;
  for (int i = 0; i < rowlen; i++) {
    testaccumulator1 += m_testmatrix->GetElem(0, i);
    testaccumulator2 += m_testmatrix->GetElem(10, i);
  }

  EXPECT_FLOAT_EQ(testaccumulator1, (float)rowlen*2);
  EXPECT_FLOAT_EQ(testaccumulator2, (float)rowlen*4);
}

// Tests Add/Get by adding a small dense matrix to the sparse matrix
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, AddSmallFullMatrix) {
  fem_float accumulator = 0;
  int i,j;
  int dim = 64;
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      m_testmatrix->AddElem(i, j, (fem_float)i+j);
      accumulator += i + j;
    }
  }
  fem_float testaccumulator = 0;
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      testaccumulator += m_testmatrix->GetElem(i, j);
    }
  }

  EXPECT_FLOAT_EQ(testaccumulator, accumulator);
}

// Tests Add/Get by adding a small dense matrix to the sparse matrix w/ omp
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, AddSmallFullMatrixOMP) {
  fem_float accumulator = 0;
  int i,j;
  int dim = 256;

  omp_set_num_threads(8);
#pragma omp parallel for private(i,j)
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      m_testmatrix->AddElem(i, j, (fem_float)i+j);
#pragma omp atomic
      accumulator += i + j;
    }
  }
  fem_float testaccumulator = 0;
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      testaccumulator += m_testmatrix->GetElem(i, j);
    }
  }

  EXPECT_FLOAT_EQ(testaccumulator, accumulator);
}

// Tests Set/Get on whole diagonal
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, AddFullDiagonal) {
  fem_float testfloat = 0;
  fem_float accumulator = 0;
  for (int i = 0; i < m_matdim; ++i) {
    m_testmatrix->AddElem(i, i, (fem_float)i);
    accumulator += i;
  }
  for (int i = 0; i < m_matdim; ++i) {
    testfloat += m_testmatrix->GetElem(i, i);
  }
  EXPECT_FLOAT_EQ(testfloat, accumulator);
}

// Tests Set/Get on whole diagonal
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, Ax_Y) {
  fem_float testfloat = 0;
  fem_float accumulator = 0;
  fem_float* xvec = (fem_float*)malloc(m_matdim * sizeof(fem_float));
  fem_float* yvec = (fem_float*)malloc(m_matdim * sizeof(fem_float));

  for (int i = 0; i < m_matdim; ++i) {
    m_testmatrix->AddElem(i, i, (fem_float)i);
    xvec[i] = (float)i;
    accumulator += i*i;
  }

  m_testmatrix->Ax_y(xvec, yvec);

  for (int i = 0; i < m_matdim; ++i) {
    testfloat += yvec[i];
  }
  EXPECT_FLOAT_EQ(testfloat, accumulator);
}

// Clear Method Test
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, Clear) {
  fem_float testfloat;

  m_testmatrix->SetElem(0, 0, 4);
  m_testmatrix->SetElem(0, 0, 7);
  m_testmatrix->SetElem(2, 3, 8);
  m_testmatrix->SetElem(73, 37, -8);
  m_testmatrix->Clear();
  testfloat = m_testmatrix->GetElem(0, 0);
  EXPECT_FLOAT_EQ(testfloat, 0);
  testfloat = m_testmatrix->GetElem(73, 37);
  EXPECT_FLOAT_EQ(testfloat, 0);
}

// Tests double teardown (function and destructor)
////////////////////////////////////////////////////////////////////////////////
TEST_P(SPRmatrixTest, DoubleTeardown) {
  m_testmatrix->Teardown();
}


INSTANTIATE_TEST_CASE_P(DENmatrix, SPRmatrixTest, Values(SPRmatrix::DEN));
INSTANTIATE_TEST_CASE_P(CSRmatrix, SPRmatrixTest, Values(SPRmatrix::CSR));
INSTANTIATE_TEST_CASE_P(DIAmatrix, SPRmatrixTest, Values(SPRmatrix::DIA));
INSTANTIATE_TEST_CASE_P(ELLmatrix, SPRmatrixTest, Values(SPRmatrix::ELL));
INSTANTIATE_TEST_CASE_P(EIGmatrix, SPRmatrixTest, Values(SPRmatrix::EIG));
