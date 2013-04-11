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
#include "SPRmatrix/ELLmatrix2.h"
#include "SPRmatrix/EIGmatrix.h"


// Tests Binary Search
////////////////////////////////////////////////////////////////////////////////
TEST(SPRSearchTest, BinarySearch) {
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(8, SPRmatrix::DEN);

  int* testintvec = (int*)calloc(10, sizeof(int));
  testintvec[0] = -1;
  testintvec[1] = 3;
  testintvec[2] = 5;
  testintvec[3] = 6;
  testintvec[4] = 11;
  int pos;
  pos = testmatrix->BinSearchInt(testintvec, 4, 5);
  EXPECT_EQ(pos, 2);
  pos = testmatrix->BinSearchInt(testintvec, -2, 5);
  EXPECT_EQ(pos, 0);
  pos = testmatrix->BinSearchInt(testintvec, 13, 5);
  EXPECT_EQ(pos, 5);
  pos = testmatrix->BinSearchInt(testintvec, 3, 5);
  EXPECT_EQ(pos, 1);
  pos = testmatrix->BinSearchInt2(testintvec, 4, 5);
  EXPECT_EQ(pos, 2);
  pos = testmatrix->BinSearchInt2(testintvec, -2, 5);
  EXPECT_EQ(pos, 0);
  pos = testmatrix->BinSearchInt2(testintvec, 13, 5);
  EXPECT_EQ(pos, 5);
  pos = testmatrix->BinSearchInt2(testintvec, 3, 5);
  EXPECT_EQ(pos, 1);

  free(testintvec);
  delete(testmatrix);
}

// Tests Linear Search
////////////////////////////////////////////////////////////////////////////////
TEST(SPRSearchTest, LinearSearch) {
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(8, SPRmatrix::DEN);

  int* testintvec = (int*)calloc(10, sizeof(int));
  testintvec[0] = -1;
  testintvec[1] = 3;
  testintvec[2] = 5;
  testintvec[3] = 6;
  testintvec[4] = 11;
  testintvec[5] = 16;
  int pos;
  pos = testmatrix->LinSearchI(testintvec, 4, 6);
  EXPECT_EQ(pos, 2);
  pos = testmatrix->LinSearchI(testintvec, -2, 6);
  EXPECT_EQ(pos, 0);
  pos = testmatrix->LinSearchI(testintvec, 13, 6);
  EXPECT_EQ(pos, 5);
  pos = testmatrix->LinSearchI(testintvec, 3, 6);
  EXPECT_EQ(pos, 1);

  free(testintvec);
  delete(testmatrix);
}

// Tests SSE Linear Search
////////////////////////////////////////////////////////////////////////////////
TEST(SPRSearchTest, LinearSearchSSE) {
  CheckMemory check;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(8, SPRmatrix::DEN);

  int* testintvec = (int*)_aligned_malloc(32 * sizeof(int), 16);
  testintvec[0] = -1;
  testintvec[1] = 3;
  testintvec[2] = 5;
  testintvec[3] = 6;
  testintvec[4] = 11;
  testintvec[5] = 12;
  testintvec[6] = 13;
  testintvec[7] = 16;
  testintvec[8] = 17;
  testintvec[9] = 19;
  testintvec[10] = 22;
  testintvec[11] = 26;
  testintvec[12] = 29;
  testintvec[13] = 37;
  testintvec[14] = 47;
  testintvec[15] = 58;
  testintvec[16] = 60;
  testintvec[17] = 61;
  testintvec[18] = 62;
  testintvec[19] = 67;
  testintvec[20] = INT_MAX;
  // Leaves garbage at end of vector to test correct handling

  // using 16 comparisons per iteration
  int pos;
  pos = testmatrix->LinSearchISSE(testintvec, 11, 32);
  EXPECT_EQ(pos, 4);
  pos = testmatrix->LinSearchISSE(testintvec, 15, 32);
  EXPECT_EQ(pos, 7);
  pos = testmatrix->LinSearchISSE(testintvec, -2, 32);
  EXPECT_EQ(pos, 0);
  pos = testmatrix->LinSearchISSE(testintvec, -1, 32);
  EXPECT_EQ(pos, 0);
  pos = testmatrix->LinSearchISSE(testintvec, 62, 32);
  EXPECT_EQ(pos, 18);
  pos = testmatrix->LinSearchISSE(testintvec, 99, 32);
  EXPECT_EQ(pos, 20);

  // using 4 comparisons per iteration
  pos = testmatrix->LinSearchISSE2(testintvec, 11, 32);
  EXPECT_EQ(pos, 4);
  pos = testmatrix->LinSearchISSE2(testintvec, 15, 32);
  EXPECT_EQ(pos, 7);
  pos = testmatrix->LinSearchISSE2(testintvec, -2, 32);
  EXPECT_EQ(pos, 0);
  pos = testmatrix->LinSearchISSE2(testintvec, -1, 32);
  EXPECT_EQ(pos, 0);
  pos = testmatrix->LinSearchISSE2(testintvec, 62, 32);
  EXPECT_EQ(pos, 18);
  pos = testmatrix->LinSearchISSE2(testintvec, 99, 32);
  EXPECT_EQ(pos, 20);

  _aligned_free(testintvec);
  delete(testmatrix);
}

// Tests Stepped binary Search
////////////////////////////////////////////////////////////////////////////////
TEST(SPRSearchTest, SteppedSearch) {
  CheckMemory chk;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(8, SPRmatrix::DEN);
  int* testintvec = (int*)calloc(16, sizeof(int));

  testintvec[0]  = 1;
  testintvec[1]  = 2;

  testintvec[2]  = 3;
  testintvec[3]  = 4;

  testintvec[4]  = 5;
  testintvec[5]  = 6;
  
  testintvec[6]  = 11;
  testintvec[7] = 16;

  testintvec[8] = 20;
  testintvec[9] = 21;
  int pos;
  pos = ELLmatrix::BinSearchIntStep(testintvec, 0, 5, 0, 2);
  EXPECT_EQ(pos, 0);
  pos = ELLmatrix::BinSearchIntStep(testintvec, 3, 5, 0, 2);
  EXPECT_EQ(pos, 2);
  pos = ELLmatrix::BinSearchIntStep(testintvec, 4, 5, 0, 2);
  EXPECT_EQ(pos, 4);
  pos = ELLmatrix::BinSearchIntStep(testintvec, 12, 5, 0, 2);
  EXPECT_EQ(pos, 8);
  pos = ELLmatrix::BinSearchIntStep(testintvec, 22, 5, 0, 2);
  EXPECT_EQ(pos, -1);
  pos = ELLmatrix::BinSearchIntStep(testintvec, 1, 5, 1, 2);
  EXPECT_EQ(pos, 1);
  pos = ELLmatrix::BinSearchIntStep(testintvec, 3, 5, 1, 2);
  EXPECT_EQ(pos, 3);
  pos = ELLmatrix::BinSearchIntStep(testintvec, 5, 5, 1, 2);
  EXPECT_EQ(pos, 5);
  pos = ELLmatrix::BinSearchIntStep(testintvec, 30, 5, 1, 2);
  EXPECT_EQ(pos, -1);

  free(testintvec);
  delete(testmatrix);
}

////////////////////////////////////////////////////////////////////////////////
TEST(SPRSearchTest, Ell2_binary_search) {
  CheckMemory chk;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(8, SPRmatrix::DEN);
  int* testintvec = (int*)calloc(16, sizeof(int));

  testintvec[0] = 0;
  testintvec[1] = 2;
  testintvec[2] = 3;
  testintvec[3] = 5;
  testintvec[4] = 7;
  testintvec[5] = 8;
  testintvec[6] = 11;
  testintvec[7] = 16;
  testintvec[8] = 18;

  int pos;
  pos = ELLmatrix2::BinSearchRow(testintvec, 0,  0, 8, 8);
  EXPECT_EQ(pos, 0);
  pos = ELLmatrix2::BinSearchRow(testintvec, 3,  0, 8, 8);
  EXPECT_EQ(pos, 2);
  pos = ELLmatrix2::BinSearchRow(testintvec, 4,  0, 8, 8);
  EXPECT_EQ(pos, 3);
  pos = ELLmatrix2::BinSearchRow(testintvec, 22, 0, 8, 8);
  EXPECT_EQ(pos, -1);
  pos = ELLmatrix2::BinSearchRow(testintvec, 1,  0, 8, 8);
  EXPECT_EQ(pos, 1);
  pos = ELLmatrix2::BinSearchRow(testintvec, -1, 0, 8, 8);
  EXPECT_EQ(pos, 0);
  pos = ELLmatrix2::BinSearchRow(testintvec, 11, 0, 6, 8);
  EXPECT_EQ(pos, 6);
  pos = ELLmatrix2::BinSearchRow(testintvec, 18, 0, 6, 8);
  EXPECT_EQ(pos, -1);

  free(testintvec);
  delete(testmatrix);
}

////////////////////////////////////////////////////////////////////////////////
TEST(SPRSearchTest, Ell2_linear_search) {
  CheckMemory chk;
  SPRmatrix* testmatrix = SPRmatrix::CreateMatrix(8, SPRmatrix::DEN);
  int* testintvec = (int*)calloc(16, sizeof(int));

  testintvec[0] = 0;
  testintvec[1] = 2;
  testintvec[2] = 3;
  testintvec[3] = 5;
  testintvec[4] = 7;
  testintvec[5] = 8;
  testintvec[6] = 11;
  testintvec[7] = 16;
  testintvec[8] = 18;

  int pos;
  pos = ELLmatrix2::LinSearchRow(testintvec, 0,  0, 9, 9);
  EXPECT_EQ(pos, 0);
  pos = ELLmatrix2::LinSearchRow(testintvec, 3,  0, 9, 9);
  EXPECT_EQ(pos, 2);
  pos = ELLmatrix2::LinSearchRow(testintvec, 4,  0, 9, 9);
  EXPECT_EQ(pos, 3);
  pos = ELLmatrix2::LinSearchRow(testintvec, 22, 0, 9, 9);
  EXPECT_EQ(pos, -1);
  pos = ELLmatrix2::LinSearchRow(testintvec, 1,  0, 9, 9);
  EXPECT_EQ(pos, 1);
  pos = ELLmatrix2::LinSearchRow(testintvec, -1, 0, 9, 9);
  EXPECT_EQ(pos, 0);
  pos = ELLmatrix2::LinSearchRow(testintvec, 11, 0, 7, 9);
  EXPECT_EQ(pos, 6);
  pos = ELLmatrix2::LinSearchRow(testintvec, 18, 0, 7, 9);
  EXPECT_EQ(pos, -1);

  free(testintvec);
  delete(testmatrix);
}
